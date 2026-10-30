//! The LPRS HTTP API: `POST /lprs/api/{method}` with a JSON body, authorised by a stream group token.

use std::net::SocketAddr;
use std::sync::Arc;

use axum::Router;
use axum::body::Bytes;
use axum::extract::{ConnectInfo, Path, State};
use axum::http::{HeaderMap, header};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use chrono::{DateTime, Utc};
use pg_cluster::HostType;
use serde_json::{Map, Value, json};

use super::Lprs;
use super::caches::vstream_key;
use super::obb;
use super::workflow;
use crate::converters;
use crate::http_api::{ApiError, ApiResult, data_response, no_content, require_member};
use crate::fetch::{FetchError, fetch_image};
use crate::imaging;
use crate::log;
use crate::logging::{Level, StreamLog};
use crate::stream_log;

pub fn router(lprs: Arc<Lprs>) -> Router {
    Router::new().route("/lprs/api/{method}", post(handle)).with_state(lprs)
}


async fn handle(
    State(lprs): State<Arc<Lprs>>,
    Path(method): Path<String>,
    ConnectInfo(remote): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let body = if body.is_empty() {
        Ok(Value::Null)
    } else {
        serde_json::from_slice::<Value>(&body)
    };
    let body = match body {
        Ok(body) => body,
        Err(e) => return ApiError::bad_request(format!("Invalid JSON body: {e}")).into_response(),
    };
    let text = body.to_string();
    let clipped: String = text.chars().take(300).collect();
    log!(lprs.logger, Level::Info, "API call from: {remote};  method: {method};  body: {clipped}");

    let id_group = match authorize(&lprs, &headers) {
        Ok(id_group) => id_group,
        Err(e) => return e.into_response(),
    };
    let result = match method.as_str() {
        "addStream" => add_stream(&lprs, id_group, &body).await,
        "removeStream" => remove_stream(&lprs, id_group, &body).await,
        "startWorkflow" => stream_id(&body).map(|ext_id| {
            workflow::start_workflow(&lprs, vstream_key(id_group, &ext_id));
            no_content()
        }),
        "stopWorkflow" => stream_id(&body).map(|ext_id| {
            workflow::stop_workflow(&lprs, &vstream_key(id_group, &ext_id), false);
            no_content()
        }),
        "listStreams" => list_streams(&lprs, id_group).await,
        "getEventData" => get_event_data(&lprs, id_group, &body).await,
        "setStreamDefaultConfig" => set_stream_default_config(&lprs, id_group, &body).await,
        "getStreamDefaultConfig" => get_stream_default_config(&lprs, id_group).await,
        "getSupportedPlateNumberFormats" => Ok(data_response(supported_plate_number_formats())),
        "detectVehiclesObb" => detect_vehicles_obb(&lprs, id_group, &body).await,
        _ => Err(ApiError::bad_request("Method not found")),
    };
    result.unwrap_or_else(IntoResponse::into_response)
}

fn authorize(lprs: &Lprs, headers: &HeaderMap) -> Result<i32, ApiError> {
    let auth = headers
        .get(header::AUTHORIZATION)
        .map(|value| value.to_str().unwrap_or_default())
        .unwrap_or_default();
    let id_group = if auth.is_empty() {
        lprs.allow_group_id_without_auth
    } else {
        let token = auth.strip_prefix("Bearer ").ok_or_else(ApiError::unauthorized)?;
        lprs.groups.get().get(token).copied().unwrap_or(-1)
    };
    if id_group <= 0 {
        return Err(ApiError::unauthorized());
    }
    Ok(id_group)
}

fn stream_id(body: &Value) -> Result<String, ApiError> {
    Ok(converters::to_string(Some(require_member(body, "streamId")?), ""))
}

async fn add_stream(lprs: &Lprs, id_group: i32, body: &Value) -> ApiResult {
    let ext_id = stream_id(body)?;
    let config = match body.get("config") {
        None => Value::Null,
        Some(config @ Value::Object(_)) => config.clone(),
        Some(_) => return Err(ApiError::bad_request("Invalid member `config`.")),
    };

    let mut client = lprs.cluster.client(HostType::Master).await?;
    let transaction = client.transaction().await?;
    let existing = transaction
        .query_opt("select id_vstream from vstreams where id_group = $1 and ext_id = $2", &[&id_group, &ext_id])
        .await?;
    match existing {
        None => {
            transaction
                .execute(
                    "insert into vstreams(id_group, ext_id, config) values($1, $2, $3)",
                    &[&id_group, &ext_id, &config],
                )
                .await?
        }
        Some(row) => {
            let id_vstream: i32 = row.get(0);
            transaction
                .execute("update vstreams set config = $1 where id_vstream = $2", &[&config, &id_vstream])
                .await?
        }
    };
    transaction.commit().await?;
    Ok(no_content())
}

async fn remove_stream(lprs: &Lprs, id_group: i32, body: &Value) -> ApiResult {
    let ext_id = stream_id(body)?;
    lprs.cluster
        .execute(
            HostType::Master,
            "delete from vstreams where id_group = $1 and ext_id = $2",
            &[&id_group, &ext_id],
        )
        .await?;
    Ok(no_content())
}

async fn list_streams(lprs: &Lprs, id_group: i32) -> ApiResult {
    let rows = lprs
        .cluster
        .query(
            HostType::Master,
            "select ext_id, config from vstreams where id_group = $1 order by ext_id",
            &[&id_group],
        )
        .await?;
    let streams = rows
        .iter()
        .map(|row| {
            let config: Option<Value> = row.get(1);
            json!({"streamId": row.get::<_, String>(0), "config": config})
        })
        .collect();
    Ok(data_response(Value::Array(streams)))
}

fn event_data(info: Option<Value>, log_date: DateTime<Utc>) -> Value {
    let mut data = match info {
        Some(Value::Object(info)) => info,
        _ => Map::new(),
    };
    data.insert("date".to_string(), Value::String(converters::format_time(log_date)));
    Value::Object(data)
}

async fn get_event_data(lprs: &Lprs, id_group: i32, body: &Value) -> ApiResult {
    if let Some(event_id) = body.get("eventId") {
        let event_id = event_id.as_i64().ok_or_else(ApiError::server_error)?;
        let rows = lprs
            .cluster
            .query(
                HostType::Master,
                "select log_date, info, id_vstream from events_log where id_event = $1::bigint",
                &[&event_id],
            )
            .await?;
        let Some(row) = rows.first() else {
            return Ok(no_content());
        };
        // An event is visible only to the stream group of its video stream
        let id_vstream: i32 = row.get(2);
        if lprs.vstream_group.get().get(&id_vstream) != Some(&id_group) {
            return Ok(no_content());
        }
        return Ok(data_response(event_data(row.get(1), row.get(0))));
    }

    if body.get("streamId").is_none() && body.get("date").is_none() {
        return Err(ApiError::bad_request(
            "Member `eventId` or both `streamId` and `date` must exist in the request.",
        ));
    }
    let ext_id = stream_id(body)?;
    let date = require_member(body, "date")?.as_str().ok_or_else(ApiError::server_error)?;
    let Some((id_vstream, before, after)) = lprs.vstreams_config.get().get(&vstream_key(id_group, &ext_id)).map(|entry| {
        let config = &entry.config;
        (entry.id_vstream, config.event_log_before.as_millis() as i64, config.event_log_after.as_millis() as i64)
    }) else {
        return Ok(no_content());
    };
    let rows = lprs
        .cluster
        .query(
            HostType::Master,
            r#"
            select
              log_date,
              info
            from
              events_log
            where
              id_vstream = $1
              and (log_date > $2::text::timestamptz - $3::bigint * interval '1 millisecond')
              and (log_date < $2::text::timestamptz + $4::bigint * interval '1 millisecond')
            order by
              abs(extract(epoch from (log_date - $2::text::timestamptz)))
            limit
              1
            "#,
            &[&id_vstream, &date, &before, &after],
        )
        .await?;
    Ok(match rows.first() {
        Some(row) => data_response(event_data(row.get(1), row.get(0))),
        None => no_content(),
    })
}

async fn set_stream_default_config(lprs: &Lprs, id_group: i32, body: &Value) -> ApiResult {
    if !body.is_object() {
        return Err(ApiError::bad_request("Body is not a valid JSON object."));
    }
    let result = lprs
        .cluster
        .execute(
            HostType::Master,
            "update default_vstream_config set config = coalesce(config, $2) || $2 where id_group = $1",
            &[&id_group, body],
        )
        .await;
    if let Err(e) = result {
        log!(lprs.logger, Level::Error, "{e}");
        return Err(ApiError::server_error());
    }
    Ok(no_content())
}

async fn get_stream_default_config(lprs: &Lprs, id_group: i32) -> ApiResult {
    let rows = lprs
        .cluster
        .query(
            HostType::Master,
            "select config from default_vstream_config where id_group = $1",
            &[&id_group],
        )
        .await
        .inspect_err(|e| log!(lprs.logger, Level::Error, "{e}"))?;
    let config = rows.first().and_then(|row| row.get::<_, Option<Value>>(0));
    Ok(data_response(config.unwrap_or(Value::Null)))
}

fn supported_plate_number_formats() -> Value {
    json!([{"countryCode": "ru", "regex": "^[ABCEHKMOPTXY][0-9]{3}[ABCEHKMOPTXY]{2}[0-9]{2,3}$"}])
}

/// An integer given as a JSON number or a numeric string.
fn to_int(value: &Value) -> Option<i64> {
    match value {
        Value::Number(number) => number.as_i64(),
        Value::String(text) => text.trim().parse().ok(),
        _ => None,
    }
}

async fn detect_vehicles_obb(lprs: &Lprs, id_group: i32, body: &Value) -> ApiResult {
    let url = converters::to_string(Some(require_member(body, "screenshotUrl")?), "");
    let image_size = match body.get("imageSize") {
        None => None,
        Some(value) => match to_int(value) {
            Some(size @ 160..=1024) => Some(size as u32),
            _ => return Err(ApiError::bad_request("Member `imageSize` must be between 160 and 1024")),
        },
    };
    let confidence_threshold = match body.get("confidenceThreshold") {
        None => None,
        Some(value) => match converters::to_f32(Some(value), -1.0) {
            threshold if (0.0..=1.0).contains(&threshold) => Some(threshold),
            _ => return Err(ApiError::bad_request("Member `confidenceThreshold` must be between 0.0 and 1.0")),
        },
    };

    // The stream group's default config
    let Some(entry) = lprs.vstreams_config.get().get(&id_group.to_string()).cloned() else {
        return Ok(no_content());
    };
    let mut config = entry.config;
    if let Some(size) = image_size {
        config.ovd_net.input_width = size;
        config.ovd_net.input_height = size;
    }
    if let Some(threshold) = confidence_threshold {
        config.vehicle_confidence = threshold;
    }

    let key = id_group.to_string();
    let stream = StreamLog {
        logger: &lprs.logger,
        key: &key,
        level: config.logs_level,
        always: false,
    };
    if config.logs(Level::Debug) {
        let frame_url = if url.starts_with("data:") { "data:base64..." } else { url.as_str() };
        log!(lprs.logger, Level::Debug, "Start detectVehiclesObb: vstream_key = {key};  frame_url = {frame_url}");
    }
    stream_log!(stream, Level::Trace, "before image acquisition");
    let data = fetch_image(&lprs.http, &url, config.max_capture_error_count, config.capture_timeout)
        .await
        .map_err(|e| match e {
            FetchError::Base64 => {
                if config.logs(Level::Error) {
                    log!(lprs.logger, Level::Error, "Error decoding image from BASE64: vstream_key = {key};");
                }
                ApiError::bad_request("Failed to decode Base64-encoded image from parameter screenshotUrl")
            }
            FetchError::Http(e) => {
                stream_log!(stream, Level::Error, "url = {url};  {e}");
                ApiError::bad_request("Failed to fetch image from parameter screenshotUrl")
            }
        })?;
    stream_log!(stream, Level::Trace, "after image acquisition");
    stream_log!(stream, Level::Trace, "image size = {} bytes", data.len());
    stream_log!(stream, Level::Trace, "before decoding the image");
    let image = tokio::task::block_in_place(|| imaging::decode(&data)).map_err(|e| {
        stream_log!(stream, Level::Error, "{e:#}");
        ApiError::server_error()
    })?;
    stream_log!(stream, Level::Trace, "after decoding the image");
    stream_log!(stream, Level::Trace, "before doInferenceOvdNet");
    let vehicles = match obb::detect(&lprs.triton, &image, &config, stream).await {
        Ok(vehicles) => vehicles,
        Err(e) => {
            log!(lprs.logger, Level::Error, "Error! Unable to send inference request: {e:#}");
            Vec::new()
        }
    };
    stream_log!(stream, Level::Trace, "after doInferenceOvdNet");
    if config.logs(Level::Debug) {
        log!(lprs.logger, Level::Debug, "End detectVehiclesObb: vstream_key = {key};");
    }

    let vehicles = vehicles
        .iter()
        .map(|v| {
            json!({
                "classId": v.class_id,
                "className": obb::CLASSES.get(v.class_id).copied().unwrap_or_default(),
                "confidence": f64::from(v.confidence),
                "box": v.bbox.map(|c| c as i32),
                "angle": f64::from(v.angle),
                "corners": v.corners.as_flattened().iter().map(|&c| c as i32).collect::<Vec<_>>(),
            })
        })
        .collect();
    Ok(data_response(Value::Array(vehicles)))
}
