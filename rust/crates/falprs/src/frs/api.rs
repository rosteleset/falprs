//! The FRS HTTP API: `POST /frs/api/{method}` with a JSON body, authorised by a stream group token
//! or, for `sg*` methods, by a special group token; `GET /frs/api/motionDetection` with query
//! arguments when the handler is configured to accept GET.

use std::collections::{HashMap, HashSet, VecDeque};
use std::net::SocketAddr;
use std::path::{Path as FsPath, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use axum::Router;
use axum::body::Bytes;
use axum::extract::{ConnectInfo, Path, Query, State};
use axum::http::{HeaderMap, header};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use base64::Engine;
use chrono::{DateTime, Local, NaiveDate, TimeZone, Utc};
use pg_cluster::HostType;
use serde_json::{Map, Value, json};

use super::Frs;
use super::caches::{descriptor_from_bytes, vstream_key};
use super::events::{EVENT_DATA_SIZE, EventData, cosine_similarity};
use super::workflow::{self, TaskData, TaskType};
use crate::converters;
use crate::http_api::{ApiError, ApiResult, data_response, no_content, require_member};
use crate::log;
use crate::logging::Level;

pub fn router(frs: Arc<Frs>) -> Router {
    let route = if frs.api_allows_get {
        post(handle_post).get(handle_get)
    } else {
        post(handle_post)
    };
    Router::new().route("/frs/api/{method}", route).with_state(frs)
}

const UNKNOWN_METHOD: &str = "Unknown API method";

/// Turns a database error into the internal server error answer, logging it.
fn db_error(frs: &Frs) -> impl Fn(pg_cluster::Error) -> ApiError + '_ {
    move |e| {
        log!(frs.logger, Level::Error, "{e}");
        ApiError::server_error()
    }
}

fn pg_error(frs: &Frs) -> impl Fn(pg_cluster::tokio_postgres::Error) -> ApiError + '_ {
    move |e| {
        log!(frs.logger, Level::Error, "{}", pg_cluster::error_message(&e));
        ApiError::server_error()
    }
}

fn ok_response() -> Response {
    crate::http_api::json_response(axum::http::StatusCode::OK, json!({"code": "200", "message": "Ok"}))
}

async fn handle_get(
    State(frs): State<Arc<Frs>>,
    Path(method): Path<String>,
    Query(args): Query<HashMap<String, String>>,
    ConnectInfo(remote): ConnectInfo<SocketAddr>,
) -> Response {
    if method != "motionDetection" {
        return ApiError::bad_request(UNKNOWN_METHOD).into_response();
    }
    let arg = |name: &str| args.get(name).cloned().unwrap_or_default();
    let token = arg("token");
    let mut id_group = frs.local.allow_group_id_without_auth;
    if token.is_empty() && id_group <= 0 {
        return ApiError::unauthorized().into_response();
    }
    if !token.is_empty() {
        id_group = frs.groups.get().get(&token).copied().unwrap_or(-1);
    }
    if id_group <= 0 {
        return ApiError::unauthorized().into_response();
    }

    let mut params = Map::new();
    params.insert("streamId".to_string(), Value::String(arg("streamId")));
    params.insert("start".to_string(), Value::String(arg("start")));
    if !arg("duration").is_empty() {
        params.insert("duration".to_string(), Value::String(arg("duration")));
    }
    log!(
        frs.logger,
        Level::Info,
        "API call from: {remote};  method: {method};  streamId = {};  start = {};  duration = {}",
        arg("streamId"),
        arg("start"),
        arg("duration")
    );
    match motion_detection(&frs, id_group, &Value::Object(params)) {
        Ok(()) => ok_response(),
        Err(e) => e.into_response(),
    }
}

/// The token of an `Authorization: Bearer <token>` header; any other form is unauthorised.
fn bearer_token(auth: &str) -> Result<&str, ApiError> {
    match auth.split_once(' ') {
        Some(("Bearer", token)) => Ok(token),
        _ => Err(ApiError::unauthorized()),
    }
}

async fn handle_post(
    State(frs): State<Arc<Frs>>,
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
    log!(frs.logger, Level::Info, "API call from: {remote};  method: {method};  body: {clipped}");

    let auth = headers
        .get(header::AUTHORIZATION)
        .map(|value| value.to_str().unwrap_or_default())
        .unwrap_or_default();
    let result = if method.starts_with("sg") {
        handle_sg_method(&frs, &method, auth, &body).await
    } else {
        handle_method(&frs, &method, auth, &body).await
    };
    result.unwrap_or_else(IntoResponse::into_response)
}

async fn handle_sg_method(frs: &Arc<Frs>, method: &str, auth: &str, body: &Value) -> ApiResult {
    if auth.is_empty() {
        return Err(ApiError::unauthorized());
    }
    let token = bearer_token(auth)?;
    let id_sgroup = frs.sg_config.get().by_token.get(token).map_or(-1, |sg| sg.id_special_group);
    if id_sgroup <= 0 {
        return Err(ApiError::unauthorized());
    }
    match method {
        "sgDeleteFaces" => sg_delete_faces(frs, id_sgroup, body).await.map(|()| no_content()),
        "sgUpdateGroup" => sg_update_group(frs, id_sgroup, body).await.map(|()| no_content()),
        "sgRegisterFace" => sg_register_face(frs, id_sgroup, body).await.map(data_response),
        "sgListFaces" => sg_list_faces(frs, id_sgroup).await.map(data_response),
        "sgRenewToken" => sg_renew_token(frs, id_sgroup).await.map(data_response),
        "sgSearchFaces" => sg_search_faces(frs, id_sgroup, body).await.map(data_response),
        _ => Err(ApiError::bad_request(UNKNOWN_METHOD)),
    }
}

async fn handle_method(frs: &Arc<Frs>, method: &str, auth: &str, body: &Value) -> ApiResult {
    let mut id_group = frs.local.allow_group_id_without_auth;
    if auth.is_empty() && id_group <= 0 {
        return Err(ApiError::unauthorized());
    }
    if !auth.is_empty() {
        let token = bearer_token(auth)?;
        id_group = frs.groups.get().get(token).copied().unwrap_or(-1);
    }
    if id_group <= 0 {
        return Err(ApiError::unauthorized());
    }

    let done = |result: Result<(), ApiError>| result.map(|()| no_content());
    let data = |result: Result<Value, ApiError>| result.map(data_response);
    match method {
        "addStream" => done(add_stream(frs, id_group, body).await),
        "motionDetection" => done(motion_detection(frs, id_group, body)),
        "doorIsOpen" => done(door_is_open(frs, id_group, body)),
        "addFaces" => done(add_faces(frs, id_group, body).await),
        "removeFaces" => done(remove_faces(frs, id_group, body).await),
        "removeStream" => done(remove_stream(frs, id_group, body).await),
        "deleteFaces" => done(delete_faces(frs, id_group, body).await),
        "testImage" => done(test_image(frs, id_group, body).await),
        "updateSpecialGroup" => done(update_special_group(frs, id_group, body).await),
        "deleteSpecialGroup" => done(delete_special_group(frs, id_group, body).await),
        "saveDnnStatsData" => {
            workflow::save_dnn_stats(frs).await;
            Ok(no_content())
        }
        "setCommonConfig" => done(set_config(frs, id_group, body, "common_config").await),
        "setStreamDefaultConfig" => done(set_config(frs, id_group, body, "default_vstream_config").await),
        "listStreams" => data(list_streams(frs, id_group).await),
        "bestQuality" => data(best_quality(frs, id_group, body).await),
        "listAllFaces" => data(list_all_faces(frs, id_group).await),
        "clusterFacesBySimilarity" => data(cluster_faces_by_similarity(frs, id_group, body).await),
        "getEvents" => data(get_events(frs, id_group, body).await),
        "registerFace" => data(register_face(frs, id_group, body).await),
        "processFrame" => data(process_frame(frs, id_group, body).await),
        "addSpecialGroup" => data(add_special_group(frs, id_group, body).await),
        "listSpecialGroups" => data(list_special_groups(frs, id_group).await),
        "getCommonConfig" => data(get_config(frs, id_group, "common_config").await),
        "getStreamDefaultConfig" => data(get_config(frs, id_group, "default_vstream_config").await),
        "getBarcodeEvent" => data(get_barcode_event(frs, id_group, body).await),
        "getAdditionalFeatures" => data(Ok(json!({"qrCodeRecognition": true, "faceClustering": true}))),
        _ => Err(ApiError::bad_request(UNKNOWN_METHOD)),
    }
}

/// The member must be present and be a non-empty array.
fn require_array<'a>(body: &'a Value, member: &str) -> Result<&'a Vec<Value>, ApiError> {
    let value = body
        .get(member)
        .ok_or_else(|| ApiError::bad_request(format!("Required array member `{member}` not found.")))?;
    match value {
        Value::Null => Err(ApiError::bad_request(format!("Member `{member}` must not be null."))),
        Value::Object(_) => Err(ApiError::bad_request(format!("Member `{member}` must not be an object."))),
        Value::Array(items) if items.is_empty() => {
            Err(ApiError::bad_request(format!("Array member `{member}` must not be empty.")))
        }
        Value::Array(items) => Ok(items),
        _ => Err(ApiError::bad_request(format!("Member `{member}` must be an array."))),
    }
}

/// Positive face ids from an array of numbers or numeric strings; other items are skipped.
fn face_ids(items: &[Value]) -> Vec<i32> {
    items
        .iter()
        .map(|item| converters::to_i64(Some(item), 0))
        .filter(|&id| id > 0 && id <= i64::from(i32::MAX))
        .map(|id| id as i32)
        .collect()
}

/// Face ids of an optional `faces` member, which must be an array if present.
fn optional_face_ids(body: &Value) -> Result<Vec<i32>, ApiError> {
    match body.get("faces") {
        None => Ok(Vec::new()),
        Some(Value::Array(items)) => Ok(face_ids(items)),
        Some(_) => Err(ApiError::bad_request("Member `faces` must be an array.")),
    }
}

fn stream_id(body: &Value) -> Result<String, ApiError> {
    Ok(converters::to_string(Some(require_member(body, "streamId")?), ""))
}

/// An optional string member; anything but a string or null is an error, as with userver's
/// `As<std::optional<std::string>>`.
fn optional_string(body: &Value, member: &str) -> Result<Option<String>, ApiError> {
    match body.get(member) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(text)) => Ok(Some(text.clone())),
        Some(_) => Err(ApiError::server_error()),
    }
}

/// An optional integer member with a default; anything but an integer or null is an error.
fn optional_int(body: &Value, member: &str, default: i32) -> Result<i32, ApiError> {
    match body.get(member) {
        None | Some(Value::Null) => Ok(default),
        Some(value) => value
            .as_i64()
            .and_then(|v| i32::try_from(v).ok())
            .ok_or_else(ApiError::server_error),
    }
}

/// A required string member: present, a non-empty scalar, and a string.
fn required_string(body: &Value, member: &str) -> Result<String, ApiError> {
    require_member(body, member)?
        .as_str()
        .map(str::to_string)
        .ok_or_else(ApiError::server_error)
}

/// The stream group's id is an integer member that fits 32 bits.
fn is_int(value: &Value) -> bool {
    value.as_i64().is_some_and(|v| i32::try_from(v).is_ok())
}

async fn vstream_id(frs: &Frs, id_group: i32, vstream_ext: &str) -> Result<i32, ApiError> {
    let row = frs
        .cluster
        .query_opt(
            HostType::Master,
            "select id_vstream from video_streams where id_group = $1 and vstream_ext = $2 and not flag_deleted",
            &[&id_group, &vstream_ext],
        )
        .await
        .map_err(db_error(frs))?;
    Ok(row.map_or(-1, |row| row.get(0)))
}

async fn vstream_ext(frs: &Frs, id_group: i32, id_vstream: i32) -> Result<String, ApiError> {
    let row = frs
        .cluster
        .query_opt(
            HostType::Master,
            "select vstream_ext from video_streams where id_group = $1 and id_vstream = $2 and not flag_deleted",
            &[&id_group, &id_vstream],
        )
        .await
        .map_err(db_error(frs))?;
    Ok(row.map(|row| row.get(0)).unwrap_or_default())
}

/// Converts a video stream config given in the old API's `params` form.
fn config_from_params(params: &[Value]) -> Result<Option<Value>, ApiError> {
    const INT_PARAMS: [&str; 1] = ["max-capture-error-count"];
    const FLOAT_PARAMS: [&str; 8] = [
        "blur",
        "blur-max",
        "tolerance",
        "title-height-ratio",
        "face-confidence",
        "face-enlarge-scale",
        "face-class-confidence",
        "margin",
    ];
    const STRING_PARAMS: [&str; 6] = [
        "osd-datetime-format",
        "dnn-fd-inference-server",
        "dnn-fc-inference-server",
        "dnn-fr-inference-server",
        "dnn-bd-inference-server",
        "title",
    ];
    const TIME_PARAMS: [&str; 8] = [
        "best-quality-interval-after",
        "best-quality-interval-before",
        "capture-timeout",
        "delay-after-error",
        "delay-between-frames",
        "open-door-duration",
        "workflow-timeout",
        "unknown-descriptor-ttl",
    ];
    const BOOL_PARAMS: [&str; 3] = ["flag-spawned-descriptors", "flag-process-faces", "flag-process-barcodes"];

    let invalid = |name: &str| ApiError::bad_request(format!("Invalid value of parameter `{name}`."));
    let mut config = Map::new();
    for param in params {
        let name = param
            .get("paramName")
            .and_then(Value::as_str)
            .ok_or_else(|| ApiError::bad_request("Member `paramName` must be a string."))?;
        let value = param.get("paramValue").unwrap_or(&Value::Null);
        let converted = if INT_PARAMS.contains(&name) {
            json!(value.as_i64().and_then(|v| i32::try_from(v).ok()).ok_or_else(|| invalid(name))?)
        } else if FLOAT_PARAMS.contains(&name) {
            json!(value.as_f64().ok_or_else(|| invalid(name))? as f32)
        } else if STRING_PARAMS.contains(&name) {
            json!(value.as_str().ok_or_else(|| invalid(name))?)
        } else if TIME_PARAMS.contains(&name) {
            let seconds = value.as_f64().ok_or_else(|| invalid(name))? as f32;
            json!(format!("{}ms", (f64::from(seconds) * 1000.0).round() as i32))
        } else if BOOL_PARAMS.contains(&name) {
            json!(value.as_bool().ok_or_else(|| invalid(name))?)
        } else if name == "logs-level" {
            let level = value.as_i64().ok_or_else(|| invalid(name))?;
            json!(match level {
                0 => "error",
                2 => "trace",
                _ => "info",
            })
        } else {
            continue;
        };
        config.insert(name.to_string(), converted);
    }
    Ok((!config.is_empty()).then_some(Value::Object(config)))
}

const SQL_ADD_LINK_DESCRIPTOR_VSTREAM: &str = r#"
    insert into link_descriptor_vstream(id_vstream, id_descriptor)
        select $2, f.id_descriptor from face_descriptors f where f.id_descriptor = $3 and f.id_group = $1
    on conflict (id_vstream, id_descriptor) do update set last_updated = now(), flag_deleted = false
"#;

async fn add_stream(frs: &Frs, id_group: i32, body: &Value) -> Result<(), ApiError> {
    let ext = stream_id(body)?;
    let mut url = optional_string(body, "url")?;
    let mut callback_url = optional_string(body, "callback")?;
    let mut callback_url_barcodes = optional_string(body, "callbackBarcodes")?;
    let faces = optional_face_ids(body)?;
    let mut config = match body.get("params") {
        Some(Value::Array(params)) => config_from_params(params)?,
        _ => None,
    };
    if let Some(value @ Value::Object(_)) = body.get("config") {
        config = Some(value.clone());
    }

    let mut client = frs.cluster.client(HostType::Master).await.map_err(db_error(frs))?;
    let result: Result<(), pg_cluster::tokio_postgres::Error> = async {
        let transaction = client.transaction().await?;
        let existing = transaction
            .query_opt(
                "select id_vstream, url, callback_url, callback_url_barcodes from video_streams where id_group = $1 and vstream_ext = $2",
                &[&id_group, &ext],
            )
            .await?;
        let id_vstream: i32 = match existing {
            None => {
                transaction
                    .query_one(
                        "insert into video_streams(id_group, vstream_ext, url, callback_url, callback_url_barcodes, config) values($1, $2, $3, $4, $5, $6) returning id_vstream",
                        &[&id_group, &ext, &url, &callback_url, &callback_url_barcodes, &config],
                    )
                    .await?
                    .get(0)
            }
            Some(row) => {
                let id_vstream = row.get(0);
                url = url.or_else(|| row.get(1));
                callback_url = callback_url.or_else(|| row.get(2));
                callback_url_barcodes = callback_url_barcodes.or_else(|| row.get(3));
                transaction
                    .execute(
                        r#"
                        update
                          video_streams
                        set
                          url = $2,
                          callback_url = $3,
                          callback_url_barcodes = $4,
                          flag_deleted = false,
                          config = $5,
                          last_updated = now()
                        where
                          id_group = $1
                          and id_vstream = $6
                        "#,
                        &[&id_group, &url, &callback_url, &callback_url_barcodes, &config, &id_vstream],
                    )
                    .await?;
                id_vstream
            }
        };
        for id_descriptor in &faces {
            transaction
                .execute(SQL_ADD_LINK_DESCRIPTOR_VSTREAM, &[&id_group, &id_vstream, id_descriptor])
                .await?;
        }
        transaction.commit().await
    }
    .await;
    result.map_err(pg_error(frs))
}

async fn list_streams(frs: &Frs, id_group: i32) -> Result<Value, ApiError> {
    let mut client = frs.cluster.client(HostType::Master).await.map_err(db_error(frs))?;
    let (streams, faces) = async {
        let transaction = client.transaction().await?;
        let streams = transaction
            .query(
                "select vstream_ext, url, callback_url, callback_url_barcodes, config from video_streams where id_group = $1 and not flag_deleted",
                &[&id_group],
            )
            .await?;
        let faces = transaction
            .query(
                r#"
                select
                  v.vstream_ext,
                  ldv.id_descriptor
                from
                  video_streams v
                  inner join link_descriptor_vstream ldv
                    on v.id_vstream = ldv.id_vstream
                    and not ldv.flag_deleted
                where
                  v.id_group = $1
                  and not v.flag_deleted
                "#,
                &[&id_group],
            )
            .await?;
        transaction.commit().await?;
        Ok::<_, pg_cluster::tokio_postgres::Error>((streams, faces))
    }
    .await
    .map_err(pg_error(frs))?;

    let mut faces_by_stream: HashMap<String, Vec<i32>> = HashMap::new();
    for row in &faces {
        if let Some(id_descriptor) = row.get::<_, Option<i32>>(1) {
            faces_by_stream.entry(row.get(0)).or_default().push(id_descriptor);
        }
    }
    let streams = streams
        .iter()
        .map(|row| {
            let ext: String = row.get(0);
            let mut stream = Map::new();
            stream.insert("streamId".to_string(), json!(ext));
            let members = [(1, "url"), (2, "callback"), (3, "callbackBarcodes")];
            for (column, member) in members {
                if let Some(text) = row.get::<_, Option<String>>(column) {
                    stream.insert(member.to_string(), json!(text));
                }
            }
            if let Some(config) = row.get::<_, Option<Value>>(4) {
                stream.insert("config".to_string(), config);
            }
            if let Some(faces) = faces_by_stream.get(&ext) {
                stream.insert("faces".to_string(), json!(faces));
            }
            Value::Object(stream)
        })
        .collect();
    Ok(Value::Array(streams))
}

fn motion_detection(frs: &Arc<Frs>, id_group: i32, body: &Value) -> Result<(), ApiError> {
    let ext = stream_id(body)?;
    let start = require_member(body, "start")?;
    let key = vstream_key(id_group, &ext);
    let duration = converters::to_duration(body.get("duration"), Duration::ZERO);
    let start = match start {
        Value::Bool(start) => *start,
        Value::String(text) => text == "t",
        _ => return Err(ApiError::server_error()),
    };
    if start {
        workflow::start_workflow(frs, key, duration);
    } else {
        workflow::stop_workflow(frs, &key, false);
    }
    Ok(())
}

fn door_is_open(frs: &Arc<Frs>, id_group: i32, body: &Value) -> Result<(), ApiError> {
    let ext = stream_id(body)?;
    workflow::stop_workflow(frs, &vstream_key(id_group, &ext), false);
    Ok(())
}

/// An integer given as a JSON number or a numeric string, like `convertToInt`.
fn to_int(value: Option<&Value>) -> Option<i32> {
    match value? {
        Value::Number(number) => number.as_i64().and_then(|v| i32::try_from(v).ok()),
        Value::String(text) => text.trim().parse().ok(),
        Value::Bool(b) => Some(i32::from(*b)),
        _ => None,
    }
}

fn is_present(body: &Value, member: &str) -> bool {
    body.get(member).is_some_and(|value| !value.is_null())
}

async fn best_quality(frs: &Frs, id_group: i32, body: &Value) -> Result<Value, ApiError> {
    let id_log = to_int(body.get("eventId"));
    if id_log.is_none() && !(is_present(body, "streamId") && is_present(body, "date")) {
        return Err(ApiError::bad_request(
            "Required members `eventId` or `streamId` and `date` not found or invalid.",
        ));
    }
    let ext = is_present(body, "streamId").then(|| converters::to_string(body.get("streamId"), ""));
    let ext_event_uuid = match body.get("uuid") {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(text)) => text.clone(),
        Some(_) => return Err(ApiError::server_error()),
    };
    let do_copy_event_data = frs
        .config
        .get()
        .common
        .get(&id_group)
        .is_some_and(|common| common.flag_copy_event_data);

    let row = match id_log {
        Some(id_log) => frs
            .cluster
            .query_opt(
                HostType::Master,
                r#"
                select
                  l.id_log,
                  l.screenshot_url,
                  l.face_left,
                  l.face_top,
                  l.face_width,
                  l.face_height,
                  l.copy_data
                from
                  log_faces l
                  join video_streams vs
                    on vs.id_vstream = l.id_vstream
                where
                  vs.id_group = $1
                  and l.id_log = $2
                "#,
                &[&id_group, &id_log],
            )
            .await
            .map_err(db_error(frs))?,
        None => {
            let ext = ext.unwrap_or_default();
            let Some((before, after)) = frs.vstreams_config.get().get(&vstream_key(id_group, &ext)).map(|config| {
                (
                    config.best_quality_interval_before.as_millis() as i64,
                    config.best_quality_interval_after.as_millis() as i64,
                )
            }) else {
                return Ok(Value::Null);
            };
            let id_vstream = vstream_id(frs, id_group, &ext).await?;
            let date = body.get("date").and_then(Value::as_str).ok_or_else(ApiError::server_error)?;
            frs.cluster
                .query_opt(
                    HostType::Master,
                    r#"
                    select
                      l.id_log,
                      l.screenshot_url,
                      l.face_left,
                      l.face_top,
                      l.face_width,
                      l.face_height,
                      l.copy_data
                    from
                      log_faces l
                    where
                      l.id_vstream = $1
                      and (l.log_date >= $2::text::timestamptz - $3::bigint * interval '1 millisecond')
                      and (l.log_date <= $2::text::timestamptz + $4::bigint * interval '1 millisecond')
                      and l.copy_data >= 0
                    order by
                      l.quality desc
                    limit
                      1
                    "#,
                    &[&id_vstream, &date, &before, &after],
                )
                .await
                .map_err(db_error(frs))?
        }
    };
    let Some(row) = row else {
        return Ok(Value::Null);
    };
    let values = (|| {
        Ok::<_, pg_cluster::tokio_postgres::Error>((
            row.try_get::<_, i32>(0)?,
            row.try_get::<_, String>(1)?,
            [row.try_get::<_, i32>(2)?, row.try_get(3)?, row.try_get(4)?, row.try_get(5)?],
            row.try_get::<_, i16>(6)?,
        ))
    })();
    let (id_event_log, screenshot, [left, top, width, height], copy_data) = values.map_err(pg_error(frs))?;
    if do_copy_event_data && copy_data == workflow::COPY_EVENT_DATA_NONE && !ext_event_uuid.is_empty() {
        frs.cluster
            .execute(
                HostType::Master,
                "update log_faces set copy_data = $1, ext_event_uuid = $2 where id_log = $3",
                &[&workflow::COPY_EVENT_DATA_SCHEDULED, &ext_event_uuid, &id_event_log],
            )
            .await
            .map_err(db_error(frs))?;
    }
    Ok(json!({"screenshot": screenshot, "left": left, "top": top, "width": width, "height": height}))
}

async fn add_faces(frs: &Frs, id_group: i32, body: &Value) -> Result<(), ApiError> {
    let ext = stream_id(body)?;
    let faces = face_ids(require_array(body, "faces")?);
    let id_vstream = vstream_id(frs, id_group, &ext).await?;
    if faces.is_empty() || id_vstream <= 0 {
        return Ok(());
    }
    let mut client = frs.cluster.client(HostType::Master).await.map_err(db_error(frs))?;
    async {
        let transaction = client.transaction().await?;
        for id_descriptor in &faces {
            transaction
                .execute(SQL_ADD_LINK_DESCRIPTOR_VSTREAM, &[&id_group, &id_vstream, id_descriptor])
                .await?;
        }
        transaction.commit().await
    }
    .await
    .map_err(pg_error(frs))
}

async fn remove_faces(frs: &Frs, id_group: i32, body: &Value) -> Result<(), ApiError> {
    let ext = stream_id(body)?;
    let faces = face_ids(require_array(body, "faces")?);
    let id_vstream = vstream_id(frs, id_group, &ext).await?;
    if faces.is_empty() || id_vstream <= 0 {
        return Ok(());
    }
    let mut client = frs.cluster.client(HostType::Master).await.map_err(db_error(frs))?;
    async {
        let transaction = client.transaction().await?;
        for id_descriptor in &faces {
            // Rows are only marked as deleted here and removed later by a maintenance task
            transaction
                .execute(
                    "update link_descriptor_vstream set last_updated = now(), flag_deleted = true where id_vstream = $1 and id_descriptor = $2",
                    &[&id_vstream, id_descriptor],
                )
                .await?;
        }
        transaction.commit().await
    }
    .await
    .map_err(pg_error(frs))
}

async fn remove_stream(frs: &Frs, id_group: i32, body: &Value) -> Result<(), ApiError> {
    let ext = stream_id(body)?;
    let id_vstream = vstream_id(frs, id_group, &ext).await?;
    if id_vstream <= 0 {
        return Ok(());
    }
    let mut client = frs.cluster.client(HostType::Master).await.map_err(db_error(frs))?;
    async {
        let transaction = client.transaction().await?;
        transaction
            .execute(
                "update link_descriptor_vstream set last_updated = now(), flag_deleted = true where id_vstream = $1",
                &[&id_vstream],
            )
            .await?;
        transaction
            .execute(
                "update video_streams set last_updated = now(), flag_deleted = true where id_group = $1 and id_vstream = $2",
                &[&id_group, &id_vstream],
            )
            .await?;
        transaction.commit().await
    }
    .await
    .map_err(pg_error(frs))
}

async fn list_all_faces(frs: &Frs, id_group: i32) -> Result<Value, ApiError> {
    let rows = frs
        .cluster
        .query(
            HostType::Master,
            r#"
            select
              id_descriptor
            from
              face_descriptors
            where
              id_group = $1
              and not flag_deleted
              and id_descriptor not in (select ldsg.id_descriptor from link_descriptor_sgroup ldsg)
              and id_parent is null
            "#,
            &[&id_group],
        )
        .await
        .map_err(db_error(frs))?;
    Ok(Value::Array(rows.iter().map(|row| json!(row.get::<_, i32>(0))).collect()))
}

/// Descriptors by id as stored, padded or truncated to the descriptor size like the C++ copy
/// into a fixed-size array.
fn stored_descriptors(rows: &[pg_cluster::tokio_postgres::Row]) -> HashMap<i32, Vec<f32>> {
    rows.iter()
        .map(|row| {
            let bytes: Option<&[u8]> = row.get(1);
            let mut descriptor = descriptor_from_bytes(bytes.unwrap_or_default());
            descriptor.resize(super::events::DESCRIPTOR_SIZE, 0.0);
            (row.get(0), descriptor)
        })
        .collect()
}

async fn cluster_faces_by_similarity(frs: &Frs, id_group: i32, body: &Value) -> Result<Value, ApiError> {
    let items = require_array(body, "faces")?;
    let similarity = require_member(body, "similarity")?;
    let threshold = converters::to_f32(Some(similarity), f32::NAN);
    if !(-1.0..=1.0).contains(&threshold) {
        return Err(ApiError::bad_request(
            "Member `similarity` must be a number within the range [-1.0, 1.0].",
        ));
    }
    let mut seen = HashSet::new();
    let faces: Vec<i32> = face_ids(items).into_iter().filter(|id| seen.insert(*id)).collect();
    if faces.is_empty() {
        return Ok(Value::Array(Vec::new()));
    }

    let rows = frs
        .cluster
        .query(
            HostType::Master,
            "select id_descriptor, descriptor_data from face_descriptors where id_group = $1 and id_descriptor = any($2) and not flag_deleted",
            &[&id_group, &faces],
        )
        .await
        .map_err(db_error(frs))?;
    let descriptors = stored_descriptors(&rows);
    let available: Vec<i32> = faces.into_iter().filter(|id| descriptors.contains_key(id)).collect();

    // Connected components of the similarity graph, found by BFS
    let n = available.len();
    let mut adjacent = vec![Vec::new(); n];
    for i in 0..n {
        for j in i + 1..n {
            if cosine_similarity(&descriptors[&available[i]], &descriptors[&available[j]]) >= f64::from(threshold) {
                adjacent[i].push(j);
                adjacent[j].push(i);
            }
        }
    }
    let mut visited = vec![false; n];
    let mut clusters = Vec::new();
    for i in 0..n {
        if visited[i] {
            continue;
        }
        let mut component = Vec::new();
        let mut queue = VecDeque::from([i]);
        visited[i] = true;
        while let Some(u) = queue.pop_front() {
            component.push(available[u]);
            for &v in &adjacent[u] {
                if !visited[v] {
                    visited[v] = true;
                    queue.push_back(v);
                }
            }
        }
        clusters.push(json!(component));
    }
    Ok(Value::Array(clusters))
}

async fn delete_faces(frs: &Frs, id_group: i32, body: &Value) -> Result<(), ApiError> {
    let faces = face_ids(require_array(body, "faces")?);
    if faces.is_empty() {
        return Ok(());
    }
    let mut client = frs.cluster.client(HostType::Master).await.map_err(db_error(frs))?;
    async {
        let transaction = client.transaction().await?;
        for id_descriptor in &faces {
            // Detach from all video streams; rows are only marked as deleted
            transaction
                .execute(
                    r#"
                    update
                      link_descriptor_vstream
                    set
                      last_updated = now(),
                      flag_deleted = true
                    where
                      id_descriptor = $2
                      and $1 = (select fd.id_group from face_descriptors fd where fd.id_descriptor = $2)
                    "#,
                    &[&id_group, id_descriptor],
                )
                .await?;
            transaction
                .execute(
                    "update face_descriptors set last_updated = now(), flag_deleted = true where id_group = $1 and id_descriptor = $2",
                    &[&id_group, id_descriptor],
                )
                .await?;
            transaction
                .execute(
                    "update face_descriptors set last_updated = now(), flag_deleted = true where id_group = $1 and id_parent = $2",
                    &[&id_group, id_descriptor],
                )
                .await?;
        }
        transaction.commit().await
    }
    .await
    .map_err(pg_error(frs))
}

async fn get_events(frs: &Frs, id_group: i32, body: &Value) -> Result<Value, ApiError> {
    let ext = stream_id(body)?;
    require_member(body, "dateStart")?;
    require_member(body, "dateEnd")?;
    let id_vstream = vstream_id(frs, id_group, &ext).await?;
    let date_start = body["dateStart"].as_str().ok_or_else(ApiError::server_error)?;
    let date_end = body["dateEnd"].as_str().ok_or_else(ApiError::server_error)?;
    let rows = frs
        .cluster
        .query(
            HostType::Master,
            r#"
            select
              l.log_date,
              l.id_descriptor,
              l.quality,
              l.screenshot_url,
              l.face_left,
              l.face_top,
              l.face_width,
              l.face_height
            from
              log_faces l
            where
              l.id_vstream = $1
              and l.log_date >= $2::text::timestamptz
              and l.log_date <= $3::text::timestamptz
            order by
              l.log_date
            "#,
            &[&id_vstream, &date_start, &date_end],
        )
        .await
        .map_err(db_error(frs))?;
    let events = rows
        .iter()
        .map(|row| {
            let log_date: DateTime<Utc> = row.try_get(0)?;
            let mut event = Map::new();
            event.insert("date".to_string(), json!(converters::format_time(log_date)));
            if let Some(id_descriptor) = row.try_get::<_, Option<i32>>(1)? {
                event.insert("faceId".to_string(), json!(id_descriptor));
            }
            event.insert("quality".to_string(), json!(row.try_get::<_, f64>(2)?));
            event.insert("screenshot".to_string(), json!(row.try_get::<_, String>(3)?));
            for (column, member) in [(4, "left"), (5, "top"), (6, "width"), (7, "height")] {
                event.insert(member.to_string(), json!(row.try_get::<_, i32>(column)?));
            }
            Ok(Value::Object(event))
        })
        .collect::<Result<Vec<_>, pg_cluster::tokio_postgres::Error>>()
        .map_err(pg_error(frs))?;
    Ok(Value::Array(events))
}

/// The answer of a successful face registration.
fn registered_face(result: workflow::RegistrationResult) -> Result<Value, ApiError> {
    if result.id_descriptor <= 0 {
        return Err(ApiError::bad_request(result.comments));
    }
    let image = base64::engine::general_purpose::STANDARD.encode(&result.face_image_jpeg);
    Ok(json!({
        "faceId": result.id_descriptor,
        "left": result.face_left,
        "top": result.face_top,
        "width": result.face_width,
        "height": result.face_height,
        "faceImage": format!("data:{};base64,{image}", workflow::MIME_IMAGE),
    }))
}

fn face_rect(body: &Value, task: &mut TaskData) -> Result<(), ApiError> {
    task.face_left = optional_int(body, "left", 0)?;
    task.face_top = optional_int(body, "top", 0)?;
    task.face_width = optional_int(body, "width", 0)?;
    task.face_height = optional_int(body, "height", 0)?;
    Ok(())
}

async fn register_face(frs: &Arc<Frs>, id_group: i32, body: &Value) -> Result<Value, ApiError> {
    let ext = stream_id(body)?;
    let url = required_string(body, "url")?;
    let mut task = TaskData::new(id_group, vstream_key(id_group, &ext), TaskType::RegisterDescriptor, url);
    face_rect(body, &mut task)?;
    registered_face(workflow::process_pipeline(frs, task).await)
}

async fn test_image(frs: &Arc<Frs>, id_group: i32, body: &Value) -> Result<(), ApiError> {
    let ext = stream_id(body)?;
    let url = required_string(body, "url")?;
    let task = TaskData::new(id_group, vstream_key(id_group, &ext), TaskType::Test, url);
    workflow::process_pipeline(frs, task).await;
    Ok(())
}

async fn process_frame(frs: &Arc<Frs>, id_group: i32, body: &Value) -> Result<Value, ApiError> {
    let has_group_id = body.get("groupId").is_some_and(is_int);
    if !(is_present(body, "streamId") || has_group_id) {
        return Err(ApiError::bad_request(
            "Required members `streamId` or `groupId` not found or invalid.",
        ));
    }
    let url = required_string(body, "url")?;
    let key = match body.get("streamId") {
        Some(value) => vstream_key(id_group, &converters::to_string(Some(value), "")),
        None => String::new(),
    };
    let mut task = TaskData::new(id_group, key, TaskType::ProcessFrame, url);
    if task.vstream_key.is_empty() {
        if let Some(id_sgroup) = body.get("groupId") {
            task.id_sgroup = id_sgroup.as_i64().and_then(|v| i32::try_from(v).ok()).ok_or_else(ApiError::server_error)?;
        }
    }
    let result = workflow::process_pipeline(frs, task).await;
    Ok(json!(result.id_descriptors))
}

async fn add_special_group(frs: &Frs, id_group: i32, body: &Value) -> Result<Value, ApiError> {
    let group_name = required_string(body, "groupName")?;
    let mut max_descriptor_count = frs
        .config
        .get()
        .common
        .get(&id_group)
        .map_or(1, |common| common.sg_max_descriptor_count);
    if let Some(value) = body.get("maxDescriptorCount") {
        if !is_int(value) {
            return Err(ApiError::bad_request("Member `maxDescriptorCount` is invalid."));
        }
        let requested = value.as_i64().unwrap_or_default() as i32;
        max_descriptor_count = requested.min(max_descriptor_count).max(1);
    }
    let result = frs
        .cluster
        .query_one(
            HostType::Master,
            "insert into special_groups(id_group, group_name, sg_api_token, max_descriptor_count) values($1, $2, gen_random_uuid(), $3) returning id_special_group, sg_api_token",
            &[&id_group, &group_name, &max_descriptor_count],
        )
        .await;
    match result {
        Ok(row) => Ok(json!({"groupId": row.get::<_, i32>(0), "accessApiToken": row.get::<_, String>(1)})),
        Err(e) => {
            log!(frs.logger, Level::Error, "{e}");
            if e.is_unique_violation() {
                Err(ApiError::bad_request("A special group with this name already exists."))
            } else {
                Err(ApiError::server_error())
            }
        }
    }
}

fn special_group_id(body: &Value) -> Result<i32, ApiError> {
    let value = require_member(body, "groupId")?;
    if !is_int(value) {
        return Err(ApiError::bad_request("Member `groupId` is invalid."));
    }
    Ok(value.as_i64().unwrap_or_default() as i32)
}

async fn update_special_group(frs: &Frs, id_group: i32, body: &Value) -> Result<(), ApiError> {
    let id_sgroup = special_group_id(body)?;
    let group_name = match body.get("groupName") {
        None => None,
        Some(Value::String(name)) => Some(name.clone()),
        Some(_) => return Err(ApiError::bad_request("Member `groupName` is invalid.")),
    };
    let max_descriptor_count = match body.get("maxDescriptorCount") {
        None => None,
        Some(value) if is_int(value) => Some(value.as_i64().unwrap_or_default() as i32),
        Some(_) => return Err(ApiError::bad_request("Member `maxDescriptorCount` is invalid.")),
    };
    frs.cluster
        .execute(
            HostType::Master,
            r#"
            update
              special_groups
            set
              group_name = coalesce($1, group_name),
              max_descriptor_count = coalesce($2, max_descriptor_count)
            where
              id_group = $3
              and id_special_group = $4
            "#,
            &[&group_name, &max_descriptor_count, &id_group, &id_sgroup],
        )
        .await
        .map_err(db_error(frs))?;
    Ok(())
}

async fn delete_special_group(frs: &Frs, id_group: i32, body: &Value) -> Result<(), ApiError> {
    let id_sgroup = special_group_id(body)?;
    let mut client = frs.cluster.client(HostType::Master).await.map_err(db_error(frs))?;
    async {
        let transaction = client.transaction().await?;
        // Rows are only marked as deleted here and removed later by a maintenance task
        transaction
            .execute(
                "update link_descriptor_sgroup set last_updated = now(), flag_deleted = true where id_sgroup = $1",
                &[&id_sgroup],
            )
            .await?;
        transaction
            .execute(
                r#"
                update
                  face_descriptors fd
                set
                  last_updated = now(),
                  flag_deleted = true
                where
                  id_descriptor in (select ldsg.id_descriptor from link_descriptor_sgroup ldsg where ldsg.id_sgroup = $1)
                "#,
                &[&id_sgroup],
            )
            .await?;
        transaction
            .execute(
                "update special_groups set last_updated = now(), flag_deleted = true where id_group = $1 and id_special_group = $2",
                &[&id_group, &id_sgroup],
            )
            .await?;
        transaction.commit().await
    }
    .await
    .map_err(pg_error(frs))
}

async fn list_special_groups(frs: &Frs, id_group: i32) -> Result<Value, ApiError> {
    let rows = frs
        .cluster
        .query(
            HostType::Master,
            "select id_special_group, group_name, sg_api_token, callback_url, max_descriptor_count from special_groups where id_group = $1 and flag_deleted = false",
            &[&id_group],
        )
        .await
        .map_err(db_error(frs))?;
    let groups = rows
        .iter()
        .map(|row| {
            let mut group = Map::new();
            group.insert("groupId".to_string(), json!(row.get::<_, i32>(0)));
            group.insert("groupName".to_string(), json!(row.get::<_, String>(1)));
            group.insert("accessApiToken".to_string(), json!(row.get::<_, String>(2)));
            if let Some(callback) = row.get::<_, Option<String>>(3) {
                group.insert("callback".to_string(), json!(callback));
            }
            group.insert("maxDescriptorCount".to_string(), json!(row.get::<_, i32>(4)));
            Value::Object(group)
        })
        .collect();
    Ok(Value::Array(groups))
}

/// Merges the body into a stream group's config in `table` (`common_config` or `default_vstream_config`).
async fn set_config(frs: &Frs, id_group: i32, body: &Value, table: &str) -> Result<(), ApiError> {
    if !body.is_object() {
        return Err(ApiError::bad_request("Body is not a valid JSON object."));
    }
    let query = format!("update {table} set config = coalesce(config, $2) || $2 where id_group = $1");
    frs.cluster
        .execute(HostType::Master, &query, &[&id_group, body])
        .await
        .map_err(db_error(frs))?;
    Ok(())
}

async fn get_config(frs: &Frs, id_group: i32, table: &str) -> Result<Value, ApiError> {
    let query = format!("select config from {table} where id_group = $1");
    let row = frs
        .cluster
        .query_opt(HostType::Master, &query, &[&id_group])
        .await
        .map_err(db_error(frs))?;
    Ok(row.and_then(|row| row.get::<_, Option<Value>>(0)).unwrap_or(Value::Null))
}

/// The stream group of a special group, from the cache.
fn special_group_owner(frs: &Frs, id_sgroup: i32) -> Result<i32, ApiError> {
    let id_group = frs.sg_config.get().by_id(id_sgroup).map_or(-1, |sg| sg.id_group);
    if id_group <= 0 {
        return Err(ApiError::server_error());
    }
    Ok(id_group)
}

async fn sg_register_face(frs: &Arc<Frs>, id_sgroup: i32, body: &Value) -> Result<Value, ApiError> {
    let url = required_string(body, "url")?;
    let id_group = special_group_owner(frs, id_sgroup)?;
    let mut task = TaskData::new(id_group, String::new(), TaskType::RegisterDescriptor, url);
    task.id_sgroup = id_sgroup;
    face_rect(body, &mut task)?;
    registered_face(workflow::process_pipeline(frs, task).await)
}

async fn sg_delete_faces(frs: &Frs, id_sgroup: i32, body: &Value) -> Result<(), ApiError> {
    let faces = face_ids(require_array(body, "faces")?);
    if faces.is_empty() {
        return Ok(());
    }
    let mut client = frs.cluster.client(HostType::Master).await.map_err(db_error(frs))?;
    async {
        let transaction = client.transaction().await?;
        for id_descriptor in &faces {
            // Rows are only marked as deleted here and removed later by a maintenance task
            transaction
                .execute(
                    "update link_descriptor_sgroup set last_updated = now(), flag_deleted = true where id_sgroup = $1 and id_descriptor = $2",
                    &[&id_sgroup, id_descriptor],
                )
                .await?;
            transaction
                .execute(
                    r#"
                    update
                      face_descriptors fd
                    set
                      last_updated = now(),
                      flag_deleted = true
                    where
                      id_descriptor = $2
                      and id_descriptor in (select ldsg.id_descriptor from link_descriptor_sgroup ldsg where ldsg.id_sgroup = $1)
                    "#,
                    &[&id_sgroup, id_descriptor],
                )
                .await?;
        }
        transaction.commit().await
    }
    .await
    .map_err(pg_error(frs))
}

async fn sg_list_faces(frs: &Frs, id_sgroup: i32) -> Result<Value, ApiError> {
    let rows = frs
        .cluster
        .query(
            HostType::Master,
            r#"
            select
              di.id_descriptor,
              concat('data:', di.mime_type, ';base64,', translate(encode(di.face_image, 'base64'), E'\n', '')) face_image
            from
              link_descriptor_sgroup ldsg
              inner join face_descriptors fd
                on ldsg.id_descriptor = fd.id_descriptor
              inner join descriptor_images di
                on fd.id_descriptor = di.id_descriptor
            where
              ldsg.id_sgroup = $1
              and not ldsg.flag_deleted
              and not fd.flag_deleted
            "#,
            &[&id_sgroup],
        )
        .await
        .map_err(db_error(frs))?;
    Ok(Value::Array(
        rows.iter()
            .map(|row| json!({"faceId": row.get::<_, i32>(0), "faceImage": row.get::<_, String>(1)}))
            .collect(),
    ))
}

async fn sg_update_group(frs: &Frs, id_sgroup: i32, body: &Value) -> Result<(), ApiError> {
    let callback_url = required_string(body, "callback")?;
    frs.cluster
        .execute(
            HostType::Master,
            "update special_groups set callback_url = $2 where id_special_group = $1",
            &[&id_sgroup, &callback_url],
        )
        .await
        .map_err(db_error(frs))?;
    Ok(())
}

async fn sg_renew_token(frs: &Frs, id_sgroup: i32) -> Result<Value, ApiError> {
    let row = frs
        .cluster
        .query_one(
            HostType::Master,
            "update special_groups set sg_api_token = gen_random_uuid() where id_special_group = $1 returning sg_api_token",
            &[&id_sgroup],
        )
        .await
        .map_err(db_error(frs))?;
    Ok(json!({"accessApiToken": row.get::<_, String>(0)}))
}

/// A search hit; results are sorted by date, newest first.
struct SearchResult {
    event_date: String,
    event_id: String,
    uuid: String,
    url_image: String,
    id_descriptor: i32,
    similarity: f64,
}

/// Parses a `%Y-%m-%d` date as midnight UTC, like `absl::ParseTime` without a time zone.
fn parse_date(text: &str) -> Option<DateTime<Utc>> {
    let date = NaiveDate::parse_from_str(text.trim(), "%Y-%m-%d").ok()?;
    Some(Utc.from_utc_datetime(&date.and_hms_opt(0, 0, 0)?))
}

/// Regular files under `dir` with the given extension, recursively.
fn files_with_extension(dir: &FsPath, extension: &str) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut pending = vec![dir.to_path_buf()];
    while let Some(dir) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            let Ok(file_type) = entry.file_type() else { continue };
            let path = entry.path();
            if file_type.is_dir() {
                pending.push(path);
            } else if file_type.is_file() && path.extension().is_some_and(|e| e == extension) {
                files.push(path);
            }
        }
    }
    files
}

/// Event records of a `.dat` file; a trailing partial record is ignored.
fn read_event_data(path: &FsPath) -> Vec<EventData> {
    std::fs::read(path)
        .map(|bytes| bytes.chunks_exact(EVENT_DATA_SIZE).map(EventData::from_bytes).collect())
        .unwrap_or_default()
}

fn read_json(path: &FsPath) -> Option<Value> {
    let bytes = std::fs::read(path).ok().filter(|bytes| !bytes.is_empty())?;
    serde_json::from_slice(&bytes).ok()
}

/// Characters of an event id used as directory levels.
fn id_chars(id: &str) -> Vec<char> {
    id.chars().collect()
}

async fn sg_search_faces(frs: &Arc<Frs>, id_sgroup: i32, body: &Value) -> Result<Value, ApiError> {
    let id_group = special_group_owner(frs, id_sgroup)?;
    let items = require_array(body, "faces")?;
    require_member(body, "dateStart")?;
    require_member(body, "dateEnd")?;
    require_member(body, "similarityThreshold")?;

    let flag_of = |member: &str| match body.get(member) {
        None | Some(Value::Null) => Ok(true),
        Some(Value::Bool(flag)) => Ok(*flag),
        Some(_) => Err(ApiError::bad_request(format!("Member `{member}` must be a boolean."))),
    };
    let (flag_logs, flag_events) = (flag_of("useLogs")?, flag_of("useEvents")?);
    if !flag_logs && !flag_events {
        return Err(ApiError::bad_request(
            "At least one of the members `useLogs` or `useEvents` must be true.",
        ));
    }
    let date_of = |member: &str| {
        body[member]
            .as_str()
            .and_then(parse_date)
            .ok_or_else(|| ApiError::bad_request(format!("Required member `{member}` is invalid.")))
    };
    let date_start = date_of("dateStart")?;
    let date_end = date_of("dateEnd")? + chrono::Duration::hours(24);
    let faces = face_ids(items);
    let threshold = match &body["similarityThreshold"] {
        Value::Number(number) => number.as_f64().unwrap_or(0.5) as f32,
        _ => return Err(ApiError::bad_request("Member `similarityThreshold` must be a number.")),
    };

    let rows = frs
        .cluster
        .query(
            HostType::Master,
            r#"
            select
              f.id_descriptor,
              f.descriptor_data
            from
              link_descriptor_sgroup l
              inner join face_descriptors f
              on f.id_descriptor = l.id_descriptor
            where
              not l.flag_deleted
              and l.id_sgroup = $1
              and f.id_descriptor = any($2)
            "#,
            &[&id_sgroup, &faces],
        )
        .await
        .map_err(db_error(frs))?;
    let descriptors = stored_descriptors(&rows);
    let frs = frs.clone();
    let results = tokio::task::spawn_blocking(move || {
        search_files(&frs, id_group, &descriptors, threshold, (date_start, date_end), (flag_logs, flag_events))
    })
    .await
    .map_err(|_| ApiError::server_error())?;
    Ok(Value::Array(
        results
            .into_iter()
            .map(|r| {
                json!({
                    "date": r.event_date,
                    "uuid": r.uuid,
                    "eventId": r.event_id,
                    "url": r.url_image,
                    "faceId": r.id_descriptor,
                    "similarity": r.similarity,
                })
            })
            .collect(),
    ))
}

/// Searches the copied events and then the face logs of a stream group for the descriptors.
fn search_files(
    frs: &Frs,
    id_group: i32,
    descriptors: &HashMap<i32, Vec<f32>>,
    threshold: f32,
    (date_start, date_end): (DateTime<Utc>, DateTime<Utc>),
    (flag_logs, flag_events): (bool, bool),
) -> Vec<SearchResult> {
    let threshold = f64::from(threshold);
    let mut results = Vec::new();
    let mut event_ids = HashSet::new();
    let local = &frs.local;

    let events_dir = PathBuf::from(format!("{}group_{id_group}/", local.events_path));
    if flag_events && events_dir.exists() {
        // Event files are named by local date and filtered by name
        let file_name = |date: DateTime<Utc>| format!("{}.dat", date.with_timezone(&Local).format("%Y-%m-%d"));
        let (first, last) = (file_name(date_start), file_name(date_end));
        for path in files_with_extension(&events_dir, "dat") {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or_default().to_string();
            if name < first || name > last {
                continue;
            }
            for data in read_event_data(&path) {
                for (&id_descriptor, descriptor) in descriptors {
                    let similarity = cosine_similarity(descriptor, &data.descriptor);
                    if similarity <= threshold {
                        continue;
                    }
                    let event_id = data.event_id.clone();
                    event_ids.insert(event_id.clone());
                    let c = id_chars(&event_id);
                    let group_dir = format!("{}group_{id_group}", local.events_path);
                    let mut json_path =
                        PathBuf::from(format!("{group_dir}/{}/{}/{}/{}/{event_id}.json", c[0], c[1], c[2], c[3]));
                    if !json_path.exists() {
                        // The shorter path of the old project
                        json_path = PathBuf::from(format!("{group_dir}/{}/{}/{}/{event_id}.json", c[0], c[1], c[2]));
                    }
                    let Some(event) = read_json(&json_path) else { continue };
                    let uuid = event.get("event_uuid").and_then(Value::as_str).unwrap_or_default().to_string();
                    let url_image = if uuid.is_empty() {
                        format!(
                            "{}group_{id_group}/{}/{}/{}/{}/{event_id}.jpg",
                            local.screenshots_url_prefix, c[0], c[1], c[2], c[3]
                        )
                    } else {
                        String::new()
                    };
                    let stem = name.trim_end_matches(".dat").to_string();
                    let event_date = event.get("event_date").and_then(Value::as_str).map_or(stem, str::to_string);
                    results.push(SearchResult {
                        event_date,
                        event_id,
                        uuid,
                        url_image,
                        id_descriptor,
                        similarity,
                    });
                }
            }
        }
    }

    let logs_dir = PathBuf::from(format!("{}group_{id_group}/", local.screenshots_path));
    if flag_logs && logs_dir.exists() {
        let (start, end) = (SystemTime::from(date_start), SystemTime::from(date_end));
        for path in files_with_extension(&logs_dir, "dat") {
            let modified = std::fs::metadata(&path).and_then(|m| m.modified());
            if !modified.is_ok_and(|modified| modified >= start && modified < end) {
                continue;
            }
            for data in read_event_data(&path) {
                for (&id_descriptor, descriptor) in descriptors {
                    let similarity = cosine_similarity(descriptor, &data.descriptor);
                    // A log entry already found among the events is skipped
                    if similarity <= threshold || event_ids.contains(&data.event_id) {
                        continue;
                    }
                    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or_default().to_string();
                    let Some(event) = read_json(&path.with_extension("json")) else { continue };
                    let Some(event_date) = event.get("event_date").and_then(Value::as_str) else { continue };
                    let c = id_chars(&stem);
                    if c.len() < 4 {
                        continue;
                    }
                    results.push(SearchResult {
                        event_date: event_date.to_string(),
                        event_id: data.event_id.clone(),
                        uuid: String::new(),
                        url_image: format!(
                            "{}group_{id_group}/{}/{}/{}/{}/{stem}.jpg",
                            local.screenshots_url_prefix, c[0], c[1], c[2], c[3]
                        ),
                        id_descriptor,
                        similarity,
                    });
                }
            }
        }
    }

    results.sort_by(|a, b| b.event_date.cmp(&a.event_date));
    results
}

async fn get_barcode_event(frs: &Frs, id_group: i32, body: &Value) -> Result<Value, ApiError> {
    let event_data = |info: Option<Value>, log_date: DateTime<Utc>| -> Result<Value, ApiError> {
        let mut data = match info {
            None | Some(Value::Null) => Map::new(),
            Some(Value::Object(info)) => info,
            Some(_) => return Err(ApiError::server_error()),
        };
        data.insert("date".to_string(), json!(converters::format_time(log_date)));
        Ok(Value::Object(data))
    };

    if let Some(event_id) = body.get("eventId") {
        let event_id = event_id.as_i64().ok_or_else(ApiError::server_error)?;
        let row = frs
            .cluster
            .query_opt(
                HostType::Master,
                "select log_date, info, id_vstream from log_barcodes where id_log = $1::bigint",
                &[&event_id],
            )
            .await
            .map_err(db_error(frs))?;
        let Some(row) = row else {
            return Ok(Value::Null);
        };
        // An event is visible only to the stream group of its video stream
        let ext = vstream_ext(frs, id_group, row.get(2)).await?;
        if !frs.vstreams_config.get().contains_key(&vstream_key(id_group, &ext)) {
            return Ok(Value::Null);
        }
        return event_data(row.get(1), row.get(0));
    }

    if body.get("streamId").is_none() && body.get("date").is_none() {
        return Err(ApiError::bad_request(
            "Member `eventId` or both `streamId` and `date` must exist in the request.",
        ));
    }
    let ext = stream_id(body)?;
    require_member(body, "date")?;
    let Some((before, after)) = frs.vstreams_config.get().get(&vstream_key(id_group, &ext)).map(|config| {
        (
            config.best_quality_interval_before.as_millis() as i64,
            config.best_quality_interval_after.as_millis() as i64,
        )
    }) else {
        return Ok(Value::Null);
    };
    let id_vstream = vstream_id(frs, id_group, &ext).await?;
    let date = body["date"].as_str().ok_or_else(ApiError::server_error)?;
    let row = frs
        .cluster
        .query_opt(
            HostType::Master,
            r#"
            select
              log_date,
              info
            from
              log_barcodes
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
        .await
        .map_err(db_error(frs))?;
    match row {
        Some(row) => event_data(row.get(1), row.get(0)),
        None => Ok(Value::Null),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn old_api_params() {
        let params = json!([
            {"paramName": "blur", "paramValue": 150},
            {"paramName": "capture-timeout", "paramValue": 1.5},
            {"paramName": "logs-level", "paramValue": 2},
            {"paramName": "title", "paramValue": "Door"},
            {"paramName": "flag-process-faces", "paramValue": false},
            {"paramName": "unknown", "paramValue": 1},
        ]);
        let config = config_from_params(params.as_array().unwrap()).unwrap().unwrap();
        assert_eq!(
            config,
            json!({
                "blur": 150.0,
                "capture-timeout": "1500ms",
                "logs-level": "trace",
                "title": "Door",
                "flag-process-faces": false,
            })
        );
        assert!(config_from_params(&[]).unwrap().is_none());
    }

    #[test]
    fn face_id_lists() {
        assert_eq!(face_ids(&[json!(1), json!("2"), json!(0), json!(-3), json!("x"), json!(4)]), vec![1, 2, 4]);
    }

    #[test]
    fn dates_parse_as_utc_midnight() {
        assert_eq!(parse_date("2024-03-05").unwrap().to_rfc3339(), "2024-03-05T00:00:00+00:00");
        assert!(parse_date("2024-03").is_none());
    }
}
