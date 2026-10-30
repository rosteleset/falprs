//! Video stream workflows: each started video stream is processed frame by frame until it is
//! stopped, times out, or fails without a configured retry delay.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use image::{Rgb, RgbImage};
use pg_cluster::HostType;
use serde_json::{Map, Value, json};

use super::Lprs;
use super::geometry::{BBox, iou};
use super::pipeline::{self, LicensePlate, PLATE_CLASSES, Vehicle};
use super::vstream_config::VStreamConfig;
use crate::converters;
use crate::fetch::{FetchError, fetch_image};
use crate::imaging;
use crate::log;
use crate::logging::{Level, StreamLog};
use crate::stream_log;


/// Workflow settings from the `config` section of the `lprs-workflow` component.
#[derive(Debug, Clone)]
pub struct WorkflowSettings {
    pub ban_maintenance_interval: Duration,
    pub events_log_maintenance_interval: Duration,
    pub events_log_ttl: Duration,
    pub screenshots_path: String,
    pub screenshots_url_prefix: String,
    pub failed_path: String,
    pub failed_ttl: Duration,
}

fn with_trailing_slash(mut path: String) -> String {
    if !path.is_empty() && !path.ends_with('/') {
        path.push('/');
    }
    path
}

impl WorkflowSettings {
    pub fn from_config(config: &Map<String, Value>) -> Self {
        let duration = |key: &str, default: Duration| converters::to_duration(config.get(key), default);
        let string = |key: &str, default: &str| converters::to_string(config.get(key), default);
        let hours = |h: u64| Duration::from_secs(h * 3600);
        Self {
            ban_maintenance_interval: duration("ban-maintenance-interval", Duration::from_secs(5)),
            events_log_maintenance_interval: duration("events-log-maintenance-interval", hours(2)),
            events_log_ttl: duration("events-log-ttl", hours(4)),
            screenshots_path: with_trailing_slash(string("screenshots-path", "/opt/falprs/static/lprs/")),
            screenshots_url_prefix: with_trailing_slash(string("screenshots-url-prefix", "http://localhost:9051/lprs/")),
            failed_path: with_trailing_slash(string("failed-path", "/opt/falprs/static/lprs/failed/")),
            failed_ttl: duration("failed-ttl", hours(60 * 24)),
        }
    }
}

/// A plate ban: until `full_until` the plate is ignored anywhere; until `area_until` it is ignored
/// while it stays at `bbox`.
struct PlateBan {
    full_until: Instant,
    area_until: Instant,
    bbox: BBox,
}

/// Workflow state shared by all video streams.
#[derive(Default)]
pub struct Workflows {
    /// Video streams with a running workflow; false once a stop was requested.
    running: Mutex<HashMap<String, bool>>,
    /// Deadlines of workflows started with a `workflow-timeout`.
    timeouts: Mutex<HashMap<String, Instant>>,
    /// Plate bans by `<video stream key>_<number>`.
    plate_bans: Mutex<HashMap<String, PlateBan>>,
    /// Special vehicle bans by video stream key.
    special_bans: Mutex<HashMap<String, Instant>>,
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

pub fn start_workflow(lprs: &Arc<Lprs>, vstream_key: String) {
    let Some(workflow_timeout) = lprs.vstreams_config.get().get(&vstream_key).map(|e| e.config.workflow_timeout) else {
        return;
    };
    let is_new = lock(&lprs.workflows.running).insert(vstream_key.clone(), true).is_none();
    if !workflow_timeout.is_zero() {
        lock(&lprs.workflows.timeouts).insert(vstream_key.clone(), Instant::now() + workflow_timeout);
    }
    if is_new {
        tokio::spawn(run(lprs.clone(), vstream_key));
    }
}

/// Requests a stop; the workflow ends after its current frame. `internal` stops forget the
/// workflow at once.
pub fn stop_workflow(lprs: &Lprs, vstream_key: &str, internal: bool) {
    {
        let mut running = lock(&lprs.workflows.running);
        if running.contains_key(vstream_key) {
            if internal {
                running.remove(vstream_key);
            } else {
                running.insert(vstream_key.to_string(), false);
            }
        }
    }
    lock(&lprs.workflows.timeouts).remove(vstream_key);
}

/// What to do after a frame.
enum Next {
    After(Duration),
    Stop,
}

async fn run(lprs: Arc<Lprs>, vstream_key: String) {
    loop {
        let delay = match process_frame(&lprs, &vstream_key).await {
            Next::After(delay) => delay,
            Next::Stop => {
                stop_workflow(&lprs, &vstream_key, true);
                return;
            }
        };
        tokio::time::sleep(delay).await;

        let timed_out = {
            let mut timeouts = lock(&lprs.workflows.timeouts);
            let expired = timeouts.get(&vstream_key).is_some_and(|deadline| *deadline < Instant::now());
            if expired {
                timeouts.remove(&vstream_key);
            }
            expired
        };
        let proceed = {
            let mut running = lock(&lprs.workflows.running);
            match running.get(&vstream_key) {
                Some(true) if !timed_out => true,
                Some(_) => {
                    running.remove(&vstream_key);
                    false
                }
                None => false,
            }
        };
        if timed_out {
            log!(lprs.logger, Level::Info, "Stopping a workflow by timeout: vstream_key = {vstream_key};");
        }
        if !proceed {
            return;
        }
    }
}

async fn process_frame(lprs: &Lprs, vstream_key: &str) -> Next {
    let Some(entry) = lprs.vstreams_config.get().get(vstream_key).cloned() else {
        return Next::Stop;
    };
    let config = &entry.config;
    if config.screenshot_url.is_empty() {
        return Next::Stop;
    }
    let after_error = || {
        if config.delay_after_error.is_zero() {
            Next::Stop
        } else {
            if config.logs(Level::Error) {
                log!(lprs.logger, Level::Error, "vstream_key = {vstream_key};  delay for {}ms", config.delay_after_error.as_millis());
            }
            Next::After(config.delay_after_error)
        }
    };

    let stream = StreamLog {
        logger: &lprs.logger,
        key: vstream_key,
        level: config.logs_level,
        always: false,
    };
    if config.logs(Level::Debug) {
        let frame_url = if config.screenshot_url.starts_with("data:") { "data:base64..." } else { config.screenshot_url.as_str() };
        log!(lprs.logger, Level::Debug, "Start processPipeline: vstream_key = {vstream_key};  frame_url = {frame_url}");
    }
    stream_log!(stream, Level::Trace, "before image acquisition");
    let image_data = match fetch_image(&lprs.http, &config.screenshot_url, config.max_capture_error_count, config.capture_timeout).await {
        Ok(data) => data,
        Err(FetchError::Base64) => {
            if config.logs(Level::Error) {
                log!(lprs.logger, Level::Error, "Error decoding image from BASE64: vstream_key = {vstream_key};");
            }
            return Next::Stop;
        }
        Err(FetchError::Http(e)) => {
            if config.logs(Level::Error) {
                log!(lprs.logger, Level::Error, "vstream_key = {vstream_key};  url = {};  {e}", config.screenshot_url);
            }
            return after_error();
        }
    };

    stream_log!(stream, Level::Trace, "after image acquisition");
    stream_log!(stream, Level::Trace, "image size = {} bytes", image_data.len());

    match recognize(lprs, stream, &entry.ext_id, entry.id_vstream, config, &image_data).await {
        Ok(()) => {
            if config.logs(Level::Debug) {
                log!(lprs.logger, Level::Debug, "End processPipeline: vstream_key = {vstream_key};");
            }
            Next::After(config.delay_between_frames)
        }
        Err(e) => {
            if config.logs(Level::Error) {
                log!(lprs.logger, Level::Error, "vstream_key = {vstream_key};  {e:#}");
            }
            after_error()
        }
    }
}

/// Recognises a frame, then records an event and notifies the callback if anything new is found.
async fn recognize(lprs: &Lprs, stream: StreamLog<'_>, ext_id: &str, id_vstream: i32, config: &VStreamConfig, image_data: &[u8]) -> Result<()> {
    let vstream_key = stream.key;
    stream_log!(stream, Level::Trace, "before decoding the image");
    let frame = tokio::task::block_in_place(|| imaging::decode(image_data))?;
    stream_log!(stream, Level::Trace, "after decoding the image");
    let triton = &lprs.triton;

    stream_log!(stream, Level::Trace, "before doInferenceVdNet");
    let mut vehicles = match pipeline::detect_vehicles(triton, &frame, config, stream).await {
        Ok(vehicles) => vehicles,
        Err(e) => {
            log!(lprs.logger, Level::Error, "Error! Unable to send inference request: {e:#}");
            Vec::new()
        }
    };
    stream_log!(stream, Level::Trace, "after doInferenceVdNet");
    if config.flag_process_special {
        stream_log!(stream, Level::Trace, "before doInferenceVcNet");
        pipeline::classify_special(triton, &frame, config, &mut vehicles, stream).await?;
        stream_log!(stream, Level::Trace, "after doInferenceVcNet");
    }
    if config.logs(Level::Trace) {
        for (i, vehicle) in vehicles.iter().enumerate() {
            stream_log!(stream, Level::Trace, "vehicle {i} confidence: {:.3}", vehicle.confidence);
        }
    }

    let is_special_banned = {
        let mut special_bans = lock(&lprs.workflows.special_bans);
        let now = Instant::now();
        match special_bans.get(vstream_key) {
            Some(until) if *until > now => {
                let left = until.saturating_duration_since(now).as_secs();
                stream_log!(stream, Level::Trace, "special vehicles are banned ({left}s left)");
                true
            }
            Some(_) => {
                special_bans.remove(vstream_key);
                stream_log!(stream, Level::Trace, "special vehicles are no longer banned");
                false
            }
            None => false,
        }
    };

    stream_log!(stream, Level::Trace, "before doInferenceLpdNet");
    pipeline::detect_plates(triton, &frame, config, &mut vehicles, stream).await?;
    stream_log!(stream, Level::Trace, "after doInferenceLpdNet");
    pipeline::remove_duplicate_plates(config, &mut vehicles, frame.width(), frame.height(), stream);
    {
        let mut plates: Vec<&mut LicensePlate> = vehicles.iter_mut().flat_map(|v| v.plates.iter_mut()).collect();
        if !plates.is_empty() {
            stream_log!(stream, Level::Trace, "before doInferenceLpcNet");
            pipeline::classify_plates(triton, &frame, config, &mut plates, stream).await?;
            stream_log!(stream, Level::Trace, "after doInferenceLpcNet");
            stream_log!(stream, Level::Trace, "before doInferenceLprNet");
            pipeline::recognize_plates(triton, &frame, config, &mut plates, stream).await?;
            stream_log!(stream, Level::Trace, "after doInferenceLprNet");
        }
    }

    let now = Instant::now();
    let mut vehicles_info = Vec::new();
    let mut callback_plates = Vec::new();
    let mut has_special = false;
    let mut has_failed = false;
    for vehicle in &vehicles {
        has_special |= vehicle.is_special;
        let mut plates_info = Vec::new();
        for plate in &vehicle.plates {
            has_failed |= plate.numbers.is_empty();
            for number in &plate.numbers {
                if is_plate_banned(lprs, vstream_key, &number.number, &plate.bbox, config, now) {
                    continue;
                }
                if config.logs(Level::Info) {
                    log!(lprs.logger, Level::Info, "vstream_key = {vstream_key};  plate number: {}", number.number);
                }
                let plate_type = PLATE_CLASSES.get(number.plate_class as usize).copied().unwrap_or_default();
                plates_info.push(json!({
                    "box": plate.bbox.map(|c| c as i32),
                    "kpts": plate.kpts.map(|c| c as i32),
                    "number": number.number,
                    "score": f64::from(number.score),
                    "type": plate_type,
                }));
                callback_plates.push(json!({"type": plate_type, "number": number.number}));
            }
        }
        if !plates_info.is_empty() || (vehicle.is_special && !is_special_banned) {
            let mut info = Map::new();
            if !plates_info.is_empty() {
                info.insert("plates".to_string(), Value::Array(plates_info));
            }
            info.insert("isSpecial".to_string(), Value::Bool(vehicle.is_special));
            info.insert("confidence".to_string(), json!(f64::from(vehicle.confidence)));
            info.insert("box".to_string(), json!(vehicle.bbox.map(|c| c as i32)));
            vehicles_info.push(Value::Object(info));
        }
    }

    if !vehicles_info.is_empty() {
        record_event(lprs, vstream_key, ext_id, id_vstream, config, image_data, vehicles_info, callback_plates, has_special).await?;
    }
    if has_special {
        lock(&lprs.workflows.special_bans).insert(vstream_key.to_string(), now + config.ban_duration);
    }
    if has_failed && config.flag_save_failed {
        save_failed(lprs, ext_id, config, image_data, &frame, &vehicles).await?;
    }
    Ok(())
}

/// The two-stage plate ban: checks the plate and updates its ban.
fn is_plate_banned(lprs: &Lprs, vstream_key: &str, number: &str, bbox: &BBox, config: &VStreamConfig, now: Instant) -> bool {
    if config.ban_duration.is_zero() || config.ban_duration_area.is_zero() {
        return false;
    }
    let key = format!("{vstream_key}_{number}");
    let mut ban = PlateBan {
        full_until: now + config.ban_duration,
        area_until: now + config.ban_duration_area,
        bbox: *bbox,
    };
    let mut bans = lock(&lprs.workflows.plate_bans);
    let mut is_banned = false;
    if let Some(existing) = bans.get(&key) {
        if existing.full_until > now {
            // Full ban: seeing the plate again extends it
            is_banned = true;
            if config.logs(Level::Debug) {
                let left = existing.full_until.saturating_duration_since(now).as_secs();
                log!(lprs.logger, Level::Debug, "vstream_key = {vstream_key};  plate number {number} is banned at the first stage ({left}s left)");
            }
        } else {
            // Area ban: ignored while the plate stays where it was
            let iou_value = iou(bbox, &existing.bbox);
            is_banned = iou_value > config.ban_iou_threshold;
            if is_banned {
                ban.bbox = existing.bbox;
                ban.full_until = existing.full_until;
            }
            if config.logs(Level::Debug) {
                let state = if is_banned { "is banned at the second stage" } else { "was removed from the the second stage ban" };
                log!(
                    lprs.logger,
                    Level::Debug,
                    "vstream_key = {vstream_key};  plate_number {number} {state} (iou = {iou_value:.2}, threshold = {:.2})",
                    config.ban_iou_threshold
                );
            }
        }
    }
    bans.insert(key, ban);
    is_banned
}

#[allow(clippy::too_many_arguments)]
async fn record_event(
    lprs: &Lprs,
    vstream_key: &str,
    ext_id: &str,
    id_vstream: i32,
    config: &VStreamConfig,
    image_data: &[u8],
    vehicles_info: Vec<Value>,
    callback_plates: Vec<Value>,
    has_special: bool,
) -> Result<()> {
    let log_date: DateTime<Utc> = Utc::now();
    let date = converters::format_time(log_date);
    let uuid = uuid::Uuid::new_v4().to_string();
    let u: Vec<char> = uuid.chars().take(4).collect();
    let path_suffix = format!("{}/{}/{}/{}/", u[0], u[1], u[2], u[3]);
    let settings = &lprs.settings;
    let info = json!({
        "vehicles": vehicles_info,
        "screenshotUrl": format!("{}{path_suffix}{uuid}.jpg", settings.screenshots_url_prefix),
        "date": date,
    });

    // A failed insert is logged and the event goes on with id -1, as in the C++ service
    let id_event = match lprs
        .cluster
        .query(
            HostType::Master,
            "insert into events_log(id_vstream, log_date, info) values($1, $2, $3) returning id_event",
            &[&id_vstream, &log_date, &info],
        )
        .await
    {
        Ok(rows) => rows.first().map_or(-1, |row| row.get::<_, i32>(0) as i64),
        Err(e) => {
            log!(lprs.logger, Level::Error, "can't add event: {e}");
            -1
        }
    };

    let directory = format!("{}{path_suffix}", settings.screenshots_path);
    write_image_file(Path::new(&directory), &format!("{uuid}.jpg"), image_data).await?;

    let mut payload = Map::new();
    payload.insert("streamId".to_string(), Value::String(ext_id.to_string()));
    payload.insert("date".to_string(), Value::String(date));
    payload.insert("eventId".to_string(), json!(id_event));
    if !callback_plates.is_empty() {
        payload.insert("plates".to_string(), Value::Array(callback_plates));
    }
    payload.insert("hasSpecial".to_string(), Value::Bool(has_special));
    let response = lprs
        .http
        .post(&config.callback_url)
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .body(Value::Object(payload).to_string())
        .timeout(config.callback_timeout)
        .send()
        .await;
    match response {
        Ok(response) if matches!(response.status().as_u16(), 200 | 204) => {}
        Ok(_) => {
            if config.logs(Level::Warning) {
                log!(lprs.logger, Level::Warning, "vstream_key = {vstream_key};  error sending data to callback {}", config.callback_url);
            }
        }
        Err(e) => log!(
            lprs.logger,
            Level::Error,
            "vstream_key = {vstream_key};  error sending data to callback {};  {e}",
            config.callback_url
        ),
    }
    Ok(())
}

/// Writes a screenshot readable and writable by the owner and others, like the C++ service.
async fn write_image_file(directory: &Path, name: &str, data: &[u8]) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    tokio::fs::create_dir_all(directory)
        .await
        .with_context(|| format!("can't create {}", directory.display()))?;
    let path = directory.join(name);
    tokio::fs::write(&path, data).await.with_context(|| format!("can't write {}", path.display()))?;
    tokio::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o606)).await?;
    Ok(())
}

/// Draws a closed polyline two pixels wide.
fn draw_polygon(image: &mut RgbImage, points: &[[f32; 2]], color: Rgb<u8>) {
    let (w, h) = (image.width() as i64, image.height() as i64);
    for i in 0..points.len() {
        let (a, b) = (points[i], points[(i + 1) % points.len()]);
        let (x0, y0, x1, y1) = (a[0] as i64, a[1] as i64, b[0] as i64, b[1] as i64);
        let steps = (x1 - x0).abs().max((y1 - y0).abs()).max(1);
        for s in 0..=steps {
            let x = x0 + (x1 - x0) * s / steps;
            let y = y0 + (y1 - y0) * s / steps;
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                if (0..w).contains(&(x + dx)) && (0..h).contains(&(y + dy)) {
                    image.put_pixel((x + dx) as u32, (y + dy) as u32, color);
                }
            }
        }
    }
}

/// Saves a frame with unrecognised plates, and a copy with the work area, vehicles and plates drawn.
async fn save_failed(lprs: &Lprs, ext_id: &str, config: &VStreamConfig, image_data: &[u8], frame: &RgbImage, vehicles: &[Vehicle]) -> Result<()> {
    let uuid = uuid::Uuid::new_v4().to_string();
    let directory = PathBuf::from(format!("{}{ext_id}/", lprs.settings.failed_path));
    write_image_file(&directory, &format!("{uuid}.jpg"), image_data).await?;

    let drawn = tokio::task::block_in_place(|| -> Result<Vec<u8>> {
        let mut image = frame.clone();
        let (width, height) = (image.width() as f32, image.height() as f32);
        for polygon in &config.work_area {
            let points: Vec<[f32; 2]> = polygon
                .iter()
                .map(|p| [(p[0] * width / 100.0) as i32 as f32, (p[1] * height / 100.0) as i32 as f32])
                .collect();
            draw_polygon(&mut image, &points, Rgb([0, 200, 0]));
        }
        for vehicle in vehicles {
            let b = vehicle.bbox;
            let color = if vehicle.is_special { Rgb([200, 0, 0]) } else { Rgb([0, 0, 200]) };
            draw_polygon(&mut image, &[[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]], color);
            for plate in &vehicle.plates {
                let k = plate.kpts;
                let color = if plate.numbers.is_empty() { Rgb([138, 43, 226]) } else { Rgb([255, 105, 2]) };
                draw_polygon(&mut image, &[[k[0], k[1]], [k[2], k[3]], [k[4], k[5]], [k[6], k[7]]], color);
            }
        }
        let mut encoded = Vec::new();
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut encoded, 95).encode_image(&image)?;
        Ok(encoded)
    })?;
    write_image_file(&directory, &format!("{uuid}_draw.jpg"), &drawn).await
}

/// Forgets plate bans whose area ban has ended.
pub fn ban_maintenance(lprs: &Lprs) {
    log!(lprs.logger, Level::Debug, "call doBanMaintenance");
    let now = Instant::now();
    lock(&lprs.workflows.plate_bans).retain(|_, ban| ban.area_until >= now);
}

const IMAGE_EXTENSIONS: [&str; 6] = ["png", "jpg", "jpeg", "bmp", "ppm", "tiff"];

/// Removes image files under `directory` last modified before `before`.
fn remove_old_images(directory: &Path, before: SystemTime) {
    let Ok(entries) = std::fs::read_dir(directory) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(metadata) = entry.metadata() else {
            continue;
        };
        if metadata.is_dir() {
            remove_old_images(&path, before);
        } else if metadata.is_file()
            && path
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| IMAGE_EXTENSIONS.contains(&e))
            && metadata.modified().is_ok_and(|modified| modified < before)
        {
            let _ = std::fs::remove_file(&path);
        }
    }
}

/// Deletes events and event screenshots older than the events TTL, and old failed screenshots.
pub async fn events_log_maintenance(lprs: &Lprs) {
    log!(lprs.logger, Level::Debug, "call doEventsLogMaintenance");
    let settings = &lprs.settings;
    let before = SystemTime::now() - settings.events_log_ttl;
    let before_date: DateTime<Utc> = before.into();
    log!(lprs.logger, Level::Debug, "delete event logs older than {before_date}");
    if let Err(e) = lprs
        .cluster
        .execute(HostType::Master, "delete from events_log where log_date < $1", &[&before_date])
        .await
    {
        log!(lprs.logger, Level::Error, "{e}");
        return;
    }
    let failed_before = SystemTime::now() - settings.failed_ttl;
    let (screenshots, failed) = (PathBuf::from(&settings.screenshots_path), PathBuf::from(&settings.failed_path));
    let _ = tokio::task::spawn_blocking(move || {
        remove_old_images(&screenshots, before);
        remove_old_images(&failed, failed_before);
    })
    .await;
}
