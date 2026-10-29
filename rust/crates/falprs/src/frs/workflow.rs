//! FRS workflows: per video stream recognition loops, the face pipeline behind face registration,
//! frame processing and image tests, maintenance tasks and inference statistics.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use anyhow::{Context, Result};
use chrono::{DateTime, Local, Utc};
use image::{Rgb, RgbImage};
use pg_cluster::HostType;
use serde_json::{Value, json};

use super::Frs;
use super::caches::{descriptor_to_bytes, normalize};
use super::config::{CommonConfig, VStreamConfig};
use super::pipeline::{self, Rect, cosine_similarity, crop, enlarge_face_rect};
use crate::converters::format_time;
use crate::fetch::{FetchError, decode_base64, fetch_image};
use crate::imaging;
use crate::log;
use crate::logging::Level;

pub const MIME_IMAGE: &str = "image/jpeg";

/// `log_faces.copy_data` states.
pub const COPY_EVENT_DATA_DISABLED: i16 = -1;
pub const COPY_EVENT_DATA_NONE: i16 = 0;
pub const COPY_EVENT_DATA_SCHEDULED: i16 = 1;

const FACE_NORMAL: i32 = 0;
const FACE_NONE: i32 = -1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskType {
    Recognize = 1,
    RegisterDescriptor = 2,
    ProcessFrame = 3,
    Test = 4,
}

#[derive(Debug, Clone)]
pub struct TaskData {
    pub id_group: i32,
    pub vstream_key: String,
    pub task_type: TaskType,
    pub frame_url: String,
    pub face_left: i32,
    pub face_top: i32,
    pub face_width: i32,
    pub face_height: i32,
    pub id_sgroup: i32,
}

impl TaskData {
    pub fn new(id_group: i32, vstream_key: String, task_type: TaskType, frame_url: String) -> Self {
        Self {
            id_group,
            vstream_key,
            task_type,
            frame_url,
            face_left: 0,
            face_top: 0,
            face_width: 0,
            face_height: 0,
            id_sgroup: 0,
        }
    }
}

#[derive(Debug, Default)]
pub struct RegistrationResult {
    pub id_descriptor: i32,
    pub comments: String,
    pub face_image_jpeg: Vec<u8>,
    pub face_left: i32,
    pub face_top: i32,
    pub face_width: i32,
    pub face_height: i32,
    pub id_descriptors: Vec<i32>,
}

impl RegistrationResult {
    fn comments(comments: impl Into<String>) -> Self {
        Self {
            comments: comments.into(),
            ..Default::default()
        }
    }
}

/// Inference counters of a video stream.
#[derive(Debug, Default, Clone, Copy)]
pub struct DnnStats {
    fd_count: i64,
    fc_count: i64,
    fr_count: i64,
}

struct UnknownDescriptor {
    expiration: Instant,
    descriptor: Vec<f32>,
    face_image: RgbImage,
}

/// State of the running workflows.
#[derive(Default)]
pub struct Workflows {
    /// Video streams with a running recognition loop; `false` asks the loop to stop.
    being_processed: Mutex<HashMap<String, bool>>,
    timeouts: Mutex<HashMap<String, Instant>>,
    unknown_descriptors: Mutex<HashMap<i32, Vec<UnknownDescriptor>>>,
    dnn_stats: Mutex<BTreeMap<i32, DnnStats>>,
    /// Inference statistics of removed video streams, counted only in the totals.
    removed_dnn_stats: Mutex<DnnStats>,
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

pub fn start_workflow(frs: &Arc<Frs>, vstream_key: String, duration: Duration) {
    let Some((id_group, mut workflow_timeout)) = frs
        .vstreams_config
        .get()
        .get(&vstream_key)
        .map(|config| (config.id_group, config.workflow_timeout))
    else {
        return;
    };
    if id_group <= 0 {
        return;
    }
    let do_pipeline = lock(&frs.workflows.being_processed).insert(vstream_key.clone(), true).is_none();
    if !duration.is_zero() {
        workflow_timeout = duration;
    }
    if !workflow_timeout.is_zero() {
        lock(&frs.workflows.timeouts).insert(vstream_key.clone(), Instant::now() + workflow_timeout);
    }
    if do_pipeline {
        let task = TaskData::new(id_group, vstream_key, TaskType::Recognize, String::new());
        tokio::spawn(recognize_loop(frs.clone(), task));
    }
}

/// Stops a video stream's workflow: an internal stop forgets it at once, an external one lets the
/// running loop finish its current frame.
pub fn stop_workflow(frs: &Frs, vstream_key: &str, is_internal: bool) {
    {
        let mut being_processed = lock(&frs.workflows.being_processed);
        if being_processed.contains_key(vstream_key) {
            if is_internal {
                being_processed.remove(vstream_key);
            } else {
                being_processed.insert(vstream_key.to_string(), false);
            }
        }
    }
    lock(&frs.workflows.timeouts).remove(vstream_key);
}

async fn recognize_loop(frs: Arc<Frs>, mut task: TaskData) {
    loop {
        let (_, next) = run_pipeline(&frs, &mut task).await;
        let Some(delay) = next else { return };
        tokio::time::sleep(delay).await;

        let is_timeout = {
            let mut timeouts = lock(&frs.workflows.timeouts);
            let expired = timeouts.get(&task.vstream_key).is_some_and(|deadline| *deadline < Instant::now());
            if expired {
                timeouts.remove(&task.vstream_key);
            }
            expired
        };
        let do_next = {
            let mut being_processed = lock(&frs.workflows.being_processed);
            match being_processed.get(&task.vstream_key) {
                Some(true) if !is_timeout => true,
                Some(_) => {
                    being_processed.remove(&task.vstream_key);
                    false
                }
                None => false,
            }
        };
        if is_timeout {
            log!(frs.logger, Level::Info, "Stopping a workflow by timeout: vstream_key = {};", task.vstream_key);
        }
        if !do_next {
            return;
        }
    }
}

/// Runs the pipeline once for a task other than recognition.
pub async fn process_pipeline(frs: &Arc<Frs>, mut task: TaskData) -> RegistrationResult {
    run_pipeline(frs, &mut task).await.0
}

/// How a pipeline run ended: with a result to return at once (and, for recognition, the delay
/// before the next frame, or none to stop), or by running to the end.
enum Flow {
    Return(RegistrationResult, Option<Duration>),
    Done(RegistrationResult),
}

struct Run<'a> {
    frs: &'a Arc<Frs>,
    task: &'a TaskData,
    common: CommonConfig,
    config: VStreamConfig,
}

impl Run<'_> {
    /// Whether to log at `level`: by the video stream's log level, or always for image tests.
    fn logs(&self, level: Level) -> bool {
        self.config.logs(level) || self.task.task_type == TaskType::Test
    }

    fn is(&self, task_type: TaskType) -> bool {
        self.task.task_type == task_type
    }
}

async fn run_pipeline(frs: &Arc<Frs>, task: &mut TaskData) -> (RegistrationResult, Option<Duration>) {
    let (common, mut config) = {
        let cache = frs.config.get();
        (
            cache.common.get(&task.id_group).cloned().unwrap_or_default(),
            cache.default_vstream.get(&task.id_group).cloned().unwrap_or_default(),
        )
    };
    if !task.vstream_key.is_empty() {
        match frs.vstreams_config.get().get(&task.vstream_key) {
            Some(stream_config) => config = VStreamConfig::clone(stream_config),
            None => {
                stop_workflow(frs, &task.vstream_key, true);
                let comments = format!("Invalid video stream key: {}", task.vstream_key);
                return (RegistrationResult::comments(comments), None);
            }
        }
    }
    let url = if task.task_type == TaskType::Recognize { config.url.clone() } else { task.frame_url.clone() };
    let mut ctx = Run {
        frs,
        task,
        common,
        config,
    };

    if ctx.logs(Level::Debug) {
        let frame_url = if url.starts_with("data:") { "data:base64..." } else { url.as_str() };
        log!(
            frs.logger,
            Level::Debug,
            "Start processPipeline: vstream_key = {};  task_type = {};  url = {frame_url}",
            ctx.task.vstream_key,
            ctx.task.task_type as i32
        );
    }

    let mut delay_between_frames = ctx.config.delay_between_frames;
    let result = match pipeline_body(&mut ctx, &url).await {
        Ok(Flow::Return(result, next)) => return (result, next),
        Ok(Flow::Done(result)) => result,
        Err(e) => {
            if ctx.logs(Level::Error) {
                log!(frs.logger, Level::Error, "vstream_key = {};  {e:#}", ctx.task.vstream_key);
            }
            if !ctx.is(TaskType::Recognize) {
                return (RegistrationResult::comments("Error during pipeline execution"), None);
            }
            if ctx.config.delay_after_error.is_zero() {
                stop_workflow(frs, &ctx.task.vstream_key, true);
                return (RegistrationResult::comments("Error during pipeline execution"), None);
            }
            if ctx.logs(Level::Error) {
                log!(
                    frs.logger,
                    Level::Error,
                    "vstream_key = {};  delay for {}ms",
                    ctx.task.vstream_key,
                    ctx.config.delay_after_error.as_millis()
                );
            }
            delay_between_frames = ctx.config.delay_after_error;
            RegistrationResult::default()
        }
    };
    if ctx.logs(Level::Debug) {
        log!(frs.logger, Level::Debug, "End processPipeline: vstream_key = {};", ctx.task.vstream_key);
    }
    let next = ctx.is(TaskType::Recognize).then_some(delay_between_frames);
    (result, next)
}

/// Gets the frame's encoded image, or the flow to return with when it can't be had.
async fn acquire_image(ctx: &Run<'_>, url: &str) -> Result<Vec<u8>, Flow> {
    let (frs, key) = (ctx.frs, &ctx.task.vstream_key);
    if url.starts_with("data:") {
        let Some(comma) = url.find(',') else { return Ok(Vec::new()) };
        if !url.contains(";base64,") {
            return Ok(Vec::new());
        }
        return decode_base64(&url[comma + 1..]).ok_or_else(|| {
            if ctx.logs(Level::Error) {
                log!(frs.logger, Level::Error, "Error decoding image from BASE64: vstream_key = {key};");
            }
            // A recognition loop stops: the frame of its video stream will not get better
            if ctx.is(TaskType::Recognize) {
                stop_workflow(frs, key, true);
            }
            Flow::Return(RegistrationResult::comments("Error decoding image from BASE64"), None)
        });
    }

    if ctx.logs(Level::Trace) {
        log!(frs.logger, Level::Trace, "vstream_key = {key};  before image acquisition");
    }
    let config = &ctx.config;
    let fetched = fetch_image(&frs.http, url, config.max_capture_error_count, config.capture_timeout).await;
    if ctx.logs(Level::Trace) {
        log!(frs.logger, Level::Trace, "vstream_key = {key};  after image acquisition");
    }
    match fetched {
        Ok(data) => Ok(data.to_vec()),
        Err(e) => {
            if ctx.logs(Level::Error) {
                let status = match &e {
                    FetchError::Http(reason) => reason.clone(),
                    FetchError::Base64 => e.to_string(),
                };
                log!(frs.logger, Level::Error, "vstream_key = {key};  url = {url};  status_code = {status}");
            }
            let mut next = None;
            if !config.delay_after_error.is_zero() {
                if ctx.is(TaskType::Recognize) {
                    if config.logs(Level::Error) {
                        log!(
                            frs.logger,
                            Level::Error,
                            "vstream_key = {key};  delay for {}ms",
                            config.delay_after_error.as_millis()
                        );
                    }
                    next = Some(config.delay_after_error);
                }
            } else if ctx.is(TaskType::Recognize) {
                stop_workflow(frs, key, true);
            }
            let comments = format!("Error when retrieving image by url: {url}");
            Err(Flow::Return(RegistrationResult::comments(comments), next))
        }
    }
}

/// What the pipeline learned about one detected face.
#[derive(Default)]
struct FaceData {
    face_rect: Rect,
    is_work_area: bool,
    is_frontal: bool,
    is_non_blurry: bool,
    face_class_index: i32,
    face_class_confidence: f32,
    cosine_similarity: f64,
    /// The descriptor as inferred, not normalized.
    descriptor: Vec<f32>,
    landmarks: Option<[[f32; 2]; 5]>,
    laplacian: f64,
    ioa: f64,
    id_descriptor: i32,
    /// Best match per special group: similarity and descriptor id.
    sg_descriptors: HashMap<i32, (f64, i32)>,
}

/// Writes a debug image of an image test into the working directory, as the C++ service does.
fn write_test_image(name: &str, image: &RgbImage) {
    let path = std::env::current_dir().unwrap_or_default().join(name);
    if let Ok(jpeg) = imaging::encode_jpeg(image) {
        let _ = std::fs::write(path, jpeg);
    }
}

async fn pipeline_body(ctx: &mut Run<'_>, url: &str) -> Result<Flow> {
    let frs = ctx.frs;
    let key = ctx.task.vstream_key.clone();
    let image_data = match acquire_image(ctx, url).await {
        Ok(data) => data,
        Err(flow) => return Ok(flow),
    };
    if ctx.logs(Level::Trace) {
        log!(frs.logger, Level::Trace, "vstream_key = {key};  image size = {} bytes", image_data.len());
        log!(frs.logger, Level::Trace, "vstream_key = {key};  before decoding the image");
    }
    let frame = tokio::task::block_in_place(|| imaging::decode(&image_data))?;
    if ctx.logs(Level::Trace) {
        log!(frs.logger, Level::Trace, "vstream_key = {key};  after decoding the image");
    }
    let (cols, rows) = (frame.width() as i32, frame.height() as i32);

    let mut work_area = Rect::default();
    if let [x, y, w, h] = ctx.config.work_area[..] {
        work_area = Rect::new(
            (x * cols as f32 / 100.0) as i32,
            (y * rows as f32 / 100.0) as i32,
            (w * cols as f32 / 100.0) as i32,
            (h * rows as f32 / 100.0) as i32,
        );
    }

    let screenshot = Screenshot::new(&ctx.config, &frame, &image_data);
    if ctx.config.flag_process_barcodes {
        let is_test = ctx.task.task_type == TaskType::Test;
        super::barcodes::process(ctx.frs, &ctx.task.vstream_key, is_test, &ctx.common, &ctx.config, &frame, &screenshot)
            .await;
    }

    let mut result = RegistrationResult::default();
    if !ctx.config.flag_process_faces {
        return Ok(Flow::Done(result));
    }
    let mut task = ctx.task.clone();
    if task.task_type == TaskType::RegisterDescriptor {
        // Without a search area the whole frame is searched
        if task.face_width == 0 {
            task.face_width = cols;
        }
        if task.face_height == 0 {
            task.face_height = rows;
        }
    }

    let detected = pipeline::detect_faces(&frs.triton, &frame, &ctx.common, &ctx.config).await;
    let detected = match detected {
        Ok(detected) => detected,
        Err(e) => {
            if ctx.logs(Level::Error) {
                log!(frs.logger, Level::Error, "Error! {e:#}");
            }
            if task.task_type == TaskType::RegisterDescriptor {
                result.id_descriptor = 0;
                result.comments = ctx.common.comments_inference_error.clone();
            }
            return Ok(Flow::Done(result));
        }
    };

    let mut stats = DnnStats {
        fd_count: 1,
        ..Default::default()
    };
    let mut face_data: Vec<FaceData> = Vec::new();
    let mut recognized_face_count = 0;
    let mut best_quality = 0.0;
    let mut best_face_index: Option<usize> = None;
    let mut best_register_quality = 0.0;
    let mut best_register_ioa = 0.0;
    let mut best_register_index: Option<usize> = None;
    let mut has_sgroup_events = false;
    let config = ctx.config.clone();
    let common = ctx.common.clone();

    if ctx.logs(Level::Trace) {
        log!(frs.logger, Level::Trace, "vstream_key = {key};  process the found faces, quantity: {}", detected.len());
    }
    for face in &detected {
        if ctx.logs(Level::Trace) {
            log!(frs.logger, Level::Trace, "vstream_key = {key};  face probability: {:.3}", face.confidence);
        }
        let margin = f64::from(config.margin);
        let mut work_region = Rect::new(
            (margin / 100.0 * f64::from(cols)) as i32,
            (margin / 100.0 * f64::from(rows)) as i32,
            (f64::from(cols) - 2.0 * f64::from(cols) * margin / 100.0) as i32,
            (f64::from(rows) - 2.0 * f64::from(rows) * margin / 100.0) as i32,
        );
        if !work_area.is_empty() {
            work_region = work_region.intersect(&work_area);
        }
        let b = face.bbox;
        let face_rect = Rect::new(b[0] as i32, b[1] as i32, (b[2] - b[0] + 1.0) as i32, (b[3] - b[1] + 1.0) as i32);
        face_data.push(FaceData {
            face_rect,
            face_class_index: FACE_NONE,
            cosine_similarity: -2.0,
            ..Default::default()
        });
        let index = face_data.len() - 1;

        if work_region.intersect(&face_rect) != face_rect {
            if ctx.logs(Level::Trace) {
                log!(frs.logger, Level::Trace, "vstream_key = {key};  the person is not in the work area");
            }
            continue;
        }
        face_data[index].is_work_area = true;
        if ctx.logs(Level::Trace) {
            log!(frs.logger, Level::Trace, "vstream_key = {key};  the person is in the work area");
        }
        face_data[index].landmarks = Some(face.landmarks);

        if !pipeline::is_frontal_face(&face.landmarks) {
            if ctx.logs(Level::Trace) {
                log!(frs.logger, Level::Trace, "vstream_key = {key};  face is not frontal according to markers");
            }
            continue;
        }

        let (fr_w, fr_h) = (common.fr_net.input_width, common.fr_net.input_height);
        let aligned_face = tokio::task::block_in_place(|| pipeline::align_face(&frame, &face.landmarks, fr_w, fr_h));
        let Some(aligned_face) = aligned_face else {
            if ctx.logs(Level::Trace) {
                log!(frs.logger, Level::Trace, "vstream_key = {key};  failed to do face alignment to get descriptor");
            }
            continue;
        };
        if task.task_type == TaskType::Test {
            write_test_image(&format!("aligned_face_{}.jpg", face_data.len()), &aligned_face);
        }
        face_data[index].is_frontal = true;

        let laplacian = imaging::variance_of_laplacian(&aligned_face);
        face_data[index].laplacian = laplacian;
        if ctx.logs(Level::Trace) {
            log!(frs.logger, Level::Trace, "vstream_key = {key};  laplacian = {laplacian:.2}");
        }
        if laplacian < f64::from(config.blur) || laplacian > f64::from(config.blur_max) {
            if ctx.logs(Level::Trace) {
                log!(frs.logger, Level::Trace, "vstream_key = {key};  the face is blurry or too clear");
            }
            continue;
        }
        face_data[index].is_non_blurry = true;
        if ctx.logs(Level::Trace) {
            log!(frs.logger, Level::Trace, "vstream_key = {key};  the face is not blurry");
        }

        let (fc_w, fc_h) = (common.fc_net.input_width, common.fc_net.input_height);
        let aligned_face_class = tokio::task::block_in_place(|| pipeline::align_face(&frame, &face.landmarks, fc_w, fc_h));
        let Some(aligned_face_class) = aligned_face_class else {
            if ctx.logs(Level::Trace) {
                log!(frs.logger, Level::Trace, "vstream_key = {key};  failed to do face alignment for inference class");
            }
            continue;
        };
        if task.task_type == TaskType::Test {
            write_test_image(&format!("aligned_face_class_{}.jpg", face_data.len()), &aligned_face_class);
        }

        match pipeline::infer_face_class(&frs.triton, &aligned_face_class, &common, &config).await {
            Ok(classes) => {
                stats.fc_count += 1;
                if let Some(&(class_index, score)) = classes.first() {
                    face_data[index].face_class_index = class_index as i32;
                    face_data[index].face_class_confidence = score;
                    if ctx.logs(Level::Trace) {
                        log!(frs.logger, Level::Trace, "vstream_key = {key};  face class: {class_index};  probability: {score:.3}");
                    }
                }
            }
            Err(e) => {
                if ctx.logs(Level::Error) {
                    log!(frs.logger, Level::Error, "Error! {e:#}");
                }
            }
        }
        let (class_index, class_confidence) = (face_data[index].face_class_index, face_data[index].face_class_confidence);
        if class_index == FACE_NONE || (class_index != FACE_NORMAL && class_confidence > config.face_class_confidence) {
            continue;
        }
        face_data[index].face_class_index = FACE_NORMAL;

        if task.task_type == TaskType::RegisterDescriptor {
            let search = Rect::new(task.face_left, task.face_top, task.face_width, task.face_height);
            let intersection = search.intersect(&face_rect).area();
            if face_rect.area() > 0 {
                face_data[index].ioa = intersection as f64 / face_rect.area() as f64;
            }
        }

        let descriptor = match pipeline::extract_face_descriptor(&frs.triton, &aligned_face, &common, &config).await {
            Ok(descriptor) => descriptor,
            Err(e) => {
                if ctx.logs(Level::Error) {
                    log!(frs.logger, Level::Error, "Error! {e:#}");
                }
                continue;
            }
        };
        stats.fr_count += 1;
        face_data[index].descriptor = descriptor.clone();
        let mut normalized = descriptor;
        normalize(&mut normalized);

        if ctx.logs(Level::Trace) {
            log!(frs.logger, Level::Trace, "vstream_key = {key};  before recognition");
        }
        let (max_similarity, id_descriptor) = recognize(ctx, &task, &normalized, &mut face_data[index], &mut has_sgroup_events);
        if ctx.logs(Level::Trace) {
            log!(frs.logger, Level::Trace, "vstream_key = {key};  after recognition");
        }
        face_data[index].cosine_similarity = max_similarity;
        if ctx.logs(Level::Trace) {
            log!(
                frs.logger,
                Level::Trace,
                "vstream_key = {key};  most similar data: cosine_similarity = {max_similarity:.3};  id_descriptor = {id_descriptor}"
            );
        }

        if id_descriptor == 0 || max_similarity < f64::from(config.tolerance) {
            // The face isn't recognized
            if face_data[index].laplacian > best_quality && recognized_face_count == 0 {
                best_quality = face_data[index].laplacian;
                best_face_index = Some(index);
            }
            if config.flag_spawned_descriptors && task.task_type == TaskType::Recognize {
                if config.logs(Level::Trace) {
                    log!(frs.logger, Level::Trace, "vstream_key = {key};  add an unknown descriptor");
                }
                let r = enlarge_face_rect(&face_rect, f64::from(config.face_enlarge_scale)).intersect(&Rect::new(0, 0, cols, rows));
                let face_image = crop(&frame, &r);
                let mut unknown = lock(&frs.workflows.unknown_descriptors);
                let list = unknown.entry(config.id_vstream).or_default();
                let now = Instant::now();
                list.retain(|item| item.expiration >= now);
                list.push(UnknownDescriptor {
                    expiration: now + config.unknown_descriptor_ttl,
                    descriptor: face_data[index].descriptor.clone(),
                    face_image,
                });
            }
        } else {
            // The face is recognized
            face_data[index].id_descriptor = id_descriptor;
            recognized_face_count += 1;
            if recognized_face_count == 1 || face_data[index].laplacian > best_quality {
                best_quality = face_data[index].laplacian;
                best_face_index = Some(index);
            }
            if task.task_type == TaskType::ProcessFrame {
                result.id_descriptors.push(id_descriptor);
            }
            if config.flag_spawned_descriptors && task.task_type == TaskType::Recognize {
                spawn_descriptor(ctx, &key, &normalized, id_descriptor).await;
            }
        }

        if task.task_type == TaskType::RegisterDescriptor {
            let current = &face_data[index];
            if current.ioa > 0.999 && current.laplacian > best_register_quality {
                best_register_quality = current.laplacian;
                best_register_index = Some(index);
            }
            if best_register_quality.abs() < 0.001 && current.ioa > best_register_ioa {
                best_register_ioa = current.ioa;
                best_register_index = Some(index);
            }
        }
    }

    {
        let mut dnn_stats = lock(&frs.workflows.dnn_stats);
        let entry = dnn_stats.entry(config.id_vstream).or_default();
        entry.fd_count += stats.fd_count;
        entry.fc_count += stats.fc_count;
        entry.fr_count += stats.fr_count;
    }

    if let (Some(best), TaskType::Recognize) = (best_face_index, task.task_type) {
        log_face_event(ctx, &key, &face_data, best, screenshot.data()).await;
    }
    if has_sgroup_events && task.task_type == TaskType::Recognize {
        log_special_group_events(ctx, &key, &face_data, screenshot.data()).await;
    }

    if task.task_type == TaskType::RegisterDescriptor {
        register_descriptor(ctx, &key, &task, &frame, &face_data, best_register_index, &mut result).await;
    }

    if task.task_type == TaskType::Test {
        let mut frame = frame;
        draw_test_markers(&mut frame, &face_data);
        let frame_index = std::time::UNIX_EPOCH.elapsed().unwrap_or_default().as_millis();
        write_test_image(&format!("frame_{frame_index}.jpg"), &frame);
        log!(
            frs.logger,
            Level::Info,
            "vstream_key = {key};  task_type = {};  frame index: {frame_index}",
            task.task_type as i32
        );
    }
    Ok(Flow::Done(result))
}

/// The image saved as the frame's screenshots: the frame with the video stream's OSD, drawn on a
/// copy once on first use, or the frame as received when the video stream has no title.
pub struct Screenshot<'a> {
    config: &'a VStreamConfig,
    frame: &'a RgbImage,
    image_data: &'a [u8],
    with_osd: std::sync::OnceLock<Option<Vec<u8>>>,
}

impl<'a> Screenshot<'a> {
    pub fn new(config: &'a VStreamConfig, frame: &'a RgbImage, image_data: &'a [u8]) -> Self {
        Self {
            config,
            frame,
            image_data,
            with_osd: std::sync::OnceLock::new(),
        }
    }

    pub fn data(&self) -> &[u8] {
        let with_osd = self.with_osd.get_or_init(|| {
            let config = self.config;
            tokio::task::block_in_place(|| {
                let drawn = crate::osd::draw(self.frame, &config.title, &config.osd_dt_format, config.title_height_ratio)?;
                imaging::encode_jpeg(&drawn).ok()
            })
        });
        with_osd.as_deref().unwrap_or(self.image_data)
    }
}

/// Finds the most similar descriptor among the video stream's faces (or the special group's, for
/// a special group task) and the best matches in the stream group's special groups.
fn recognize(
    ctx: &Run<'_>,
    task: &TaskData,
    descriptor: &[f32],
    face: &mut FaceData,
    has_sgroup_events: &mut bool,
) -> (f64, i32) {
    let frs = ctx.frs;
    let config = &ctx.config;
    let vstream_descriptors = frs.vstream_descriptors.get();
    let face_descriptors = frs.face_descriptors.get();
    let sg_config = frs.sg_config.get();
    let sg_descriptors = frs.sg_descriptors.get();

    let mut max_similarity = -2.0;
    let mut id_descriptor = 0;
    if config.id_vstream > 0 {
        for id in vstream_descriptors.get(&config.id_vstream).into_iter().flatten() {
            if let Some(known) = face_descriptors.descriptors.get(id) {
                let similarity = cosine_similarity(descriptor, known);
                if similarity > max_similarity {
                    max_similarity = similarity;
                    id_descriptor = *id;
                }
            }
        }
    }
    if let Some(&id_parent) = face_descriptors.spawned_to_parent.get(&id_descriptor) {
        if ctx.logs(Level::Trace) {
            log!(
                frs.logger,
                Level::Trace,
                "vstream_key = {};  spawned id_descriptor = {id_descriptor};  parent id_descriptor = {id_parent}",
                task.vstream_key
            );
        }
        id_descriptor = id_parent;
    }

    if task.id_sgroup > 0 {
        for id in sg_descriptors.get(&task.id_sgroup).into_iter().flatten() {
            if let Some(known) = face_descriptors.descriptors.get(id) {
                let similarity = cosine_similarity(descriptor, known);
                if similarity > max_similarity {
                    max_similarity = similarity;
                    id_descriptor = *id;
                }
            }
        }
    } else {
        for id_sgroup in sg_config.by_group.get(&config.id_group).into_iter().flatten() {
            let Some(group_descriptors) = sg_descriptors.get(id_sgroup) else { continue };
            let mut best = (-2.0, 0);
            for id in group_descriptors {
                if let Some(known) = face_descriptors.descriptors.get(id) {
                    let similarity = cosine_similarity(descriptor, known);
                    if similarity > best.0 {
                        best = (similarity, *id);
                    }
                }
            }
            if best.1 > 0 && best.0 >= f64::from(config.tolerance) {
                face.sg_descriptors.insert(*id_sgroup, best);
                *has_sgroup_events = true;
            }
        }
    }
    (max_similarity, id_descriptor)
}

/// Turns the most similar recent unknown descriptor into a descriptor spawned from the recognized one.
async fn spawn_descriptor(ctx: &Run<'_>, key: &str, descriptor: &[f32], id_parent: i32) {
    let frs = ctx.frs;
    let config = &ctx.config;
    let spawned = {
        let mut unknown = lock(&frs.workflows.unknown_descriptors);
        let list = unknown.entry(config.id_vstream).or_default();
        let now = Instant::now();
        list.retain(|item| item.expiration >= now);
        let mut best: Option<(f64, usize)> = None;
        for (i, item) in list.iter().enumerate() {
            let mut normalized = item.descriptor.clone();
            normalize(&mut normalized);
            let similarity = cosine_similarity(descriptor, &normalized);
            if best.is_none_or(|(max, _)| similarity > max) {
                best = Some((similarity, i));
            }
        }
        let spawned = match best {
            Some((similarity, i)) if similarity > f64::from(config.tolerance) => {
                if config.logs(Level::Trace) {
                    log!(
                        frs.logger,
                        Level::Trace,
                        "vstream_key = {key};  unknown descriptors cosine similarity = {similarity:.3};  index = {i};  size = {}",
                        list.len()
                    );
                }
                let item = list.swap_remove(i);
                Some((item.descriptor, item.face_image))
            }
            _ => None,
        };
        // The unknown descriptors are cleared anyway
        list.clear();
        spawned
    };
    if let Some((spawned_descriptor, face_image)) = spawned {
        let id_spawned =
            add_face_descriptor(ctx, config.id_group, config.id_vstream, &spawned_descriptor, &face_image, id_parent).await;
        if config.logs(Level::Trace) {
            log!(
                frs.logger,
                Level::Trace,
                "vstream_key = {key};  created spawned descriptor with id = {id_spawned};  id_parent = {id_parent}"
            );
        }
    }
}

/// `group_<id>/<c0>/<c1>/<c2>/<c3>/` for an id written without dashes.
fn path_suffix(id_group: i32, s_uuid: &str) -> String {
    let c: Vec<char> = s_uuid.chars().take(4).collect();
    format!("group_{id_group}/{}/{}/{}/{}/", c[0], c[1], c[2], c[3])
}

/// Writes a file readable and writable by its owner and others, as the C++ service does.
fn write_shared_file(path: &Path, contents: &[u8]) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, contents)?;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o606))
}

async fn add_log_face(
    ctx: &Run<'_>,
    log_date: DateTime<Utc>,
    id_descriptor: i32,
    quality: f64,
    rect: &Rect,
    screenshot_url: &str,
    log_uuid: &uuid::Uuid,
    copy_data: i16,
) -> i64 {
    let frs = ctx.frs;
    let id_descriptor = (id_descriptor > 0).then_some(id_descriptor);
    let log_uuid = log_uuid.to_string();
    let result = frs
        .cluster
        .query_one(
            HostType::Master,
            r#"
            insert into log_faces(id_vstream, log_date, id_descriptor, quality, face_left, face_top, face_width, face_height, screenshot_url, log_uuid, copy_data)
            values($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::text::uuid, $11) returning id_log
            "#,
            &[
                &ctx.config.id_vstream,
                &log_date,
                &id_descriptor,
                &quality,
                &rect.x,
                &rect.y,
                &rect.width,
                &rect.height,
                &screenshot_url,
                &log_uuid,
                &copy_data,
            ],
        )
        .await;
    match result {
        Ok(row) => i64::from(row.get::<_, i32>(0)),
        Err(e) => {
            log!(frs.logger, Level::Error, "{e}");
            -1
        }
    }
}

/// Posts a JSON event; delivered when answered with 200 or 204.
async fn send_callback(ctx: &Run<'_>, url: &str, body: &Value) -> bool {
    let response = ctx
        .frs
        .http
        .post(url)
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .body(body.to_string())
        .timeout(ctx.common.callback_timeout)
        .send()
        .await;
    match response {
        Ok(response) => matches!(response.status().as_u16(), 200 | 204),
        Err(e) => {
            log!(ctx.frs.logger, Level::Error, "{e}");
            false
        }
    }
}

/// Logs the best face of a recognition frame: database row, screenshot, callback and event files.
async fn log_face_event(ctx: &Run<'_>, key: &str, face_data: &[FaceData], best: usize, image_data: &[u8]) {
    let frs = ctx.frs;
    let config = &ctx.config;
    if config.logs(Level::Info) {
        log!(frs.logger, Level::Info, "vstream_key = {key};  faces detected: id_vstream = {}", config.id_vstream);
    }
    let log_uuid = uuid::Uuid::new_v4();
    let s_uuid = log_uuid.simple().to_string();
    let suffix = path_suffix(config.id_group, &s_uuid);
    let local = &frs.local;
    let log_date = Utc::now();
    let best_face = &face_data[best];
    let screenshot_url = format!("{}{suffix}{s_uuid}.jpg", local.screenshots_url_prefix);
    let id_log = add_log_face(
        ctx,
        log_date,
        best_face.id_descriptor,
        best_face.laplacian,
        &best_face.face_rect,
        &screenshot_url,
        &log_uuid,
        COPY_EVENT_DATA_NONE,
    )
    .await;

    let path_prefix = format!("{}{suffix}", local.screenshots_path);
    let screenshot = PathBuf::from(format!("{path_prefix}{s_uuid}.jpg"));
    if let Err(e) = write_shared_file(&screenshot, image_data) {
        log!(frs.logger, Level::Error, "{e}");
    }

    if id_log > 0 && best_face.id_descriptor > 0 && !config.callback_url.is_empty() {
        let body = json!({"faceId": best_face.id_descriptor, "eventId": id_log});
        if send_callback(ctx, &config.callback_url, &body).await {
            if config.logs(Level::Info) {
                log!(
                    frs.logger,
                    Level::Info,
                    "vstream_key = {key};  facial recognition event sent: id_vstream = {}; id_descriptor = {}",
                    config.id_vstream,
                    best_face.id_descriptor
                );
            }
        } else {
            log!(
                frs.logger,
                Level::Error,
                "vstream_key = {key};  error sending facial recognition event data to callback {}",
                config.callback_url
            );
        }
    }

    // The event's data files: descriptors and a JSON description of all faces
    let output_size = ctx.common.fr_net.output_size;
    let mut dat = Vec::new();
    let mut faces = Vec::new();
    for (i, face) in face_data.iter().enumerate() {
        let landmarks: Vec<f64> = face
            .landmarks
            .iter()
            .flat_map(|lm| lm.iter().flatten().map(|&v| f64::from(v)))
            .collect();
        faces.push(json!({
            "left": face.face_rect.x,
            "top": face.face_rect.y,
            "width": face.face_rect.width,
            "height": face.face_rect.height,
            "laplacian": face.laplacian,
            "landmarks5": landmarks,
            "face_class_index": face.face_class_index,
            "id_descriptor": face.id_descriptor,
            "face_class_confidence": f64::from(face.face_class_confidence),
            "is_frontal": face.is_frontal,
            "is_non_blurry": face.is_non_blurry,
            "is_work_area": face.is_work_area,
        }));
        if !face.descriptor.is_empty() {
            let mut descriptor = face.descriptor.clone();
            descriptor.resize(output_size, 0.0);
            dat.extend_from_slice(s_uuid.as_bytes());
            dat.extend_from_slice(&(i as i32).to_ne_bytes());
            dat.extend_from_slice(&descriptor_to_bytes(&descriptor));
        }
    }
    let event = json!({
        "id_vstream": config.id_vstream,
        "event_date": format_time(log_date),
        "best_face_index": best as i32,
        "faces": faces,
    });
    let written = std::fs::write(format!("{path_prefix}{s_uuid}.dat"), &dat)
        .and_then(|()| std::fs::write(format!("{path_prefix}{s_uuid}.json"), event.to_string()));
    if let Err(e) = written {
        log!(frs.logger, Level::Error, "{e}");
    }
}

/// Logs and reports faces recognized in the stream group's special groups.
async fn log_special_group_events(ctx: &Run<'_>, key: &str, face_data: &[FaceData], image_data: &[u8]) {
    let frs = ctx.frs;
    let config = &ctx.config;
    let local = &frs.local;
    for face in face_data {
        for (&id_sgroup, &(_, id_descriptor)) in &face.sg_descriptors {
            let log_uuid = uuid::Uuid::new_v4();
            let s_uuid = log_uuid.simple().to_string();
            let suffix = path_suffix(config.id_group, &s_uuid);
            let screenshot_url = format!("{}{suffix}{s_uuid}.jpg", local.screenshots_url_prefix);
            let log_date = Utc::now();
            let id_log = add_log_face(
                ctx,
                log_date,
                id_descriptor,
                face.laplacian,
                &face.face_rect,
                &screenshot_url,
                &log_uuid,
                COPY_EVENT_DATA_DISABLED,
            )
            .await;
            let screenshot = PathBuf::from(format!("{}{suffix}{s_uuid}.jpg", local.screenshots_path));
            if let Err(e) = write_shared_file(&screenshot, image_data) {
                log!(frs.logger, Level::Error, "{e}");
            }

            let callback_url = frs.sg_config.get().by_id(id_sgroup).map(|sg| sg.callback_url.clone()).unwrap_or_default();
            if id_log > 0 && !callback_url.is_empty() {
                let body = json!({"faceId": id_descriptor, "screenshot": screenshot_url, "date": format_time(log_date)});
                if send_callback(ctx, &callback_url, &body).await {
                    if config.logs(Level::Info) {
                        log!(
                            frs.logger,
                            Level::Info,
                            "vstream_key = {key};  an event was sent about facial recognition in a special group: id_sgroup = {id_sgroup}; id_vstream = {}; id_descriptor = {id_descriptor}",
                            config.id_vstream
                        );
                    }
                } else {
                    log!(
                        frs.logger,
                        Level::Error,
                        "vstream_key = {key};  failed to send face recognition event data to special group by callback {callback_url}"
                    );
                }
            }
        }
    }
}

async fn register_descriptor(
    ctx: &Run<'_>,
    key: &str,
    task: &TaskData,
    frame: &RgbImage,
    face_data: &[FaceData],
    best_register_index: Option<usize>,
    result: &mut RegistrationResult,
) {
    let frs = ctx.frs;
    let config = &ctx.config;
    let common = &ctx.common;
    let Some(best) = best_register_index else {
        result.comments = match face_data.first() {
            None => common.comments_no_faces.clone(),
            Some(face) if !face.is_work_area => common.comments_partial_face.clone(),
            Some(face) if !face.is_frontal => common.comments_non_frontal_face.clone(),
            Some(face) if !face.is_non_blurry => common.comments_blurry_face.clone(),
            Some(face) if face.face_class_index != FACE_NORMAL => common.comments_non_normal_face_class.clone(),
            Some(_) => common.comments_inference_error.clone(),
        };
        return;
    };
    let face = &face_data[best];
    let (cols, rows) = (frame.width() as i32, frame.height() as i32);
    let r = enlarge_face_rect(&face.face_rect, f64::from(config.face_enlarge_scale)).intersect(&Rect::new(0, 0, cols, rows));
    let face_image = crop(frame, &r);
    result.id_descriptor = if face.cosine_similarity > 0.999 {
        face.id_descriptor
    } else if task.id_sgroup > 0 {
        add_sgroup_face_descriptor(ctx, task.id_sgroup, &face.descriptor, &face_image).await
    } else {
        add_face_descriptor(ctx, config.id_group, config.id_vstream, &face.descriptor, &face_image, 0).await
    };
    if result.id_descriptor <= 0 {
        result.comments = common.comments_descriptor_creation_error.clone();
        return;
    }
    if face.id_descriptor != result.id_descriptor {
        result.comments = common.comments_new_descriptor.clone();
        if config.logs(Level::Info) {
            log!(frs.logger, Level::Info, "vstream_key = {key};  descriptor created: id = {}", result.id_descriptor);
        }
    } else {
        result.comments = common.comments_descriptor_exists.clone();
        if config.logs(Level::Info) {
            log!(frs.logger, Level::Info, "vstream_key = {key};  descriptor already exists: id = {}", result.id_descriptor);
        }
    }
    result.face_image_jpeg = imaging::encode_jpeg(&face_image).unwrap_or_default();
    result.face_left = face.face_rect.x;
    result.face_top = face.face_rect.y;
    result.face_width = face.face_rect.width;
    result.face_height = face.face_rect.height;
}

/// Stores a descriptor with its face image; a descriptor without a parent is attached to the video stream.
async fn add_face_descriptor(
    ctx: &Run<'_>,
    id_group: i32,
    id_vstream: i32,
    descriptor: &[f32],
    face_image: &RgbImage,
    id_parent: i32,
) -> i32 {
    let frs = ctx.frs;
    let output_size = frs.config.get().common.get(&id_group).map_or(512, |common| common.fr_net.output_size);
    let mut data = descriptor.to_vec();
    data.resize(output_size, 0.0);
    let bytes = descriptor_to_bytes(&data);
    let jpeg = imaging::encode_jpeg(face_image).unwrap_or_default();
    let id_parent = (id_parent > 0).then_some(id_parent);
    let result: Result<i32> = async {
        let mut client = frs.cluster.client(HostType::Master).await?;
        let transaction = client.transaction().await?;
        let id_descriptor: i32 = transaction
            .query_one(
                "insert into face_descriptors(id_group, descriptor_data, id_parent) values($1, $2, $3) returning id_descriptor",
                &[&id_group, &bytes, &id_parent],
            )
            .await?
            .get(0);
        transaction
            .execute(
                "insert into descriptor_images(id_descriptor, mime_type, face_image) values($1, $2, $3)",
                &[&id_descriptor, &MIME_IMAGE, &jpeg],
            )
            .await?;
        if id_parent.is_none() {
            transaction
                .execute(
                    r#"
                    insert into link_descriptor_vstream(id_vstream, id_descriptor)
                        select $2, f.id_descriptor from face_descriptors f where f.id_descriptor = $3 and f.id_group = $1
                    on conflict (id_vstream, id_descriptor) do update set last_updated = now(), flag_deleted = false
                    "#,
                    &[&id_group, &id_vstream, &id_descriptor],
                )
                .await?;
        }
        transaction.commit().await?;
        Ok(id_descriptor)
    }
    .await;
    result.unwrap_or_else(|e| {
        log!(frs.logger, Level::Error, "{e:#}");
        -1
    })
}

async fn add_sgroup_face_descriptor(ctx: &Run<'_>, id_sgroup: i32, descriptor: &[f32], face_image: &RgbImage) -> i32 {
    let frs = ctx.frs;
    let id_group = frs.sg_config.get().by_id(id_sgroup).map_or(-1, |sg| sg.id_group);
    if id_group <= 0 {
        return -1;
    }
    let output_size = frs.config.get().common.get(&id_group).map_or(512, |common| common.fr_net.output_size);
    let mut data = descriptor.to_vec();
    data.resize(output_size, 0.0);
    let bytes = descriptor_to_bytes(&data);
    let jpeg = imaging::encode_jpeg(face_image).unwrap_or_default();
    let result: Result<i32> = async {
        let mut client = frs.cluster.client(HostType::Master).await?;
        let transaction = client.transaction().await?;
        let id_descriptor: i32 = transaction
            .query_one(
                "insert into face_descriptors(id_group, descriptor_data, id_parent) values($1, $2, null) returning id_descriptor",
                &[&id_group, &bytes],
            )
            .await?
            .get(0);
        transaction
            .execute(
                "insert into descriptor_images(id_descriptor, mime_type, face_image) values($1, $2, $3)",
                &[&id_descriptor, &MIME_IMAGE, &jpeg],
            )
            .await?;
        transaction
            .execute(
                "insert into link_descriptor_sgroup(id_sgroup, id_descriptor) values($1, $2)",
                &[&id_sgroup, &id_descriptor],
            )
            .await?;
        transaction.commit().await?;
        Ok(id_descriptor)
    }
    .await;
    result.unwrap_or_else(|e| {
        log!(frs.logger, Level::Error, "{e:#}");
        -1
    })
}

/// Draws the face rectangles and landmarks of an image test.
fn draw_test_markers(frame: &mut RgbImage, face_data: &[FaceData]) {
    let (width, height) = (frame.width() as i32, frame.height() as i32);
    let put = |frame: &mut RgbImage, x: i32, y: i32, color: Rgb<u8>| {
        if (0..width).contains(&x) && (0..height).contains(&y) {
            frame.put_pixel(x as u32, y as u32, color);
        }
    };
    for face in face_data {
        if let Some(landmarks) = &face.landmarks {
            for (k, point) in landmarks.iter().enumerate() {
                // OpenCV's BGR colours of the C++ service, written as RGB
                let k2 = k * 2;
                let blue = if k2 > 2 { 255 } else { 0 };
                let green = if k2 > 0 && k2 < 8 { 255 } else { 0 };
                let red = if k2 < 6 { 255 } else { 0 };
                let (cx, cy) = (point[0] as i32, point[1] as i32);
                for dy in -2..=2 {
                    for dx in -2..=2 {
                        if dx * dx + dy * dy <= 5 {
                            put(frame, cx + dx, cy + dy, Rgb([red, green, blue]));
                        }
                    }
                }
            }
        }
        let r = face.face_rect;
        let color = Rgb([0, 200, 0]);
        for x in r.x..r.x + r.width {
            put(frame, x, r.y, color);
            put(frame, x, r.y + r.height - 1, color);
        }
        for y in r.y..r.y + r.height {
            put(frame, r.x, y, color);
            put(frame, r.x + r.width - 1, y, color);
        }
    }
}

fn dnn_stats_path() -> PathBuf {
    std::env::current_dir().unwrap_or_default().join("dnn_stats_data.json")
}

/// Loads inference statistics saved by a previous run from the working directory.
pub fn load_dnn_stats(frs: &Frs) {
    let path = dnn_stats_path();
    let Ok(text) = std::fs::read_to_string(&path) else { return };
    let parsed = || -> Option<BTreeMap<i32, DnnStats>> {
        let json: Value = serde_json::from_str(&text).ok()?;
        json.get("data")?
            .as_array()?
            .iter()
            .map(|item| {
                let field = |name: &str| item.get(name).and_then(Value::as_i64);
                Some((
                    i32::try_from(field("id_vstream")?).ok()?,
                    DnnStats {
                        fd_count: field("fd_count")?,
                        fc_count: field("fc_count")?,
                        fr_count: field("fr_count")?,
                    },
                ))
            })
            .collect()
    };
    match parsed() {
        Some(stats) => {
            // The totals also include removed video streams
            let all = |name: &str| {
                serde_json::from_str::<Value>(&text)
                    .ok()
                    .and_then(|json| json.get("all")?.get(name)?.as_i64())
                    .unwrap_or(0)
            };
            let sum = |field: fn(&DnnStats) -> i64| stats.values().map(field).sum::<i64>();
            *lock(&frs.workflows.removed_dnn_stats) = DnnStats {
                fd_count: (all("fd_count") - sum(|s| s.fd_count)).max(0),
                fc_count: (all("fc_count") - sum(|s| s.fc_count)).max(0),
                fr_count: (all("fr_count") - sum(|s| s.fr_count)).max(0),
            };
            *lock(&frs.workflows.dnn_stats) = stats;
        }
        None => log!(frs.logger, Level::Error, "invalid inference statistics in {}", path.display()),
    }
}

/// Saves inference statistics per video stream and in total to the working directory.
pub async fn save_dnn_stats(frs: &Frs) {
    let (all, data) = {
        let stats = lock(&frs.workflows.dnn_stats);
        let mut all = *lock(&frs.workflows.removed_dnn_stats);
        let data: Vec<Value> = stats
            .iter()
            .map(|(id_vstream, s)| {
                all.fd_count += s.fd_count;
                all.fc_count += s.fc_count;
                all.fr_count += s.fr_count;
                json!({"id_vstream": id_vstream, "fd_count": s.fd_count, "fc_count": s.fc_count, "fr_count": s.fr_count})
            })
            .collect();
        (all, data)
    };
    let json = json!({
        "all": {"fd_count": all.fd_count, "fc_count": all.fc_count, "fr_count": all.fr_count},
        "data": data,
    });
    if let Err(e) = tokio::fs::write(dnn_stats_path(), json.to_string()).await {
        log!(frs.logger, Level::Error, "{e}");
    }
}

/// Image and data file extensions removed by the screenshot and event cleanups.
const LOG_FILE_EXTENSIONS: [&str; 8] = ["png", "jpg", "jpeg", "bmp", "ppm", "tiff", "dat", "json"];

/// Removes files with one of `extensions` under `dir` last modified before `before`.
pub fn remove_old_files(frs: &Frs, dir: &str, extensions: &[&str], before: SystemTime) {
    let mut pending = vec![PathBuf::from(dir)];
    while let Some(dir) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(file_type) = entry.file_type() else { continue };
            if file_type.is_dir() {
                pending.push(path);
                continue;
            }
            let matches = path.extension().and_then(|e| e.to_str()).is_some_and(|e| extensions.contains(&e));
            let old = entry.metadata().and_then(|m| m.modified()).is_ok_and(|modified| modified < before);
            if file_type.is_file() && matches && old {
                if let Err(e) = std::fs::remove_file(&path) {
                    log!(frs.logger, Level::Error, "{e}");
                }
            }
        }
    }
}

pub async fn old_log_faces_maintenance(frs: &Frs) {
    log!(frs.logger, Level::Info, "Removing obsolete entries from the log_faces table");
    let before = SystemTime::now() - frs.local.log_faces_ttl;
    let result = frs
        .cluster
        .execute(HostType::Master, "delete from log_faces where log_date < $1", &[&DateTime::<Utc>::from(before)])
        .await;
    if let Err(e) = result {
        log!(frs.logger, Level::Error, "{e}");
        return;
    }
    log!(frs.logger, Level::Info, "Removing outdated face screenshots and data");
    let dir = frs.local.screenshots_path.clone();
    tokio::task::block_in_place(|| remove_old_files(frs, &dir, &LOG_FILE_EXTENSIONS, before));
}

pub async fn flag_deleted_maintenance(frs: &Frs) {
    log!(frs.logger, Level::Debug, "Deleting marked records from the database");
    let before = DateTime::<Utc>::from(SystemTime::now() - frs.local.flag_deleted_ttl);
    let result: Result<()> = async {
        let mut client = frs.cluster.client(HostType::Master).await?;
        let transaction = client.transaction().await?;
        for table in [
            "video_streams",
            "face_descriptors",
            "link_descriptor_vstream",
            "special_groups",
            "link_descriptor_sgroup",
        ] {
            let query = format!("delete from {table} where flag_deleted and last_updated < $1");
            transaction.execute(query.as_str(), &[&before]).await?;
        }
        transaction.commit().await?;
        Ok(())
    }
    .await;
    if let Err(e) = result {
        log!(frs.logger, Level::Error, "{e:#}");
    }
    clean_up_workflow_data(frs);
}

/// Drops the in-memory data of video streams that no longer need it, so that it does not grow
/// with time: expired unknown descriptors, and the unknown descriptors and per-stream inference
/// statistics of removed video streams (their statistics are kept in the totals).
fn clean_up_workflow_data(frs: &Frs) {
    let id_vstreams: HashSet<i32> = frs.vstreams_config.get().values().map(|config| config.id_vstream).collect();
    prune_unknown_descriptors(&mut lock(&frs.workflows.unknown_descriptors), &id_vstreams, Instant::now());
    let mut removed = lock(&frs.workflows.removed_dnn_stats);
    prune_dnn_stats(&mut lock(&frs.workflows.dnn_stats), &mut removed, &id_vstreams);
}

/// Drops expired unknown descriptors, then empty lists and the lists of removed video streams.
fn prune_unknown_descriptors(unknown: &mut HashMap<i32, Vec<UnknownDescriptor>>, id_vstreams: &HashSet<i32>, now: Instant) {
    unknown.retain(|id_vstream, list| {
        list.retain(|item| item.expiration >= now);
        !list.is_empty() && id_vstreams.contains(id_vstream)
    });
}

/// Moves the statistics of removed video streams into the totals; zero is the id of tasks without
/// a video stream and is kept.
fn prune_dnn_stats(stats: &mut BTreeMap<i32, DnnStats>, removed: &mut DnnStats, id_vstreams: &HashSet<i32>) {
    stats.retain(|id_vstream, s| {
        let keep = *id_vstream == 0 || id_vstreams.contains(id_vstream);
        if !keep {
            removed.fd_count += s.fd_count;
            removed.fc_count += s.fc_count;
            removed.fr_count += s.fr_count;
        }
        keep
    });
}

/// Copies the data of events marked by `bestQuality` from the face logs to the events directory.
pub async fn copy_events_maintenance(frs: &Frs) {
    log!(frs.logger, Level::Debug, "Copying event data");
    let rows = frs
        .cluster
        .query(
            HostType::Master,
            r#"
            select
              l.id_log,
              v.id_group,
              l.id_vstream,
              l.log_uuid::varchar,
              l.ext_event_uuid,
              l.log_date
            from
              log_faces l
              inner join video_streams v
                on v.id_vstream = l.id_vstream
            where
              l.copy_data = 1
            "#,
            &[],
        )
        .await;
    let rows = match rows {
        Ok(rows) => rows,
        Err(e) => {
            log!(frs.logger, Level::Error, "{e}");
            return;
        }
    };
    for row in rows {
        if let Err(e) = copy_event(frs, &row).await {
            log!(frs.logger, Level::Error, "{e:#}");
        }
    }
}

async fn copy_event(frs: &Frs, row: &pg_cluster::tokio_postgres::Row) -> Result<()> {
    let local = &frs.local;
    let id_log: i32 = row.try_get(0)?;
    let id_group: i32 = row.try_get::<_, Option<i32>>(1)?.context("video stream without a stream group")?;
    let s_uuid = row.try_get::<_, Option<String>>(3)?.unwrap_or_default().replace('-', "");
    let ext_event_uuid: Option<String> = row.try_get(4)?;
    let log_date: DateTime<Utc> = row.try_get(5)?;
    let c: Vec<char> = s_uuid.chars().collect();
    if c.len() < 4 {
        return Ok(());
    }
    let mut suffix = format!("group_{id_group}/{}/{}/{}/{}/", c[0], c[1], c[2], c[3]);
    let mut source = format!("{}{suffix}", local.screenshots_path);
    if !Path::new(&format!("{source}{s_uuid}.json")).exists() {
        // The shorter path of the old project
        suffix = format!("group_{id_group}/{}/{}/{}/", c[0], c[1], c[2]);
        source = format!("{}{suffix}", local.screenshots_path);
    }
    let source_json = format!("{source}{s_uuid}.json");
    if !Path::new(&source_json).exists() {
        return Ok(());
    }
    let target = format!("{}{suffix}", local.events_path);
    std::fs::create_dir_all(&target)?;

    // The JSON data with the external event uuid added
    let mut event: Value = serde_json::from_slice(&std::fs::read(&source_json)?)?;
    if let Value::Object(members) = &mut event {
        members.insert("event_uuid".to_string(), json!(ext_event_uuid.unwrap_or_default()));
    }
    std::fs::write(format!("{target}{s_uuid}.json"), event.to_string())?;

    // The descriptors are appended to the stream group's file of the event's local date
    let Ok(data) = std::fs::read(format!("{source}{s_uuid}.dat")) else { return Ok(()) };
    if data.is_empty() {
        return Ok(());
    }
    let events_file = format!(
        "{}group_{id_group}/{}.dat",
        local.events_path,
        log_date.with_timezone(&Local).format("%Y-%m-%d")
    );
    {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new().create(true).append(true).open(&events_file)?;
        file.write_all(&data)?;
    }
    frs.cluster
        .execute(HostType::Master, "update log_faces set copy_data = 2 where id_log = $1", &[&id_log])
        .await?;
    Ok(())
}

pub async fn old_events_maintenance(frs: &Frs) {
    log!(frs.logger, Level::Info, "Removing outdated events");
    let before = SystemTime::now() - frs.local.events_ttl;
    let dir = frs.local.events_path.clone();
    tokio::task::block_in_place(|| remove_old_files(frs, &dir, &LOG_FILE_EXTENSIONS, before));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unknown(expiration: Instant) -> UnknownDescriptor {
        UnknownDescriptor {
            expiration,
            descriptor: vec![1.0],
            face_image: RgbImage::new(1, 1),
        }
    }

    #[test]
    fn unknown_descriptors_are_pruned() {
        let now = Instant::now();
        let (past, future) = (now - Duration::from_secs(1), now + Duration::from_secs(5));
        let mut map = HashMap::from([
            (1, vec![unknown(past), unknown(future)]),
            (2, vec![unknown(past)]),
            (3, vec![unknown(future)]),
        ]);
        // Video stream 3 was removed
        prune_unknown_descriptors(&mut map, &HashSet::from([1, 2]), now);
        assert_eq!(map.keys().copied().collect::<Vec<_>>(), vec![1]);
        assert_eq!(map[&1].len(), 1);
    }

    #[test]
    fn statistics_of_removed_streams_move_to_totals() {
        let stats = |n| DnnStats {
            fd_count: n,
            fc_count: n,
            fr_count: n,
        };
        let mut map = BTreeMap::from([(0, stats(1)), (5, stats(10)), (6, stats(100))]);
        let mut removed = stats(1000);
        prune_dnn_stats(&mut map, &mut removed, &HashSet::from([5]));
        assert_eq!(map.keys().copied().collect::<Vec<_>>(), vec![0, 5]);
        assert_eq!(removed.fd_count, 1100);
    }
}
