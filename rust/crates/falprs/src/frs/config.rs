//! FRS configs: the service-wide local config from `config.yaml`, a stream group's common config
//! and a video stream's config, read leniently from their JSON forms with the C++ service's defaults.

use std::time::Duration;

use serde_json::{Map, Value};

use crate::converters::{to_bool, to_duration, to_f32, to_i64, to_string};
use crate::logging::Level;

/// Settings of the `frs-workflow` component.
#[derive(Debug, Clone)]
pub struct LocalConfig {
    pub allow_group_id_without_auth: i32,
    pub events_path: String,
    pub screenshots_path: String,
    pub screenshots_url_prefix: String,
    pub clear_old_log_faces: Duration,
    pub flag_deleted_maintenance_interval: Duration,
    pub flag_deleted_ttl: Duration,
    pub copy_events_maintenance_interval: Duration,
    pub clear_old_events: Duration,
    pub log_faces_ttl: Duration,
    pub events_ttl: Duration,
    pub barcodes_path: String,
    pub barcodes_url_prefix: String,
    pub clear_old_log_barcodes: Duration,
    pub log_barcodes_ttl: Duration,
}

impl LocalConfig {
    pub fn from_config(config: &Map<String, Value>) -> Self {
        let secs = Duration::from_secs;
        let duration_of = |key: &str, default: Duration| to_duration(config.get(key), default);
        let string_of = |key: &str, default: &str| to_string(config.get(key), default);
        // Directories and URL prefixes are used with a trailing slash
        let dir_of = |key: &str, default: &str| {
            let mut dir = string_of(key, default);
            if !dir.is_empty() && !dir.ends_with('/') {
                dir.push('/');
            }
            dir
        };
        Self {
            allow_group_id_without_auth: to_i64(config.get("allow-group-id-without-auth"), 1) as i32,
            events_path: dir_of("events-path", "/opt/falprs/static/frs/events/"),
            screenshots_path: dir_of("screenshots-path", "/opt/falprs/static/frs/screenshots/"),
            screenshots_url_prefix: dir_of("screenshots-url-prefix", "http://localhost:9051/frs/screenshots/"),
            clear_old_log_faces: duration_of("clear-old-log-faces", secs(3600)),
            flag_deleted_maintenance_interval: duration_of("flag-deleted-maintenance-interval", secs(10)),
            flag_deleted_ttl: duration_of("flag-deleted-ttl", secs(300)),
            copy_events_maintenance_interval: duration_of("copy-events-maintenance-interval", secs(30)),
            clear_old_events: duration_of("clear-old-events", secs(86400)),
            log_faces_ttl: duration_of("log-faces-ttl", secs(4 * 3600)),
            events_ttl: duration_of("events-ttl", secs(30 * 86400)),
            barcodes_path: dir_of("barcodes-path", "/opt/falprs/static/frs/barcodes/"),
            barcodes_url_prefix: dir_of("barcodes-url-prefix", "http://localhost:9051/frs/barcodes/"),
            clear_old_log_barcodes: duration_of("clear-old-log-barcodes", secs(3600)),
            log_barcodes_ttl: duration_of("log-barcodes-ttl", secs(4 * 3600)),
        }
    }
}

/// A model served by Triton; the inference server comes from the video stream config.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub model_name: String,
    pub input_width: u32,
    pub input_height: u32,
    pub input_tensor_name: String,
    pub output_tensor_name: String,
    pub output_size: usize,
}

/// A stream group's common config. Like the C++ service (which reads these with userver's typed
/// `As<T>(default)`), values are taken only when they have the expected JSON type.
#[derive(Debug, Clone)]
pub struct CommonConfig {
    pub callback_timeout: Duration,
    pub flag_copy_event_data: bool,
    /// Face detection.
    pub fd_net: ModelConfig,
    /// Face classification.
    pub fc_net: ModelConfig,
    /// Face recognition (descriptor extraction).
    pub fr_net: ModelConfig,
    /// Barcode detection.
    pub bd_net: ModelConfig,
    pub comments_blurry_face: String,
    pub comments_descriptor_creation_error: String,
    pub comments_descriptor_exists: String,
    pub comments_inference_error: String,
    pub comments_new_descriptor: String,
    pub comments_no_faces: String,
    pub comments_non_frontal_face: String,
    pub comments_non_normal_face_class: String,
    pub comments_partial_face: String,
    pub sg_max_descriptor_count: i32,
}

fn typed_string(json: &Value, key: &str, default: &str) -> String {
    json.get(key).and_then(Value::as_str).unwrap_or(default).to_string()
}

fn typed_i64(json: &Value, key: &str, default: i64) -> i64 {
    json.get(key).and_then(Value::as_i64).unwrap_or(default)
}

impl CommonConfig {
    pub fn from_json(json: &Value) -> Self {
        let string_of = |key: &str, default: &str| typed_string(json, key, default);
        let model = |prefix: &str, defaults: (&str, u32, u32, &str, &str, usize)| {
            let (name, width, height, input, output, size) = defaults;
            let key = |suffix: &str| format!("{prefix}-{suffix}");
            ModelConfig {
                model_name: string_of(&key("model-name"), name),
                input_width: typed_i64(json, &key("input-width"), width as i64) as u32,
                input_height: typed_i64(json, &key("input-height"), height as i64) as u32,
                input_tensor_name: string_of(&key("input-tensor-name"), input),
                output_tensor_name: string_of(&key("output-tensor-name"), output),
                output_size: typed_i64(json, &key("output-size"), size as i64).max(0) as usize,
            }
        };
        Self {
            callback_timeout: to_duration(json.get("callback-timeout"), Duration::from_secs(2)),
            flag_copy_event_data: json.get("flag-copy-event-data").and_then(Value::as_bool).unwrap_or(false),
            fd_net: model("dnn-fd", ("scrfd", 320, 320, "input.1", "", 0)),
            fc_net: model("dnn-fc", ("genet", 192, 192, "input.1", "419", 3)),
            fr_net: model("dnn-fr", ("arcface", 112, 112, "input.1", "683", 512)),
            bd_net: model("dnn-bd", ("barcode_detection", 320, 320, "images", "output0", 0)),
            comments_blurry_face: string_of("comments-blurry-face", "The face image is not clear enough for registration."),
            comments_descriptor_creation_error: string_of("comments-descriptor-creation-error", "Failed to register descriptor."),
            comments_descriptor_exists: string_of("comments-descriptor-exists", "The descriptor already exists."),
            comments_inference_error: string_of("comments-inference-error", "Error: Triton Inference Server request failed."),
            comments_new_descriptor: string_of("comments-new-descriptor", "A new descriptor has been created."),
            comments_no_faces: string_of("comments-no-faces", "There are no faces in the image."),
            comments_non_frontal_face: string_of("comments-non-frontal-face", "The face in the image must be frontal."),
            comments_non_normal_face_class: string_of("comments-non-normal-face-class", "A person wearing a mask or dark glasses."),
            comments_partial_face: string_of("comments-partial-face", "The face must be fully visible in the image."),
            sg_max_descriptor_count: typed_i64(json, "sg-max-descriptor-count", 1000) as i32,
        }
    }
}

impl Default for CommonConfig {
    fn default() -> Self {
        Self::from_json(&Value::Null)
    }
}

#[derive(Debug, Clone)]
pub struct VStreamConfig {
    pub best_quality_interval_after: Duration,
    pub best_quality_interval_before: Duration,
    pub blur: f32,
    pub blur_max: f32,
    pub capture_timeout: Duration,
    pub delay_after_error: Duration,
    pub delay_between_frames: Duration,
    pub fd_inference_server: String,
    pub fc_inference_server: String,
    pub fr_inference_server: String,
    pub bd_inference_server: String,
    pub face_class_confidence: f32,
    pub face_confidence: f32,
    pub face_iou_threshold: f32,
    pub face_enlarge_scale: f32,
    pub logs_level: Level,
    pub margin: f32,
    pub max_capture_error_count: u32,
    pub tolerance: f32,
    /// Drawn with the date on saved screenshots (OSD); no OSD when empty.
    pub title: String,
    pub title_height_ratio: f32,
    pub osd_dt_format: String,
    /// Work area in percent of the frame: left, top, width, height.
    pub work_area: Vec<f32>,
    pub workflow_timeout: Duration,
    pub flag_spawned_descriptors: bool,
    pub flag_process_faces: bool,
    pub flag_process_barcodes: bool,
    pub unknown_descriptor_ttl: Duration,
    pub barcode_confidence: f32,
    pub inference_timeout: Duration,

    // Video stream data
    pub id_group: i32,
    pub id_vstream: i32,
    pub vstream_ext: String,
    pub url: String,
    pub callback_url: String,
    pub callback_url_barcodes: String,
}

fn read_level(value: Option<&Value>, default: Level) -> Level {
    value.and_then(Value::as_str).and_then(|text| text.parse().ok()).unwrap_or(default)
}

/// The C++ service converts `work-area` with a typed `As<std::vector<float>>`, which fails (and
/// fails the whole cache update) on anything but an array of numbers; here such a value is ignored.
fn read_work_area(value: Option<&Value>) -> Vec<f32> {
    value
        .and_then(Value::as_array)
        .and_then(|items| items.iter().map(|item| item.as_f64().map(|v| v as f32)).collect())
        .unwrap_or_default()
}

impl VStreamConfig {
    pub fn from_json(json: &Value) -> Self {
        let f32_of = |key: &str, default: f32| to_f32(json.get(key), default);
        let duration_of = |key: &str, default: Duration| to_duration(json.get(key), default);
        let server_of = |key: &str| to_string(json.get(key), "127.0.0.1:8001");
        let secs = Duration::from_secs;
        Self {
            best_quality_interval_after: duration_of("best-quality-interval-after", secs(2)),
            best_quality_interval_before: duration_of("best-quality-interval-before", secs(5)),
            blur: f32_of("blur", 300.0),
            blur_max: f32_of("blur-max", 13_000.0),
            capture_timeout: duration_of("capture-timeout", secs(2)),
            delay_after_error: duration_of("delay-after-error", secs(30)),
            delay_between_frames: duration_of("delay-between-frames", secs(1)),
            fd_inference_server: server_of("dnn-fd-inference-server"),
            fc_inference_server: server_of("dnn-fc-inference-server"),
            fr_inference_server: server_of("dnn-fr-inference-server"),
            bd_inference_server: server_of("dnn-bd-inference-server"),
            face_class_confidence: f32_of("face-class-confidence", 0.7),
            face_confidence: f32_of("face-confidence", 0.7),
            face_iou_threshold: f32_of("face-iou-threshold", 0.4),
            face_enlarge_scale: f32_of("face-enlarge-scale", 1.5),
            logs_level: read_level(json.get("logs-level"), Level::Info),
            margin: f32_of("margin", 5.0),
            max_capture_error_count: to_i64(json.get("max-capture-error-count"), 3).clamp(0, i16::MAX as i64) as u32,
            tolerance: f32_of("tolerance", 0.5),
            title: to_string(json.get("title"), ""),
            title_height_ratio: f32_of("title-height-ratio", 0.033),
            osd_dt_format: to_string(json.get("osd-datetime-format"), "%Y-%m-%d %H:%M:%S"),
            work_area: read_work_area(json.get("work-area")),
            workflow_timeout: duration_of("workflow-timeout", Duration::ZERO),
            flag_spawned_descriptors: to_bool(json.get("flag-spawned-descriptors"), false),
            flag_process_faces: to_bool(json.get("flag-process-faces"), true),
            flag_process_barcodes: to_bool(json.get("flag-process-barcodes"), false),
            unknown_descriptor_ttl: duration_of("unknown-descriptor-ttl", secs(5)),
            barcode_confidence: f32_of("barcode-confidence", 0.8),
            inference_timeout: duration_of("inference-timeout", secs(1)),
            id_group: 0,
            id_vstream: 0,
            vstream_ext: String::new(),
            url: String::new(),
            callback_url: String::new(),
            callback_url_barcodes: String::new(),
        }
    }

    /// Whether a message of `level` should be logged for this video stream.
    pub fn logs(&self, level: Level) -> bool {
        self.logs_level <= level
    }
}

impl Default for VStreamConfig {
    fn default() -> Self {
        Self::from_json(&Value::Null)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn defaults() {
        let common = CommonConfig::default();
        assert_eq!(common.fr_net.model_name, "arcface");
        assert_eq!(common.fr_net.output_size, 512);
        assert_eq!(common.sg_max_descriptor_count, 1000);
        let config = VStreamConfig::default();
        assert_eq!(config.blur_max, 13_000.0);
        assert_eq!(config.logs_level, Level::Info);
        assert!(config.flag_process_faces);
        assert!(!config.flag_process_barcodes);
    }

    #[test]
    fn overrides() {
        let common = CommonConfig::from_json(&json!({
            "callback-timeout": "500ms",
            "dnn-fc-output-size": 4,
            "sg-max-descriptor-count": "5",
            "comments-no-faces": "none",
        }));
        assert_eq!(common.callback_timeout, Duration::from_millis(500));
        assert_eq!(common.fc_net.output_size, 4);
        // A string is not taken where a number is expected
        assert_eq!(common.sg_max_descriptor_count, 1000);
        assert_eq!(common.comments_no_faces, "none");

        let config = VStreamConfig::from_json(&json!({
            "blur": "150",
            "logs-level": "trace",
            "work-area": [10, 20, 50.5, 60],
            "flag-process-barcodes": "true",
        }));
        assert_eq!(config.blur, 150.0);
        assert_eq!(config.logs_level, Level::Trace);
        assert_eq!(config.work_area, vec![10.0, 20.0, 50.5, 60.0]);
        assert!(config.flag_process_barcodes);
    }
}
