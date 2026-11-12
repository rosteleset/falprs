//! Typed video stream config, read from its stored JSON form through its schema, with the C++
//! service's defaults.

use std::sync::LazyLock;
use std::time::Duration;

use serde_json::{Value, json};

use crate::config_schema::{Kind, Legacy, Schema, SchemaBuilder};
use crate::logging::Level;

/// The models: key prefix, model name, input width and height, input and output tensor names.
const MODELS: [(&str, &str, i64, i64, &str, &str); 6] = [
    ("vd-net", "vdnet_yolo", 640, 640, "images", "output0"),
    ("vc-net", "vcnet_vit", 224, 224, "input", "output"),
    ("lpd-net", "lpdnet_yolo", 640, 640, "images", "output0"),
    ("lpc-net", "lpcnet_vit", 224, 224, "input", "output"),
    ("lpr-net", "lprnet_yolo", 320, 320, "images", "output0"),
    ("uod-net", "uodnet", 640, 640, "images", "output0"),
];

/// The schema of a video stream config and of a default video stream config.
pub static SCHEMA: LazyLock<Schema> = LazyLock::new(|| {
    let mut builder = SchemaBuilder::new("LPRS video stream config", Legacy::Lenient);
    for (prefix, name, width, height, input, output) in MODELS {
        builder = builder
            .string(format!("{prefix}-inference-server"), "127.0.0.1:8001")
            .string(format!("{prefix}-model-name"), name)
            .int(format!("{prefix}-input-width"), 1, width)
            .int(format!("{prefix}-input-height"), 1, height)
            .string(format!("{prefix}-input-tensor-name"), input)
            .string(format!("{prefix}-output-tensor-name"), output);
    }
    builder
        .fraction("uod-net-nms-iou-threshold", 0.7)
        .fraction("uod-net-keypoint-confidence", 0.5)
        .fraction("uod-net-nested-box-overlap", 0.85)
        .fraction("uod-net-nested-footprint-overlap", 0.3)
        .duration("callback-timeout", "2s")
        .fraction("plate-confidence", 0.6)
        .fraction("char-score", 0.4)
        .fraction("char-iou-threshold", 0.7)
        .int("max-capture-error-count", 0, 3)
        .fraction("vehicle-confidence", 0.6)
        .fraction("vehicle-iou-threshold", 0.45)
        .fraction("vehicle-area-ratio-threshold", 0.01)
        .fraction("special-confidence", 0.7)
        .duration("capture-timeout", "2s")
        .duration("event-log-before", "10s")
        .duration("event-log-after", "5s")
        .duration("delay-between-frames", "1s")
        .duration("ban-duration", "30s")
        .duration("ban-duration-area", "12h")
        .fraction("ban-iou-threshold", 0.9)
        .duration("delay-after-error", "30s")
        .key("logs-level", Kind::Level, json!("info"))
        .int("min-plate-height", 0, 0)
        .key("work-area", Kind::LprsWorkArea, json!([]))
        .bool("flag-save-failed", false)
        .bool("flag-process-special", false)
        .duration("workflow-timeout", "0s")
        .duration("inference-timeout", "1s")
        .string("screenshot-url", "")
        .string("callback-url", "")
        .build()
});

/// A model served by Triton.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub inference_server: String,
    pub model_name: String,
    pub input_width: u32,
    pub input_height: u32,
    pub input_tensor_name: String,
    pub output_tensor_name: String,
}

impl ModelConfig {
    /// Reads the `<prefix>-*` keys.
    fn read(json: &Value, prefix: &str) -> Self {
        let key = |suffix: &str| format!("{prefix}-{suffix}");
        Self {
            inference_server: SCHEMA.string(json, &key("inference-server")),
            model_name: SCHEMA.string(json, &key("model-name")),
            input_width: SCHEMA.i64(json, &key("input-width")) as u32,
            input_height: SCHEMA.i64(json, &key("input-height")) as u32,
            input_tensor_name: SCHEMA.string(json, &key("input-tensor-name")),
            output_tensor_name: SCHEMA.string(json, &key("output-tensor-name")),
        }
    }
}

/// A polygon of the work area in percent of the frame size.
pub type WorkAreaPolygon = Vec<[f32; 2]>;

#[derive(Debug, Clone)]
pub struct VStreamConfig {
    /// Vehicle detection.
    pub vd_net: ModelConfig,
    /// Special vehicle classification.
    pub vc_net: ModelConfig,
    /// Plate detection.
    pub lpd_net: ModelConfig,
    /// Plate classification.
    pub lpc_net: ModelConfig,
    /// Plate recognition.
    pub lpr_net: ModelConfig,
    /// Vehicle footprint detection.
    pub uod_net: ModelConfig,
    /// NMS of vehicle footprint detections suppresses a box whose IoU with a more confident one of its class is
    /// above this.
    pub uod_net_nms_iou_threshold: f32,
    /// The probability every cuboid keypoint of a vehicle footprint needs; 0 disables the test.
    pub uod_net_keypoint_confidence: f32,
    /// How much of a vehicle's box must lie in a larger vehicle's box to make it a nested duplicate; 0 disables
    /// the suppression of nested duplicates.
    pub uod_net_nested_box_overlap: f32,
    /// How much a nested duplicate's footprint must overlap the larger vehicle's (of the smaller footprint).
    pub uod_net_nested_footprint_overlap: f32,

    pub callback_timeout: Duration,
    pub plate_confidence: f32,
    pub char_score: f32,
    pub char_iou_threshold: f32,
    pub max_capture_error_count: u32,
    pub vehicle_confidence: f32,
    pub vehicle_iou_threshold: f32,
    pub vehicle_area_ratio_threshold: f32,
    pub special_confidence: f32,
    pub capture_timeout: Duration,
    pub event_log_before: Duration,
    pub event_log_after: Duration,
    pub delay_between_frames: Duration,
    pub ban_duration: Duration,
    pub ban_duration_area: Duration,
    pub ban_iou_threshold: f32,
    pub delay_after_error: Duration,
    pub logs_level: Level,
    pub min_plate_height: f32,
    pub work_area: Vec<WorkAreaPolygon>,
    pub flag_save_failed: bool,
    pub flag_process_special: bool,
    pub workflow_timeout: Duration,
    pub inference_timeout: Duration,
    pub screenshot_url: String,
    pub callback_url: String,
}

fn read_work_area(value: &Value) -> Vec<WorkAreaPolygon> {
    let point = |point: &Value| [point[0].as_f64().unwrap_or_default() as f32, point[1].as_f64().unwrap_or_default() as f32];
    value
        .as_array()
        .into_iter()
        .flatten()
        .map(|polygon| polygon.as_array().into_iter().flatten().map(point).collect())
        .collect()
}

impl VStreamConfig {
    pub fn from_json(json: &Value) -> Self {
        let schema = &*SCHEMA;
        let f32_of = |key: &str| schema.f32(json, key);
        let duration_of = |key: &str| schema.duration(json, key);
        Self {
            vd_net: ModelConfig::read(json, "vd-net"),
            vc_net: ModelConfig::read(json, "vc-net"),
            lpd_net: ModelConfig::read(json, "lpd-net"),
            lpc_net: ModelConfig::read(json, "lpc-net"),
            lpr_net: ModelConfig::read(json, "lpr-net"),
            uod_net: ModelConfig::read(json, "uod-net"),
            uod_net_nms_iou_threshold: f32_of("uod-net-nms-iou-threshold"),
            uod_net_keypoint_confidence: f32_of("uod-net-keypoint-confidence"),
            uod_net_nested_box_overlap: f32_of("uod-net-nested-box-overlap"),
            uod_net_nested_footprint_overlap: f32_of("uod-net-nested-footprint-overlap"),

            callback_timeout: duration_of("callback-timeout"),
            plate_confidence: f32_of("plate-confidence"),
            char_score: f32_of("char-score"),
            char_iou_threshold: f32_of("char-iou-threshold"),
            max_capture_error_count: schema.i64(json, "max-capture-error-count") as u32,
            vehicle_confidence: f32_of("vehicle-confidence"),
            vehicle_iou_threshold: f32_of("vehicle-iou-threshold"),
            vehicle_area_ratio_threshold: f32_of("vehicle-area-ratio-threshold"),
            special_confidence: f32_of("special-confidence"),
            capture_timeout: duration_of("capture-timeout"),
            event_log_before: duration_of("event-log-before"),
            event_log_after: duration_of("event-log-after"),
            delay_between_frames: duration_of("delay-between-frames"),
            ban_duration: duration_of("ban-duration"),
            ban_duration_area: duration_of("ban-duration-area"),
            ban_iou_threshold: f32_of("ban-iou-threshold"),
            delay_after_error: duration_of("delay-after-error"),
            logs_level: schema.level(json, "logs-level"),
            min_plate_height: schema.i64(json, "min-plate-height") as f32,
            work_area: read_work_area(&schema.value(json, "work-area")),
            flag_save_failed: schema.bool(json, "flag-save-failed"),
            flag_process_special: schema.bool(json, "flag-process-special"),
            workflow_timeout: duration_of("workflow-timeout"),
            inference_timeout: duration_of("inference-timeout"),
            screenshot_url: schema.string(json, "screenshot-url"),
            callback_url: schema.string(json, "callback-url"),
        }
    }

    /// Whether a message of `level` should be logged for this video stream.
    pub fn logs(&self, level: Level) -> bool {
        self.logs_level <= level
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn defaults() {
        let config = VStreamConfig::from_json(&json!({}));
        assert_eq!(config.vd_net.model_name, "vdnet_yolo");
        assert_eq!(config.lpr_net.input_width, 320);
        assert_eq!(config.uod_net.model_name, "uodnet");
        assert_eq!((config.uod_net.input_width, config.uod_net.input_height), (640, 640));
        assert_eq!(config.uod_net_nms_iou_threshold, 0.7);
        assert_eq!(config.uod_net_keypoint_confidence, 0.5);
        assert_eq!(config.uod_net_nested_box_overlap, 0.85);
        assert_eq!(config.uod_net_nested_footprint_overlap, 0.3);
        assert_eq!(config.ban_duration_area, Duration::from_secs(12 * 3600));
        assert_eq!(config.logs_level, Level::Info);
        assert!(config.work_area.is_empty());
    }

    #[test]
    fn overrides() {
        let config = VStreamConfig::from_json(&json!({
            "vd-net-inference-server": "triton:8001",
            "lpc-net-inference-server": "lpc:8001",
            "lpd-net-inference-server": "other:8001",
            "ban-duration": "5s",
            "min-plate-height": 25,
            "char-score": "0.3",
            "logs-level": "trace",
            "work-area": [[[0, 0], [100, 0], [100, 50.5], [0, 50.5]]],
            "flag-process-special": true,
            "uod-net-inference-server": "uod:8001",
            "uod-net-input-width": 960,
            "uod-net-nested-box-overlap": 0,
        }));
        assert_eq!(config.vd_net.inference_server, "triton:8001");
        // Each model has its own server key
        assert_eq!(config.vc_net.inference_server, "127.0.0.1:8001");
        assert_eq!(config.lpc_net.inference_server, "lpc:8001");
        assert_eq!(config.lpd_net.inference_server, "other:8001");
        assert_eq!(config.ban_duration, Duration::from_secs(5));
        assert_eq!(config.min_plate_height, 25.0);
        assert_eq!(config.char_score, 0.3);
        assert_eq!(config.logs_level, Level::Trace);
        assert_eq!(config.work_area, vec![vec![[0.0, 0.0], [100.0, 0.0], [100.0, 50.5], [0.0, 50.5]]]);
        assert!(config.flag_process_special);
        assert_eq!(config.uod_net.inference_server, "uod:8001");
        assert_eq!(config.uod_net.input_width, 960);
        assert_eq!(config.uod_net_nested_box_overlap, 0.0);
    }

    #[test]
    fn invalid_work_area_is_empty() {
        let config = VStreamConfig::from_json(&json!({"work-area": [[[0, 0], [100]]]}));
        assert!(config.work_area.is_empty());
    }
}
