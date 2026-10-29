//! Typed video stream config, read leniently from its JSON form with the C++ service's defaults.

use std::time::Duration;

use serde_json::Value;

use crate::converters::{to_bool, to_duration, to_f32, to_i64, to_string};
use crate::logging::Level;

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
    /// Reads `<prefix>-*` keys; `server_key` names the key holding the inference server address.
    fn read(json: &Value, prefix: &str, server_key: &str, defaults: (&str, u32, u32, &str, &str)) -> Self {
        let (model_name, width, height, input, output) = defaults;
        let key = |suffix: &str| json.get(format!("{prefix}-{suffix}"));
        Self {
            inference_server: to_string(json.get(server_key), "127.0.0.1:8001"),
            model_name: to_string(key("model-name"), model_name),
            input_width: to_i64(key("input-width"), width as i64) as u32,
            input_height: to_i64(key("input-height"), height as i64) as u32,
            input_tensor_name: to_string(key("input-tensor-name"), input),
            output_tensor_name: to_string(key("output-tensor-name"), output),
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
    /// Oriented vehicle detection.
    pub ovd_net: ModelConfig,

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

fn read_work_area(value: Option<&Value>) -> Vec<WorkAreaPolygon> {
    let parse = |value: &Value| -> Option<Vec<WorkAreaPolygon>> {
        value
            .as_array()?
            .iter()
            .map(|polygon| {
                polygon
                    .as_array()?
                    .iter()
                    .map(|point| {
                        let point = point.as_array()?;
                        Some([point.first()?.as_f64()? as f32, point.get(1)?.as_f64()? as f32])
                    })
                    .collect()
            })
            .collect()
    };
    value.and_then(parse).unwrap_or_default()
}

fn read_level(value: Option<&Value>, default: Level) -> Level {
    value.and_then(Value::as_str).and_then(|text| text.parse().ok()).unwrap_or(default)
}

impl VStreamConfig {
    pub fn from_json(json: &Value) -> Self {
        let f32_of = |key: &str, default: f32| to_f32(json.get(key), default);
        let duration_of = |key: &str, default: Duration| to_duration(json.get(key), default);
        let secs = Duration::from_secs;
        Self {
            vd_net: ModelConfig::read(json, "vd-net", "vd-net-inference-server", ("vdnet_yolo", 640, 640, "images", "output0")),
            vc_net: ModelConfig::read(json, "vc-net", "vc-net-inference-server", ("vcnet_vit", 224, 224, "input", "output")),
            lpd_net: ModelConfig::read(json, "lpd-net", "lpd-net-inference-server", ("lpdnet_yolo", 640, 640, "images", "output0")),
            lpc_net: ModelConfig::read(json, "lpc-net", "lpc-net-inference-server", ("lpcnet_vit", 224, 224, "input", "output")),
            lpr_net: ModelConfig::read(json, "lpr-net", "lpr-net-inference-server", ("lprnet_yolo", 320, 320, "images", "output0")),
            ovd_net: ModelConfig::read(json, "ovd-net", "ovd-net-inference-server", ("ovdnet", 320, 320, "images", "output0")),

            callback_timeout: duration_of("callback-timeout", secs(2)),
            plate_confidence: f32_of("plate-confidence", 0.6),
            char_score: f32_of("char-score", 0.4),
            char_iou_threshold: f32_of("char-iou-threshold", 0.7),
            max_capture_error_count: to_i64(json.get("max-capture-error-count"), 3).max(0) as u32,
            vehicle_confidence: f32_of("vehicle-confidence", 0.6),
            vehicle_iou_threshold: f32_of("vehicle-iou-threshold", 0.45),
            vehicle_area_ratio_threshold: f32_of("vehicle-area-ratio-threshold", 0.01),
            special_confidence: f32_of("special-confidence", 0.7),
            capture_timeout: duration_of("capture-timeout", secs(2)),
            event_log_before: duration_of("event-log-before", secs(10)),
            event_log_after: duration_of("event-log-after", secs(5)),
            delay_between_frames: duration_of("delay-between-frames", secs(1)),
            ban_duration: duration_of("ban-duration", secs(30)),
            ban_duration_area: duration_of("ban-duration-area", secs(12 * 3600)),
            ban_iou_threshold: f32_of("ban-iou-threshold", 0.9),
            delay_after_error: duration_of("delay-after-error", secs(30)),
            logs_level: read_level(json.get("logs-level"), Level::Info),
            min_plate_height: to_i64(json.get("min-plate-height"), 0) as f32,
            work_area: read_work_area(json.get("work-area")),
            flag_save_failed: to_bool(json.get("flag-save-failed"), false),
            flag_process_special: to_bool(json.get("flag-process-special"), false),
            workflow_timeout: duration_of("workflow-timeout", Duration::ZERO),
            inference_timeout: duration_of("inference-timeout", secs(1)),
            screenshot_url: to_string(json.get("screenshot-url"), ""),
            callback_url: to_string(json.get("callback-url"), ""),
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
    }

    #[test]
    fn invalid_work_area_is_empty() {
        let config = VStreamConfig::from_json(&json!({"work-area": [[[0, 0], [100]]]}));
        assert!(config.work_area.is_empty());
    }
}
