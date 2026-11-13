//! FRS configs: a stream group's common config and a video stream's config, read from their
//! stored JSON forms through their schemas, with the C++ service's defaults.

use std::sync::LazyLock;
use std::time::Duration;

use serde_json::{Value, json};

use crate::config_schema::{Kind, Legacy, Schema, SchemaBuilder};
use crate::logging::Level;

/// The models of the common config: key prefix, model name, input width and height, input and
/// output tensor names, output size.
const MODELS: [(&str, &str, i64, i64, &str, &str, i64); 4] = [
    ("dnn-fd", "scrfd", 320, 320, "input.1", "", 0),
    ("dnn-fc", "genet", 192, 192, "input.1", "419", 3),
    ("dnn-fr", "arcface", 112, 112, "input.1", "683", 512),
    ("dnn-bd", "barcode_detection", 320, 320, "images", "output0", 0),
];

/// API messages of the common config with their defaults.
const COMMENTS: [(&str, &str); 9] = [
    ("comments-blurry-face", "The face image is not clear enough for registration."),
    ("comments-descriptor-creation-error", "Failed to register descriptor."),
    ("comments-descriptor-exists", "The descriptor already exists."),
    ("comments-inference-error", "Error: Triton Inference Server request failed."),
    ("comments-new-descriptor", "A new descriptor has been created."),
    ("comments-no-faces", "There are no faces in the image."),
    ("comments-non-frontal-face", "The face in the image must be frontal."),
    ("comments-non-normal-face-class", "A person wearing a mask or dark glasses."),
    ("comments-partial-face", "The face must be fully visible in the image."),
];

/// The schema of a stream group's common config. The C++ service read it with typed `As<T>`.
pub static COMMON_SCHEMA: LazyLock<Schema> = LazyLock::new(|| {
    let mut builder = SchemaBuilder::new("FRS common config", Legacy::Typed)
        .duration("callback-timeout", "2s")
        .bool("flag-copy-event-data", false);
    for (prefix, name, width, height, input, output, size) in MODELS {
        builder = builder
            .string(format!("{prefix}-model-name"), name)
            .int(format!("{prefix}-input-width"), 1, width)
            .int(format!("{prefix}-input-height"), 1, height)
            .string(format!("{prefix}-input-tensor-name"), input)
            .string(format!("{prefix}-output-tensor-name"), output)
            .int(format!("{prefix}-output-size"), 0, size);
    }
    for (key, default) in COMMENTS {
        builder = builder.string(key, default);
    }
    builder.int("sg-max-descriptor-count", 0, 1000).build()
});

/// The schema of a video stream config and of a default video stream config.
pub static VSTREAM_SCHEMA: LazyLock<Schema> = LazyLock::new(|| {
    SchemaBuilder::new("FRS video stream config", Legacy::Lenient)
        .duration("best-quality-interval-after", "2s")
        .duration("best-quality-interval-before", "5s")
        .non_negative("blur", 300.0)
        .non_negative("blur-max", 13_000.0)
        .duration("capture-timeout", "2s")
        .duration("delay-after-error", "30s")
        .duration("delay-between-frames", "1s")
        .string("dnn-fd-inference-server", "127.0.0.1:8001")
        .string("dnn-fc-inference-server", "127.0.0.1:8001")
        .string("dnn-fr-inference-server", "127.0.0.1:8001")
        .string("dnn-bd-inference-server", "127.0.0.1:8001")
        .fraction("face-class-confidence", 0.7)
        .fraction("face-confidence", 0.7)
        .fraction("face-iou-threshold", 0.4)
        .non_negative("face-enlarge-scale", 1.5)
        .key("logs-level", Kind::Level, json!("info"))
        .non_negative("margin", 5.0)
        .key(
            "max-capture-error-count",
            Kind::Int {
                min: 0,
                max: i64::from(i16::MAX),
            },
            json!(3),
        )
        .non_negative("tolerance", 0.5)
        .string("title", "")
        .fraction("title-height-ratio", 0.033)
        .string("osd-datetime-format", "%Y-%m-%d %H:%M:%S")
        .key("work-area", Kind::FrsWorkArea, json!([]))
        .duration("workflow-timeout", "0s")
        .bool("flag-spawned-descriptors", false)
        .bool("flag-process-faces", true)
        .bool("flag-process-barcodes", false)
        .duration("unknown-descriptor-ttl", "5s")
        .fraction("barcode-confidence", 0.8)
        .duration("inference-timeout", "1s")
        .build()
});

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

impl CommonConfig {
    pub fn from_json(json: &Value) -> Self {
        let schema = &*COMMON_SCHEMA;
        let string_of = |key: &str| schema.string(json, key);
        let model = |prefix: &str| {
            let key = |suffix: &str| format!("{prefix}-{suffix}");
            ModelConfig {
                model_name: string_of(&key("model-name")),
                input_width: schema.i64(json, &key("input-width")) as u32,
                input_height: schema.i64(json, &key("input-height")) as u32,
                input_tensor_name: string_of(&key("input-tensor-name")),
                output_tensor_name: string_of(&key("output-tensor-name")),
                output_size: schema.i64(json, &key("output-size")) as usize,
            }
        };
        Self {
            callback_timeout: schema.duration(json, "callback-timeout"),
            flag_copy_event_data: schema.bool(json, "flag-copy-event-data"),
            fd_net: model("dnn-fd"),
            fc_net: model("dnn-fc"),
            fr_net: model("dnn-fr"),
            bd_net: model("dnn-bd"),
            comments_blurry_face: string_of("comments-blurry-face"),
            comments_descriptor_creation_error: string_of("comments-descriptor-creation-error"),
            comments_descriptor_exists: string_of("comments-descriptor-exists"),
            comments_inference_error: string_of("comments-inference-error"),
            comments_new_descriptor: string_of("comments-new-descriptor"),
            comments_no_faces: string_of("comments-no-faces"),
            comments_non_frontal_face: string_of("comments-non-frontal-face"),
            comments_non_normal_face_class: string_of("comments-non-normal-face-class"),
            comments_partial_face: string_of("comments-partial-face"),
            sg_max_descriptor_count: schema.i64(json, "sg-max-descriptor-count") as i32,
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

impl VStreamConfig {
    pub fn from_json(json: &Value) -> Self {
        let schema = &*VSTREAM_SCHEMA;
        let f32_of = |key: &str| schema.f32(json, key);
        let duration_of = |key: &str| schema.duration(json, key);
        let string_of = |key: &str| schema.string(json, key);
        let bool_of = |key: &str| schema.bool(json, key);
        let work_area = schema.value(json, "work-area");
        Self {
            best_quality_interval_after: duration_of("best-quality-interval-after"),
            best_quality_interval_before: duration_of("best-quality-interval-before"),
            blur: f32_of("blur"),
            blur_max: f32_of("blur-max"),
            capture_timeout: duration_of("capture-timeout"),
            delay_after_error: duration_of("delay-after-error"),
            delay_between_frames: duration_of("delay-between-frames"),
            fd_inference_server: string_of("dnn-fd-inference-server"),
            fc_inference_server: string_of("dnn-fc-inference-server"),
            fr_inference_server: string_of("dnn-fr-inference-server"),
            bd_inference_server: string_of("dnn-bd-inference-server"),
            face_class_confidence: f32_of("face-class-confidence"),
            face_confidence: f32_of("face-confidence"),
            face_iou_threshold: f32_of("face-iou-threshold"),
            face_enlarge_scale: f32_of("face-enlarge-scale"),
            logs_level: schema.level(json, "logs-level"),
            margin: f32_of("margin"),
            max_capture_error_count: schema.i64(json, "max-capture-error-count") as u32,
            tolerance: f32_of("tolerance"),
            title: string_of("title"),
            title_height_ratio: f32_of("title-height-ratio"),
            osd_dt_format: string_of("osd-datetime-format"),
            work_area: work_area
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|n| n.as_f64().map(|n| n as f32))
                .collect(),
            workflow_timeout: duration_of("workflow-timeout"),
            flag_spawned_descriptors: bool_of("flag-spawned-descriptors"),
            flag_process_faces: bool_of("flag-process-faces"),
            flag_process_barcodes: bool_of("flag-process-barcodes"),
            unknown_descriptor_ttl: duration_of("unknown-descriptor-ttl"),
            barcode_confidence: f32_of("barcode-confidence"),
            inference_timeout: duration_of("inference-timeout"),
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
        // A number given as a string is a valid form
        assert_eq!(common.sg_max_descriptor_count, 5);
        assert_eq!(common.comments_no_faces, "none");

        let config = VStreamConfig::from_json(&json!({
            "blur": "150",
            "logs-level": "trace",
            "work-area": [10, 20, 50.5, 60],
            "flag-process-barcodes": true,
            "flag-process-faces": "false",
        }));
        assert_eq!(config.blur, 150.0);
        assert_eq!(config.logs_level, Level::Trace);
        assert_eq!(config.work_area, vec![10.0, 20.0, 50.5, 60.0]);
        assert!(config.flag_process_barcodes);
        // A boolean given as a string is invalid: the default is used
        assert!(config.flag_process_faces);
    }
}
