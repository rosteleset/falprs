//! Barcode detection and QR code decoding in the FRS workflow. Only QR codes are decoded, with
//! `rxing`'s port of the zxing-cpp QR reader: unlike ZXing for other formats, it reads QR codes at
//! least as well as ZXing does, and QR codes are the only format in use.

use std::path::PathBuf;
use std::time::SystemTime;

use anyhow::{Context, Result, bail};
use chrono::{DateTime, Utc};
use image::RgbImage;
use pg_cluster::HostType;
use rxing::multi::MultipleBarcodeReader;
use serde_json::{Value, json};

use super::Frs;
use super::config::{CommonConfig, VStreamConfig};
use super::pipeline::{Rect, crop};
use super::workflow::Screenshot;
use crate::converters::format_time;
use crate::imaging;
use crate::log;
use crate::logging::{Level, StreamLog};
use crate::stream_log;
use crate::triton::{InferRequest, Tensor};

/// A detected barcode: absolute x_min, y_min, x_max, y_max and confidence.
#[derive(Debug, Clone, Copy)]
struct Detection {
    bbox: [f32; 4],
    confidence: f32,
}

/// Columns of the detector output, a `[5, 2100]` tensor of box centre, size and confidence.
const NUM_COLS: usize = 2100;

fn has_intersection(a: &[f32; 4], b: &[f32; 4]) -> bool {
    a[0].max(b[0]) < a[2].min(b[2]) && a[1].max(b[1]) < a[3].min(b[3])
}

/// Keeps the most confident of intersecting detections.
fn nms_barcodes(mut detections: Vec<Detection>) -> Vec<Detection> {
    detections.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
    let mut kept: Vec<Detection> = Vec::with_capacity(detections.len());
    for detection in detections {
        if kept.iter().all(|k| !has_intersection(&k.bbox, &detection.bbox)) {
            kept.push(detection);
        }
    }
    kept
}

async fn detect_barcodes(frs: &Frs, frame: &RgbImage, common: &CommonConfig, config: &VStreamConfig, log: StreamLog<'_>) -> Result<Vec<Detection>> {
    let model = &common.bd_net;
    let (width, height) = (model.input_width, model.input_height);
    let input = tokio::task::block_in_place(|| imaging::prepare_yolo(frame, width, height));
    stream_log!(log, Level::Trace, "after image preprocessing for face detection");
    stream_log!(log, Level::Trace, "before inference barcode detection");
    let outputs = frs
        .triton
        .infer(InferRequest {
            endpoint: &config.bd_inference_server,
            model_name: &model.model_name,
            inputs: vec![Tensor::fp32(&model.input_tensor_name, vec![1, 3, height as i64, width as i64], &input.tensor)],
            outputs: vec![&model.output_tensor_name],
            timeout: config.inference_timeout,
        })
        .await
        .context("Unable to send inference request")?;
    stream_log!(log, Level::Trace, "after inference barcode detection");
    let Some(output) = outputs.first() else {
        bail!("Unable to receive inference result");
    };
    stream_log!(log, Level::Debug, "inference barcode detection OK");
    let data = output.as_f32();
    if data.len() < 5 * NUM_COLS {
        bail!("unexpected barcode detection output size {}", data.len());
    }
    let inv_scale = 1.0 / input.scale as f32;
    let (shift_x, shift_y) = input.shift;
    let (max_x, max_y) = ((frame.width() - 1) as f32, (frame.height() - 1) as f32);
    stream_log!(log, Level::Trace, "barcode confidence threshold: {:.3}", config.barcode_confidence);
    let detections: Vec<Detection> = (0..NUM_COLS)
        .filter(|&j| data[4 * NUM_COLS + j] > config.barcode_confidence)
        .map(|j| {
            let (cx, cy, bw, bh) = (data[j], data[NUM_COLS + j], data[2 * NUM_COLS + j], data[3 * NUM_COLS + j]);
            // The box is widened by a tenth of its size on each side
            Detection {
                bbox: [
                    ((cx - bw / 2.0 - bw / 10.0 - shift_x) * inv_scale).max(0.0),
                    ((cy - bh / 2.0 - bh / 10.0 - shift_y) * inv_scale).max(0.0),
                    ((cx + bw / 2.0 + bw / 10.0 - shift_x) * inv_scale).min(max_x),
                    ((cy + bh / 2.0 + bh / 10.0 - shift_y) * inv_scale).min(max_y),
                ],
                confidence: data[4 * NUM_COLS + j],
            }
        })
        .collect();
    stream_log!(log, Level::Trace, "before nms_barcodes count: {}", detections.len());
    let detections = nms_barcodes(detections);
    stream_log!(log, Level::Trace, "after nms_barcodes count: {}", detections.len());
    Ok(detections)
}

/// ZXing's luminance of the frame as the C++ service passes it: OpenCV's BGR bytes labelled as
/// RGB, so blue is weighted as red.
fn luminance(image: &RgbImage) -> Vec<u8> {
    image
        .pixels()
        .map(|p| ((306 * u32::from(p[2]) + 601 * u32::from(p[1]) + 117 * u32::from(p[0]) + 0x200) >> 10) as u8)
        .collect()
}

/// A decoded QR code: its text and corners (top left, top right, bottom right, bottom left).
struct QrCode {
    text: String,
    corners: [[f32; 2]; 4],
}

fn read_qr_codes(luma: Vec<u8>, width: u32, height: u32) -> Vec<QrCode> {
    let hints = rxing::DecodeHints {
        TryHarder: Some(true),
        PossibleFormats: Some([rxing::BarcodeFormat::QR_CODE].into_iter().collect()),
        ..Default::default()
    };
    let Ok(source) = rxing::Luma8LuminanceSource::new(luma, width, height) else {
        return Vec::new();
    };
    let mut bitmap = rxing::BinaryBitmap::new(rxing::common::HybridBinarizer::new(source));
    let results = rxing::qrcode::cpp_port::QrReader
        .decode_multiple_with_hints(&mut bitmap, &hints)
        .unwrap_or_default();
    results
        .iter()
        .filter_map(|result| {
            // The reader gives the corners in reading order: top left, top right, bottom left, bottom right
            let points = result.getPoints();
            let [tl, tr, bl, br] = [points.first()?, points.get(1)?, points.get(2)?, points.get(3)?];
            Some(QrCode {
                text: result.getText().to_string(),
                corners: [[tl.x, tl.y], [tr.x, tr.y], [br.x, br.y], [bl.x, bl.y]],
            })
        })
        .collect()
}

/// Decodes QR codes, trying the inverted image when the image itself has none, like ZXing's `tryInvert`.
fn decode_qr_codes(image: &RgbImage) -> Vec<QrCode> {
    let luma = luminance(image);
    let codes = read_qr_codes(luma.clone(), image.width(), image.height());
    if !codes.is_empty() {
        return codes;
    }
    read_qr_codes(luma.iter().map(|v| 255 - v).collect(), image.width(), image.height())
}

/// The barcodes' JSON as reported by the C++ service, with positions in frame coordinates.
fn barcodes_json(frame: &RgbImage, detections: &[Detection]) -> Vec<Value> {
    let mut barcodes = Vec::new();
    for detection in detections {
        let b = detection.bbox;
        let (x1, y1, x2, y2) = (b[0] as i32, b[1] as i32, b[2] as i32, b[3] as i32);
        let roi = Rect::new(x1.min(x2), y1.min(y2), (x2 - x1).abs(), (y2 - y1).abs());
        if roi.is_empty() {
            continue;
        }
        for code in decode_qr_codes(&crop(frame, &roi)) {
            // ZXing reports integer positions; the C++ service rounds them shifted by the box origin
            let position: Vec<i64> = code
                .corners
                .iter()
                .flat_map(|p| [(p[0].floor() + b[0]).round() as i64, (p[1].floor() + b[1]).round() as i64])
                .collect();
            barcodes.push(json!({"text": code.text, "format": "QR Code", "position": position}));
        }
    }
    barcodes
}

/// Detects barcodes in a frame, decodes QR codes, logs them and sends the barcode callback.
pub async fn process(
    frs: &Frs,
    vstream_key: &str,
    is_test: bool,
    common: &CommonConfig,
    config: &VStreamConfig,
    frame: &RgbImage,
    screenshot: &Screenshot<'_>,
) {
    let logs = |level: Level| config.logs(level) || is_test;
    let key = vstream_key;
    if logs(Level::Trace) {
        log!(frs.logger, Level::Trace, "vstream_key = {key};  before image preprocessing for barcode detection");
    }
    let stream = StreamLog {
        logger: &frs.logger,
        key,
        level: config.logs_level,
        always: is_test,
    };
    let detections = match detect_barcodes(frs, frame, common, config, stream).await {
        Ok(detections) => detections,
        Err(e) => {
            log!(frs.logger, Level::Error, "Error! {e:#}");
            return;
        }
    };
    if logs(Level::Trace) {
        log!(frs.logger, Level::Trace, "vstream_key = {key};  before decoding barcodes");
    }
    let barcodes = tokio::task::block_in_place(|| barcodes_json(frame, &detections));
    if logs(Level::Trace) {
        log!(frs.logger, Level::Trace, "vstream_key = {key};  after decoding barcodes");
        log!(frs.logger, Level::Trace, "vstream_key = {key};  found barcodes: {}", barcodes.len());
    }
    if barcodes.is_empty() {
        return;
    }

    let log_date = Utc::now();
    let uuid = uuid::Uuid::new_v4().to_string();
    let c: Vec<char> = uuid.chars().take(4).collect();
    let path_suffix = format!("{}/{}/{}/{}/", c[0], c[1], c[2], c[3]);
    let local = &frs.local;
    let info = json!({
        "screenshot": format!("{}{path_suffix}{uuid}.jpg", local.barcodes_url_prefix),
        "date": format_time(log_date),
        "barcodes": barcodes,
    });
    let id_log = add_log_barcode(frs, config.id_vstream, log_date, &info).await;

    let path = PathBuf::from(format!("{}{path_suffix}{uuid}.jpg", local.barcodes_path));
    if let Err(e) = write_screenshot(&path, screenshot.data()) {
        log!(frs.logger, Level::Error, "{e}");
    }

    if id_log > 0 && !config.callback_url_barcodes.is_empty() {
        let body = json!({
            "streamId": config.vstream_ext,
            "date": format_time(log_date),
            "eventId": id_log,
            "barcodes": barcodes,
        });
        let response = frs
            .http
            .post(&config.callback_url_barcodes)
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(body.to_string())
            .timeout(common.callback_timeout)
            .send()
            .await;
        let delivered = match response {
            Ok(response) => matches!(response.status().as_u16(), 200 | 204),
            Err(e) => {
                log!(frs.logger, Level::Error, "{e}");
                false
            }
        };
        if delivered {
            if config.logs(Level::Info) {
                log!(
                    frs.logger,
                    Level::Info,
                    "vstream_key = {key};  barcode recognition event sent: id_vstream = {};",
                    config.id_vstream
                );
            }
        } else if config.logs(Level::Warning) {
            log!(
                frs.logger,
                Level::Warning,
                "vstream_key = {key};  error sending data to callback {}",
                config.callback_url_barcodes
            );
        }
    }
}

fn write_screenshot(path: &std::path::Path, contents: &[u8]) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, contents)?;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o606))
}

async fn add_log_barcode(frs: &Frs, id_vstream: i32, log_date: DateTime<Utc>, info: &Value) -> i64 {
    let result = frs
        .cluster
        .query_one(
            HostType::Master,
            "insert into log_barcodes(id_vstream, log_date, info) values($1, $2, $3) returning id_log",
            &[&id_vstream, &log_date, info],
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

pub async fn old_log_barcodes_maintenance(frs: &Frs) {
    log!(frs.logger, Level::Info, "Removing obsolete entries from the log_barcodes table");
    let before = SystemTime::now() - frs.local.log_barcodes_ttl;
    let result = frs
        .cluster
        .execute(HostType::Master, "delete from log_barcodes where log_date < $1", &[&DateTime::<Utc>::from(before)])
        .await;
    if let Err(e) = result {
        log!(frs.logger, Level::Error, "{e}");
        return;
    }
    log!(frs.logger, Level::Info, "Removing outdated barcode screenshots");
    let dir = frs.local.barcodes_path.clone();
    const IMAGE_EXTENSIONS: [&str; 6] = ["png", "jpg", "jpeg", "bmp", "ppm", "tiff"];
    tokio::task::block_in_place(|| super::workflow::remove_old_files(frs, &dir, &IMAGE_EXTENSIONS, before));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nms_drops_intersecting_boxes() {
        let detection = |x: f32, confidence: f32| Detection {
            bbox: [x, 0.0, x + 10.0, 10.0],
            confidence,
        };
        let kept = nms_barcodes(vec![detection(0.0, 0.85), detection(5.0, 0.95), detection(20.0, 0.9)]);
        let confidences: Vec<f32> = kept.iter().map(|d| d.confidence).collect();
        assert_eq!(confidences, vec![0.95, 0.9]);
    }

    #[test]
    fn qr_code_from_test_image() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../tests/images/qr_code_001.jpg");
        let image = imaging::decode(&std::fs::read(path).unwrap()).unwrap();
        let codes = decode_qr_codes(&image);
        assert_eq!(codes.len(), 1);
        assert_eq!(codes[0].text, "https://example.com/test");
        // ZXing reports the corners (78, 78), (562, 78), (562, 562), (78, 562)
        for (corner, expected) in codes[0].corners.iter().zip([[78.0, 78.0], [562.0, 78.0], [562.0, 562.0], [78.0, 562.0]]) {
            assert!((corner[0] - expected[0]).abs() <= 1.0 && (corner[1] - expected[1]).abs() <= 1.0, "{corner:?}");
        }
    }
}
