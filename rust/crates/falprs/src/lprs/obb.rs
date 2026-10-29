//! Vehicle detection with oriented bounding boxes (OVDNet, a YOLO OBB model).

use anyhow::{Result, bail};
use image::RgbImage;

use super::vstream_config::VStreamConfig;
use crate::imaging::letterbox_yolo_obb;
use crate::triton::{InferRequest, Tensor, Triton};

pub const CLASSES: [&str; 5] = ["car", "van", "truck", "bus", "freight car"];

#[derive(Debug, Clone)]
pub struct VehicleObb {
    /// Axis-aligned box `[x1, y1, x2, y2]` clipped to the image.
    pub bbox: [f32; 4],
    pub angle: f32,
    /// Corner points `[x, y]` of the rotated box.
    pub corners: [[f32; 2]; 4],
    pub confidence: f32,
    pub class_id: usize,
}

/// Rows of the model output: `[cx, cy, w, h, confidence, class id, angle]`.
const OUTPUT_ROWS: usize = 300;
const OUTPUT_COLUMNS: usize = 7;

pub async fn detect(triton: &Triton, image: &RgbImage, config: &VStreamConfig) -> Result<Vec<VehicleObb>> {
    let model = &config.ovd_net;
    let target_size = model.input_width.max(model.input_height);
    let letterbox = tokio::task::block_in_place(|| letterbox_yolo_obb(image, target_size));
    let shape = vec![1, 3, letterbox.height as i64, letterbox.width as i64];
    let outputs = triton
        .infer(InferRequest {
            endpoint: &model.inference_server,
            model_name: &model.model_name,
            inputs: vec![Tensor::fp32(&model.input_tensor_name, shape, &letterbox.tensor)],
            outputs: vec![&model.output_tensor_name],
            timeout: config.inference_timeout,
        })
        .await?;
    let Some(output) = outputs.first() else {
        bail!("inference returned no outputs");
    };
    let data = output.as_f32();
    if data.len() < OUTPUT_ROWS * OUTPUT_COLUMNS {
        bail!("unexpected output size {}", data.len());
    }

    let inv_scale = 1.0 / letterbox.ratio;
    let (pad_x, pad_y) = letterbox.padding;
    let (max_x, max_y) = ((image.width() - 1) as f32, (image.height() - 1) as f32);
    let vehicles = data
        .chunks_exact(OUTPUT_COLUMNS)
        .take(OUTPUT_ROWS)
        .filter(|row| row[4] > config.vehicle_confidence)
        .map(|row| {
            let [cx, cy, w, h, confidence, class_id, angle] = [row[0], row[1], row[2], row[3], row[4], row[5], row[6]];
            let (dx, dy) = (w * 0.5, h * 0.5);
            let bbox = [
                ((cx - dx - pad_x) * inv_scale).max(0.0),
                ((cy - dy - pad_y) * inv_scale).max(0.0),
                ((cx + dx - pad_x) * inv_scale).min(max_x),
                ((cy + dy - pad_y) * inv_scale).min(max_y),
            ];
            // Undo the letterbox for the rotated box
            let (cx, cy) = ((cx - pad_x) * inv_scale, (cy - pad_y) * inv_scale);
            let (dx, dy) = (dx * inv_scale, dy * inv_scale);
            let (sin, cos) = angle.sin_cos();
            VehicleObb {
                bbox,
                angle,
                corners: [
                    [cx + dx * cos - dy * sin, cy + dx * sin + dy * cos],
                    [cx - dx * cos - dy * sin, cy - dx * sin + dy * cos],
                    [cx - dx * cos + dy * sin, cy - dx * sin - dy * cos],
                    [cx + dx * cos + dy * sin, cy + dx * sin - dy * cos],
                ],
                confidence,
                class_id: class_id as usize,
            }
        })
        .collect();
    Ok(vehicles)
}
