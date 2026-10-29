//! Face inference on Triton (detection, face class, descriptor) and the face geometry checks of the
//! C++ service.

use anyhow::{Context, Result, bail};
use image::RgbImage;

use super::config::{CommonConfig, VStreamConfig};
use crate::imaging;
use crate::triton::{InferRequest, Tensor, Triton};

#[derive(Debug, Clone)]
pub struct FaceDetection {
    /// x1, y1, x2, y2
    pub bbox: [f32; 4],
    pub confidence: f32,
    /// Right eye, left eye, nose, right and left mouth corners.
    pub landmarks: [[f32; 2]; 5],
}

/// An integer rectangle with OpenCV's `cv::Rect` semantics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

impl Rect {
    pub fn new(x: i32, y: i32, width: i32, height: i32) -> Self {
        Self { x, y, width, height }
    }

    pub fn is_empty(&self) -> bool {
        self.width <= 0 || self.height <= 0
    }

    pub fn area(&self) -> i64 {
        i64::from(self.width) * i64::from(self.height)
    }

    /// `a & b`: the intersection, or an all-zero rectangle when there is none.
    pub fn intersect(&self, other: &Rect) -> Rect {
        let x1 = self.x.max(other.x);
        let y1 = self.y.max(other.y);
        let x2 = (self.x + self.width).min(other.x + other.width);
        let y2 = (self.y + self.height).min(other.y + other.height);
        if x2 - x1 <= 0 || y2 - y1 <= 0 {
            return Rect::default();
        }
        Rect::new(x1, y1, x2 - x1, y2 - y1)
    }
}

/// The square around the face rectangle's centre with the longer side scaled by `k`.
pub fn enlarge_face_rect(rect: &Rect, k: f64) -> Rect {
    let cx = rect.x + rect.width / 2;
    let cy = rect.y + rect.height / 2;
    let side = (f64::from(rect.width.max(rect.height)) * k) as i32;
    Rect::new(cx - side / 2, cy - side / 2, side, side)
}

/// Crops a rectangle lying inside the image.
pub fn crop(image: &RgbImage, rect: &Rect) -> RgbImage {
    image::imageops::crop_imm(image, rect.x as u32, rect.y as u32, rect.width as u32, rect.height as u32).to_image()
}

fn dist(a: [f32; 2], b: [f32; 2]) -> f64 {
    let (dx, dy) = (f64::from(a[0]) - f64::from(b[0]), f64::from(a[1]) - f64::from(b[1]));
    (dx * dx + dy * dy).sqrt()
}

/// Whether the face is frontal judging by its landmarks' symmetry.
pub fn is_frontal_face(lm: &[[f32; 2]; 5]) -> bool {
    let [right_eye, left_eye, nose, right_mouth, left_mouth] = *lm;
    // The nose should be between and below the eyes
    if nose[0] <= right_eye[0] || nose[0] >= left_eye[0] || nose[1] <= right_eye[1] || nose[1] <= left_eye[1] {
        return false;
    }
    // Each eye should not be beyond the opposite mouth corner
    if right_eye[0] >= left_mouth[0] || left_eye[0] <= right_mouth[0] {
        return false;
    }
    const EQUAL_THRESHOLD: f64 = 0.62;
    let similar = |d1: f64, d2: f64| d1.min(d2) / d1.max(d2) > EQUAL_THRESHOLD;
    let mut is_frontal = similar(dist(right_eye, nose), dist(left_eye, nose));
    is_frontal = is_frontal && similar(dist(right_mouth, nose), dist(left_mouth, nose));
    is_frontal = is_frontal && similar(dist(right_mouth, right_eye), dist(left_mouth, left_eye));
    let width = dist(right_eye, left_eye).max(dist(right_mouth, left_mouth));
    let height = dist(right_eye, right_mouth).max(dist(left_eye, left_mouth));
    is_frontal && similar(width, height)
}

fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let area = |r: &[f32; 4]| (r[2] - r[0]) * (r[3] - r[1]);
    let (left, top) = (a[0].max(b[0]), a[1].max(b[1]));
    let (right, bottom) = (a[2].min(b[2]), a[3].min(b[3]));
    if left >= right || top >= bottom {
        return 0.0;
    }
    let inter = (right - left) * (bottom - top);
    inter / (area(a) + area(b) - inter + 1e-6)
}

/// Non-maximum suppression, keeping the most confident of overlapping faces.
fn nms_faces(mut faces: Vec<FaceDetection>, threshold: f32) -> Vec<FaceDetection> {
    // A stable sort; the C++ service's std::ranges::sort may order equal confidences differently
    faces.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
    let mut kept: Vec<FaceDetection> = Vec::with_capacity(faces.len());
    for face in faces {
        if kept.iter().all(|k| iou(&k.bbox, &face.bbox) <= threshold) {
            kept.push(face);
        }
    }
    kept
}

/// Output tensors of SCRFD per stride 8, 16, 32: scores, boxes, key points.
const SCRFD_OUTPUTS: [[&str; 3]; 3] = [["448", "451", "454"], ["471", "474", "477"], ["494", "497", "500"]];

pub async fn detect_faces(
    triton: &Triton,
    frame: &RgbImage,
    common: &CommonConfig,
    config: &VStreamConfig,
) -> Result<Vec<FaceDetection>> {
    let model = &common.fd_net;
    let (width, height) = (model.input_width, model.input_height);
    let (tensor, scale) = tokio::task::block_in_place(|| imaging::prepare_scrfd(frame, width, height));
    let names: Vec<&str> = [0, 1, 2].iter().flat_map(|&k| SCRFD_OUTPUTS.iter().map(move |s| s[k])).collect();
    let outputs = triton
        .infer(InferRequest {
            endpoint: &config.fd_inference_server,
            model_name: &model.model_name,
            inputs: vec![Tensor::fp32(&model.input_tensor_name, vec![1, 3, height as i64, width as i64], &tensor)],
            outputs: names.clone(),
            timeout: config.inference_timeout,
        })
        .await
        .context("Unable to send inference request")?;
    if outputs.len() != names.len() {
        bail!("Unexpected number of output tensors in face detection response: {}", outputs.len());
    }
    let output = |name: &str| outputs.iter().find(|t| t.name == name).map(|t| t.as_f32()).unwrap_or_default();

    let mut faces = Vec::new();
    for (i, stride) in [8u32, 16, 32].into_iter().enumerate() {
        let [scores, boxes, kps] = SCRFD_OUTPUTS[i].map(output);
        let stride_f = stride as f32;
        let (grid_h, grid_w) = ((height / stride) as usize, (width / stride) as usize);
        for k in 0..grid_h * grid_w {
            // The C++ service takes both grid coordinates modulo the grid height
            let px = stride_f * (k % grid_h) as f32;
            let py = stride_f * (k / grid_h) as f32;
            for anchor in [2 * k, 2 * k + 1] {
                let Some(&score) = scores.get(anchor) else { continue };
                if score < config.face_confidence {
                    continue;
                }
                let b = |j: usize| boxes.get(anchor * 4 + j).copied().unwrap_or_default() * stride_f;
                let p = |j: usize| kps.get(anchor * 10 + j).copied().unwrap_or_default() * stride_f;
                let mut landmarks = [[0f32; 2]; 5];
                for (j, landmark) in landmarks.iter_mut().enumerate() {
                    *landmark = [(px + p(2 * j)) / scale, (py + p(2 * j + 1)) / scale];
                }
                faces.push(FaceDetection {
                    bbox: [(px - b(0)) / scale, (py - b(1)) / scale, (px + b(2)) / scale, (py + b(3)) / scale],
                    confidence: score,
                    landmarks,
                });
            }
        }
    }
    Ok(nms_faces(faces, config.face_iou_threshold))
}

/// The aligned face template for a 112x112 face.
const FACE_TEMPLATE: [[f64; 2]; 5] = [
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
];

/// Warps the face so its landmarks match the template scaled to `width` x `height`.
pub fn align_face(frame: &RgbImage, landmarks: &[[f32; 2]; 5], width: u32, height: u32) -> Option<RgbImage> {
    let template: Vec<[f32; 2]> = FACE_TEMPLATE
        .iter()
        .map(|p| [(p[0] / 112.0 * f64::from(width)) as f32, (p[1] / 112.0 * f64::from(height)) as f32])
        .collect();
    let m = imaging::estimate_affine_partial_lmeds(landmarks, &template)?;
    Some(imaging::warp_affine(frame, &m, width, height))
}

/// Face class index and its probability, most probable first.
pub fn soft_max(values: &[f32]) -> Vec<(usize, f32)> {
    let sum: f32 = values.iter().map(|v| v.exp()).sum();
    let mut classes: Vec<(usize, f32)> = values.iter().enumerate().map(|(i, v)| (i, v.exp() / sum)).collect();
    classes.sort_by(|a, b| b.1.total_cmp(&a.1));
    classes
}

pub async fn infer_face_class(
    triton: &Triton,
    aligned_face: &RgbImage,
    common: &CommonConfig,
    config: &VStreamConfig,
) -> Result<Vec<(usize, f32)>> {
    let model = &common.fc_net;
    let (width, height) = (model.input_width, model.input_height);
    let tensor = imaging::prepare_genet(aligned_face, width, height);
    let outputs = triton
        .infer(InferRequest {
            endpoint: &config.fc_inference_server,
            model_name: &model.model_name,
            inputs: vec![Tensor::fp32(&model.input_tensor_name, vec![1, 3, height as i64, width as i64], &tensor)],
            outputs: vec![&model.output_tensor_name],
            timeout: config.inference_timeout,
        })
        .await
        .context("Unable to send inference request")?;
    let Some(output) = outputs.first() else {
        bail!("Unable to receive inference result");
    };
    let mut scores = output.as_f32();
    scores.resize(model.output_size, 0.0);
    Ok(soft_max(&scores))
}

pub async fn extract_face_descriptor(
    triton: &Triton,
    aligned_face: &RgbImage,
    common: &CommonConfig,
    config: &VStreamConfig,
) -> Result<Vec<f32>> {
    let model = &common.fr_net;
    let (width, height) = (model.input_width, model.input_height);
    let tensor = imaging::prepare_arcface(aligned_face, width, height);
    let outputs = triton
        .infer(InferRequest {
            endpoint: &config.fr_inference_server,
            model_name: &model.model_name,
            inputs: vec![Tensor::fp32(&model.input_tensor_name, vec![1, 3, height as i64, width as i64], &tensor)],
            outputs: vec![&model.output_tensor_name],
            timeout: config.inference_timeout,
        })
        .await
        .context("Unable to send inference request")?;
    let Some(output) = outputs.first() else {
        bail!("Unable to receive inference result");
    };
    let expected = model.output_size * 4;
    if output.data.len() != expected {
        bail!(
            "unexpected inference face descriptor extractor result size: must be {expected}, but is {}",
            output.data.len()
        );
    }
    Ok(output.as_f32())
}

/// Dot product of two normalized descriptors, or -1 for descriptors of different or zero size.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return -1.0;
    }
    let mut sums = [[0f32; 8]; 4];
    let blocks = a.chunks_exact(32).zip(b.chunks_exact(32));
    for (block_a, block_b) in blocks {
        for (k, sum) in sums.iter_mut().enumerate() {
            for (lane, s) in sum.iter_mut().enumerate() {
                *s += block_a[k * 8 + lane] * block_b[k * 8 + lane];
            }
        }
    }
    let tail: f32 = a.chunks_exact(32).remainder().iter().zip(b.chunks_exact(32).remainder()).map(|(x, y)| x * y).sum();
    let lanes: f32 = (0..8).map(|lane| sums[0][lane] + sums[1][lane] + sums[2][lane] + sums[3][lane]).sum();
    f64::from(lanes + tail)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rect_intersection() {
        let a = Rect::new(0, 0, 10, 10);
        assert_eq!(a.intersect(&Rect::new(5, 5, 10, 10)), Rect::new(5, 5, 5, 5));
        assert_eq!(a.intersect(&Rect::new(20, 20, 5, 5)), Rect::default());
        assert!(Rect::default().is_empty());
    }

    #[test]
    fn enlarged_rect() {
        assert_eq!(enlarge_face_rect(&Rect::new(10, 20, 40, 60), 1.5), Rect::new(-15, 5, 90, 90));
    }

    #[test]
    fn frontal_faces() {
        let frontal = [[40.0, 50.0], [70.0, 50.0], [55.0, 65.0], [43.0, 80.0], [67.0, 80.0]];
        assert!(is_frontal_face(&frontal));
        let turned = [[40.0, 50.0], [70.0, 50.0], [66.0, 65.0], [43.0, 80.0], [67.0, 80.0]];
        assert!(!is_frontal_face(&turned));
    }

    #[test]
    fn nms_keeps_most_confident() {
        let face = |x: f32, confidence: f32| FaceDetection {
            bbox: [x, 0.0, x + 10.0, 10.0],
            confidence,
            landmarks: [[0.0; 2]; 5],
        };
        let kept = nms_faces(vec![face(0.0, 0.8), face(1.0, 0.9), face(30.0, 0.75)], 0.4);
        let confidences: Vec<f32> = kept.iter().map(|f| f.confidence).collect();
        assert_eq!(confidences, vec![0.9, 0.75]);
    }

    #[test]
    fn soft_max_sorts_by_probability() {
        let classes = soft_max(&[0.0, 2.0, 1.0]);
        assert_eq!(classes.iter().map(|c| c.0).collect::<Vec<_>>(), vec![1, 2, 0]);
        assert!((classes.iter().map(|c| c.1).sum::<f32>() - 1.0).abs() < 1e-6);
    }
}
