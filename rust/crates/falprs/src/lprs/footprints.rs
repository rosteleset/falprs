//! Vehicle footprints (UODNet: UrbanOmniDetect-2, a YOLO26 detector with 3D-cuboid keypoints).
//!
//! The footprint is the quadrilateral where a vehicle touches the ground: the bottom face of the vehicle's
//! cuboid projected onto the frame (keypoints 0-3; keypoints 4-7 are the top face).
//!
//! The model runs its one-to-many head, which finds more vehicles than the end-to-end one (dark cars above all),
//! so its raw output is decoded and suppressed with NMS here, as ultralytics does after `predict`.

use anyhow::{Result, bail};
use image::RgbImage;

use super::vstream_config::VStreamConfig;
use crate::geometry::{intersection_area, iou, polygon_area};
use crate::imaging::{Letterbox, letterbox_yolo};
use crate::logging::{Level, StreamLog};
use crate::stream_log;
use crate::triton::{InferRequest, Tensor, Triton};

/// Model classes (COCO-80).
const MODEL_CLASSES: usize = 80;
const KEYPOINTS: usize = 8;
/// Output channels per anchor: `cx, cy, w, h`, the class probabilities, then `(x, y, v)` per keypoint, all in
/// letterboxed input pixels with probabilities already sigmoided; the output is `[1, channels, anchors]`.
const OUTPUT_CHANNELS: usize = 4 + MODEL_CLASSES + 3 * KEYPOINTS;
/// A detection after NMS: `[x1, y1, x2, y2, score, class, (x, y, v) * 8]`, in letterboxed input pixels.
const DETECTION_COLUMNS: usize = 6 + 3 * KEYPOINTS;

/// NMS as ultralytics `predict` runs it: the most confident candidates only take part...
const NMS_MAX_CANDIDATES: usize = 30_000;
/// ...and this many boxes are kept at most.
const NMS_MAX_DETECTIONS: usize = 300;

/// How far outside its box a keypoint may lie, in box widths/heights (UrbanOmniDetect's cuboid check).
const KEYPOINT_BOX_MARGIN: f32 = 0.75;

/// The model takes sides that are multiples of this (its largest stride).
const STRIDE: u32 = 32;

/// Reported classes: COCO-80 index of the model, then the class id and name of the API.
const CLASSES: [(usize, usize, &str); 3] = [(2, 0, "car"), (7, 2, "truck"), (5, 3, "bus")];

#[derive(Debug, Clone, PartialEq)]
pub struct VehicleFootprint {
    pub class_id: usize,
    pub class_name: &'static str,
    pub confidence: f32,
    /// The footprint, clockwise on the frame starting from the corner with the smallest `x + y`.
    pub corners: [[f32; 2]; 4],
    /// The cuboid in the model's order: bottom corners 0-3, then the top corners 4-7 above them.
    pub cuboid: [[f32; 2]; KEYPOINTS],
}

/// The thresholds of [`detections_from_output`] and [`footprints_from_detections`].
#[derive(Debug, Clone, Copy)]
pub struct Thresholds {
    /// A vehicle is reported when its score is above this.
    pub confidence: f32,
    /// NMS suppresses a box whose IoU with a more confident kept box of its class is above this.
    pub nms_iou: f32,
    /// Every cuboid keypoint needs at least this probability; 0 disables the test.
    pub keypoint_confidence: f32,
    /// A vehicle whose box lies at least this much inside a larger vehicle's box...
    pub nested_box_overlap: f32,
    /// ...and whose footprint overlaps that vehicle's at least this much (of the smaller footprint) is a
    /// duplicate and is dropped; a zero `nested_box_overlap` disables the suppression.
    pub nested_footprint_overlap: f32,
}

/// The letterbox size for a configured input: its longer side rounded up to a multiple of [`STRIDE`], as
/// ultralytics' `check_imgsz` does. The letterbox pads only the shorter side to a multiple of the stride, so a
/// longer side that is not one would make an input the model cannot take.
fn input_size(width: u32, height: u32) -> u32 {
    width.max(height).div_ceil(STRIDE) * STRIDE
}

/// A candidate vehicle in frame pixels.
struct Candidate {
    bbox: [f32; 4],
    footprint: VehicleFootprint,
    /// The area of the footprint.
    footprint_area: f32,
}

pub async fn detect(triton: &Triton, image: &RgbImage, config: &VStreamConfig, log: StreamLog<'_>) -> Result<Vec<VehicleFootprint>> {
    let model = &config.uod_net;
    let target_size = input_size(model.input_width, model.input_height);
    stream_log!(log, Level::Trace, "before preprocess image for UODNet");
    let letterbox = tokio::task::block_in_place(|| letterbox_yolo(image, target_size));
    stream_log!(log, Level::Trace, "after preprocess image for UODNet");
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
    stream_log!(log, Level::Trace, "after inference UODNet");
    let Some(output) = outputs.first() else {
        bail!("Unable to receive inference result");
    };
    stream_log!(log, Level::Debug, "inference UODNet OK");
    let data = output.as_f32();
    let anchors = match output.shape[..] {
        [1, channels, anchors] if channels == OUTPUT_CHANNELS as i64 && anchors > 0 => anchors as usize,
        _ => bail!("unexpected output shape {:?}", output.shape),
    };
    if data.len() != OUTPUT_CHANNELS * anchors {
        bail!("unexpected output size {} for shape {:?}", data.len(), output.shape);
    }

    let thresholds = Thresholds {
        confidence: config.vehicle_confidence,
        nms_iou: config.uod_net_nms_iou_threshold,
        keypoint_confidence: config.uod_net_keypoint_confidence,
        nested_box_overlap: config.uod_net_nested_box_overlap,
        nested_footprint_overlap: config.uod_net_nested_footprint_overlap,
    };
    stream_log!(log, Level::Trace, "vehicle confidence threshold: {:.3}", thresholds.confidence);
    let vehicles = tokio::task::block_in_place(|| {
        let detections = detections_from_output(&data, anchors, &thresholds);
        footprints_from_detections(&detections, &letterbox, &thresholds)
    });
    stream_log!(log, Level::Trace, "found vehicles: {}", vehicles.len());
    Ok(vehicles)
}

/// The detections of the one-to-many head's output `[channels, anchors]` (channel-major), as ultralytics' NMS
/// gives them: each anchor takes its most probable class, the ones scoring above the confidence threshold are
/// suppressed class by class at the NMS IoU threshold, most confident first.
fn detections_from_output(data: &[f32], anchors: usize, thresholds: &Thresholds) -> Vec<[f32; DETECTION_COLUMNS]> {
    let threshold = thresholds.confidence;
    let at = |channel: usize, anchor: usize| data[channel * anchors + anchor];
    // The most probable class of each anchor, reading the class channels row by row (the first class wins a tie)
    let mut best_score = vec![f32::MIN; anchors];
    let mut best_class = vec![0u8; anchors];
    for class in 0..MODEL_CLASSES {
        let channel = &data[(4 + class) * anchors..][..anchors];
        for ((score, best), best_class) in channel.iter().zip(&mut best_score).zip(&mut best_class) {
            if *score > *best {
                *best = *score;
                *best_class = class as u8;
            }
        }
    }
    let mut candidates: Vec<(f32, usize, usize)> = best_score
        .iter()
        .zip(&best_class)
        .enumerate()
        // NaN scores fail too
        .filter(|(_, (score, _))| **score > threshold)
        .map(|(anchor, (score, class))| (*score, *class as usize, anchor))
        .collect();
    candidates.sort_by(|a, b| b.0.total_cmp(&a.0));
    candidates.truncate(NMS_MAX_CANDIDATES);

    let mut kept: Vec<([f32; 4], usize)> = Vec::new();
    let mut detections = Vec::new();
    for (score, class, anchor) in candidates {
        let (cx, cy, w, h) = (at(0, anchor), at(1, anchor), at(2, anchor), at(3, anchor));
        let bbox = [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0];
        if kept
            .iter()
            .any(|(other, other_class)| *other_class == class && iou(&bbox, other) > thresholds.nms_iou)
        {
            continue;
        }
        kept.push((bbox, class));
        let mut detection = [0.0; DETECTION_COLUMNS];
        detection[..4].copy_from_slice(&bbox);
        detection[4] = score;
        detection[5] = class as f32;
        for (i, value) in detection[6..].iter_mut().enumerate() {
            *value = at(4 + MODEL_CLASSES + i, anchor);
        }
        detections.push(detection);
        if detections.len() == NMS_MAX_DETECTIONS {
            break;
        }
    }
    detections
}

/// The reported vehicles of one image from its detections, in the order of the detections.
fn footprints_from_detections(detections: &[[f32; DETECTION_COLUMNS]], letterbox: &Letterbox, thresholds: &Thresholds) -> Vec<VehicleFootprint> {
    let candidates: Vec<Candidate> = detections.iter().filter_map(|row| candidate(row, letterbox, thresholds)).collect();
    let dropped = nested_duplicates(&candidates, thresholds);
    candidates
        .into_iter()
        .zip(dropped)
        .filter(|(_, dropped)| !dropped)
        .map(|(candidate, _)| candidate.footprint)
        .collect()
}

/// A detection turned into a vehicle, unless it is filtered out by its class, cuboid or footprint; its score is
/// above the threshold already.
fn candidate(row: &[f32; DETECTION_COLUMNS], letterbox: &Letterbox, thresholds: &Thresholds) -> Option<Candidate> {
    let confidence = row[4];
    let class = row[5] as usize;
    let &(_, class_id, class_name) = CLASSES.iter().find(|(coco, _, _)| *coco == class)?;

    // Undo the letterbox; nothing is clipped to the frame
    let scale = 1.0 / letterbox.ratio;
    let (left, top) = (letterbox.offset.0 as f32, letterbox.offset.1 as f32);
    let to_frame = |x: f32, y: f32| [(x - left) * scale, (y - top) * scale];
    let [x1, y1] = to_frame(row[0], row[1]);
    let [x2, y2] = to_frame(row[2], row[3]);
    let mut cuboid = [[0.0; 2]; KEYPOINTS];
    for (i, point) in cuboid.iter_mut().enumerate() {
        let [x, y, v] = [row[6 + 3 * i], row[7 + 3 * i], row[8 + 3 * i]];
        if v.is_nan() || v < thresholds.keypoint_confidence {
            return None;
        }
        // A keypoint the model left at the input's origin is missing; once the letterbox is undone it no longer
        // lies at (0, 0), so the raw coordinates are tested
        if x.abs() + y.abs() <= 1e-6 {
            return None;
        }
        *point = to_frame(x, y);
    }

    // The cuboid must be plausible: finite and near its box
    let (width, height) = ((x2 - x1).max(1.0), (y2 - y1).max(1.0));
    let near_box = |[x, y]: [f32; 2]| {
        x >= x1 - KEYPOINT_BOX_MARGIN * width
            && x <= x2 + KEYPOINT_BOX_MARGIN * width
            && y >= y1 - KEYPOINT_BOX_MARGIN * height
            && y <= y2 + KEYPOINT_BOX_MARGIN * height
    };
    let valid = |p: &[f32; 2]| p[0].is_finite() && p[1].is_finite() && near_box(*p);
    if !cuboid.iter().all(valid) {
        return None;
    }

    let corners = clockwise([cuboid[0], cuboid[1], cuboid[2], cuboid[3]]);
    if !is_convex(&corners) {
        return None;
    }

    Some(Candidate {
        bbox: [x1, y1, x2, y2],
        footprint_area: polygon_area(&corners),
        footprint: VehicleFootprint {
            class_id,
            class_name,
            confidence,
            corners,
            cuboid,
        },
    })
}

/// The corners sorted clockwise on the frame (y down) around their centroid, starting from the corner with
/// the smallest `x + y`.
fn clockwise(mut corners: [[f32; 2]; 4]) -> [[f32; 2]; 4] {
    let cx = corners.iter().map(|p| p[0]).sum::<f32>() / 4.0;
    let cy = corners.iter().map(|p| p[1]).sum::<f32>() / 4.0;
    let angle = |p: &[f32; 2]| (p[1] - cy).atan2(p[0] - cx);
    corners.sort_by(|a, b| angle(a).total_cmp(&angle(b)));
    let first = (0..4)
        .min_by(|&a, &b| (corners[a][0] + corners[a][1]).total_cmp(&(corners[b][0] + corners[b][1])))
        .unwrap_or(0);
    corners.rotate_left(first);
    corners
}

/// Whether clockwise corners make a convex quadrilateral: every turn is to the same side, none is straight.
fn is_convex(corners: &[[f32; 2]; 4]) -> bool {
    (0..4).all(|i| {
        let [a, b, c] = [corners[i], corners[(i + 1) % 4], corners[(i + 2) % 4]];
        (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]) > 0.0
    })
}

/// Which candidates are duplicates nested in another vehicle, which NMS leaves: it suppresses boxes of one class
/// that overlap much, not a box truck reported as the whole vehicle and as its cab inside it, nor one car reported
/// as a car and as a truck with nearly the same box.
/// A vehicle standing in front of a larger one is nested in the image but not on the ground, so its footprint
/// doesn't overlap the larger one's and it stays.
///
/// A box nested in a larger one is dropped (the cab, as UrbanOmniDetect does); of two boxes nested in each other
/// (one vehicle twice) the less confident one is dropped, since which of them is larger is down to noise.
fn nested_duplicates(candidates: &[Candidate], thresholds: &Thresholds) -> Vec<bool> {
    let mut dropped = vec![false; candidates.len()];
    if thresholds.nested_box_overlap <= 0.0 {
        return dropped;
    }
    let box_area = |b: &[f32; 4]| (b[2] - b[0]).max(1.0) * (b[3] - b[1]).max(1.0);
    for (i, a) in candidates.iter().enumerate() {
        let area_a = box_area(&a.bbox);
        let nested = candidates.iter().enumerate().any(|(j, b)| {
            if i == j || dropped[j] {
                return false;
            }
            let width = (a.bbox[2].min(b.bbox[2]) - a.bbox[0].max(b.bbox[0])).max(0.0);
            let height = (a.bbox[3].min(b.bbox[3]) - a.bbox[1].max(b.bbox[1])).max(0.0);
            let intersection = width * height;
            if intersection / area_a < thresholds.nested_box_overlap || footprint_overlap(a, b) < thresholds.nested_footprint_overlap {
                return false;
            }
            if intersection / box_area(&b.bbox) >= thresholds.nested_box_overlap {
                // Nested in each other: keep the more confident one, the earlier row on a tie
                let (score_a, score_b) = (a.footprint.confidence, b.footprint.confidence);
                score_a < score_b || (score_a == score_b && j < i)
            } else {
                // Only `a` is nested, so `b` is the larger one
                true
            }
        });
        dropped[i] = nested;
    }
    dropped
}

/// The intersection of two candidates' footprints as a fraction of the smaller one.
fn footprint_overlap(a: &Candidate, b: &Candidate) -> f32 {
    let smaller = a.footprint_area.min(b.footprint_area);
    if smaller <= 1e-6 {
        return 0.0;
    }
    intersection_area(&a.footprint.corners, &b.footprint.corners) / smaller
}

#[cfg(test)]
mod tests {
    use super::*;

    const THRESHOLDS: Thresholds = Thresholds {
        confidence: 0.5,
        nms_iou: 0.7,
        keypoint_confidence: 0.5,
        nested_box_overlap: 0.85,
        nested_footprint_overlap: 0.3,
    };

    /// A letterbox that maps input pixels to frame pixels 1:1.
    fn identity() -> Letterbox {
        Letterbox {
            tensor: Vec::new(),
            width: 640,
            height: 640,
            offset: (0, 0),
            ratio: 1.0,
        }
    }

    /// The filtering pipeline on detections given as rows of [`DETECTION_COLUMNS`] values.
    fn run(rows: &[f32], letterbox: &Letterbox, thresholds: &Thresholds) -> Vec<VehicleFootprint> {
        footprints_from_detections(rows.as_chunks::<DETECTION_COLUMNS>().0, letterbox, thresholds)
    }

    /// An anchor of a test output: box `[cx, cy, w, h]`, `(class, probability)` pairs, keypoint `v`.
    type Anchor = ([f32; 4], Vec<(usize, f32)>, f32);

    /// A model output (channel-major) of anchors given as `(box [cx, cy, w, h], [(class, probability)], v)`,
    /// with keypoint i of each anchor at `(cx + i, cy + 2i)`.
    fn output(anchors: &[Anchor]) -> Vec<f32> {
        let n = anchors.len();
        let mut data = vec![0.0; OUTPUT_CHANNELS * n];
        for (a, (bbox, classes, v)) in anchors.iter().enumerate() {
            for (channel, value) in bbox.iter().enumerate() {
                data[channel * n + a] = *value;
            }
            for &(class, probability) in classes {
                data[(4 + class) * n + a] = probability;
            }
            for i in 0..KEYPOINTS {
                let base = 4 + MODEL_CLASSES + 3 * i;
                data[base * n + a] = bbox[0] + i as f32;
                data[(base + 1) * n + a] = bbox[1] + 2.0 * i as f32;
                data[(base + 2) * n + a] = *v;
            }
        }
        data
    }

    #[test]
    fn rounds_the_input_size_up_to_the_stride() {
        assert_eq!(input_size(640, 640), 640);
        assert_eq!(input_size(1000, 1000), 1024);
        assert_eq!(input_size(161, 160), 192);
        assert_eq!(input_size(320, 1024), 1024);
    }

    #[test]
    fn decodes_the_one_to_many_layout() {
        let data = output(&[
            ([50.0, 60.0, 20.0, 10.0], vec![(0, 0.2), (2, 0.9), (7, 0.3)], 0.8),
            ([300.0, 300.0, 20.0, 10.0], vec![(2, 0.4)], 0.8),
        ]);
        let detections = detections_from_output(&data, 2, &THRESHOLDS);
        assert_eq!(detections.len(), 1);
        let d = detections[0];
        assert_eq!(d[..6], [40.0, 55.0, 60.0, 65.0, 0.9, 2.0]);
        // Keypoint 3: x, y, v
        assert_eq!(d[6 + 9..6 + 12], [53.0, 66.0, 0.8]);
    }

    #[test]
    fn keeps_the_best_class_above_the_threshold_only() {
        let data = output(&[
            ([50.0, 50.0, 20.0, 20.0], vec![(2, 0.5), (7, 0.6)], 0.9),
            ([200.0, 50.0, 20.0, 20.0], vec![(2, 0.5)], 0.9), // at the threshold: out
            ([350.0, 50.0, 20.0, 20.0], vec![(2, f32::NAN)], 0.9),
        ]);
        let detections = detections_from_output(&data, 3, &THRESHOLDS);
        assert_eq!(detections.len(), 1);
        assert_eq!(detections[0][4..6], [0.6, 7.0]);
    }

    #[test]
    fn suppresses_overlapping_boxes_class_by_class() {
        // [0, 0, 10, 10] against boxes inside it: the IoU is their area / 100
        let data = output(&[
            ([4.0, 5.0, 8.0, 10.0], vec![(2, 0.7)], 0.9),  // IoU 0.8 with the first car: suppressed
            ([5.0, 5.0, 10.0, 10.0], vec![(2, 0.9)], 0.9), // the most confident car
            ([3.5, 5.0, 7.0, 10.0], vec![(2, 0.8)], 0.9),  // IoU exactly 0.7: kept
            ([5.0, 5.0, 10.0, 10.0], vec![(7, 0.6)], 0.9), // a truck on the same box: another class, kept
        ]);
        let kept = |thresholds: &Thresholds| -> Vec<_> { detections_from_output(&data, 4, thresholds).iter().map(|d| (d[4], d[5])).collect() };
        // Most confident first
        assert_eq!(kept(&THRESHOLDS), [(0.9, 2.0), (0.8, 2.0), (0.6, 7.0)]);
        // A higher IoU threshold keeps the car at IoU 0.8 (0.875 with the 0.8 car) too, a lower one suppresses
        // the one at 0.7
        let loose = Thresholds { nms_iou: 0.9, ..THRESHOLDS };
        assert_eq!(kept(&loose), [(0.9, 2.0), (0.8, 2.0), (0.7, 2.0), (0.6, 7.0)]);
        let strict = Thresholds { nms_iou: 0.6, ..THRESHOLDS };
        assert_eq!(kept(&strict), [(0.9, 2.0), (0.6, 7.0)]);
    }

    #[test]
    fn keeps_300_detections_at_most() {
        let anchors: Vec<_> = (0..301)
            .map(|i| ([20.0 * i as f32, 10.0, 10.0, 10.0], vec![(2, 0.5 + i as f32 / 1000.0)], 0.9))
            .collect();
        let low = Thresholds { confidence: 0.4, ..THRESHOLDS };
        let detections = detections_from_output(&output(&anchors), anchors.len(), &low);
        assert_eq!(detections.len(), 300);
        // The least confident one is left out
        assert!(detections.iter().all(|d| d[4] > 0.5));
    }

    /// A model row of a vehicle standing on `ground` (any order), with its roof 10 px above.
    fn row(class: f32, score: f32, ground: [[f32; 2]; 4], v: f32) -> Vec<f32> {
        let xs = ground.map(|p| p[0]);
        let ys = ground.map(|p| p[1]);
        let min = |values: [f32; 4]| values.into_iter().fold(f32::MAX, f32::min);
        let max = |values: [f32; 4]| values.into_iter().fold(f32::MIN, f32::max);
        let mut row = vec![min(xs), min(ys) - 10.0, max(xs), max(ys), score, class];
        for p in ground.iter().chain(ground.map(|p| [p[0], p[1] - 10.0]).iter()) {
            row.extend([p[0], p[1], v]);
        }
        row
    }

    const SQUARE: [[f32; 2]; 4] = [[100.0, 100.0], [140.0, 100.0], [140.0, 180.0], [100.0, 180.0]];

    #[test]
    fn maps_classes() {
        let data = [
            row(2.0, 0.9, SQUARE, 0.9),                               // car
            row(7.0, 0.8, SQUARE.map(|p| [p[0] + 200.0, p[1]]), 0.9), // truck
            row(5.0, 0.7, SQUARE.map(|p| [p[0] + 400.0, p[1]]), 0.9), // bus
            row(0.0, 0.9, SQUARE.map(|p| [p[0], p[1] + 300.0]), 0.9), // person: not a vehicle
        ]
        .concat();
        let vehicles = run(&data, &identity(), &THRESHOLDS);
        let classes: Vec<_> = vehicles.iter().map(|v| (v.class_id, v.class_name)).collect();
        assert_eq!(classes, [(0, "car"), (2, "truck"), (3, "bus")]);
    }

    #[test]
    fn orders_corners_clockwise_from_top_left() {
        // Counter-clockwise on the frame, starting elsewhere
        let ground = [[140.0, 180.0], [140.0, 100.0], [100.0, 100.0], [100.0, 180.0]];
        let vehicles = run(&row(2.0, 0.9, ground, 0.9), &identity(), &THRESHOLDS);
        assert_eq!(vehicles[0].corners, SQUARE);
        // The cuboid keeps the model's order
        assert_eq!(vehicles[0].cuboid[..4], ground);
        assert_eq!(vehicles[0].cuboid[4], [140.0, 170.0]);
    }

    #[test]
    fn undoes_the_letterbox_without_clipping() {
        let letterbox = Letterbox {
            offset: (0, 12),
            ratio: 0.5,
            ..identity()
        };
        let ground = [[0.0, 12.0], [20.0, 12.0], [20.0, 52.0], [0.0, 52.0]];
        let mut data = row(2.0, 0.9, ground, 0.9);
        data[6] = -4.0; // the first ground corner lies 8 px left of the frame
        let vehicles = run(&data, &letterbox, &THRESHOLDS);
        assert_eq!(vehicles[0].corners, [[-8.0, 0.0], [40.0, 0.0], [40.0, 80.0], [0.0, 80.0]]);
    }

    #[test]
    fn drops_unreliable_cuboids() {
        let mut far = row(2.0, 0.9, SQUARE, 0.9);
        far[9] = 1000.0; // keypoint 1 far outside the box
        let mut zero = row(2.0, 0.9, SQUARE, 0.9);
        zero[9] = 0.0;
        zero[10] = 0.0;
        let mut nan = row(2.0, 0.9, SQUARE, 0.9);
        nan[18] = f32::NAN; // x of the first top corner
        let mut invisible = row(2.0, 0.9, SQUARE, 0.9);
        invisible[29] = 0.4; // v of the last top corner
        for data in [far, zero, nan, invisible] {
            assert!(run(&data, &identity(), &THRESHOLDS).is_empty());
        }
        // A keypoint at the input's origin is dropped behind a letterbox too, where it maps to (0, -12): the box
        // reaches the frame's top left corner, so the point lies near it
        let letterbox = Letterbox { offset: (0, 12), ..identity() };
        let ground = [[0.0, 22.0], [40.0, 22.0], [40.0, 102.0], [0.0, 102.0]];
        let mut padded = row(2.0, 0.9, ground, 0.9);
        assert_eq!(run(&padded, &letterbox, &THRESHOLDS).len(), 1);
        padded[9] = 0.0;
        padded[10] = 0.0;
        assert!(run(&padded, &letterbox, &THRESHOLDS).is_empty());
        // A zero keypoint threshold disables the v test only
        let mut invisible = row(2.0, 0.9, SQUARE, 0.1);
        let no_v = Thresholds {
            keypoint_confidence: 0.0,
            ..THRESHOLDS
        };
        assert_eq!(run(&invisible, &identity(), &no_v).len(), 1);
        invisible[9] = 1000.0;
        assert!(run(&invisible, &identity(), &no_v).is_empty());
    }

    #[test]
    fn drops_non_convex_footprints() {
        // The fourth corner lies inside the triangle of the other three
        let dart = [[100.0, 100.0], [140.0, 100.0], [120.0, 110.0], [120.0, 180.0]];
        assert!(run(&row(2.0, 0.9, dart, 0.9), &identity(), &THRESHOLDS).is_empty());
        let collapsed = [[100.0, 100.0], [120.0, 100.0], [140.0, 100.0], [100.0, 100.0]];
        assert!(run(&row(2.0, 0.9, collapsed, 0.9), &identity(), &THRESHOLDS).is_empty());
    }

    #[test]
    fn drops_nested_duplicates_only() {
        let truck = [[100.0, 100.0], [200.0, 100.0], [200.0, 300.0], [100.0, 300.0]];
        // The cab: nested in the truck's box and on the truck's ground
        let cab = [[110.0, 110.0], [190.0, 110.0], [190.0, 180.0], [110.0, 180.0]];
        let data = [row(7.0, 0.9, truck, 0.9), row(2.0, 0.8, cab, 0.9)].concat();
        let vehicles = run(&data, &identity(), &THRESHOLDS);
        assert_eq!(vehicles.len(), 1);
        assert_eq!(vehicles[0].class_name, "truck");

        // Disabled by a zero box overlap
        let off = Thresholds {
            nested_box_overlap: 0.0,
            ..THRESHOLDS
        };
        assert_eq!(run(&data, &identity(), &off).len(), 2);

        // A car in front of the truck: nested in the image, on its own ground
        let mut front = row(2.0, 0.8, cab, 0.9);
        let ground_in_front = [[110.0, 250.0], [190.0, 250.0], [190.0, 299.0], [110.0, 299.0]];
        for (i, p) in ground_in_front.iter().enumerate() {
            front[6 + 3 * i] = p[0];
            front[7 + 3 * i] = p[1];
        }
        front[3] = 299.0;
        let truck_short = [[100.0, 100.0], [200.0, 100.0], [200.0, 200.0], [100.0, 200.0]];
        let mut truck_row = row(7.0, 0.9, truck_short, 0.9);
        truck_row[3] = 300.0;
        let data = [truck_row, front].concat();
        assert_eq!(run(&data, &identity(), &THRESHOLDS).len(), 2);
    }

    #[test]
    fn keeps_the_more_confident_of_one_vehicle_twice() {
        // The same car twice; the less confident detection has the slightly larger box
        let car = [[100.0, 100.0], [140.0, 100.0], [140.0, 180.0], [100.0, 180.0]];
        let mut larger = row(2.0, 0.6, car, 0.9);
        larger[2] += 0.5;
        for data in [[row(2.0, 0.9, car, 0.9), larger.clone()].concat(), [larger, row(2.0, 0.9, car, 0.9)].concat()] {
            let vehicles = run(&data, &identity(), &THRESHOLDS);
            assert_eq!(vehicles.len(), 1);
            assert_eq!(vehicles[0].confidence, 0.9);
        }
        // Equal scores: the earlier row stays
        let data = [row(2.0, 0.9, car, 0.9), row(2.0, 0.9, car, 0.8)].concat();
        let vehicles = run(&data, &identity(), &THRESHOLDS);
        assert_eq!(vehicles.len(), 1);
    }
}
