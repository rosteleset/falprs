//! Vehicle footprints (UODNet: UrbanOmniDetect-2, a YOLO26 detector with 3D-cuboid keypoints).
//!
//! The footprint is the quadrilateral where a vehicle touches the ground: the bottom face of the vehicle's
//! cuboid projected onto the frame (keypoints 0-3; keypoints 4-7 are the top face).

use anyhow::{Result, bail};
use image::RgbImage;

use super::geometry::intersection_area;
use super::vstream_config::VStreamConfig;
use crate::imaging::{Letterbox, letterbox_yolo};
use crate::log;
use crate::logging::{Level, StreamLog};
use crate::stream_log;
use crate::triton::{InferRequest, Tensor, Triton};

/// Rows of the end-to-end head, sorted by score: `[x1, y1, x2, y2, score, class, (x, y, v) * 8]`,
/// in letterboxed input pixels, with `v` already a probability.
const OUTPUT_ROWS: usize = 300;
const OUTPUT_COLUMNS: usize = 30;
const KEYPOINTS: usize = 8;

/// How far outside its box a keypoint may lie, in box widths/heights (UrbanOmniDetect's cuboid check).
const KEYPOINT_BOX_MARGIN: f32 = 0.75;

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

/// The thresholds of [`footprints_from_output`].
#[derive(Debug, Clone, Copy)]
pub struct Thresholds {
    /// A vehicle is reported when its score is above this.
    pub confidence: f32,
    /// Every cuboid keypoint needs at least this probability; 0 disables the test.
    pub keypoint_confidence: f32,
    /// A vehicle whose box lies at least this much inside a larger vehicle's box...
    pub nested_box_overlap: f32,
    /// ...and whose footprint overlaps that vehicle's at least this much (of the smaller footprint) is a
    /// duplicate and is dropped; a zero `nested_box_overlap` disables the suppression.
    pub nested_footprint_overlap: f32,
}

/// A candidate vehicle in frame pixels.
struct Candidate {
    bbox: [f32; 4],
    footprint: VehicleFootprint,
}

pub async fn detect(triton: &Triton, image: &RgbImage, config: &VStreamConfig, log: StreamLog<'_>) -> Result<Vec<VehicleFootprint>> {
    let model = &config.uod_net;
    let target_size = model.input_width.max(model.input_height);
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
        log!(log.logger, Level::Error, "Error! Unable to receive inference result");
        return Ok(Vec::new());
    };
    stream_log!(log, Level::Debug, "inference UODNet OK");
    let data = output.as_f32();
    if data.len() < OUTPUT_ROWS * OUTPUT_COLUMNS {
        bail!("unexpected output size {}", data.len());
    }

    let thresholds = Thresholds {
        confidence: config.vehicle_confidence,
        keypoint_confidence: config.uod_net_keypoint_confidence,
        nested_box_overlap: config.uod_net_nested_box_overlap,
        nested_footprint_overlap: config.uod_net_nested_footprint_overlap,
    };
    stream_log!(log, Level::Trace, "vehicle confidence threshold: {:.3}", thresholds.confidence);
    let vehicles = footprints_from_output(&data[..OUTPUT_ROWS * OUTPUT_COLUMNS], &letterbox, &thresholds);
    stream_log!(log, Level::Trace, "found vehicles: {}", vehicles.len());
    Ok(vehicles)
}

/// The reported vehicles of one image from the model output, in the order of the rows.
fn footprints_from_output(data: &[f32], letterbox: &Letterbox, thresholds: &Thresholds) -> Vec<VehicleFootprint> {
    let candidates: Vec<Candidate> = data
        .as_chunks::<OUTPUT_COLUMNS>()
        .0
        .iter()
        .filter_map(|row| candidate(row, letterbox, thresholds))
        .collect();
    let dropped = nested_duplicates(&candidates, thresholds);
    candidates
        .into_iter()
        .zip(dropped)
        .filter(|(_, dropped)| !dropped)
        .map(|(candidate, _)| candidate.footprint)
        .collect()
}

/// A row turned into a vehicle, unless it is filtered out by its score, class, cuboid or footprint.
fn candidate(row: &[f32; OUTPUT_COLUMNS], letterbox: &Letterbox, thresholds: &Thresholds) -> Option<Candidate> {
    let confidence = row[4];
    // NaN scores fail too
    if confidence.is_nan() || confidence <= thresholds.confidence {
        return None;
    }
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
        *point = to_frame(x, y);
    }

    // The cuboid must be plausible: finite, non-zero and near its box
    let (width, height) = ((x2 - x1).max(1.0), (y2 - y1).max(1.0));
    let near_box = |[x, y]: [f32; 2]| {
        x >= x1 - KEYPOINT_BOX_MARGIN * width
            && x <= x2 + KEYPOINT_BOX_MARGIN * width
            && y >= y1 - KEYPOINT_BOX_MARGIN * height
            && y <= y2 + KEYPOINT_BOX_MARGIN * height
    };
    let valid = |p: &[f32; 2]| p[0].is_finite() && p[1].is_finite() && p[0].abs() + p[1].abs() > 1e-6 && near_box(*p);
    if !cuboid.iter().all(valid) {
        return None;
    }

    let corners = clockwise([cuboid[0], cuboid[1], cuboid[2], cuboid[3]]);
    if !is_convex(&corners) {
        return None;
    }

    Some(Candidate {
        bbox: [x1, y1, x2, y2],
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

/// Which candidates are duplicates nested in another vehicle: the end-to-end head runs no NMS and can report
/// a box truck twice, as the whole vehicle and as its cab inside it, or one car twice with nearly the same box.
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
            if intersection / area_a < thresholds.nested_box_overlap
                || footprint_overlap(&a.footprint.corners, &b.footprint.corners) < thresholds.nested_footprint_overlap
            {
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

/// The intersection of two footprints as a fraction of the smaller one.
fn footprint_overlap(a: &[[f32; 2]; 4], b: &[[f32; 2]; 4]) -> f32 {
    let smaller = intersection_area(a, a).min(intersection_area(b, b));
    if smaller <= 1e-6 {
        return 0.0;
    }
    intersection_area(a, b) / smaller
}

#[cfg(test)]
mod tests {
    use super::*;

    const THRESHOLDS: Thresholds = Thresholds {
        confidence: 0.5,
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
            padding: (0.0, 0.0),
            offset: (0, 0),
            ratio: 1.0,
        }
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
    fn maps_classes_and_filters_scores() {
        let data = [
            row(2.0, 0.9, SQUARE, 0.9),                                       // car
            row(7.0, 0.8, SQUARE.map(|p| [p[0] + 200.0, p[1]]), 0.9),         // truck
            row(5.0, 0.7, SQUARE.map(|p| [p[0] + 400.0, p[1]]), 0.9),         // bus
            row(0.0, 0.9, SQUARE.map(|p| [p[0], p[1] + 300.0]), 0.9),         // person: not a vehicle
            row(2.0, 0.5, SQUARE.map(|p| [p[0] + 200.0, p[1] + 300.0]), 0.9), // at the threshold
        ]
        .concat();
        let vehicles = footprints_from_output(&data, &identity(), &THRESHOLDS);
        let classes: Vec<_> = vehicles.iter().map(|v| (v.class_id, v.class_name)).collect();
        assert_eq!(classes, [(0, "car"), (2, "truck"), (3, "bus")]);
    }

    #[test]
    fn orders_corners_clockwise_from_top_left() {
        // Counter-clockwise on the frame, starting elsewhere
        let ground = [[140.0, 180.0], [140.0, 100.0], [100.0, 100.0], [100.0, 180.0]];
        let vehicles = footprints_from_output(&row(2.0, 0.9, ground, 0.9), &identity(), &THRESHOLDS);
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
        let vehicles = footprints_from_output(&data, &letterbox, &THRESHOLDS);
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
            assert!(footprints_from_output(&data, &identity(), &THRESHOLDS).is_empty());
        }
        // A zero keypoint threshold disables the v test only
        let mut invisible = row(2.0, 0.9, SQUARE, 0.1);
        let no_v = Thresholds {
            keypoint_confidence: 0.0,
            ..THRESHOLDS
        };
        assert_eq!(footprints_from_output(&invisible, &identity(), &no_v).len(), 1);
        invisible[9] = 1000.0;
        assert!(footprints_from_output(&invisible, &identity(), &no_v).is_empty());
    }

    #[test]
    fn drops_non_convex_footprints() {
        // The fourth corner lies inside the triangle of the other three
        let dart = [[100.0, 100.0], [140.0, 100.0], [120.0, 110.0], [120.0, 180.0]];
        assert!(footprints_from_output(&row(2.0, 0.9, dart, 0.9), &identity(), &THRESHOLDS).is_empty());
        let collapsed = [[100.0, 100.0], [120.0, 100.0], [140.0, 100.0], [100.0, 100.0]];
        assert!(footprints_from_output(&row(2.0, 0.9, collapsed, 0.9), &identity(), &THRESHOLDS).is_empty());
    }

    #[test]
    fn drops_nested_duplicates_only() {
        let truck = [[100.0, 100.0], [200.0, 100.0], [200.0, 300.0], [100.0, 300.0]];
        // The cab: nested in the truck's box and on the truck's ground
        let cab = [[110.0, 110.0], [190.0, 110.0], [190.0, 180.0], [110.0, 180.0]];
        let data = [row(7.0, 0.9, truck, 0.9), row(2.0, 0.8, cab, 0.9)].concat();
        let vehicles = footprints_from_output(&data, &identity(), &THRESHOLDS);
        assert_eq!(vehicles.len(), 1);
        assert_eq!(vehicles[0].class_name, "truck");

        // Disabled by a zero box overlap
        let off = Thresholds {
            nested_box_overlap: 0.0,
            ..THRESHOLDS
        };
        assert_eq!(footprints_from_output(&data, &identity(), &off).len(), 2);

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
        assert_eq!(footprints_from_output(&data, &identity(), &THRESHOLDS).len(), 2);
    }

    #[test]
    fn keeps_the_more_confident_of_one_vehicle_twice() {
        // The same car twice; the less confident detection has the slightly larger box
        let car = [[100.0, 100.0], [140.0, 100.0], [140.0, 180.0], [100.0, 180.0]];
        let mut larger = row(2.0, 0.6, car, 0.9);
        larger[2] += 0.5;
        for data in [[row(2.0, 0.9, car, 0.9), larger.clone()].concat(), [larger, row(2.0, 0.9, car, 0.9)].concat()] {
            let vehicles = footprints_from_output(&data, &identity(), &THRESHOLDS);
            assert_eq!(vehicles.len(), 1);
            assert_eq!(vehicles[0].confidence, 0.9);
        }
        // Equal scores: the earlier row stays
        let data = [row(2.0, 0.9, car, 0.9), row(2.0, 0.9, car, 0.8)].concat();
        let vehicles = footprints_from_output(&data, &identity(), &THRESHOLDS);
        assert_eq!(vehicles.len(), 1);
    }
}
