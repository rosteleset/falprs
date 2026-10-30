//! The licence plate recognition pipeline: vehicle detection (VDNet), special vehicle
//! classification (VCNet), plate detection (LPDNet), plate classification (LPCNet) and
//! plate recognition (LPRNet), all served by Triton.

use std::collections::HashSet;

use anyhow::{Context, Result, bail};
use futures::future::join_all;
use image::RgbImage;

use super::geometry::{BBox, area, distance, has_intersection, intersection_area, iou};
use super::vstream_config::{ModelConfig, VStreamConfig, WorkAreaPolygon};
use crate::log;
use crate::logging::{Level, StreamLog};
use crate::stream_log;
use crate::imaging::{crop, perspective_transform, prepare_lpc, prepare_vc, prepare_yolo, warp_perspective};
use crate::triton::{InferRequest, Tensor, Triton};

pub const PLATE_CLASSES: [&str; 4] = ["ru_1", "ru_1a", "by", "am"];
const PLATE_CLASS_RU_1: i32 = 0;
const PLATE_CLASS_RU_1A: i32 = 1;
const PLATE_CLASS_BY: i32 = 2;
const PLATE_CLASS_AM: i32 = 3;

#[derive(Debug, Clone)]
pub struct PlateNumber {
    pub plate_class: i32,
    pub number: String,
    pub score: f32,
}

#[derive(Debug, Clone)]
pub struct LicensePlate {
    pub bbox: BBox,
    pub confidence: f32,
    /// Four corners clockwise from the top left, as `x, y` pairs.
    pub kpts: [f32; 8],
    pub plate_class_common: i32,
    pub numbers: Vec<PlateNumber>,
}

impl LicensePlate {
    fn corners(&self) -> [[f32; 2]; 4] {
        let k = &self.kpts;
        [[k[0], k[1]], [k[2], k[3]], [k[4], k[5]], [k[6], k[7]]]
    }
}

#[derive(Debug, Clone)]
pub struct Vehicle {
    pub bbox: BBox,
    pub confidence: f32,
    pub is_special: bool,
    pub plates: Vec<LicensePlate>,
}

fn request<'a>(model: &'a ModelConfig, shape: Vec<i64>, tensor: &[f32], config: &VStreamConfig) -> InferRequest<'a> {
    InferRequest {
        endpoint: &model.inference_server,
        model_name: &model.model_name,
        inputs: vec![Tensor::fp32(&model.input_tensor_name, shape, tensor)],
        outputs: vec![&model.output_tensor_name],
        timeout: config.inference_timeout,
    }
}

fn nchw(model: &ModelConfig) -> Vec<i64> {
    vec![1, 3, model.input_height as i64, model.input_width as i64]
}

/// Runs the requests concurrently; like userver's `WaitAllChecked`, any failure fails them all.
async fn infer_all(triton: &Triton, requests: Vec<InferRequest<'_>>, log: StreamLog<'_>) -> Result<Vec<Vec<f32>>> {
    join_all(requests.into_iter().map(|r| triton.infer(r)))
        .await
        .into_iter()
        .enumerate()
        .map(|(index, outputs)| {
            let Some(output) = outputs?.first().map(|output| output.as_f32()) else {
                log!(log.logger, Level::Error, "Error! Unable to receive inference result (vindex = {index})");
                bail!("inference returned no outputs");
            };
            Ok(output)
        })
        .collect()
}

/// The ROI of a box as the C++ service takes it: corners truncated to integers.
fn roi(image: &RgbImage, b: &BBox) -> Result<RgbImage> {
    crop(image, b[0] as i32, b[1] as i32, b[2] as i32, b[3] as i32)
}

/// Stable sort, descending by confidence.
fn sort_by_confidence<T>(items: &mut [T], confidence: impl Fn(&T) -> f32) {
    items.sort_by(|a, b| confidence(b).total_cmp(&confidence(a)));
}

/// Keeps the most confident of overlapping items; `overlaps` is asked about every later item.
fn suppress<T>(items: &mut Vec<T>, overlaps: impl Fn(&T, &T) -> bool) {
    let mut m = 0;
    while m < items.len() {
        let mut n = m + 1;
        while n < items.len() {
            if overlaps(&items[m], &items[n]) {
                items.remove(n);
            } else {
                n += 1;
            }
        }
        m += 1;
    }
}

/// VDNet: vehicles in the frame. An inference failure yields no vehicles, as in the C++ service.
pub async fn detect_vehicles(triton: &Triton, image: &RgbImage, config: &VStreamConfig, log: StreamLog<'_>) -> Result<Vec<Vehicle>> {
    stream_log!(log, Level::Trace, "before preprocess image for VDNet");
    let input = tokio::task::block_in_place(|| prepare_yolo(image, config.vd_net.input_width, config.vd_net.input_height));
    stream_log!(log, Level::Trace, "after preprocess image for VDNet");
    stream_log!(log, Level::Trace, "before inference VDNet");
    let outputs = triton
        .infer(request(&config.vd_net, nchw(&config.vd_net), &input.tensor, config))
        .await?;
    stream_log!(log, Level::Trace, "after inference VDNet");
    let Some(output) = outputs.first() else {
        log!(log.logger, Level::Error, "Error! Unable to receive inference result");
        return Ok(Vec::new());
    };
    stream_log!(log, Level::Debug, "inference VDNet OK");
    let data = output.as_f32();

    // Output [5, 8400]: centre x, centre y, width, height, confidence
    const COLUMNS: usize = 8400;
    if data.len() < 5 * COLUMNS {
        bail!("unexpected VDNet output size {}", data.len());
    }
    let inv_scale = 1.0 / input.scale as f32;
    let (max_x, max_y) = ((image.width() - 1) as f32, (image.height() - 1) as f32);
    let screen_area = (image.width() * image.height()) as f32;
    let mut vehicles = Vec::new();
    stream_log!(log, Level::Trace, "vehicle confidence threshold: {:.3}", config.vehicle_confidence);
    for j in 0..COLUMNS {
        let confidence = data[4 * COLUMNS + j];
        if confidence <= config.vehicle_confidence {
            continue;
        }
        let (cx, cy, w, h) = (data[j], data[COLUMNS + j], data[2 * COLUMNS + j], data[3 * COLUMNS + j]);
        let bbox = [
            ((cx - w / 2.0 - input.shift.0) * inv_scale).max(0.0),
            ((cy - h / 2.0 - input.shift.1) * inv_scale).max(0.0),
            ((cx + w / 2.0 - input.shift.0) * inv_scale).min(max_x),
            ((cy + h / 2.0 - input.shift.1) * inv_scale).min(max_y),
        ];
        // Drop small vehicles
        let vehicle_area = (bbox[2] - bbox[0] + 1.0) * (bbox[3] - bbox[1] + 1.0);
        if vehicle_area / screen_area < config.vehicle_area_ratio_threshold {
            continue;
        }
        vehicles.push(Vehicle {
            bbox,
            confidence,
            is_special: false,
            plates: Vec::new(),
        });
    }
    stream_log!(log, Level::Trace, "before nms_vehicles count: {}", vehicles.len());
    sort_by_confidence(&mut vehicles, |v| v.confidence);
    suppress(&mut vehicles, |a, b| iou(&a.bbox, &b.bbox) > config.vehicle_iou_threshold);
    stream_log!(log, Level::Trace, "after nms_vehicles count: {}", vehicles.len());
    Ok(vehicles)
}

fn softmax(values: &[f32]) -> Vec<f32> {
    let sum: f32 = values.iter().map(|v| v.exp()).sum();
    values.iter().map(|v| v.exp() / sum).collect()
}

/// VCNet: marks special vehicles.
pub async fn classify_special(triton: &Triton, image: &RgbImage, config: &VStreamConfig, vehicles: &mut [Vehicle], log: StreamLog<'_>) -> Result<()> {
    let model = &config.vc_net;
    stream_log!(log, Level::Trace, "before inference VcNet");
    let tensors = tokio::task::block_in_place(|| {
        vehicles
            .iter()
            .enumerate()
            .map(|(vindex, v)| {
                stream_log!(log, Level::Trace, "before preprocess image {vindex} for VcNet");
                let tensor = prepare_vc(&roi(image, &v.bbox)?, model.input_width, model.input_height);
                stream_log!(log, Level::Trace, "after preprocess image {vindex} for VcNet");
                Ok(tensor)
            })
            .collect::<Result<Vec<_>>>()
    })?;
    let requests = tensors.iter().map(|t| request(model, nchw(model), t, config)).collect();
    let outputs = infer_all(triton, requests, log).await?;
    stream_log!(log, Level::Trace, "after inference VcNet");
    stream_log!(log, Level::Trace, "special confidence threshold: {:.3}", config.special_confidence);
    for (vindex, (vehicle, scores)) in vehicles.iter_mut().zip(outputs).enumerate() {
        stream_log!(log, Level::Trace, "vindex = {vindex};  softmax scores: {scores:?}");
        let scores = softmax(&scores);
        let b = vehicle.bbox;
        stream_log!(
            log,
            Level::Trace,
            "vindex = {vindex};  bbox = [{:.2}, {:.2}, {:.2}, {:.2}];  scores: {scores:?}",
            b[0],
            b[1],
            b[2],
            b[3]
        );
        if scores.len() >= 2 {
            vehicle.is_special = scores[1] > scores[0] && scores[1] > config.special_confidence;
        }
    }
    Ok(())
}

/// Work area polygons in frame pixels, truncated to integers like the C++ service.
fn absolute_work_area(work_area: &[WorkAreaPolygon], width: u32, height: u32) -> Vec<Vec<[f32; 2]>> {
    work_area
        .iter()
        .map(|polygon| {
            polygon
                .iter()
                .map(|p| [(p[0] * width as f32 / 100.0) as i32 as f32, (p[1] * height as f32 / 100.0) as i32 as f32])
                .collect()
        })
        .collect()
}

/// LPDNet: plates of every vehicle, filtered by the work area and the minimum plate height.
pub async fn detect_plates(triton: &Triton, image: &RgbImage, config: &VStreamConfig, vehicles: &mut [Vehicle], log: StreamLog<'_>) -> Result<()> {
    let model = &config.lpd_net;
    stream_log!(log, Level::Trace, "before inference LPDNet");
    let inputs = tokio::task::block_in_place(|| {
        vehicles
            .iter()
            .enumerate()
            .map(|(vindex, v)| {
                stream_log!(log, Level::Trace, "before preprocess image {vindex} for LPDNet");
                let input = prepare_yolo(&roi(image, &v.bbox)?, model.input_width, model.input_height);
                stream_log!(log, Level::Trace, "after preprocess image {vindex} for LPDNet");
                Ok(input)
            })
            .collect::<Result<Vec<_>>>()
    })?;
    let requests = inputs.iter().map(|input| request(model, nchw(model), &input.tensor, config)).collect();
    let outputs = infer_all(triton, requests, log).await?;
    stream_log!(log, Level::Trace, "after inference LPDNet");

    let work_area = absolute_work_area(&config.work_area, image.width(), image.height());
    for (vindex, ((vehicle, input), data)) in vehicles.iter_mut().zip(&inputs).zip(outputs).enumerate() {
        // Rows of 14: left, top, right, bottom, confidence, class, 4 key points clockwise from the top left
        const COLUMNS: usize = 14;
        let inv_scale = 1.0 / input.scale as f32;
        let (shift_x, shift_y) = input.shift;
        let [bx, by, _, _] = vehicle.bbox;
        stream_log!(log, Level::Trace, "license plate confidence threshold (vindex = {vindex}): {}", config.plate_confidence);
        let mut plates: Vec<LicensePlate> = data
            .chunks_exact(COLUMNS)
            .filter(|row| row[4] > config.plate_confidence)
            .map(|row| {
                let mut kpts = [0.0; 8];
                for (l, kpt) in kpts.iter_mut().enumerate() {
                    let (shift, delta) = if l % 2 == 1 { (shift_y, by) } else { (shift_x, bx) };
                    *kpt = delta + ((row[6 + l] - shift) as f64 / input.scale) as f32;
                }
                LicensePlate {
                    bbox: [
                        bx + (row[0] - shift_x) * inv_scale,
                        by + (row[1] - shift_y) * inv_scale,
                        bx + (row[2] - shift_x) * inv_scale,
                        by + (row[3] - shift_y) * inv_scale,
                    ],
                    confidence: row[4],
                    kpts,
                    plate_class_common: -1,
                    numbers: Vec::new(),
                }
            })
            .collect();
        stream_log!(log, Level::Trace, "before nms_plates count (vindex = {vindex}): {}", plates.len());
        sort_by_confidence(&mut plates, |p| p.confidence);
        suppress(&mut plates, |a, b| a.plate_class_common == b.plate_class_common && has_intersection(&a.bbox, &b.bbox));
        stream_log!(log, Level::Trace, "after nms_plates count (vindex = {vindex}): {}", plates.len());

        if !work_area.is_empty() || config.min_plate_height > 0.0 {
            plates.retain(|plate| {
                let corners = plate.corners();
                if !work_area.is_empty() {
                    // The plate must lie inside one of the work area polygons
                    let plate_area = intersection_area(&corners, &corners);
                    let inside = work_area.iter().any(|polygon| {
                        let common = intersection_area(polygon, &corners);
                        plate_area.min(common) / plate_area.max(common) > 0.999
                    });
                    if !inside {
                        stream_log!(
                            log,
                            Level::Trace,
                            "license plate does not fall completely into the work area;  key points: {}",
                            format_key_points(&plate.kpts)
                        );
                        return false;
                    }
                }
                let too_small = config.min_plate_height > 0.0
                    && distance(corners[0], corners[3]).min(distance(corners[1], corners[2])) < config.min_plate_height;
                if too_small {
                    stream_log!(log, Level::Trace, "license plate height is too small;  key points: {}", format_key_points(&plate.kpts));
                }
                !too_small
            });
        }
        stream_log!(log, Level::Trace, "detected plate count (vindex = {vindex}): {}", plates.len());
        vehicle.plates = plates;
    }
    Ok(())
}

/// Key points as the C++ service logs them: `[(x, y) (x, y) (x, y) (x, y)]`.
fn format_key_points(kpts: &[f32; 8]) -> String {
    let points: Vec<String> = kpts.chunks_exact(2).map(|p| format!("({:.2}, {:.2})", p[0], p[1])).collect();
    format!("[{}]", points.join(" "))
}

/// Removes a plate detected on two overlapping vehicles from one of them, then drops vehicles
/// that have neither plates nor a special status, or lie outside the work area.
pub fn remove_duplicate_plates(config: &VStreamConfig, vehicles: &mut Vec<Vehicle>, width: u32, height: u32, log: StreamLog<'_>) {
    for i in 0..vehicles.len().saturating_sub(1) {
        for j in i + 1..vehicles.len() {
            if has_intersection(&vehicles[i].bbox, &vehicles[j].bbox) {
                remove_duplicates_of_pair(vehicles, i, j, log);
            }
        }
    }

    let work_area = absolute_work_area(&config.work_area, width, height);
    vehicles.retain(|vehicle| {
        if vehicle.plates.is_empty() && !vehicle.is_special {
            return false;
        }
        if work_area.is_empty() {
            return true;
        }
        let b = vehicle.bbox;
        let polygon = [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]];
        work_area.iter().any(|area| intersection_area(area, &polygon) > 0.0)
    });
}

/// Removes the plates two overlapping vehicles share from one of them. Duplicates are marked
/// first and removed afterwards; the counts of the remaining plates are updated after each mark,
/// as the choice of the vehicle depends on them: the vehicle with more plates loses the plate, or,
/// with equal counts, the larger vehicle.
fn remove_duplicates_of_pair(vehicles: &mut [Vehicle], i: usize, j: usize, log: StreamLog<'_>) {
    let (plates_i, plates_j) = (&vehicles[i].plates, &vehicles[j].plates);
    let mut remove_i = vec![false; plates_i.len()];
    let mut remove_j = vec![false; plates_j.len()];
    let (mut remaining_i, mut remaining_j) = (plates_i.len(), plates_j.len());
    for m in 0..plates_i.len() {
        for n in 0..plates_j.len() {
            if remove_i[m] {
                break;
            }
            if remove_j[n] || iou(&plates_i[m].bbox, &plates_j[n].bbox) <= 0.7 {
                continue;
            }
            let remove_from_i = if remaining_i == remaining_j {
                area(&vehicles[i].bbox) > area(&vehicles[j].bbox)
            } else {
                remaining_i > remaining_j
            };
            stream_log!(log, Level::Trace, "remove duplicate plate number from vehicle {}", if remove_from_i { i } else { j });
            if remove_from_i {
                remove_i[m] = true;
                remaining_i -= 1;
            } else {
                remove_j[n] = true;
                remaining_j -= 1;
            }
        }
    }
    let mut marked = remove_i.into_iter();
    vehicles[i].plates.retain(|_| !marked.next().unwrap_or(false));
    let mut marked = remove_j.into_iter();
    vehicles[j].plates.retain(|_| !marked.next().unwrap_or(false));
}

/// LPCNet: the plate type of every plate.
pub async fn classify_plates(triton: &Triton, image: &RgbImage, config: &VStreamConfig, plates: &mut [&mut LicensePlate], log: StreamLog<'_>) -> Result<()> {
    let model = &config.lpc_net;
    stream_log!(log, Level::Trace, "before inference LpcNet");
    let tensors = tokio::task::block_in_place(|| {
        plates
            .iter()
            .enumerate()
            .map(|(pindex, p)| {
                stream_log!(log, Level::Trace, "before preprocess image {pindex} for LpcNet");
                let tensor = prepare_lpc(&roi(image, &p.bbox)?, model.input_width, model.input_height);
                stream_log!(log, Level::Trace, "after preprocess image {pindex} for LpcNet");
                Ok(tensor)
            })
            .collect::<Result<Vec<_>>>()
    })?;
    let requests = tensors.iter().map(|t| request(model, nchw(model), t, config)).collect();
    let outputs = infer_all(triton, requests, log).await?;
    stream_log!(log, Level::Trace, "after inference LpcNet");
    for (pindex, (plate, scores)) in plates.iter_mut().zip(outputs).enumerate() {
        stream_log!(log, Level::Trace, "pindex = {pindex};  softmax scores: {scores:?}");
        let scores = softmax(&scores);
        let b = plate.bbox;
        stream_log!(
            log,
            Level::Trace,
            "pindex = {pindex};  bbox = [{:.2}, {:.2}, {:.2}, {:.2}];  scores: {scores:?}",
            b[0],
            b[1],
            b[2],
            b[3]
        );
        let mut best = 0;
        for (i, score) in scores.iter().enumerate().skip(1) {
            if *score > scores[best] {
                best = i;
            }
        }
        plate.plate_class_common = best as i32;
    }
    Ok(())
}

struct CharData {
    bbox: BBox,
    confidence: f32,
    char_class: i32,
    plate_class: i32,
}

/// Reading order of characters: left to right, and top line first on two-line plates.
fn char_precedes(a: &CharData, b: &CharData) -> bool {
    if a.plate_class == PLATE_CLASS_RU_1A {
        let (ymin, ymax) = (1, 3);
        if a.bbox[ymax] < b.bbox[ymin] {
            return true;
        }
        if a.bbox[ymin] > b.bbox[ymax] {
            return false;
        }
        let y = a.bbox[ymin] as f64 + 0.5 * (a.bbox[ymax] - a.bbox[ymin]) as f64;
        if y < b.bbox[ymin] as f64 {
            return true;
        }
        if y > b.bbox[ymax] as f64 {
            return false;
        }
    }
    a.bbox[0] < b.bbox[0]
}

/// Insertion sort: the reading order is not a strict weak ordering for two-line plates,
/// and `slice::sort_by` may panic on such orderings.
fn sort_chars(chars: &mut [CharData]) {
    for i in 1..chars.len() {
        let mut j = i;
        while j > 0 && char_precedes(&chars[j], &chars[j - 1]) {
            chars.swap(j, j - 1);
            j -= 1;
        }
    }
}

const RU_LETTERS: &str = "ABCEHKMOPTXY";
const BY_LETTERS: &str = "ABCEHIKMOPTX";

/// Russian format `L NNN LL NN(N)`, ignoring characters that can't appear on such plates.
fn check_ru(number: &str) -> Option<String> {
    let number: String = number.chars().filter(|c| c.is_ascii_digit() || RU_LETTERS.contains(*c)).collect();
    if !(8..=9).contains(&number.len()) {
        return None;
    }
    let valid = number.chars().enumerate().all(|(i, c)| match i {
        0 | 4 | 5 => RU_LETTERS.contains(c),
        _ => c.is_ascii_digit(),
    });
    valid.then_some(number)
}

/// Drops leading non-digits down to 7 characters.
fn trim_to_seven(number: &str) -> Option<String> {
    let mut number = number.to_string();
    while number.len() > 7 {
        if number.starts_with(|c: char| c.is_ascii_digit()) {
            return None;
        }
        number.remove(0);
    }
    (number.len() == 7).then_some(number)
}

/// Belarusian format `NNNN LL N`.
fn check_by(number: &str) -> Option<String> {
    let number = trim_to_seven(number)?;
    let valid = number.chars().enumerate().all(|(i, c)| match i {
        4 | 5 => BY_LETTERS.contains(c),
        _ => c.is_ascii_digit(),
    });
    valid.then_some(number)
}

/// Armenian format `NN LL NNN` or `NNN LL NN`, checked as loosely as the C++ service does.
fn check_am(number: &str) -> Option<String> {
    let number = trim_to_seven(number)?;
    let valid = number.chars().enumerate().all(|(i, c)| match i {
        0 | 1 | 5 | 6 => c.is_ascii_digit(),
        3 => c.is_ascii_uppercase(),
        _ => true,
    });
    valid.then_some(number)
}

/// Normalises a number to a supported format and sets its plate class; false if none fits.
fn validate_plate_number(plate: &mut PlateNumber) -> bool {
    if plate.number.is_empty() {
        return false;
    }
    if let Some(number) = check_ru(&plate.number) {
        plate.number = number;
        if plate.plate_class != PLATE_CLASS_RU_1 && plate.plate_class != PLATE_CLASS_RU_1A {
            plate.plate_class = PLATE_CLASS_RU_1;
        }
        return true;
    }
    if let Some(number) = check_by(&plate.number) {
        plate.number = number;
        plate.plate_class = PLATE_CLASS_BY;
        return true;
    }
    if let Some(number) = check_am(&plate.number) {
        plate.number = number;
        plate.plate_class = PLATE_CLASS_AM;
        return true;
    }
    false
}

const LABELS: [&str; 37] = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "",
];

/// Height to width ratio of the straightened plate image, by plate class.
const HEIGHT_RATIO_BY_PLATE_CLASS: [f32; 4] = [112.0 / 520.0, 170.0 / 290.0, 112.0 / 520.0, 112.0 / 520.0];

/// Every reading of the plate: overlapping characters at one position branch the number.
fn assemble_numbers(chars: &[CharData], plate_class: i32, char_iou_threshold: f32) -> Vec<PlateNumber> {
    let mut numbers = vec![PlateNumber {
        plate_class,
        number: String::new(),
        score: 1.0,
    }];
    let mut used = HashSet::new();
    for i in 0..chars.len() {
        if used.contains(&i) {
            continue;
        }
        let mut alternatives = vec![i];
        for j in i + 1..chars.len() {
            if !used.contains(&j) && iou(&chars[i].bbox, &chars[j].bbox) > char_iou_threshold {
                alternatives.push(j);
                used.insert(j);
            }
        }
        if alternatives.len() > 1 {
            let copy = numbers.clone();
            for _ in 1..alternatives.len() {
                numbers.extend(copy.iter().cloned());
            }
        }
        let count = numbers.len();
        for (k, number) in numbers.iter_mut().enumerate() {
            let c = &chars[alternatives[k * alternatives.len() / count]];
            number.number += LABELS.get(c.char_class as usize).copied().unwrap_or_default();
            number.score *= c.confidence;
        }
    }
    numbers
}

/// LPRNet: straightens every plate and reads its number.
pub async fn recognize_plates(triton: &Triton, image: &RgbImage, config: &VStreamConfig, plates: &mut [&mut LicensePlate], log: StreamLog<'_>) -> Result<()> {
    let model = &config.lpr_net;
    stream_log!(log, Level::Trace, "before inference LPRNet");
    let inputs = tokio::task::block_in_place(|| {
        plates
            .iter()
            .enumerate()
            .map(|(pindex, plate)| {
                stream_log!(log, Level::Trace, "before preprocess image {pindex} for LPRNet");
                let class = plate.plate_class_common.clamp(0, HEIGHT_RATIO_BY_PLATE_CLASS.len() as i32 - 1) as usize;
                let width = model.input_width as f32;
                let height = width * HEIGHT_RATIO_BY_PLATE_CLASS[class];
                let dst = [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]];
                let transform = perspective_transform(plate.corners(), dst).context("degenerate plate corners")?;
                let straightened = warp_perspective(image, &transform, model.input_width, height as u32)
                    .context("degenerate plate transform")?;
                let input = prepare_yolo(&straightened, model.input_width, model.input_height);
                stream_log!(log, Level::Trace, "after preprocess image {pindex} for LPRNet");
                Ok(input)
            })
            .collect::<Result<Vec<_>>>()
    })?;
    let requests = inputs.iter().map(|input| request(model, nchw(model), &input.tensor, config)).collect();
    let outputs = infer_all(triton, requests, log).await?;
    stream_log!(log, Level::Trace, "after inference LPRNet");

    for (pindex, ((plate, input), data)) in plates.iter_mut().zip(&inputs).zip(outputs).enumerate() {
        // Rows of 6: left, top, right, bottom, confidence, class
        let inv_scale = 1.0 / input.scale as f32;
        let (shift_x, shift_y) = input.shift;
        stream_log!(log, Level::Trace, "char score threshold: {:.2}", config.char_score);
        let mut chars: Vec<CharData> = data
            .chunks_exact(6)
            .take(300)
            .filter(|row| row[4] > config.char_score)
            .map(|row| CharData {
                bbox: [
                    (row[0] - shift_x) * inv_scale,
                    (row[1] - shift_y) * inv_scale,
                    (row[2] - shift_x) * inv_scale,
                    (row[3] - shift_y) * inv_scale,
                ],
                confidence: row[4],
                char_class: row[5] as i32,
                plate_class: plate.plate_class_common,
            })
            .collect();
        stream_log!(log, Level::Trace, "before nms_chars count (pindex = {pindex}): {}", chars.len());
        sort_by_confidence(&mut chars, |c| c.confidence);
        suppress(&mut chars, |a, b| a.char_class == b.char_class && iou(&a.bbox, &b.bbox) > config.char_iou_threshold);
        stream_log!(log, Level::Trace, "after nms_chars count (pindex = {pindex}): {}", chars.len());
        sort_chars(&mut chars);

        let mut numbers = assemble_numbers(&chars, plate.plate_class_common, config.char_iou_threshold);
        numbers.retain_mut(|number| {
            let is_valid = validate_plate_number(number);
            if !is_valid {
                stream_log!(log, Level::Trace, "invalid plate number: {}", number.number);
            }
            is_valid
        });
        numbers.sort_by(|a, b| b.score.total_cmp(&a.score));
        plate.numbers = numbers;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn number(text: &str, plate_class: i32) -> PlateNumber {
        PlateNumber {
            plate_class,
            number: text.to_string(),
            score: 1.0,
        }
    }

    #[test]
    fn plate_number_formats() {
        let mut ru = number("O588OA68", PLATE_CLASS_BY);
        assert!(validate_plate_number(&mut ru));
        assert_eq!((ru.number.as_str(), ru.plate_class), ("O588OA68", PLATE_CLASS_RU_1));

        let mut too_long = number("XO588OA168", PLATE_CLASS_RU_1A);
        assert!(!validate_plate_number(&mut too_long));

        let mut with_noise = number("O588-OA68", PLATE_CLASS_RU_1A);
        assert!(validate_plate_number(&mut with_noise));
        assert_eq!((with_noise.number.as_str(), with_noise.plate_class), ("O588OA68", PLATE_CLASS_RU_1A));

        let mut by = number("1234AB7", PLATE_CLASS_RU_1);
        assert!(validate_plate_number(&mut by));
        assert_eq!((by.number.as_str(), by.plate_class), ("1234AB7", PLATE_CLASS_BY));

        let mut am = number("12AB345", PLATE_CLASS_RU_1);
        assert!(validate_plate_number(&mut am));
        assert_eq!(am.plate_class, PLATE_CLASS_AM);

        assert!(!validate_plate_number(&mut number("", PLATE_CLASS_RU_1)));
        assert!(!validate_plate_number(&mut number("ABC", PLATE_CLASS_RU_1)));
    }

    #[test]
    fn overlapping_characters_branch_numbers() {
        let char_at = |x: f32, class: i32, confidence: f32| CharData {
            bbox: [x, 0.0, x + 10.0, 20.0],
            confidence,
            char_class: class,
            plate_class: PLATE_CLASS_RU_1,
        };
        // "1", then "2" or "3" at the same place
        let chars = [char_at(0.0, 1, 0.9), char_at(20.0, 2, 0.8), char_at(20.5, 3, 0.5)];
        let numbers = assemble_numbers(&chars, PLATE_CLASS_RU_1, 0.7);
        let texts: Vec<&str> = numbers.iter().map(|n| n.number.as_str()).collect();
        assert_eq!(texts, vec!["12", "13"]);
        assert!((numbers[0].score - 0.72).abs() < 1e-6);
    }

    #[test]
    fn two_line_reading_order() {
        let char_at = |x: f32, y: f32| CharData {
            bbox: [x, y, x + 10.0, y + 20.0],
            confidence: 1.0,
            char_class: 0,
            plate_class: PLATE_CLASS_RU_1A,
        };
        let mut chars = vec![char_at(30.0, 40.0), char_at(0.0, 40.0), char_at(50.0, 0.0), char_at(10.0, 0.0)];
        sort_chars(&mut chars);
        let order: Vec<(f32, f32)> = chars.iter().map(|c| (c.bbox[0], c.bbox[1])).collect();
        assert_eq!(order, vec![(10.0, 0.0), (50.0, 0.0), (0.0, 40.0), (30.0, 40.0)]);
    }

    fn plate(x: f32, id: f32) -> LicensePlate {
        LicensePlate {
            bbox: [x, 100.0, x + 50.0, 115.0],
            // The confidence tells test plates apart
            confidence: id,
            kpts: [0.0; 8],
            plate_class_common: 0,
            numbers: Vec::new(),
        }
    }

    fn vehicle(bbox: BBox, plates: Vec<LicensePlate>) -> Vehicle {
        Vehicle {
            bbox,
            confidence: 1.0,
            is_special: false,
            plates,
        }
    }

    /// Ids of the remaining plates per remaining vehicle.
    fn dedupe(vehicles: Vec<Vehicle>) -> Vec<Vec<f32>> {
        let mut vehicles = vehicles;
        let config = VStreamConfig::from_json(&serde_json::json!({}));
        let logger = crate::logging::Logger::null();
        let log = StreamLog {
            logger: &logger,
            key: "1_test",
            level: Level::Trace,
            always: false,
        };
        remove_duplicate_plates(&config, &mut vehicles, 1920, 1080, log);
        vehicles.iter().map(|v| v.plates.iter().map(|p| p.confidence).collect()).collect()
    }

    const SMALL: BBox = [0.0, 0.0, 400.0, 300.0];
    const LARGE: BBox = [100.0, 50.0, 700.0, 500.0];

    #[test]
    fn duplicate_on_equal_counts_is_removed_from_the_larger_vehicle() {
        let result = dedupe(vec![vehicle(LARGE, vec![plate(200.0, 1.0)]), vehicle(SMALL, vec![plate(201.0, 2.0)])]);
        // The large vehicle has no plates left and is dropped
        assert_eq!(result, vec![vec![2.0]]);
        let result = dedupe(vec![vehicle(SMALL, vec![plate(200.0, 1.0)]), vehicle(LARGE, vec![plate(201.0, 2.0)])]);
        assert_eq!(result, vec![vec![1.0]]);
    }

    #[test]
    fn duplicate_is_removed_from_the_vehicle_with_more_plates() {
        let result = dedupe(vec![
            vehicle(SMALL, vec![plate(200.0, 1.0), plate(10.0, 2.0)]),
            vehicle(LARGE, vec![plate(201.0, 3.0)]),
        ]);
        assert_eq!(result, vec![vec![2.0], vec![3.0]]);
    }

    #[test]
    fn several_duplicates_recount_remaining_plates() {
        // Two shared plates; vehicle 0 has three plates, vehicle 1 two. The first duplicate goes
        // from vehicle 0 (3 > 2); then both have two plates and the second goes from the larger one.
        let result = dedupe(vec![
            vehicle(SMALL, vec![plate(200.0, 1.0), plate(300.0, 2.0), plate(10.0, 3.0)]),
            vehicle(LARGE, vec![plate(201.0, 4.0), plate(301.0, 5.0)]),
        ]);
        assert_eq!(result, vec![vec![2.0, 3.0], vec![4.0]]);
    }

    #[test]
    fn plate_matching_several_candidates() {
        // One plate of the small vehicle matches two plates of the large one, which loses both:
        // it has more plates at the first match and is larger at the second
        let result = dedupe(vec![
            vehicle(SMALL, vec![plate(200.0, 1.0)]),
            vehicle(LARGE, vec![plate(201.0, 2.0), plate(202.0, 3.0)]),
        ]);
        assert_eq!(result, vec![vec![1.0]]);

        let result = dedupe(vec![
            vehicle(SMALL, vec![plate(200.0, 1.0), plate(10.0, 2.0)]),
            vehicle(LARGE, vec![plate(201.0, 3.0), plate(202.0, 4.0), plate(600.0, 5.0)]),
        ]);
        // 2 < 3: the large vehicle loses plate 3; then 2 == 2 and it is larger: it loses plate 4
        assert_eq!(result, vec![vec![1.0, 2.0], vec![5.0]]);

        let result = dedupe(vec![
            vehicle(SMALL, vec![plate(200.0, 1.0), plate(10.0, 2.0), plate(20.0, 6.0)]),
            vehicle(LARGE, vec![plate(201.0, 3.0), plate(202.0, 4.0)]),
        ]);
        // 3 > 2: plate 1 is removed from its own vehicle at the first match and not compared with plate 4 again
        assert_eq!(result, vec![vec![2.0, 6.0], vec![3.0, 4.0]]);
    }

    #[test]
    fn plates_of_separate_vehicles_are_kept() {
        let far: BBox = [1000.0, 0.0, 1400.0, 300.0];
        let result = dedupe(vec![vehicle(SMALL, vec![plate(200.0, 1.0)]), vehicle(far, vec![plate(200.0, 2.0)])]);
        assert_eq!(result, vec![vec![1.0], vec![2.0]]);
    }
}
