//! Face alignment probe.
//!   probe-face-alignment detect <bgr dir> <landmarks out>
//!     finds faces with SCRFD and writes "<image> <face> x1 y1 ... x5 y5" lines
//!   probe-face-alignment compare <bgr dir> <landmarks> <reference matrices> <reference aligned dir>
//!     aligns the faces in Rust and compares transforms, pixels and ArcFace descriptors with OpenCV's

use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use falprs::imaging::{Affine, estimate_affine_partial_lmeds, prepare_arcface, prepare_scrfd, warp_affine};
use falprs::triton::{InferRequest, Tensor, Triton};
use image::{Rgb, RgbImage};

const TRITON: &str = "127.0.0.1:8001";
const TIMEOUT: Duration = Duration::from_secs(5);
const FACE_CONFIDENCE: f32 = 0.7;
const FACE_IOU_THRESHOLD: f32 = 0.4;
const FACE_SIZE: u32 = 112;

/// The aligned face template of the FRS workflow, for a 112x112 face.
const TEMPLATE: [[f32; 2]; 5] = [
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
];

fn read_bgr(path: &Path) -> Result<RgbImage> {
    let stem = path.file_stem().and_then(|s| s.to_str()).context("bad file name")?;
    let (w, h) = stem.rsplit('_').next().and_then(|s| s.split_once('x')).context("no size in name")?;
    let mut data = std::fs::read(path)?;
    data.chunks_exact_mut(3).for_each(|p| p.swap(0, 2));
    RgbImage::from_raw(w.parse()?, h.parse()?, data).context("size mismatch")
}

struct Face {
    bbox: [f32; 4],
    confidence: f32,
    landmarks: [f32; 10],
}

fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let w = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let h = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let inter = w * h;
    inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
}

/// SCRFD 10G with key points as the FRS workflow decodes it.
async fn detect_faces(triton: &Triton, image: &RgbImage) -> Result<Vec<Face>> {
    const SIZE: u32 = 320;
    let (tensor, scale) = prepare_scrfd(image, SIZE, SIZE);
    // Per stride 8, 16, 32: scores, boxes, key points
    let names = [["448", "451", "454"], ["471", "474", "477"], ["494", "497", "500"]];
    let outputs = triton
        .infer(InferRequest {
            endpoint: TRITON,
            model_name: "scrfd",
            inputs: vec![Tensor::fp32("input.1", vec![1, 3, SIZE as i64, SIZE as i64], &tensor)],
            outputs: names.iter().flatten().copied().collect(),
            timeout: TIMEOUT,
        })
        .await?;
    let by_name: HashMap<&str, Vec<f32>> = outputs.iter().map(|t| (t.name.as_str(), t.as_f32())).collect();
    let mut faces = Vec::new();
    for (stride, [scores, boxes, kps]) in [8u32, 16, 32].into_iter().zip(names) {
        let (scores, boxes, kps) = (&by_name[scores], &by_name[boxes], &by_name[kps]);
        let cells = SIZE / stride;
        let s = stride as f32;
        for k in 0..(cells * cells) as usize {
            let px = s * (k % cells as usize) as f32;
            let py = s * (k / cells as usize) as f32;
            for anchor in 2 * k..2 * k + 2 {
                if scores[anchor] < FACE_CONFIDENCE {
                    continue;
                }
                let b = &boxes[anchor * 4..anchor * 4 + 4];
                let mut landmarks = [0.0; 10];
                for j in 0..5 {
                    landmarks[2 * j] = (px + kps[anchor * 10 + 2 * j] * s) / scale;
                    landmarks[2 * j + 1] = (py + kps[anchor * 10 + 2 * j + 1] * s) / scale;
                }
                faces.push(Face {
                    bbox: [(px - b[0] * s) / scale, (py - b[1] * s) / scale, (px + b[2] * s) / scale, (py + b[3] * s) / scale],
                    confidence: scores[anchor],
                    landmarks,
                });
            }
        }
    }
    faces.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
    let mut kept: Vec<Face> = Vec::new();
    for face in faces {
        if kept.iter().all(|k| iou(&k.bbox, &face.bbox) <= FACE_IOU_THRESHOLD) {
            kept.push(face);
        }
    }
    Ok(kept)
}

async fn descriptor(triton: &Triton, face: &RgbImage) -> Result<Vec<f32>> {
    let tensor = prepare_arcface(face, FACE_SIZE, FACE_SIZE);
    let outputs = triton
        .infer(InferRequest {
            endpoint: TRITON,
            model_name: "arcface",
            inputs: vec![Tensor::fp32("input.1", vec![1, 3, FACE_SIZE as i64, FACE_SIZE as i64], &tensor)],
            outputs: vec!["683"],
            timeout: TIMEOUT,
        })
        .await?;
    Ok(outputs.first().context("no output")?.as_f32())
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let norm = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    dot / (norm(a) * norm(b))
}

async fn detect(dir: &str, out: &str) -> Result<()> {
    let triton = Triton::default();
    let mut lines = Vec::new();
    let mut files: Vec<_> = std::fs::read_dir(dir)?.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|e| e == "bgr")).collect();
    files.sort();
    for path in files {
        let image = read_bgr(&path)?;
        let file = path.file_name().and_then(|f| f.to_str()).unwrap_or_default();
        for (index, face) in detect_faces(&triton, &image).await?.iter().enumerate() {
            let points: Vec<String> = face.landmarks.iter().map(|v| v.to_string()).collect();
            lines.push(format!("{file} {index} {}", points.join(" ")));
        }
    }
    println!("faces: {}", lines.len());
    std::fs::write(out, lines.join("\n") + "\n")?;
    Ok(())
}

async fn compare(dir: &str, landmarks_file: &str, matrices_file: &str, aligned_dir: &str) -> Result<()> {
    let triton = Triton::default();
    let reference: HashMap<String, Affine> = std::fs::read_to_string(matrices_file)?
        .lines()
        .filter_map(|line| {
            let parts: Vec<&str> = line.split_whitespace().collect();
            let m: Vec<f64> = parts.get(2..8)?.iter().filter_map(|v| v.parse().ok()).collect();
            Some((format!("{} {}", parts[0], parts[1]), m.try_into().ok()?))
        })
        .collect();
    let mut cosines = Vec::new();
    println!("face                                   matrix diff  pixels differ (max)  cosine");
    for line in std::fs::read_to_string(landmarks_file)?.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        let (file, index) = (parts[0], parts[1]);
        let values: Vec<f32> = parts[2..12].iter().map(|v| v.parse()).collect::<Result<_, _>>()?;
        let landmarks: Vec<[f32; 2]> = values.chunks_exact(2).map(|p| [p[0], p[1]]).collect();
        let template: Vec<[f32; 2]> = TEMPLATE.iter().map(|p| [p[0] / 112.0 * FACE_SIZE as f32, p[1] / 112.0 * FACE_SIZE as f32]).collect();

        let image = read_bgr(&Path::new(dir).join(file))?;
        let Some(m) = estimate_affine_partial_lmeds(&landmarks, &template) else {
            bail!("no transform for {file} {index}");
        };
        let ours = warp_affine(&image, &m, FACE_SIZE, FACE_SIZE);

        let key = format!("{file} {index}");
        let reference_m = reference.get(&key).with_context(|| format!("no reference for {key}"))?;
        let stem = Path::new(file).file_stem().and_then(|s| s.to_str()).unwrap_or_default();
        let mut theirs_raw = std::fs::read(Path::new(aligned_dir).join(format!("{stem}_{index}.aligned")))?;
        theirs_raw.chunks_exact_mut(3).for_each(|p| p.swap(0, 2));
        let theirs = RgbImage::from_raw(FACE_SIZE, FACE_SIZE, theirs_raw).context("reference size")?;

        let matrix_diff = m.iter().zip(reference_m).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        let differ = ours.pixels().zip(theirs.pixels()).filter(|(a, b)| a != b).count();
        let max_diff = ours
            .pixels()
            .zip(theirs.pixels())
            .flat_map(|(a, b): (&Rgb<u8>, &Rgb<u8>)| (0..3).map(move |c| a[c].abs_diff(b[c])))
            .max()
            .unwrap_or(0);
        let similarity = cosine(&descriptor(&triton, &ours).await?, &descriptor(&triton, &theirs).await?);
        cosines.push(similarity);
        println!("{key:38} {matrix_diff:11.2e}  {differ:13} ({max_diff:3})  {similarity:.6}");
    }
    let min = cosines.iter().copied().fold(f64::INFINITY, f64::min);
    let mean = cosines.iter().sum::<f64>() / cosines.len() as f64;
    println!("\nfaces: {}, cosine min {min:.6}, mean {mean:.6}: {}", cosines.len(), if min > 0.99 { "PASS" } else { "FAIL" });
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("detect") if args.len() == 4 => detect(&args[2], &args[3]).await,
        Some("compare") if args.len() == 6 => compare(&args[2], &args[3], &args[4], &args[5]).await,
        _ => bail!("usage: probe-face-alignment detect <bgr dir> <landmarks out> | compare <bgr dir> <landmarks> <matrices> <aligned dir>"),
    }
}
