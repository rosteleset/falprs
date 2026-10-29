//! Barcode probe: decodes raw BGR images (`<name>_<W>x<H>.bgr`) with rxing the way the FRS
//! workflow decodes them with ZXing, printing one JSON line per image like the reference tool.

use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};
use rxing::DecodeHints;
use serde_json::json;

/// ZXing's luminance of an RGB pixel. The FRS workflow hands ZXing OpenCV's BGR data labelled
/// as RGB, so the first byte is weighted as red.
fn luminance(bgr: &[u8]) -> Vec<u8> {
    bgr.chunks_exact(3)
        .map(|p| ((306 * p[0] as u32 + 601 * p[1] as u32 + 117 * p[2] as u32 + 0x200) >> 10) as u8)
        .collect()
}

fn size_of(name: &str) -> Option<(u32, u32)> {
    let size = name.rsplit('_').next()?;
    let (w, h) = size.split_once('x')?;
    Some((w.parse().ok()?, h.parse().ok()?))
}

fn main() -> Result<()> {
    let dir = std::env::args().nth(1).context("usage: probe-barcodes <dir>")?;
    let mut files: Vec<_> = std::fs::read_dir(&dir)?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "bgr"))
        .collect();
    files.sort();
    for path in files {
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or_default();
        let (w, h) = size_of(stem).with_context(|| format!("no size in {}", path.display()))?;
        let luma = luminance(&std::fs::read(&path)?);
        let mut hints = DecodeHints {
            TryHarder: Some(true),
            AlsoInverted: Some(true),
            ..Default::default()
        };
        let start = Instant::now();
        let results = rxing::helpers::detect_multiple_in_luma_with_hints(luma, w, h, &mut hints).unwrap_or_default();
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        let results: Vec<_> = results
            .iter()
            .map(|r| {
                let points: Vec<f32> = r.getPoints().iter().flat_map(|p| [p.x, p.y]).collect();
                json!({"format": r.getBarcodeFormat().to_string(), "text": r.getText(), "position": points})
            })
            .collect();
        let file = Path::new(&path).file_name().and_then(|f| f.to_str()).unwrap_or_default();
        println!("{}", json!({"file": file, "ms": ms, "results": results}));
    }
    Ok(())
}
