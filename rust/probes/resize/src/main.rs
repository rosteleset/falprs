//! Resize probe: compares `imaging::resize_linear` and `imaging::resize_area` with OpenCV's `cv::resize`
//! (`INTER_LINEAR`, `INTER_AREA`) byte by byte.
//!   probe-resize <dir written by reference/resize_ref.cpp> [case to dump]
//! With a case number, prints that case's differing values as `x y channel rust opencv`.
//! Prints every case with differing values, with where in the row they lie: OpenCV rounds the vertical pass
//! of whole SIMD blocks and of the row's tail with different formulas.

use std::path::Path;

use anyhow::{Context, Result, bail};
use falprs::imaging::{resize_area, resize_linear};
use image::RgbImage;

/// A raw BGR file as an RGB image.
fn read_bgr(path: &Path, width: u32, height: u32) -> Result<RgbImage> {
    let mut data = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    data.as_chunks_mut::<3>().0.iter_mut().for_each(|p| p.swap(0, 2));
    RgbImage::from_raw(width, height, data).context("size mismatch")
}

fn main() -> Result<()> {
    let dir = std::env::args().nth(1).context("usage: probe-resize <dir>")?;
    let dir = Path::new(&dir);
    let dump = std::env::args().nth(2);
    let list = std::fs::read_to_string(dir.join("cases.txt"))?;
    let (mut cases, mut failed) = (0, 0);
    for line in list.lines() {
        let fields: Vec<&str> = line.split_whitespace().collect();
        let [case, mode, sw, sh, dw, dh] = fields[..] else {
            bail!("bad line: {line}")
        };
        let [sw, sh, dw, dh]: [u32; 4] = [sw.parse()?, sh.parse()?, dw.parse()?, dh.parse()?];
        let src = read_bgr(&dir.join(format!("{case}.in")), sw, sh)?;
        let expected = read_bgr(&dir.join(format!("{case}.out")), dw, dh)?;
        let actual = match mode {
            "linear" => resize_linear(&src, dw, dh),
            "area" => resize_area(&src, dw, dh),
            _ => bail!("bad mode: {mode}"),
        };
        cases += 1;
        if dump.as_deref() == Some(case) {
            for (i, (a, e)) in actual.as_raw().iter().zip(expected.as_raw()).enumerate() {
                if a != e {
                    let pixel = i / 3;
                    println!("{} {} {} {a} {e}", pixel % dw as usize, pixel / dw as usize, i % 3);
                }
            }
        }
        let row_len = dw as usize * 3;
        // Differing values: count, largest difference, and the leftmost index within a row
        let (mut count, mut max_diff, mut first_index) = (0, 0, usize::MAX);
        for (i, (a, e)) in actual.as_raw().iter().zip(expected.as_raw()).enumerate() {
            if a != e {
                count += 1;
                max_diff = max_diff.max(a.abs_diff(*e));
                first_index = first_index.min(i % row_len);
            }
        }
        if count > 0 {
            failed += 1;
            println!("case {case} {mode} {sw}x{sh} -> {dw}x{dh}: {count} values differ (max {max_diff}), from index {first_index} of {row_len} in a row");
        }
    }
    println!("{cases} cases, {failed} with differences");
    if failed > 0 {
        std::process::exit(1);
    }
    Ok(())
}
