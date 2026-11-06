//! Warp probe: compares `imaging::perspective_transform` and `imaging::warp_perspective` with
//! OpenCV's `getPerspectiveTransform` and `warpPerspective` (`INTER_LINEAR`, black border), the
//! plate straightening of LPRS, bit for bit.
//!   probe-warp <dir written by reference/warp_ref.cpp>
//! Corners OpenCV finds no transform for (all zeros but the last element) must be rejected; LPRS
//! skips such plates.

use std::path::Path;

use anyhow::{Context, Result, bail};
use falprs::imaging::{perspective_transform, warp_perspective};
use image::RgbImage;

/// A raw BGR file as an RGB image.
fn read_bgr(path: &Path, width: u32, height: u32) -> Result<RgbImage> {
    let mut data = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    data.as_chunks_mut::<3>().0.iter_mut().for_each(|p| p.swap(0, 2));
    RgbImage::from_raw(width, height, data).context("size mismatch")
}

fn main() -> Result<()> {
    let dir = std::env::args().nth(1).context("usage: probe-warp <dir>")?;
    let dir = Path::new(&dir);
    let list = std::fs::read_to_string(dir.join("cases.txt"))?;
    let (mut cases, mut degenerate, mut failed) = (0, 0, 0);
    for line in list.lines() {
        let fields: Vec<u64> = line.split_whitespace().map(str::parse).collect::<Result<_, _>>()?;
        let [case, sw, sh, dw, dh] = [fields[0], fields[1], fields[2], fields[3], fields[4]].map(|v| v as u32);
        if fields.len() != 22 {
            bail!("bad line: {line}");
        }
        let corner = |i: usize| f32::from_bits(fields[5 + i] as u32);
        let src_points = [[corner(0), corner(1)], [corner(2), corner(3)], [corner(4), corner(5)], [corner(6), corner(7)]];
        let reference: Vec<u64> = fields[13..22].to_vec();
        let (w, h) = ((dw - 1) as f32, (dh - 1) as f32);
        let dst_points = [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]];
        cases += 1;

        let Some(m) = perspective_transform(src_points, dst_points) else {
            let zero = reference[..8].iter().all(|&b| f64::from_bits(b) == 0.0) && f64::from_bits(reference[8]) == 1.0;
            if zero {
                degenerate += 1;
            } else {
                failed += 1;
                println!("case {case}: rejected as degenerate, OpenCV found a transform");
            }
            continue;
        };
        if m.map(f64::to_bits)[..] != reference[..] {
            failed += 1;
            println!("case {case}: transform differs");
            continue;
        }
        let src = read_bgr(&dir.join(format!("{case}.in")), sw, sh)?;
        let expected = read_bgr(&dir.join(format!("{case}.out")), dw, dh)?;
        let Some(actual) = warp_perspective(&src, &m, dw, dh) else {
            failed += 1;
            println!("case {case}: warp rejected a transform OpenCV inverts");
            continue;
        };
        let differing: Vec<usize> = (0..(dw * dh) as usize)
            .filter(|&p| actual.as_raw()[p * 3..p * 3 + 3] != expected.as_raw()[p * 3..p * 3 + 3])
            .collect();
        if let Some(&first) = differing.first() {
            failed += 1;
            let first_column = differing.iter().map(|p| p % dw as usize).min().unwrap_or_default();
            println!(
                "case {case} {sw}x{sh} -> {dw}x{dh}: {} pixels differ, first at ({}, {}), leftmost column {first_column}",
                differing.len(),
                first % dw as usize,
                first / dw as usize
            );
        }
    }
    println!("{cases} cases ({degenerate} degenerate, rejected as expected), {failed} with differences");
    if failed > 0 {
        std::process::exit(1);
    }
    Ok(())
}
