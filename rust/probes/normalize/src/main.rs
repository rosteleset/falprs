//! Descriptor normalisation probe: compares `similarity::normalize` with `fd / cv::norm(fd)` of
//! OpenCV bit for bit.
//!   probe-normalize <dir written by reference/normalize_ref.cpp>
//! Prints how many descriptors and values differ, and the first differing values.

use std::path::Path;

use anyhow::{Context, Result, ensure};
use falprs::similarity::normalize;

const DIM: usize = 512;

fn read_f32(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    ensure!(bytes.len() % (DIM * 4) == 0, "{} is not a whole number of descriptors", path.display());
    Ok(bytes.as_chunks::<4>().0.iter().map(|b| f32::from_ne_bytes(*b)).collect())
}

fn main() -> Result<()> {
    let dir = std::env::args().nth(1).context("usage: probe-normalize <dir>")?;
    let dir = Path::new(&dir);
    let input = read_f32(&dir.join("input.bin"))?;
    let expected = read_f32(&dir.join("output.bin"))?;
    ensure!(input.len() == expected.len(), "input and output sizes differ");

    let (mut descriptors, mut differing_descriptors, mut differing_values, mut shown) = (0, 0, 0, 0);
    for (fd, reference) in input.as_chunks::<DIM>().0.iter().zip(expected.as_chunks::<DIM>().0) {
        let mut normalized = fd.to_vec();
        normalize(&mut normalized);
        descriptors += 1;
        let mut differs = false;
        for (i, (rust, opencv)) in normalized.iter().zip(reference).enumerate() {
            if rust.to_bits() != opencv.to_bits() {
                differs = true;
                differing_values += 1;
                if shown < 10 {
                    shown += 1;
                    println!("descriptor {} value {i}: rust {rust:e} opencv {opencv:e}", descriptors - 1);
                }
            }
        }
        differing_descriptors += usize::from(differs);
    }
    println!(
        "{descriptors} descriptors: {differing_descriptors} differ, {differing_values} of {} values differ",
        descriptors * DIM
    );
    ensure!(differing_values == 0, "normalisation differs from OpenCV");
    Ok(())
}
