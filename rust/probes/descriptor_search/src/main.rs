//! Descriptor search probe: the FRS linear search for the most similar pre-normalised face
//! descriptor, on stable Rust without `unsafe` or SIMD intrinsics.
//!   probe-descriptor-search <descriptors.bin> <queries.bin> <dimension>
//! Prints the best of 3 timings and, per query, the index and similarity of the best match.

use std::time::Instant;

use anyhow::{Context, Result};

/// Lanes of independent partial sums; enough for the compiler to keep several vector
/// registers busy, like the four accumulators of the C++ version.
const LANES: usize = 16;

/// Dot product of two equally long slices whose length is a multiple of `LANES`.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut sums = [0.0f32; LANES];
    for (x, y) in a.chunks_exact(LANES).zip(b.chunks_exact(LANES)) {
        for lane in 0..LANES {
            sums[lane] += x[lane] * y[lane];
        }
    }
    sums.iter().sum()
}

fn read(path: &str) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path).with_context(|| format!("can't read {path}"))?;
    Ok(bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let [_, db_path, queries_path, dim] = args.as_slice() else {
        anyhow::bail!("usage: probe-descriptor-search <descriptors.bin> <queries.bin> <dimension>");
    };
    let (db, queries, dim): (Vec<f32>, Vec<f32>, usize) = (read(db_path)?, read(queries_path)?, dim.parse()?);
    anyhow::ensure!(dim % LANES == 0, "dimension must be a multiple of {LANES}");

    let mut best = Vec::new();
    let mut best_ms = f64::INFINITY;
    for _ in 0..3 {
        let start = Instant::now();
        best = queries
            .chunks_exact(dim)
            .map(|query| {
                let mut max = (0, -1.0f32);
                for (i, descriptor) in db.chunks_exact(dim).enumerate() {
                    let value = dot(query, descriptor);
                    if value > max.1 {
                        max = (i, value);
                    }
                }
                max
            })
            .collect();
        best_ms = best_ms.min(start.elapsed().as_secs_f64() * 1000.0);
    }
    println!("ms {best_ms}");
    for (index, value) in best {
        println!("{index} {value}");
    }
    Ok(())
}
