//! Cosine similarity probe: the two FRS descriptor similarity functions, scalar (autovectorized)
//! versus SIMD through `pulp` runtime dispatch, compared with the C++ reference.
//!   probe-cosine-similarity generate <pairs.bin>
//!   probe-cosine-similarity <pairs.bin> [<reference_values.bin>]
//! `pairs.bin` holds interleaved normalised descriptor pairs (a0 b0 a1 b1 ...) as raw f32;
//! `reference_values.bin` holds the C++ values as f64: all full cosines, then all dot products.

use std::hint::black_box;
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use pulp::{Arch, Simd, WithSimd};

const DIM: usize = 512;
const PAIRS: usize = 10_000;
/// Calls per timing run, the same for the memory-bound and the cache-resident data set
const CALLS: usize = 1_000_000;
/// Pairs of the cache-resident data set (256 KiB, fits L2)
const HOT_PAIRS: usize = 64;

// --- Before: the scalar code falprs had before `falprs::similarity`, autovectorized to SSE2 by the compiler ---

fn reduce_sum8(lanes: &[f32; 8]) -> f32 {
    ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]))
}

/// `events::cosine_similarity` before
fn cosine_before(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = [[0f32; 8]; 4];
    let mut sqr_a = [[0f32; 8]; 4];
    let mut sqr_b = [[0f32; 8]; 4];
    for (block_a, block_b) in a.as_chunks::<32>().0.iter().zip(b.as_chunks::<32>().0) {
        for k in 0..4 {
            for lane in 0..8 {
                let (x, y) = (block_a[k * 8 + lane], block_b[k * 8 + lane]);
                dot[k][lane] += x * y;
                sqr_a[k][lane] += x * x;
                sqr_b[k][lane] += y * y;
            }
        }
    }
    let combine = |sums: &[[f32; 8]; 4]| {
        let mut lanes = [0f32; 8];
        for lane in 0..8 {
            lanes[lane] = (sums[0][lane] + sums[1][lane]) + (sums[2][lane] + sums[3][lane]);
        }
        reduce_sum8(&lanes)
    };
    f64::from(combine(&dot)) / f64::from(combine(&sqr_a)).sqrt() / f64::from(combine(&sqr_b)).sqrt()
}

/// `pipeline::cosine_similarity` before (without the length check and the tail)
fn dot_before(a: &[f32], b: &[f32]) -> f64 {
    let mut sums = [[0f32; 8]; 4];
    for (block_a, block_b) in a.as_chunks::<32>().0.iter().zip(b.as_chunks::<32>().0) {
        for (k, sum) in sums.iter_mut().enumerate() {
            for (lane, s) in sum.iter_mut().enumerate() {
                *s += block_a[k * 8 + lane] * block_b[k * 8 + lane];
            }
        }
    }
    let lanes: f32 = (0..8).map(|lane| sums[0][lane] + sums[1][lane] + sums[2][lane] + sums[3][lane]).sum();
    f64::from(lanes)
}

// --- After: pulp kernels with ACC independent vector accumulators per sum ---

/// Pairwise sum of the accumulators, then of the lanes
#[inline(always)]
fn reduce<S: Simd, const ACC: usize>(simd: S, mut acc: [S::f32s; ACC]) -> f32 {
    let mut n = ACC;
    while n > 1 {
        n /= 2;
        for k in 0..n {
            acc[k] = simd.add_f32s(acc[k], acc[k + n]);
        }
    }
    simd.reduce_sum_f32s(acc[0])
}

struct Dot<'a, const ACC: usize>(&'a [f32], &'a [f32]);

impl<const ACC: usize> WithSimd for Dot<'_, ACC> {
    type Output = f32;

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> f32 {
        if S::F32_LANES == 1 {
            // No AVX2 + FMA: the autovectorized scalar code is SSE2 already
            return dot_before(self.0, self.1) as f32;
        }
        let (a, _) = S::as_simd_f32s(self.0);
        let (b, _) = S::as_simd_f32s(self.1);
        let mut acc = [simd.splat_f32s(0.0); ACC];
        let ((blocks_a, rest_a), (blocks_b, rest_b)) = (a.as_chunks::<ACC>(), b.as_chunks::<ACC>());
        for (va, vb) in blocks_a.iter().zip(blocks_b) {
            for k in 0..ACC {
                acc[k] = simd.mul_add_e_f32s(va[k], vb[k], acc[k]);
            }
        }
        for (k, (&x, &y)) in rest_a.iter().zip(rest_b).enumerate() {
            acc[k] = simd.mul_add_e_f32s(x, y, acc[k]);
        }
        reduce(simd, acc)
    }
}

struct Cosine<'a, const ACC: usize>(&'a [f32], &'a [f32]);

impl<const ACC: usize> WithSimd for Cosine<'_, ACC> {
    type Output = f64;

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> f64 {
        if S::F32_LANES == 1 {
            return cosine_before(self.0, self.1);
        }
        let (a, _) = S::as_simd_f32s(self.0);
        let (b, _) = S::as_simd_f32s(self.1);
        let zero = simd.splat_f32s(0.0);
        let (mut dot, mut sqr_a, mut sqr_b) = ([zero; ACC], [zero; ACC], [zero; ACC]);
        let ((blocks_a, rest_a), (blocks_b, rest_b)) = (a.as_chunks::<ACC>(), b.as_chunks::<ACC>());
        for (va, vb) in blocks_a.iter().zip(blocks_b) {
            for k in 0..ACC {
                dot[k] = simd.mul_add_e_f32s(va[k], vb[k], dot[k]);
                sqr_a[k] = simd.mul_add_e_f32s(va[k], va[k], sqr_a[k]);
                sqr_b[k] = simd.mul_add_e_f32s(vb[k], vb[k], sqr_b[k]);
            }
        }
        for (k, (&x, &y)) in rest_a.iter().zip(rest_b).enumerate() {
            dot[k] = simd.mul_add_e_f32s(x, y, dot[k]);
            sqr_a[k] = simd.mul_add_e_f32s(x, x, sqr_a[k]);
            sqr_b[k] = simd.mul_add_e_f32s(y, y, sqr_b[k]);
        }
        let (dot, sqr_a, sqr_b) = (reduce(simd, dot), reduce(simd, sqr_a), reduce(simd, sqr_b));
        f64::from(dot) / f64::from(sqr_a).sqrt() / f64::from(sqr_b).sqrt()
    }
}

fn dot_pulp<const ACC: usize>(a: &[f32], b: &[f32]) -> f64 {
    f64::from(Arch::new().dispatch(Dot::<ACC>(a, b)))
}

fn cosine_pulp<const ACC: usize>(a: &[f32], b: &[f32]) -> f64 {
    Arch::new().dispatch(Cosine::<ACC>(a, b))
}

// --- Exact values in double precision, for the accuracy columns ---

fn dot_exact(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| f64::from(x) * f64::from(y)).sum()
}

fn cosine_exact(a: &[f32], b: &[f32]) -> f64 {
    dot_exact(a, b) / dot_exact(a, a).sqrt() / dot_exact(b, b).sqrt()
}

// --- Harness ---

type Similarity = fn(&[f32], &[f32]) -> f64;
/// Title, exact reference, offset of the C++ values and the measured variants
type Group = (&'static str, Similarity, usize, Vec<(&'static str, Similarity)>);

fn read_f32(path: &str) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path).with_context(|| format!("can't read {path}"))?;
    Ok(bytes.as_chunks::<4>().0.iter().map(|&b| f32::from_le_bytes(b)).collect())
}

fn read_f64(path: &str) -> Result<Vec<f64>> {
    let bytes = std::fs::read(path).with_context(|| format!("can't read {path}"))?;
    Ok(bytes.as_chunks::<8>().0.iter().map(|&b| f64::from_le_bytes(b)).collect())
}

/// Normalised pseudo-random descriptors (xorshift64*), the same on every run
fn generate(path: &str) -> Result<()> {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        (state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    };
    let mut bytes = Vec::with_capacity(2 * PAIRS * DIM * 4);
    for _ in 0..2 * PAIRS {
        let v: Vec<f32> = (0..DIM).map(|_| next()).collect();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in v {
            bytes.extend_from_slice(&(x / norm).to_le_bytes());
        }
    }
    std::fs::write(path, bytes).with_context(|| format!("can't write {path}"))
}

/// Best of 3 runs of `CALLS` calls over the first `pairs` pairs, in nanoseconds per call
fn time(f: Similarity, data: &[f32], pairs: usize) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let mut sink = 0.0;
        let start = Instant::now();
        let mut done = 0;
        while done < CALLS {
            for pair in data.as_chunks::<{ 2 * DIM }>().0.iter().take(pairs.min(CALLS - done)) {
                sink += f(black_box(&pair[..DIM]), black_box(&pair[DIM..]));
                done += 1;
            }
        }
        best = best.min(start.elapsed().as_secs_f64() * 1e9 / CALLS as f64);
        black_box(sink);
    }
    best
}

fn values(f: Similarity, data: &[f32]) -> Vec<f64> {
    data.as_chunks::<{ 2 * DIM }>().0.iter().map(|pair| f(&pair[..DIM], &pair[DIM..])).collect()
}

fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let (data_path, reference) = match args.as_slice() {
        [_, cmd, path] if cmd == "generate" => return generate(path),
        [_, path] => (path, None),
        [_, path, reference] => (path, Some(read_f64(reference)?)),
        _ => bail!("usage: probe-cosine-similarity generate <pairs.bin> | <pairs.bin> [<reference_values.bin>]"),
    };
    let data = read_f32(data_path)?;
    ensure!(data.len() == 2 * PAIRS * DIM, "expected {PAIRS} pairs of {DIM} floats");
    if let Some(reference) = &reference {
        ensure!(reference.len() == 2 * PAIRS, "expected {} reference values", 2 * PAIRS);
    }

    println!("dispatch: {:?}", Arch::new());
    let groups: [Group; 2] = [
        (
            "events::cosine_similarity (full cosine)",
            cosine_exact,
            0,
            vec![
                ("before", cosine_before),
                ("pulp, 4 acc", cosine_pulp::<4>),
                ("pulp, 8 acc", cosine_pulp::<8>),
                ("falprs", falprs::similarity::cosine),
            ],
        ),
        (
            "pipeline::cosine_similarity (dot product)",
            dot_exact,
            PAIRS,
            vec![
                ("before", dot_before),
                ("pulp, 4 acc", dot_pulp::<4>),
                ("pulp, 8 acc", dot_pulp::<8>),
                ("falprs", falprs::similarity::dot),
            ],
        ),
    ];
    for (title, exact, offset, variants) in groups {
        let exact = values(exact, &data);
        println!("\n{title}");
        println!(
            "{:<14} {:>12} {:>12} {:>14} {:>14}",
            "variant", "ns (memory)", "ns (cache)", "max |Δ| exact", "max |Δ| C++"
        );
        for (name, f) in variants {
            let v = values(f, &data);
            let cpp = reference
                .as_ref()
                .map_or(String::from("-"), |r| format!("{:.3e}", max_diff(&v, &r[offset..offset + PAIRS])));
            println!(
                "{name:<14} {:>12.2} {:>12.2} {:>14.3e} {cpp:>14}",
                time(f, &data, PAIRS),
                time(f, &data, HOT_PAIRS),
                max_diff(&v, &exact)
            );
        }
        if let Some(r) = &reference {
            println!("{:<14} {:>12} {:>12} {:>14.3e}", "C++", "", "", max_diff(&r[offset..offset + PAIRS], &exact));
        }
    }
    Ok(())
}
