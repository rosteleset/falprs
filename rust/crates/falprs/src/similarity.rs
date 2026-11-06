//! Similarity of face descriptors with SIMD chosen at runtime: AVX2 + FMA through `pulp` where
//! the CPU has them, otherwise scalar code that the compiler vectorizes for the baseline SSE2.
//! Both functions expect lengths that are a multiple of 32 floats; the remainder is ignored.
//! Results differ from the C++ service in the last bits only (float summation order).

use pulp::{Arch, Simd, WithSimd};

/// Independent vector accumulators per sum; 8 spill registers in the full cosine (3 sums)
const ACC: usize = 4;

/// Cosine similarity of two descriptors in single precision, as `cosineSimilaritySIMD` of the
/// C++ service.
pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    Arch::new().dispatch(Cosine(a, b))
}

/// Dot product of two descriptors in single precision, as `cosineSimilarity` of the C++ FRS
/// workflow (the cosine similarity of normalized descriptors).
pub fn dot(a: &[f32], b: &[f32]) -> f64 {
    f64::from(Arch::new().dispatch(Dot(a, b)))
}

/// The norm of a descriptor exactly as `cosine` computes it: the square root, in double precision,
/// of the single-precision sum of squares.
pub fn norm(a: &[f32]) -> f64 {
    f64::from(Arch::new().dispatch(Dot(a, a))).sqrt()
}

/// `cosine(a, b)` with the norms of `a` and `b` precomputed by `norm`, bit-identical to it. When the
/// descriptors differ in whole 32-float blocks, `cosine` sums the squares over the shorter one only,
/// so it is called directly.
pub fn cosine_with_norms(a: &[f32], norm_a: f64, b: &[f32], norm_b: f64) -> f64 {
    if a.len() / 32 != b.len() / 32 {
        return cosine(a, b);
    }
    dot(a, b) / norm_a / norm_b
}

/// The first row with the largest `dot` with `query` among the rows whose product exceeds `above`,
/// as its index and product; all products are computed inside one SIMD dispatch.
pub fn max_dot<'a>(query: &[f32], rows: impl IntoIterator<Item = &'a [f32]>, above: f64) -> Option<(usize, f64)> {
    Arch::new().dispatch(MaxDot {
        query,
        rows: rows.into_iter(),
        above,
    })
}

/// Pairwise sum of the accumulators, then of the lanes
#[inline(always)]
fn reduce<S: Simd>(simd: S, mut acc: [S::f32s; ACC]) -> f32 {
    let mut n = ACC;
    while n > 1 {
        n /= 2;
        for k in 0..n {
            acc[k] = simd.add_f32s(acc[k], acc[k + n]);
        }
    }
    simd.reduce_sum_f32s(acc[0])
}

struct Cosine<'a>(&'a [f32], &'a [f32]);

impl WithSimd for Cosine<'_> {
    type Output = f64;

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> f64 {
        if S::F32_LANES == 1 {
            return scalar::cosine(self.0, self.1);
        }
        let (a, _) = S::as_simd_f32s(self.0);
        let (b, _) = S::as_simd_f32s(self.1);
        let zero = simd.splat_f32s(0.0);
        let (mut dot, mut sqr_a, mut sqr_b) = ([zero; ACC], [zero; ACC], [zero; ACC]);
        for (va, vb) in a.as_chunks::<ACC>().0.iter().zip(b.as_chunks::<ACC>().0) {
            for k in 0..ACC {
                dot[k] = simd.mul_add_e_f32s(va[k], vb[k], dot[k]);
                sqr_a[k] = simd.mul_add_e_f32s(va[k], va[k], sqr_a[k]);
                sqr_b[k] = simd.mul_add_e_f32s(vb[k], vb[k], sqr_b[k]);
            }
        }
        let (dot, sqr_a, sqr_b) = (reduce(simd, dot), reduce(simd, sqr_a), reduce(simd, sqr_b));
        f64::from(dot) / f64::from(sqr_a).sqrt() / f64::from(sqr_b).sqrt()
    }
}

struct Dot<'a>(&'a [f32], &'a [f32]);

impl WithSimd for Dot<'_> {
    type Output = f32;

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> f32 {
        if S::F32_LANES == 1 {
            return scalar::dot(self.0, self.1);
        }
        let (a, _) = S::as_simd_f32s(self.0);
        let (b, _) = S::as_simd_f32s(self.1);
        let mut acc = [simd.splat_f32s(0.0); ACC];
        for (va, vb) in a.as_chunks::<ACC>().0.iter().zip(b.as_chunks::<ACC>().0) {
            for k in 0..ACC {
                acc[k] = simd.mul_add_e_f32s(va[k], vb[k], acc[k]);
            }
        }
        reduce(simd, acc)
    }
}

struct MaxDot<'q, I> {
    query: &'q [f32],
    rows: I,
    above: f64,
}

impl<'a, I: Iterator<Item = &'a [f32]>> WithSimd for MaxDot<'_, I> {
    type Output = Option<(usize, f64)>;

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let (mut best, mut above) = (None, self.above);
        for (i, row) in self.rows.enumerate() {
            let product = f64::from(Dot(self.query, row).with_simd(simd));
            if product > above {
                above = product;
                best = Some((i, product));
            }
        }
        best
    }
}

/// Fallback without AVX2 + FMA: four 8-float accumulators per sum over 32-float blocks, written
/// for the compiler to vectorize.
mod scalar {
    pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
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
        f64::from(combine(&dot)) / f64::from(combine(&sqr_a)).sqrt() / f64::from(combine(&sqr_b)).sqrt()
    }

    pub fn dot(a: &[f32], b: &[f32]) -> f32 {
        let mut sums = [[0f32; 8]; 4];
        for (block_a, block_b) in a.as_chunks::<32>().0.iter().zip(b.as_chunks::<32>().0) {
            for (k, sum) in sums.iter_mut().enumerate() {
                for (lane, s) in sum.iter_mut().enumerate() {
                    *s += block_a[k * 8 + lane] * block_b[k * 8 + lane];
                }
            }
        }
        combine(&sums)
    }

    fn combine(sums: &[[f32; 8]; 4]) -> f32 {
        let mut lanes = [0f32; 8];
        for (lane, l) in lanes.iter_mut().enumerate() {
            *l = (sums[0][lane] + sums[1][lane]) + (sums[2][lane] + sums[3][lane]);
        }
        ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor(seed: usize) -> Vec<f32> {
        (0..512).map(|i| ((i * 7 + seed * 13) % 23) as f32 - 11.0).collect()
    }

    fn exact(a: &[f32], b: &[f32]) -> f64 {
        a.iter().zip(b).map(|(&x, &y)| f64::from(x) * f64::from(y)).sum()
    }

    #[test]
    fn cosine_matches_exact() {
        let (a, b) = (descriptor(1), descriptor(2));
        let expected = exact(&a, &b) / exact(&a, &a).sqrt() / exact(&b, &b).sqrt();
        assert!((cosine(&a, &b) - expected).abs() < 1e-6);
        assert!((scalar::cosine(&a, &b) - expected).abs() < 1e-6);
        assert!((cosine(&a, &a) - 1.0).abs() < 1e-6);
        let negated: Vec<f32> = a.iter().map(|v| -v).collect();
        assert!((cosine(&a, &negated) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn precomputed_norms_are_bit_identical() {
        let descriptors: Vec<Vec<f32>> = (0..6)
            .map(|seed| (0..512).map(|i| (((i * 31 + seed * 17) % 101) as f32 - 50.0) * 0.0137).collect())
            .collect();
        for a in &descriptors {
            for b in &descriptors {
                assert_eq!(cosine_with_norms(a, norm(a), b, norm(b)).to_bits(), cosine(a, b).to_bits());
                // The fallback without AVX2 + FMA accumulates the squares the same way
                let scalar_norm = |v: &[f32]| f64::from(scalar::dot(v, v)).sqrt();
                let scalar_split = f64::from(scalar::dot(a, b)) / scalar_norm(a) / scalar_norm(b);
                assert_eq!(scalar_split.to_bits(), scalar::cosine(a, b).to_bits());
            }
        }
        let short = &descriptors[1][..256];
        assert_eq!(
            cosine_with_norms(&descriptors[0], norm(&descriptors[0]), short, norm(short)).to_bits(),
            cosine(&descriptors[0], short).to_bits()
        );
    }

    #[test]
    fn max_dot_finds_first_largest_above() {
        let rows: Vec<Vec<f32>> = [1.0f32, 3.0, 2.0, 3.0].iter().map(|&v| vec![v; 32]).collect();
        let query = vec![1.0f32; 32];
        let slices = || rows.iter().map(Vec::as_slice);
        assert_eq!(max_dot(&query, slices(), -2.0), Some((1, 96.0)));
        assert_eq!(max_dot(&query, slices(), 96.0), None);
        assert_eq!(max_dot(&query, slices(), 50.0), Some((1, 96.0)));
        for (i, row) in rows.iter().enumerate() {
            assert_eq!(max_dot(&query, [row.as_slice()], -2.0), Some((0, dot(&query, row))), "row {i}");
        }
    }

    #[test]
    fn dot_matches_exact() {
        let (a, b) = (descriptor(3), descriptor(4));
        let expected = exact(&a, &b);
        assert!((dot(&a, &b) - expected).abs() < 1e-6 * expected.abs().max(1.0));
        assert!((f64::from(scalar::dot(&a, &b)) - expected).abs() < 1e-6 * expected.abs().max(1.0));
    }
}
