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
    fn dot_matches_exact() {
        let (a, b) = (descriptor(3), descriptor(4));
        let expected = exact(&a, &b);
        assert!((dot(&a, &b) - expected).abs() < 1e-6 * expected.abs().max(1.0));
        assert!((f64::from(scalar::dot(&a, &b)) - expected).abs() < 1e-6 * expected.abs().max(1.0));
    }
}
