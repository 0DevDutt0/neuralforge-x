//! Low-level numeric kernels with runtime CPU-feature dispatch.
//!
//! Each primitive has a portable scalar implementation (the correctness
//! reference) and, on `x86_64`, a hand-vectorised AVX2 + FMA implementation
//! selected at runtime via [`std::is_x86_feature_detected`]. The vectorised
//! paths carry four independent 8-wide accumulators to hide the latency of the
//! fused-multiply-add, then perform a single horizontal reduction.
//!
//! AVX-512 is intentionally not targeted: it is fused-off on Intel Core Ultra
//! (Meteor/Arrow Lake) client parts, so AVX2 + FMA (8 × `f32` per lane group)
//! is the widest width that is actually profitable on the target hardware.
//!
//! # Safety
//! The `*_avx2_fma` functions are `unsafe` and annotated with
//! `#[target_feature(enable = "avx2,fma")]`. They are only ever reached through
//! the safe [`dot`] / [`l2_sq`] dispatchers, which call them solely after
//! `is_x86_feature_detected!` confirms the features are present at runtime.

/// Inner product `Σ aᵢ·bᵢ`.
///
/// `a` and `b` must have equal length (checked with `debug_assert!`; callers in
/// this crate validate lengths before calling).
#[inline]
#[must_use]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: the AVX2 and FMA target features were just detected.
            return unsafe { dot_avx2_fma(a, b) };
        }
    }
    dot_scalar(a, b)
}

/// Squared Euclidean distance `Σ (aᵢ − bᵢ)²`.
#[inline]
#[must_use]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: the AVX2 and FMA target features were just detected.
            return unsafe { l2_sq_avx2_fma(a, b) };
        }
    }
    l2_sq_scalar(a, b)
}

/// Squared L2 norm `Σ aᵢ²`.
#[inline]
#[must_use]
pub fn norm_sq(a: &[f32]) -> f32 {
    dot(a, a)
}

// --------------------------------------------------------------------------
// Portable scalar reference implementations.
// --------------------------------------------------------------------------

/// Scalar inner product. Reference implementation used as the correctness
/// oracle for the vectorised path and as the fallback on non-x86 targets.
#[inline]
#[must_use]
pub fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Scalar squared Euclidean distance. See [`dot_scalar`].
#[inline]
#[must_use]
pub fn l2_sq_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

// --------------------------------------------------------------------------
// AVX2 + FMA implementations (x86_64).
// --------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2_fma(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::x86_64::*;
    unsafe {
        let n = a.len();
        let pa = a.as_ptr();
        let pb = b.as_ptr();
        let mut i = 0usize;

        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();

        // Main body: 32 lanes (4 × 8) per iteration for instruction-level parallelism.
        while i + 32 <= n {
            let a0 = _mm256_loadu_ps(pa.add(i));
            let a1 = _mm256_loadu_ps(pa.add(i + 8));
            let a2 = _mm256_loadu_ps(pa.add(i + 16));
            let a3 = _mm256_loadu_ps(pa.add(i + 24));
            let b0 = _mm256_loadu_ps(pb.add(i));
            let b1 = _mm256_loadu_ps(pb.add(i + 8));
            let b2 = _mm256_loadu_ps(pb.add(i + 16));
            let b3 = _mm256_loadu_ps(pb.add(i + 24));
            acc0 = _mm256_fmadd_ps(a0, b0, acc0);
            acc1 = _mm256_fmadd_ps(a1, b1, acc1);
            acc2 = _mm256_fmadd_ps(a2, b2, acc2);
            acc3 = _mm256_fmadd_ps(a3, b3, acc3);
            i += 32;
        }

        let mut acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));

        // Tail in 8-lane chunks.
        while i + 8 <= n {
            let av = _mm256_loadu_ps(pa.add(i));
            let bv = _mm256_loadu_ps(pb.add(i));
            acc = _mm256_fmadd_ps(av, bv, acc);
            i += 8;
        }

        let mut sum = hsum256(acc);

        // Scalar remainder (< 8 elements).
        while i < n {
            sum += *pa.add(i) * *pb.add(i);
            i += 1;
        }
        sum
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn l2_sq_avx2_fma(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::x86_64::*;
    unsafe {
        let n = a.len();
        let pa = a.as_ptr();
        let pb = b.as_ptr();
        let mut i = 0usize;

        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();

        while i + 32 <= n {
            let d0 = _mm256_sub_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)));
            let d1 = _mm256_sub_ps(
                _mm256_loadu_ps(pa.add(i + 8)),
                _mm256_loadu_ps(pb.add(i + 8)),
            );
            let d2 = _mm256_sub_ps(
                _mm256_loadu_ps(pa.add(i + 16)),
                _mm256_loadu_ps(pb.add(i + 16)),
            );
            let d3 = _mm256_sub_ps(
                _mm256_loadu_ps(pa.add(i + 24)),
                _mm256_loadu_ps(pb.add(i + 24)),
            );
            acc0 = _mm256_fmadd_ps(d0, d0, acc0);
            acc1 = _mm256_fmadd_ps(d1, d1, acc1);
            acc2 = _mm256_fmadd_ps(d2, d2, acc2);
            acc3 = _mm256_fmadd_ps(d3, d3, acc3);
            i += 32;
        }

        let mut acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));

        while i + 8 <= n {
            let d = _mm256_sub_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)));
            acc = _mm256_fmadd_ps(d, d, acc);
            i += 8;
        }

        let mut sum = hsum256(acc);

        while i < n {
            let d = *pa.add(i) - *pb.add(i);
            sum += d * d;
            i += 1;
        }
        sum
    }
}

/// Horizontal sum of the eight `f32` lanes of an AVX register.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum256(v: core::arch::x86_64::__m256) -> f32 {
    use core::arch::x86_64::*;
    // All operations below are register-only intrinsics, made safe to call by
    // the enabled `avx2` target feature, so no `unsafe` block is required (the
    // pointer-loading kernels above genuinely need theirs).
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps::<1>(v);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128); // [a1, a1, a3, a3]
    let sums = _mm_add_ps(sum128, shuf); // [a0+a1, _, a2+a3, _]
    let shuf2 = _mm_movehl_ps(shuf, sums); // bring a2+a3 down
    let result = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// Relative tolerance for SIMD-vs-scalar comparisons. FMA reassociation can
    /// shift the last few ULPs, so we compare with a magnitude-scaled epsilon.
    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-3 * (1.0 + a.abs().max(b.abs()))
    }

    #[test]
    fn dot_known_values() {
        assert_eq!(dot_scalar(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0);
        assert!(close(dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0));
    }

    #[test]
    fn l2_known_values() {
        assert_eq!(l2_sq_scalar(&[0.0, 0.0], &[3.0, 4.0]), 25.0);
        assert!(close(l2_sq(&[0.0, 0.0], &[3.0, 4.0]), 25.0));
    }

    #[test]
    fn handles_all_lengths_across_simd_boundaries() {
        // Exercise the 32-lane body, the 8-lane tail and the scalar remainder.
        for len in [1usize, 7, 8, 9, 31, 32, 33, 64, 100, 257] {
            let a: Vec<f32> = (0..len).map(|i| (i as f32) * 0.5 - 3.0).collect();
            let b: Vec<f32> = (0..len).map(|i| (i as f32).sin()).collect();
            assert!(
                close(dot(&a, &b), dot_scalar(&a, &b)),
                "dot mismatch at len {len}"
            );
            assert!(
                close(l2_sq(&a, &b), l2_sq_scalar(&a, &b)),
                "l2 mismatch at len {len}"
            );
        }
    }

    proptest! {
        #[test]
        fn simd_dot_matches_scalar(a in prop::collection::vec(-10.0f32..10.0, 1..600)) {
            let b: Vec<f32> = a.iter().rev().copied().collect();
            prop_assert!(close(dot(&a, &b), dot_scalar(&a, &b)));
        }

        #[test]
        fn simd_l2_matches_scalar(a in prop::collection::vec(-10.0f32..10.0, 1..600)) {
            let b: Vec<f32> = a.iter().map(|x| x * 0.5 + 1.0).collect();
            prop_assert!(close(l2_sq(&a, &b), l2_sq_scalar(&a, &b)));
        }
    }
}
