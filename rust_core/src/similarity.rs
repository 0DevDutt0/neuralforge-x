//! Pairwise and batched similarity kernels.

use rayon::prelude::*;

use crate::error::{CoreError, Result};
use crate::matrix::MatrixView;
use crate::metric::Metric;
use crate::simd;

#[inline]
fn check_pair(a: &[f32], b: &[f32]) -> Result<()> {
    if a.is_empty() || b.is_empty() {
        return Err(CoreError::EmptyInput { what: "vector" });
    }
    if a.len() != b.len() {
        return Err(CoreError::DimensionMismatch {
            left: a.len(),
            right: b.len(),
        });
    }
    Ok(())
}

/// Inner product `⟨a, b⟩`.
///
/// # Errors
/// [`CoreError::EmptyInput`] if either vector is empty, or
/// [`CoreError::DimensionMismatch`] if the lengths differ.
pub fn dot_product(a: &[f32], b: &[f32]) -> Result<f32> {
    check_pair(a, b)?;
    Ok(simd::dot(a, b))
}

/// Euclidean (L2) distance `‖a − b‖₂`.
///
/// # Errors
/// As [`dot_product`].
pub fn l2_distance(a: &[f32], b: &[f32]) -> Result<f32> {
    check_pair(a, b)?;
    Ok(simd::l2_sq(a, b).sqrt())
}

/// Cosine similarity `⟨a, b⟩ / (‖a‖·‖b‖)`.
///
/// If either vector has zero norm the similarity is defined as `0.0` (matching
/// the convention used by scikit-learn) rather than producing `NaN`.
///
/// # Errors
/// As [`dot_product`].
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32> {
    check_pair(a, b)?;
    let dot = simd::dot(a, b);
    let denom = simd::norm_sq(a).sqrt() * simd::norm_sq(b).sqrt();
    Ok(if denom == 0.0 { 0.0 } else { dot / denom })
}

/// Computes the full `queries.rows × corpus.rows` similarity matrix in row-major
/// order, parallelised across queries with `rayon`.
///
/// For [`Metric::Cosine`], the per-vector L2 norms of the corpus are precomputed
/// once and reused for every query, turning an `O(q·n·d)` normalisation cost into
/// an `O(n·d)` one.
///
/// # Errors
/// [`CoreError::DimensionMismatch`] if the query and corpus dimensionalities differ.
pub fn batch_similarity(
    queries: MatrixView<'_>,
    corpus: MatrixView<'_>,
    metric: Metric,
) -> Result<Vec<f32>> {
    if queries.cols() != corpus.cols() {
        return Err(CoreError::DimensionMismatch {
            left: queries.cols(),
            right: corpus.cols(),
        });
    }

    let n_corpus = corpus.rows();
    let mut out = vec![0.0f32; queries.rows() * n_corpus];

    let corpus_norms: Option<Vec<f32>> = match metric {
        Metric::Cosine => Some(precompute_norms(corpus)),
        _ => None,
    };

    // Each query owns a disjoint, contiguous output row → safe parallel writes.
    out.par_chunks_mut(n_corpus)
        .enumerate()
        .for_each(|(qi, row_out)| {
            let q = queries.row(qi);
            let q_norm = match metric {
                Metric::Cosine => simd::norm_sq(q).sqrt(),
                _ => 0.0,
            };
            for (ci, slot) in row_out.iter_mut().enumerate() {
                let c = corpus.row(ci);
                *slot = match metric {
                    Metric::DotProduct => simd::dot(q, c),
                    Metric::L2 => simd::l2_sq(q, c).sqrt(),
                    Metric::Cosine => {
                        // `unwrap` is sound: `corpus_norms` is `Some` for Cosine.
                        let denom = q_norm * corpus_norms.as_ref().unwrap()[ci];
                        if denom == 0.0 {
                            0.0
                        } else {
                            simd::dot(q, c) / denom
                        }
                    }
                };
            }
        });

    Ok(out)
}

/// Computes the L2 norm of every row of `m` in parallel.
fn precompute_norms(m: MatrixView<'_>) -> Vec<f32> {
    (0..m.rows())
        .into_par_iter()
        .map(|i| simd::norm_sq(m.row(i)).sqrt())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    #[test]
    fn cosine_identity_orthogonal_opposite() {
        let a = [1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &a).unwrap() - 1.0).abs() < EPS);
        assert!(cosine_similarity(&a, &[0.0, 1.0, 0.0]).unwrap().abs() < EPS);
        assert!((cosine_similarity(&a, &[-1.0, 0.0, 0.0]).unwrap() + 1.0).abs() < EPS);
    }

    #[test]
    fn cosine_zero_norm_is_zero_not_nan() {
        let s = cosine_similarity(&[0.0, 0.0], &[1.0, 2.0]).unwrap();
        assert_eq!(s, 0.0);
    }

    #[test]
    fn l2_and_dot_known_values() {
        assert!((l2_distance(&[0.0, 0.0], &[3.0, 4.0]).unwrap() - 5.0).abs() < EPS);
        assert!((dot_product(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]).unwrap() - 32.0).abs() < EPS);
    }

    #[test]
    fn dimension_mismatch_is_reported() {
        assert_eq!(
            dot_product(&[1.0, 2.0], &[1.0]),
            Err(CoreError::DimensionMismatch { left: 2, right: 1 })
        );
    }

    #[test]
    fn batch_matches_pairwise() {
        let queries = [1.0, 0.0, 0.0, 1.0]; // 2×2
        let corpus = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0]; // 3×2
        let q = MatrixView::new(&queries, 2, 2).unwrap();
        let c = MatrixView::new(&corpus, 3, 2).unwrap();
        let m = batch_similarity(q, c, Metric::Cosine).unwrap();
        assert_eq!(m.len(), 6);
        // query0 = [1,0] vs corpus0 = [1,0] → cosine 1.0
        assert!((m[0] - 1.0).abs() < EPS);
        // query0 vs corpus1 = [0,1] → 0.0
        assert!(m[1].abs() < EPS);
        // query0 vs corpus2 = [1,1] → 1/√2
        assert!((m[2] - std::f32::consts::FRAC_1_SQRT_2).abs() < EPS);
    }
}
