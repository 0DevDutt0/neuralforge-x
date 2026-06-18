//! Parallel top-k nearest-neighbour search.

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use rayon::prelude::*;

use crate::error::{CoreError, Result};
use crate::matrix::MatrixView;
use crate::metric::Metric;
use crate::simd;

/// A single search result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Neighbor {
    /// Row index of the matched corpus vector.
    pub index: usize,
    /// Metric value for the match (cosine/dot similarity, or L2 distance).
    pub score: f32,
}

/// Internal ranking element. The kernel always ranks on a "higher key is
/// better" basis so that a single fixed-size min-heap can evict the worst
/// current candidate; for L2 the key is the *negated* squared distance.
#[derive(Debug, Clone, Copy)]
struct Ranked {
    key: f32,
    index: usize,
}

impl PartialEq for Ranked {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Ranked {}
impl PartialOrd for Ranked {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Ranked {
    /// Total order over `(key, index)`. `total_cmp` gives a well-defined order
    /// even in the presence of `NaN`; ties on key break on index for
    /// deterministic, reproducible results.
    fn cmp(&self, other: &Self) -> Ordering {
        self.key
            .total_cmp(&other.key)
            .then(self.index.cmp(&other.index))
    }
}

/// Pushes `r` into a bounded min-heap that retains the `k` largest keys.
///
/// The heap stores `Reverse<Ranked>`, so its top is the *smallest* retained
/// element. When the heap is full we replace that smallest element only if the
/// candidate is strictly better, keeping the operation `O(log k)`.
#[inline]
fn push_bounded(heap: &mut BinaryHeap<Reverse<Ranked>>, r: Ranked, k: usize) {
    if heap.len() < k {
        heap.push(Reverse(r));
    } else if let Some(&Reverse(worst)) = heap.peek() {
        if r.cmp(&worst) == Ordering::Greater {
            heap.pop();
            heap.push(Reverse(r));
        }
    }
}

/// Returns the `k` vectors of `corpus` most similar to `query` under `metric`,
/// best match first.
///
/// # Algorithm
/// The corpus is scanned in parallel with `rayon`. Each worker maintains a
/// size-`k` bounded min-heap of its best candidates; the per-worker heaps are
/// then merged. Work is `O(n·d + n·log k)` and peak auxiliary memory is
/// `O(k · threads)` — never `O(n)` — which matters for million-scale corpora.
///
/// Result scores are reported in the metric's natural units: cosine/dot
/// similarity (higher is better) or L2 distance (lower is better).
///
/// # Errors
/// [`CoreError::EmptyInput`] for an empty query, [`CoreError::DimensionMismatch`]
/// if the query width differs from the corpus width, or [`CoreError::InvalidK`]
/// if `k == 0` or `k > corpus.rows()`.
pub fn top_k_search(
    query: &[f32],
    corpus: MatrixView<'_>,
    k: usize,
    metric: Metric,
) -> Result<Vec<Neighbor>> {
    if query.is_empty() {
        return Err(CoreError::EmptyInput { what: "query" });
    }
    if query.len() != corpus.cols() {
        return Err(CoreError::DimensionMismatch {
            left: query.len(),
            right: corpus.cols(),
        });
    }
    let n = corpus.rows();
    if k == 0 || k > n {
        return Err(CoreError::InvalidK { k, n });
    }

    let q_norm = match metric {
        Metric::Cosine => simd::norm_sq(query).sqrt(),
        _ => 1.0,
    };

    let heap = (0..n)
        .into_par_iter()
        .fold(
            || BinaryHeap::<Reverse<Ranked>>::with_capacity(k + 1),
            |mut heap, ci| {
                let c = corpus.row(ci);
                let key = match metric {
                    Metric::DotProduct => simd::dot(query, c),
                    Metric::L2 => -simd::l2_sq(query, c),
                    Metric::Cosine => {
                        let denom = q_norm * simd::norm_sq(c).sqrt();
                        if denom == 0.0 {
                            0.0
                        } else {
                            simd::dot(query, c) / denom
                        }
                    }
                };
                push_bounded(&mut heap, Ranked { key, index: ci }, k);
                heap
            },
        )
        .reduce(
            || BinaryHeap::<Reverse<Ranked>>::with_capacity(k + 1),
            |mut acc, other| {
                for Reverse(r) in other {
                    push_bounded(&mut acc, r, k);
                }
                acc
            },
        );

    let mut results: Vec<Neighbor> = heap
        .into_iter()
        .map(|Reverse(r)| Neighbor {
            index: r.index,
            // Undo the L2 key negation and recover the real distance.
            score: match metric {
                Metric::L2 => (-r.key).max(0.0).sqrt(),
                _ => r.key,
            },
        })
        .collect();

    // Best first: descending similarity, or ascending distance.
    results.sort_unstable_by(|a, b| match metric {
        Metric::L2 => a.score.total_cmp(&b.score),
        _ => b.score.total_cmp(&a.score),
    });

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corpus() -> Vec<f32> {
        // 4 rows × 2 cols.
        vec![
            1.0, 0.0, // 0
            0.0, 1.0, // 1
            1.0, 1.0, // 2
            -1.0, 0.0, // 3
        ]
    }

    #[test]
    fn cosine_orders_by_similarity() {
        let data = corpus();
        let c = MatrixView::new(&data, 4, 2).unwrap();
        let res = top_k_search(&[1.0, 0.0], c, 3, Metric::Cosine).unwrap();
        assert_eq!(res.len(), 3);
        // Most similar to [1,0] is itself (row 0), then [1,1] (row 2), then [0,1] (row 1).
        assert_eq!(res[0].index, 0);
        assert!((res[0].score - 1.0).abs() < 1e-5);
        assert_eq!(res[1].index, 2);
        // Least similar of the top-3 is the orthogonal one, not the opposite.
        assert_eq!(res[2].index, 1);
    }

    #[test]
    fn l2_orders_by_ascending_distance() {
        let data = corpus();
        let c = MatrixView::new(&data, 4, 2).unwrap();
        let res = top_k_search(&[1.0, 0.0], c, 2, Metric::L2).unwrap();
        assert_eq!(res[0].index, 0);
        assert!(res[0].score < 1e-5);
        // Distances are sorted ascending.
        assert!(res[0].score <= res[1].score);
    }

    #[test]
    fn invalid_k_is_rejected() {
        let data = corpus();
        let c = MatrixView::new(&data, 4, 2).unwrap();
        assert_eq!(
            top_k_search(&[1.0, 0.0], c, 0, Metric::Cosine),
            Err(CoreError::InvalidK { k: 0, n: 4 })
        );
        assert_eq!(
            top_k_search(&[1.0, 0.0], c, 5, Metric::Cosine),
            Err(CoreError::InvalidK { k: 5, n: 4 })
        );
    }

    #[test]
    fn matches_brute_force_argsort_on_random_data() {
        // Deterministic pseudo-random corpus via a small LCG.
        let (n, d, k) = (500usize, 16usize, 10usize);
        let mut state = 0x9E3779B9u32;
        let mut next = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        };
        let data: Vec<f32> = (0..n * d).map(|_| next()).collect();
        let query: Vec<f32> = (0..d).map(|_| next()).collect();
        let c = MatrixView::new(&data, n, d).unwrap();

        let got = top_k_search(&query, c, k, Metric::DotProduct).unwrap();

        // Brute-force reference.
        let mut all: Vec<(usize, f32)> = (0..n).map(|i| (i, simd::dot(&query, c.row(i)))).collect();
        all.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

        for (rank, neighbor) in got.iter().enumerate() {
            assert_eq!(neighbor.index, all[rank].0, "rank {rank} index mismatch");
            assert!((neighbor.score - all[rank].1).abs() < 1e-5);
        }
    }
}
