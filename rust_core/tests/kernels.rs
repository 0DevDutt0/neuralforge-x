//! Integration tests exercising only the public crate API.

use neuralforge_core::{
    batch_similarity, cosine_similarity, dot_product, l2_distance, top_k_search, CoreError,
    MatrixView, Metric,
};

#[test]
fn pairwise_kernels_agree_with_hand_computation() {
    let a = [1.0f32, 2.0, 2.0]; // ‖a‖ = 3
    let b = [2.0f32, 0.0, 0.0]; // ‖b‖ = 2
    assert!((dot_product(&a, &b).unwrap() - 2.0).abs() < 1e-5);
    assert!((l2_distance(&a, &b).unwrap() - 3.0).abs() < 1e-5); // √(1+4+4)
    assert!((cosine_similarity(&a, &b).unwrap() - (2.0 / 6.0)).abs() < 1e-5);
}

#[test]
fn top_k_is_consistent_with_batch_similarity() {
    // 6 vectors × 4 dims.
    let data: Vec<f32> = vec![
        1.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        0.9, 0.1, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0, //
        0.7, 0.7, 0.0, 0.0, //
        0.0, 0.0, 0.0, 1.0, //
    ];
    let corpus = MatrixView::new(&data, 6, 4).unwrap();
    let query = [1.0f32, 0.0, 0.0, 0.0];

    let top = top_k_search(&query, corpus, 3, Metric::Cosine).unwrap();

    // Full similarity row for the same query (1 × 6 matrix).
    let qmat = MatrixView::new(&query, 1, 4).unwrap();
    let sims = batch_similarity(qmat, corpus, Metric::Cosine).unwrap();

    // The top-k indices must be the 3 highest entries of `sims`, in order.
    let mut ranked: Vec<usize> = (0..6).collect();
    ranked.sort_unstable_by(|&i, &j| sims[j].total_cmp(&sims[i]));
    let expected: Vec<usize> = ranked.into_iter().take(3).collect();
    let got: Vec<usize> = top.iter().map(|n| n.index).collect();
    assert_eq!(got, expected);

    // And each reported score must equal the corresponding batch entry.
    for n in &top {
        assert!((n.score - sims[n.index]).abs() < 1e-5);
    }
}

#[test]
fn shape_errors_surface_as_typed_variants() {
    let data = [1.0f32, 2.0, 3.0, 4.0];
    let corpus = MatrixView::new(&data, 2, 2).unwrap();
    // Query width 3 against corpus width 2.
    assert_eq!(
        top_k_search(&[1.0, 2.0, 3.0], corpus, 1, Metric::L2),
        Err(CoreError::DimensionMismatch { left: 3, right: 2 })
    );
}
