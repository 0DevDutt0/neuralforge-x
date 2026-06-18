//! Criterion benchmark for parallel top-k retrieval.
//!
//! Run with `cargo bench -p neuralforge_core --bench topk`. Scales the corpus
//! from 10k to 500k 768-dimensional vectors to show throughput under the
//! rayon-parallel bounded-heap selection.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use neuralforge_core::{top_k_search, MatrixView, Metric};

fn make_vec(len: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..len)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32) / ((1u32 << 31) as f32) - 1.0
        })
        .collect()
}

fn bench_topk(c: &mut Criterion) {
    let mut g = c.benchmark_group("top_k_search_cosine");
    let (d, k) = (768usize, 10usize);
    let query = make_vec(d, 7);
    for &n in &[10_000usize, 100_000, 500_000] {
        let corpus = make_vec(n * d, 3);
        let cv = MatrixView::new(&corpus, n, d).unwrap();
        g.throughput(Throughput::Elements(n as u64));
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |bch, _| {
            bch.iter(|| black_box(top_k_search(black_box(&query), cv, k, Metric::Cosine).unwrap()));
        });
    }
    g.finish();
}

criterion_group!(benches, bench_topk);
criterion_main!(benches);
