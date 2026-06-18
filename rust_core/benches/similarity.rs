//! Criterion benchmarks for the similarity kernels.
//!
//! Run with `cargo bench -p neuralforge_core --bench similarity`. The
//! `scalar` vs `simd_avx2` comparison quantifies the SIMD speed-up across the
//! embedding dimensionalities common in practice (MiniLM 384, BERT 768,
//! OpenAI-style 1536).

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use neuralforge_core::{batch_similarity, simd, MatrixView, Metric};

/// Deterministic pseudo-random buffer (a 64-bit LCG; values land in `[-1, 1)`).
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

fn bench_dot(c: &mut Criterion) {
    let mut g = c.benchmark_group("dot_product");
    for &d in &[128usize, 384, 768, 1536] {
        let a = make_vec(d, 1);
        let b = make_vec(d, 2);
        g.throughput(Throughput::Elements(d as u64));
        g.bench_with_input(BenchmarkId::new("scalar", d), &d, |bch, _| {
            bch.iter(|| black_box(simd::dot_scalar(black_box(&a), black_box(&b))));
        });
        g.bench_with_input(BenchmarkId::new("simd_avx2", d), &d, |bch, _| {
            bch.iter(|| black_box(simd::dot(black_box(&a), black_box(&b))));
        });
    }
    g.finish();
}

fn bench_batch(c: &mut Criterion) {
    let mut g = c.benchmark_group("batch_similarity_cosine");
    let (q, d) = (64usize, 768usize);
    let queries = make_vec(q * d, 10);
    for &n in &[1_000usize, 10_000, 50_000] {
        let corpus = make_vec(n * d, 20);
        let qv = MatrixView::new(&queries, q, d).unwrap();
        let cv = MatrixView::new(&corpus, n, d).unwrap();
        g.throughput(Throughput::Elements((q * n) as u64));
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |bch, _| {
            bch.iter(|| black_box(batch_similarity(qv, cv, Metric::Cosine).unwrap()));
        });
    }
    g.finish();
}

criterion_group!(benches, bench_dot, bench_batch);
criterion_main!(benches);
