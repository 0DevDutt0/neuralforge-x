//! Criterion benchmarks for the HNSW index.
//!
//! Two axes matter for an ANN index and both are measured here against a
//! deterministic synthetic corpus:
//!
//! * **Build throughput** — wall-clock to construct the graph over `N` vectors.
//! * **Query latency & recall** — single-query search time as `ef_search` grows,
//!   alongside the recall@k that extra beam width buys (printed once at startup,
//!   since recall is a quality number rather than a timing).
//!
//! Recall is measured against exact brute-force top-k from `neuralforge_core`, so
//! the benchmark doubles as a correctness sanity check on real-sized data.

use std::collections::HashSet;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use neuralforge_core::{top_k_search, MatrixView, Metric};
use neuralforge_vector_db::{HnswConfig, Metadata, VectorStore};

/// A small deterministic LCG so the corpus is reproducible without `rand`.
struct Lcg(u32);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (self.0 >> 8) as f32 / (1u32 << 24) as f32 - 0.5
    }
}

fn corpus(n: usize, d: usize) -> Vec<f32> {
    let mut lcg = Lcg(0x1234_5678);
    (0..n * d).map(|_| lcg.next_f32()).collect()
}

fn build_store(data: &[f32], n: usize, d: usize, ef_search: usize) -> VectorStore {
    let cfg = HnswConfig::new(Metric::Cosine).with_ef_search(ef_search);
    let mut store = VectorStore::with_config(d, cfg);
    for i in 0..n {
        store
            .insert(i as u64, &data[i * d..(i + 1) * d], Metadata::new())
            .unwrap();
    }
    store
}

/// Mean recall@k of the index against exact brute force over `queries`.
fn recall_at_k(store: &VectorStore, data: &[f32], n: usize, d: usize, k: usize, ef: usize) -> f64 {
    let view = MatrixView::new(data, n, d).unwrap();
    let queries = 64usize;
    let mut hit = 0usize;
    let mut lcg = Lcg(0xCAFE_F00D);
    for _ in 0..queries {
        let q: Vec<f32> = (0..d).map(|_| lcg.next_f32()).collect();
        let exact: HashSet<usize> = top_k_search(&q, view, k, Metric::Cosine)
            .unwrap()
            .into_iter()
            .map(|nb| nb.index)
            .collect();
        let approx = store.search(&q, k, ef, None).unwrap();
        hit += approx
            .iter()
            .filter(|h| exact.contains(&(h.id as usize)))
            .count();
    }
    hit as f64 / (queries * k) as f64
}

fn bench_build(c: &mut Criterion) {
    let d = 128;
    let mut group = c.benchmark_group("hnsw_build");
    group.measurement_time(Duration::from_secs(8));
    for &n in &[2_000usize, 10_000] {
        let data = corpus(n, d);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| build_store(&data, n, d, 64));
        });
    }
    group.finish();
}

fn bench_query(c: &mut Criterion) {
    let (n, d, k) = (20_000usize, 128usize, 10usize);
    let data = corpus(n, d);
    let store = build_store(&data, n, d, 64);

    // Report recall@k vs ef once, as a quality readout.
    eprintln!("\nHNSW recall@{k} on N={n}, D={d}:");
    for &ef in &[10usize, 32, 64, 128] {
        let r = recall_at_k(&store, &data, n, d, k, ef);
        eprintln!("  ef_search={ef:>4} -> recall@{k} = {:.3}", r);
    }

    let mut lcg = Lcg(0x0BAD_F00D);
    let query: Vec<f32> = (0..d).map(|_| lcg.next_f32()).collect();

    let mut group = c.benchmark_group("hnsw_query");
    for &ef in &[32usize, 64, 128] {
        group.bench_with_input(BenchmarkId::from_parameter(ef), &ef, |b, &ef| {
            b.iter(|| store.search(&query, k, ef, None).unwrap());
        });
    }
    group.finish();
}

criterion_group!(benches, bench_build, bench_query);
criterion_main!(benches);
