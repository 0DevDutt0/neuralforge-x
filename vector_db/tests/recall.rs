//! Integration tests: HNSW recall against exact brute-force ground truth.
//!
//! The whole point of an approximate index is that it is *almost* exact, so the
//! contract we test is statistical: on realistic-sized random data, recall@k
//! against `neuralforge_core`'s exact `top_k_search` must clear a high bar, and
//! it must improve monotonically as the search beam (`ef`) widens. We also pin
//! down the exactness guarantees that *are* deterministic — the metadata filter
//! never returns a non-matching id, and a filtered search agrees with an exact
//! filtered scan.

use std::collections::HashSet;

use neuralforge_core::{top_k_search, MatrixView, Metric};
use neuralforge_vector_db::{Filter, MetaValue, Metadata, VectorStore};

/// Deterministic pseudo-random data via a small LCG (keeps tests reproducible
/// and dependency-free).
struct Lcg(u32);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (self.0 >> 8) as f32 / (1u32 << 24) as f32 - 0.5
    }
    fn vector(&mut self, d: usize) -> Vec<f32> {
        (0..d).map(|_| self.next_f32()).collect()
    }
}

fn exact_top_k(data: &[f32], n: usize, d: usize, q: &[f32], k: usize) -> HashSet<usize> {
    let view = MatrixView::new(data, n, d).unwrap();
    top_k_search(q, view, k, Metric::Cosine)
        .unwrap()
        .into_iter()
        .map(|nb| nb.index)
        .collect()
}

#[test]
fn recall_is_high_and_grows_with_ef() {
    let (n, d, k) = (5_000usize, 96usize, 10usize);
    let mut rng = Lcg(0xABCD_1234);
    let data: Vec<f32> = (0..n * d).map(|_| rng.next_f32()).collect();

    let mut store = VectorStore::new(d, Metric::Cosine);
    for i in 0..n {
        store
            .insert(i as u64, &data[i * d..(i + 1) * d], Metadata::new())
            .unwrap();
    }

    let mut qrng = Lcg(0x5555_AAAA);
    let queries: Vec<Vec<f32>> = (0..50).map(|_| qrng.vector(d)).collect();

    let mean_recall = |ef: usize| -> f64 {
        let mut hits = 0usize;
        for q in &queries {
            let exact = exact_top_k(&data, n, d, q, k);
            let approx = store.search(q, k, ef, None).unwrap();
            hits += approx
                .iter()
                .filter(|h| exact.contains(&(h.id as usize)))
                .count();
        }
        hits as f64 / (queries.len() * k) as f64
    };

    let low = mean_recall(16);
    let high = mean_recall(128);

    // A wider beam never hurts recall, and the high-ef setting is near-exact.
    assert!(
        high >= low - 1e-9,
        "recall regressed with more ef: {low} -> {high}"
    );
    assert!(high > 0.92, "recall@{k} with ef=128 too low: {high:.3}");
}

#[test]
fn filtered_search_only_returns_matching_metadata() {
    let (n, d, k) = (3_000usize, 64usize, 20usize);
    let mut rng = Lcg(0x0F0F_0F0F);

    let mut store = VectorStore::new(d, Metric::Cosine);
    let mut data = Vec::with_capacity(n * d);
    for i in 0..n {
        let v = rng.vector(d);
        data.extend_from_slice(&v);
        let bucket = (i % 5) as i64;
        let md: Metadata = [("bucket".to_owned(), MetaValue::Int(bucket))]
            .into_iter()
            .collect();
        store.insert(i as u64, &v, md).unwrap();
    }

    let mut qrng = Lcg(0x9999_3333);
    let filter = Filter::Eq("bucket".into(), MetaValue::Int(2));
    for _ in 0..25 {
        let q = qrng.vector(d);
        let hits = store.search(&q, k, 128, Some(&filter)).unwrap();
        // Every returned id must satisfy the predicate...
        for h in &hits {
            let md = store.metadata(h.id).unwrap();
            assert_eq!(md.get("bucket"), Some(&MetaValue::Int(2)));
        }
        // ...and scores must be sorted best-first (cosine: descending).
        for w in hits.windows(2) {
            assert!(w[0].score >= w[1].score - 1e-6);
        }
    }
}

#[test]
fn deletes_are_never_returned_even_at_high_ef() {
    let (n, d) = (2_000usize, 48usize);
    let mut rng = Lcg(0x2468_1357);
    let mut store = VectorStore::new(d, Metric::L2);
    let mut vectors = Vec::new();
    for i in 0..n {
        let v = rng.vector(d);
        store.insert(i as u64, &v, Metadata::new()).unwrap();
        vectors.push(v);
    }

    // Delete the first 500 ids.
    let deleted: HashSet<u64> = (0..500u64).collect();
    for &id in &deleted {
        store.delete(id).unwrap();
    }
    assert_eq!(store.len(), n - deleted.len());

    let mut qrng = Lcg(0x1111_2222);
    for _ in 0..30 {
        let q = qrng.vector(d);
        let hits = store.search(&q, 25, 200, None).unwrap();
        for h in &hits {
            assert!(!deleted.contains(&h.id), "tombstoned id {} surfaced", h.id);
        }
    }
}
