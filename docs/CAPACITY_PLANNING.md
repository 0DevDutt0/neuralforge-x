# Capacity Planning

Sizing guidance derived from the data model and the measured numbers in
[PERFORMANCE.md](PERFORMANCE.md). Rules of thumb, not guarantees — always
benchmark with your own data.

## Memory

A corpus is `n × d` `float32`: **`bytes = n · d · 4`**.

| Vectors (n) | d = 384 | d = 768 | d = 1536 |
|------------:|--------:|--------:|---------:|
| 100k  | 154 MB | 307 MB | 614 MB |
| 1M    | 1.5 GB | 2.9 GB | 5.9 GB |
| 5M    | 7.7 GB | 14.7 GB | 29.5 GB |

On the 32 GB dev box, brute-force exact search is comfortable to ~1–2M × 768 in
RAM. Beyond that, use HNSW (Phase 4) to avoid scanning every vector, and/or the
GPU (Phase 3) — the 24 GB RTX 5090 holds ≈ 8M × 768 vectors resident.

Retrieval working memory is **`O(k · threads)`**, negligible next to the corpus.
`batch_similarity` additionally allocates a `q · n` `float32` output — size it:
e.g. 64 queries × 1M = 256 MB.

## Throughput (measured, CPU)

From the benchmarks (768-dim, Core Ultra 9, AVX2):

- Exact scan ≈ **2.3M vector-comparisons/ms** at 100k (`top_k` 100k ≈ 3.4 ms).
- Batched cosine ≈ **72M pair-scores/s per query** sustained at 100k corpus.
- Single `dot`/`cosine`/`l2` ≈ **33 Gelem/s** on the SIMD path.

Linear extrapolation for exact top-k (768-dim, single query):

| Corpus | Est. latency | Notes |
|-------:|-------------:|-------|
| 100k | ~3.4 ms | measured |
| 1M | ~34 ms | scan-bound, scales ~linearly |
| 10M | ~340 ms | use HNSW below this latency target |

## Approximate search (HNSW) — measured

On clustered (realistic) data the `vector_db` HNSW index answers top-10 at
**recall 1.0 in ~0.06 ms** for a 50k × 768 corpus — about **28× faster than the
exact Rust scan** and **170× over NumPy** — because it never scans the corpus
(see [PERFORMANCE.md](PERFORMANCE.md) §5). Graph overhead is ≈ `M · 8` bytes
of edges per vector (`M = 16` → ~256 B) on top of the `d · 4` vector bytes;
budget ~10–15% over the raw corpus size. Build is a Python-loop insert, so for
very large indexes build offline and `save()`/`load()` the Parquet snapshot.
`ef_search` trades latency for recall at query time; raise it under selective
metadata filters, where recall degrades with predicate selectivity.

## When to switch strategy

| Situation | Use |
|-----------|-----|
| n ≤ ~1M, need exact / high recall | CPU exact kernels (`rust_core`) |
| Large batched scoring, GPU present | `cuda_engine` (Phase 3) |
| n ≫ 1M, latency-sensitive, ANN acceptable | HNSW (`vector_db`) |
| Need metadata filters + persistence | `vector_db` (HNSW) + Parquet (DuckDB-queryable) |

## Thread scaling

rayon scales batch/top-k across physical cores; expect near-linear gains until
memory bandwidth saturates (the corpus must stream from RAM). Past that point,
adding threads helps little — that is the signal to move to the GPU or to an ANN
index. Cap threads with `RAYON_NUM_THREADS` to leave headroom for a co-located
service.

## Latency budget (illustrative service SLO)

For a p99 < 10 ms exact-search SLO at 768-dim, keep the in-RAM corpus ≲ ~300k per
node and scale horizontally by sharding the corpus across nodes, merging top-k at
a coordinator. Above that, adopt HNSW or GPU offload before sharding further.
