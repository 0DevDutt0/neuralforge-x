# research

Design notes, algorithm write-ups, and experiment logs that inform the
implementation. Engineering rationale that doesn't belong in code comments lives
here.

## Index

- **SIMD reduction strategy** — why 4× FMA accumulators and AVX2 (not AVX-512) on
  Core Ultra; horizontal-reduction choices. See [PERFORMANCE.md](../docs/PERFORMANCE.md)
  §1 and `rust_core/src/simd.rs`.
- **Parallel top-k** — per-thread bounded heaps vs full-sort vs `argpartition`;
  memory/latency trade-offs. See [ARCHITECTURE.md](../docs/ARCHITECTURE.md) §3.
- **Planned:** HNSW parameter study (M, efConstruction, efSearch) vs recall@k
  (Phase 4); CUDA vs Triton kernel comparison on Blackwell (Phase 3).

Findings graduate into the design docs and the benchmark lab once validated.
