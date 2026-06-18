# Roadmap

NeuralForge-X is built in vertical slices: each phase is real, tested, and
benchmarked before the next begins. Status reflects the current `main`.

| Phase | Module | Scope | Status |
|------:|--------|-------|--------|
| 0 | Foundation | Workspace, packaging, CI, licensing, docs | ✅ Done |
| 1 | `rust_core` | SIMD + rayon similarity & top-k kernels | ✅ Done |
| 2 | `python_sdk` | PyO3/maturin typed Python SDK | ✅ Done |
| 3 | `cuda_engine` | CUDA C++ **and** Triton kernels (+ PyTorch baseline); GPU benchmarks | ✅ Done |
| 4 | `vector_db` | HNSW index, metadata filtering, Parquet persistence (+ DuckDB) | ✅ Done |
| 5 | `benchmark_lab` | Cross-stack harness, SVG charts, reports | ✅ Done |
| 6 | `profiling` | flamegraph/criterion + Nsight Systems/Compute | ✅ Done |
| 7 | `observability` | OTel tracing, Prometheus metrics, Grafana dashboards | ✅ Done |
| 8 | `docs` | MkDocs Material site, full design docs | ✅ Done |
| 9 | README excellence | Hero, animated assets, full visual suite | 🚧 In progress |

## Phase 3 — GPU engine (done)

Shipped for the local **RTX 5090 (Blackwell, `sm_120`)** as `neuralforge_cuda`:
`gpu_cosine_similarity`, `gpu_batch_similarity`, `gpu_topk_search` across three
backends — hand-written **CUDA C++** (CuPy NVRTC, native `sm_120`), **Triton**,
and a **PyTorch** baseline — with a four-way benchmark and a GPU test suite
validated against the CPU core. Notably, native `sm_120` codegen was achieved
**without** upgrading the system CUDA toolkit (12.6) by using a Blackwell-capable
NVRTC (≥ 12.8); see [cuda_engine/README.md](cuda_engine/README.md).

## Phase 4 — Vector DB (done)

A from-scratch Rust **HNSW** index shipped as `neuralforge_vector_db`:
`insert`/`delete`/`update`/`search` over external ids, a composable metadata
`Filter` language applied during traversal, soft-delete tombstones with
`compact()`, and self-describing **Parquet** persistence (pure-Rust `arrow`/
`parquet`) that rebuilds the graph on load. Recall@k is validated against exact
brute force from `rust_core`, and the snapshot is queryable directly by
**DuckDB** — `examples/03_vector_db.py` cross-checks the index against an exact
`list_cosine_similarity` SQL baseline. Exposed to Python as `VectorIndex`. See
[vector_db/README.md](vector_db/README.md).

## Phase 5 — Benchmark lab (done)

`benchmark_lab` is a unified harness measuring the **same** workloads across
Python · NumPy · Rust · GPU (CUDA/Triton/PyTorch) · HNSW, capturing latency,
throughput, host/device memory, CPU% and GPU% into a self-describing JSON, then
rendering committed **SVG charts** and a Markdown report. Accelerated backends
are verified against a NumPy oracle; the ANN backend reports recall@k. Driven by
`python -m benchmark_lab {run,charts,report,all}`. Headlines: SIMD up to **77×**
over pure Python, GPU up to **7.4×** over the multi-core CPU on batch, and HNSW
top-k at **recall 1.0** ~**28×** faster than the exact Rust scan. See
[benchmark_lab/README.md](benchmark_lab/README.md) and PERFORMANCE.md §5.

## Phase 6 — Profiling lab (done)

`profiling` splits into a portable **analysis** half and an external-tool
**capture** half. `analyze.py` turns criterion medians into `CPU_PROFILE.md`
(scalar-vs-SIMD effectiveness, a roofline read — the AVX2+FMA kernel saturates
~33 Gelem/s and is load-bound past 384-dim — and an optimization log) plus the
committed `docs/assets/profile_cpu.svg`. The `neuralforge_profile` (Rust) and
`gpu_workload.py` targets drive the real kernels under sustained load for
`cargo flamegraph` and Nsight, wired up by `scripts/{cpu_flamegraph,gpu_nsight}.ps1`.
Honest environment notes: Windows flamegraph capture needs an elevated shell
(ETW), and Blackwell `sm_120` kernel tracing needs a newer Nsight than the one
installed — the same toolchain-freshness gap as Phase 3. See
[profiling/README.md](profiling/README.md).

## Phase 7 — Observability (done)

`neuralforge_service` is an **axum** HTTP service exposing the HNSW engine
in-process (`/v1/search`, `/v1/vectors`, `/v1/stats`) with production telemetry:
a Prometheus `/metrics` endpoint (RED signals + index gauges), structured
`tracing` logs with optional **OpenTelemetry** OTLP/gRPC span export (the `otel`
feature), and `/healthz` + `/readyz` probes with graceful shutdown. A
`docker compose` stack runs it alongside an **OTel Collector → Jaeger** trace
pipeline and **Prometheus → Grafana** with a provisioned dashboard. Verified
natively (real metrics + ranked search over HTTP) and with tower-oneshot
integration tests; the stack config is compose-validated. See
[observability/README.md](observability/README.md).

## Phase 8 — Documentation (done)

A **MkDocs Material** site (`mkdocs.yml`, `python -m mkdocs serve`) over the full
design-doc set — [Architecture](docs/ARCHITECTURE.md), [System
Design](docs/SYSTEM_DESIGN.md), [API Reference](docs/API_REFERENCE.md),
[Developer Guide](docs/DEVELOPER_GUIDE.md), [Performance](docs/PERFORMANCE.md),
[Capacity Planning](docs/CAPACITY_PLANNING.md), [Security](docs/SECURITY.md) —
all brought current with the as-built system (HNSW, the benchmark/profiling labs,
the axum service), with Mermaid diagrams and the generated SVG charts. The site
builds under `--strict` and deploys to GitHub Pages via
`.github/workflows/docs.yml`.

## Guiding principles

1. **No placeholders** — a module ships only when it is real and tested.
2. **Benchmark-driven** — every kernel has a reproducible benchmark.
3. **Local-first & free** — no paid APIs, no cloud dependency.
4. **Production posture** — typed errors, CI gates, observability, docs.
