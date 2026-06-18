# API Reference

The public Python API lives in the `neuralforge` package. All array inputs are
coerced to **contiguous `float32`**; Python lists and other dtypes are accepted
and converted (a copy is made only when necessary).

## Metrics

`metric` accepts (case-insensitive): `"cosine"` (aliases `cos`), `"dot"`
(`dot_product`, `ip`, `inner_product`), `"l2"` (`euclidean`, `euclid`).
Cosine/dot are *higher-is-better*; L2 is *lower-is-better*.

## Functions

### `cosine_similarity(a, b) -> float`
Cosine similarity in `[-1, 1]`. A zero-norm input yields `0.0` (never `NaN`).

### `dot_product(a, b) -> float`
Inner product `⟨a, b⟩`.

### `l2_distance(a, b) -> float`
Euclidean distance `‖a - b‖₂`.

For all three: `a`, `b` are 1-D arrays of equal length.
Raises `DimensionMismatchError` on unequal lengths, `InvalidInputError` on
empty/non-1-D input.

### `batch_similarity(queries, corpus, metric="cosine") -> np.ndarray`
Full similarity matrix.

| Param | Type | Notes |
|-------|------|-------|
| `queries` | `(q, d)` array | query vectors |
| `corpus` | `(n, d)` array | corpus vectors |
| `metric` | str | see *Metrics* |

**Returns** a `(q, n)` `float32` array; `[i, j]` compares query `i` with corpus
`j`. Raises `DimensionMismatchError` if `d` differs, `InvalidMetricError`,
`InvalidInputError`.

```python
sims = nf.batch_similarity(queries, corpus, metric="cosine")  # (q, n) float32
```

### `top_k_search(query, corpus, k, metric="cosine") -> SearchResult`
The `k` corpus vectors most similar to `query`, best-first.

| Param | Type | Notes |
|-------|------|-------|
| `query` | `(d,)` array | the query |
| `corpus` | `(n, d)` array | corpus |
| `k` | int | `1 <= k <= n` |
| `metric` | str | see *Metrics* |

**Returns** a [`SearchResult`](#searchresult). Raises `DimensionMismatchError`,
`InvalidMetricError`, or `InvalidInputError` (bad `k`/shape).

```python
res = nf.top_k_search(query, corpus, k=10, metric="cosine")
res.indices   # np.int64[k], best-first
res.scores    # np.float32[k]
for idx, score in res:
    ...
```

## Types

### `SearchResult`
Frozen dataclass with `indices: NDArray[int64]` and `scores: NDArray[float32]`.
Supports `len(result)` and iteration yielding `(index, score)` tuples; `repr`
shows the top 5.

## Exceptions

```
NeuralForgeError (Exception)
├── InvalidInputError (ValueError)
│   └── DimensionMismatchError
└── InvalidMetricError (ValueError)
```

All custom errors subclass `ValueError`, so `except ValueError` keeps working.
Low-level errors from the native layer also surface as `ValueError`.

## Module attributes

- `neuralforge.__version__` — version string (sourced from the Rust crate).

---

## Vector database — `neuralforge.VectorIndex`

A stateful HNSW index with metadata filtering and Parquet persistence.

```python
import numpy as np
from neuralforge import VectorIndex, Filter

idx = VectorIndex(dim=768, metric="cosine", m=16, ef_construction=200, ef_search=64)
idx.add(1, np.random.rand(768).astype(np.float32), {"lang": "rust", "year": 2026})
hits = idx.search(query, k=5, ef=128,
                  filter=Filter.eq("lang", "rust") & Filter.ge("year", 2024))
hits[0].id, hits[0].score          # VectorHit(id: int, score: float)
idx.save("snap.parquet"); VectorIndex.load("snap.parquet")
```

| Method | Notes |
|--------|-------|
| `VectorIndex(dim, metric="cosine", *, m, ef_construction, ef_search)` | construct |
| `add(id, vector, metadata=None)` | insert; `KeyError` on duplicate id |
| `search(query, k, *, ef=0, filter=None) -> list[VectorHit]` | best-first; `ef=0` uses the default |
| `delete(id)` / `update(id, vector=None, metadata=None)` | soft-delete / replace |
| `get_metadata(id) -> dict \| None` | stored metadata |
| `compact()` / `tombstones()` | reclaim deleted nodes / count them |
| `save(path)` / `VectorIndex.load(path)` | self-describing Parquet snapshot |
| `dim`, `metric`, `len(idx)`, `id in idx` | introspection |

**`Filter`** builds metadata predicates, composed with `&` / `|` / `~`:
`Filter.eq/ne(field, value)`, `Filter.lt/le/gt/ge(field, bound)`,
`Filter.is_in(field, values)`, `Filter.exists/missing(field)`.

---

## GPU engine — `neuralforge_cuda`

A separately-installed package (`pip install -e ./cuda_engine`) exposing the same
operations on the GPU across three backends (`"cuda"`, `"triton"`, `"torch"`).

```python
import neuralforge_cuda as gpu
gpu.cuda_available()                 # bool
gpu.device_info()                    # {name, compute_capability, total_mem_mb, ...}
gpu.available_backends()             # subset of ["cuda", "triton", "torch"]
gpu.gpu_batch_similarity(queries, corpus, "cosine", backend="cuda")   # (q, n)
gpu.gpu_topk_search(query, corpus, k, "cosine", backend="triton")     # (idx, scores)
```

`gpu_cosine_similarity` / `gpu_dot_product` / `gpu_l2_distance` mirror the CPU
pairwise kernels. A capability gate raises `GpuError` when CUDA is unavailable.
See [cuda_engine/README.md](https://github.com/0DevDutt0/neuralforge-x/blob/main/cuda_engine/README.md).

---

## HTTP service — `neuralforge_service`

The Phase-7 axum service exposes the engine over HTTP (see
[observability/README.md](https://github.com/0DevDutt0/neuralforge-x/blob/main/observability/README.md)).

| Method & path | Body | Result |
|---------------|------|--------|
| `POST /v1/search` | `{query, k, ef?, filter?}` | `{count, hits:[{id, score}]}` |
| `POST /v1/vectors` | `{id, vector, metadata?}` | `201` / `409` duplicate |
| `DELETE /v1/vectors/{id}` | — | `204` / `404` |
| `GET /v1/stats` | — | `{vectors, dim, metric, tombstones}` |
| `GET /healthz` · `/readyz` · `/metrics` | — | liveness · readiness · Prometheus |

`filter` is the same predicate language as JSON, e.g.
`{"And": [{"Eq": ["lang", "rust"]}, {"Ge": ["year", 2024]}]}`.

---

## Rust core (`neuralforge_core`)

For embedding the engine directly in Rust.

```rust
use neuralforge_core::{
    cosine_similarity, dot_product, l2_distance, batch_similarity, top_k_search,
    MatrixView, Metric, Neighbor, CoreError,
};

let a = [1.0f32, 2.0, 3.0];
let b = [4.0f32, 5.0, 6.0];
let s = cosine_similarity(&a, &b)?;                 // Result<f32, CoreError>

let data: Vec<f32> = /* n*d row-major */ vec![/* ... */];
let corpus = MatrixView::new(&data, n, d)?;
let hits: Vec<Neighbor> = top_k_search(&query, corpus, 10, Metric::Cosine)?;
// Neighbor { index: usize, score: f32 }
```

- `MatrixView::new(&[f32], rows, cols)` / `MatrixView::from_flat(&[f32], cols)` —
  zero-copy, validated row-major view.
- `Metric::{Cosine, DotProduct, L2}`, `Metric::from_name(&str) -> Option<Metric>`.
- `CoreError::{DimensionMismatch, EmptyInput, InvalidShape, InvalidK}`.

Full rustdoc: `cargo doc -p neuralforge_core --open`.
