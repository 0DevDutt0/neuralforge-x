# neuralforge_core

The CPU **domain core** of NeuralForge-X: cache-friendly, SIMD-accelerated,
data-parallel kernels for dense vector similarity and top-k retrieval. I/O-free
and dependency-light (`rayon`, `thiserror`).

## Kernels

| Function | Operation | Time | Space |
|----------|-----------|------|-------|
| `dot_product` | `⟨a, b⟩` | `O(d)` | `O(1)` |
| `cosine_similarity` | `⟨a, b⟩ / (‖a‖·‖b‖)` | `O(d)` | `O(1)` |
| `l2_distance` | `‖a - b‖₂` | `O(d)` | `O(1)` |
| `batch_similarity` | `Q × Cᵀ` matrix | `O(q·n·d)` | `O(q·n)` out |
| `top_k_search` | k nearest of n | `O(n·d + n·log k)` | `O(k·threads)` |

## Example

```rust
use neuralforge_core::{top_k_search, MatrixView, Metric};

let data: Vec<f32> = /* n*d, row-major */ vec![/* ... */];
let corpus = MatrixView::new(&data, n, d)?;
let hits = top_k_search(&query, corpus, 10, Metric::Cosine)?; // Vec<Neighbor>
```

## Design highlights

- **AVX2 + FMA** kernels with runtime dispatch (`is_x86_feature_detected!`) and a
  scalar fallback; four FMA accumulators for ILP. `unsafe` is confined to
  `src/simd.rs` and guarded by that runtime check.
- **rayon** parallel batch + a parallel bounded-min-heap top-k.
- Zero-copy `MatrixView` over contiguous row-major `f32` (NumPy-compatible).
- Typed `CoreError`; no panics on bad input.

## Develop

```bash
cargo test -p neuralforge_core
cargo bench -p neuralforge_core          # criterion: similarity, topk
cargo doc  -p neuralforge_core --open
```

Property tests assert the SIMD path matches the scalar reference within a
relative epsilon. Licensed MIT OR Apache-2.0.
