//! # neuralforge_core
//!
//! The CPU domain core of **NeuralForge-X**: cache-friendly, SIMD-accelerated,
//! data-parallel kernels for dense vector similarity and top-k retrieval.
//!
//! This crate is deliberately I/O-free and dependency-light. It is the hexagonal
//! *domain* layer — the Python SDK (`python_sdk`), the future HTTP service, and
//! the GPU accelerator all sit at the edges and call into these primitives.
//!
//! ## Kernels
//!
//! | Function | Operation | Time | Space |
//! |----------|-----------|------|-------|
//! | [`dot_product`] | `⟨a, b⟩` | `O(d)` | `O(1)` |
//! | [`cosine_similarity`] | `⟨a, b⟩ / (‖a‖·‖b‖)` | `O(d)` | `O(1)` |
//! | [`l2_distance`] | `‖a − b‖₂` | `O(d)` | `O(1)` |
//! | [`batch_similarity`] | `Q × Cᵀ` similarity matrix | `O(q·n·d)` | `O(q·n)` out |
//! | [`top_k_search`] | k nearest of `n` | `O(n·d + n·log k)` | `O(k·threads)` |
//!
//! ## Optimization strategy
//!
//! 1. **Data layout.** Corpora are passed as a single contiguous `&[f32]` in
//!    row-major order (see [`MatrixView`]), never as `Vec<Vec<f32>>`. This keeps
//!    each vector in a cache-line-friendly run and enables wide vector loads.
//! 2. **SIMD.** The inner products and squared distances are hand-vectorised with
//!    AVX2 + FMA and dispatched at runtime via `is_x86_feature_detected!`, with a
//!    portable scalar fallback (see [`simd`]). Four independent accumulators hide
//!    FMA latency.
//! 3. **Parallelism.** [`batch_similarity`] and [`top_k_search`] fan out across
//!    cores with `rayon`. Top-k uses per-thread bounded min-heaps merged at the
//!    end, so peak memory is `O(k · threads)` rather than `O(n)`.
//! 4. **Allocation discipline.** The hot loops borrow slices and write into
//!    pre-sized buffers; no per-element heap traffic.
//!
//! ## Numerical contract
//!
//! All kernels operate in `f32`. Because floating-point addition is not
//! associative, the SIMD and scalar paths can disagree in the last few ULPs;
//! the test suite asserts agreement within a relative epsilon rather than bit
//! equality. Cosine similarity of a zero-norm vector is defined as `0.0`.

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod error;
pub mod matrix;
pub mod metric;
pub mod simd;

mod similarity;
mod topk;

pub use error::{CoreError, Result};
pub use matrix::MatrixView;
pub use metric::Metric;
pub use similarity::{batch_similarity, cosine_similarity, dot_product, l2_distance};
pub use topk::{top_k_search, Neighbor};
