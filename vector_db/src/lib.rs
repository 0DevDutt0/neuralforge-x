//! # neuralforge_vector_db
//!
//! An approximate-nearest-neighbour vector store for **NeuralForge-X**, built on
//! a hand-written **HNSW** graph with metadata filtering and Parquet persistence.
//!
//! This is the *outbound* adapter in the project's hexagonal layering: it turns
//! the dependency-light kernels of [`neuralforge_core`] into a stateful, mutable
//! store. The graph reuses the core's SIMD distance kernels verbatim, so the
//! approximate index and any exact brute-force baseline rank by one identical
//! numeric implementation — which is exactly what makes recall measurable.
//!
//! ## Layers
//!
//! | Module | Responsibility |
//! |--------|----------------|
//! | [`hnsw`] | The multi-layer proximity graph: construction heuristic, filtered beam search. |
//! | [`store`] | Id mapping, metadata, soft-delete/update, compaction — the repository surface. |
//! | [`metadata`] | Scalar metadata values and the composable [`Filter`] predicate language. |
//! | [`persistence`] | Self-describing Parquet snapshot / restore. |
//!
//! ## Example
//!
//! ```
//! use neuralforge_core::Metric;
//! use neuralforge_vector_db::{Filter, MetaValue, Metadata, VectorStore};
//!
//! let mut store = VectorStore::new(3, Metric::Cosine);
//! let md = |lang: &str| Metadata::from([("lang".to_owned(), MetaValue::from(lang))]);
//! store.insert(1, &[1.0, 0.0, 0.0], md("rust")).unwrap();
//! store.insert(2, &[0.9, 0.1, 0.0], md("python")).unwrap();
//! store.insert(3, &[0.0, 1.0, 0.0], md("rust")).unwrap();
//!
//! // Nearest to +x, restricted to rust-tagged vectors.
//! let filter = Filter::Eq("lang".into(), "rust".into());
//! let hits = store.search(&[1.0, 0.05, 0.0], 1, 0, Some(&filter)).unwrap();
//! assert_eq!(hits[0].id, 1);
//! ```
//!
//! ## Design notes
//!
//! - **Append-only graph, soft deletes.** Deletions tombstone a node; updates
//!   that change a vector tombstone-and-reinsert. [`VectorStore::compact`]
//!   rebuilds to reclaim the space. This keeps the hot insert path simple and the
//!   graph invariants easy to reason about.
//! - **Filtered search.** A metadata predicate gates *result membership* only;
//!   filtered-out and tombstoned nodes still route the walk, so the graph stays
//!   connected and recall degrades gracefully with selectivity.
//! - **Reproducibility.** Level assignment uses a seeded `splitmix64`, so a given
//!   insertion order yields a bit-identical graph — the recall tests depend on it.

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod error;
pub mod hnsw;
pub mod metadata;
pub mod persistence;
pub mod store;

pub use error::{Result, VectorDbError};
pub use hnsw::{Hnsw, HnswConfig};
pub use metadata::{Filter, MetaValue, Metadata};
pub use store::{Hit, VectorStore};
