//! Error types for the vector database.
//!
//! Like the domain core, the store validates inputs eagerly and returns a typed
//! [`VectorDbError`] instead of panicking, so the Python bindings can translate
//! each variant into the appropriate Python exception without unwinding across
//! the FFI boundary. The [`VectorDbError::Core`] variant forwards failures from
//! the underlying `neuralforge_core` kernels unchanged.

use neuralforge_core::CoreError;
use thiserror::Error;

/// Errors returned by the HNSW index and the vector store.
#[derive(Debug, Error)]
pub enum VectorDbError {
    /// A vector did not match the index's configured dimensionality.
    #[error("dimension mismatch: index holds {expected}-d vectors, got {actual}")]
    DimensionMismatch {
        /// Dimensionality the index was built with.
        expected: usize,
        /// Dimensionality of the offending vector.
        actual: usize,
    },

    /// An `insert` reused an external id that is already live in the index.
    #[error("duplicate id: {id} is already present (use `update` to replace it)")]
    DuplicateId {
        /// The conflicting external id.
        id: u64,
    },

    /// A `delete`, `update`, or `get` referenced an id the index does not hold.
    #[error("unknown id: {id} is not present in the index")]
    UnknownId {
        /// The missing external id.
        id: u64,
    },

    /// A vector contained a non-finite component (`NaN`/`±∞`), which would poison
    /// the graph's distance ordering.
    #[error("non-finite value at position {pos} of the input vector")]
    NonFinite {
        /// Index of the first offending component.
        pos: usize,
    },

    /// `k` was zero or exceeded the number of live vectors.
    #[error("invalid k: requested {k} neighbours from {live} live vectors")]
    InvalidK {
        /// Requested number of neighbours.
        k: usize,
        /// Number of non-deleted vectors currently in the index.
        live: usize,
    },

    /// A metadata filter expression could not be parsed or applied.
    #[error("invalid filter: {0}")]
    InvalidFilter(String),

    /// A persistence (Parquet/Arrow) operation failed.
    #[error("persistence error: {0}")]
    Persistence(String),

    /// A failure surfaced by an underlying `neuralforge_core` kernel.
    #[error(transparent)]
    Core(#[from] CoreError),
}

/// Convenience result alias for the vector database.
pub type Result<T> = core::result::Result<T, VectorDbError>;
