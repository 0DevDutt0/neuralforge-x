//! Error types for the core kernels.
//!
//! Kernels validate their inputs eagerly and return a typed [`CoreError`] rather
//! than panicking, so that the Python bindings can translate failures into the
//! appropriate Python exception without unwinding across the FFI boundary.

use thiserror::Error;

/// Errors returned by the core similarity and retrieval kernels.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CoreError {
    /// Two operands that must share a dimensionality did not.
    #[error("dimension mismatch: left has {left} elements, right has {right}")]
    DimensionMismatch {
        /// Length of the left-hand operand.
        left: usize,
        /// Length of the right-hand operand.
        right: usize,
    },

    /// A vector or matrix was empty where at least one element was required.
    #[error("empty input: {what} must be non-empty")]
    EmptyInput {
        /// Human-readable name of the offending argument.
        what: &'static str,
    },

    /// A flat buffer length was not an exact multiple of the row dimension.
    #[error("invalid shape: buffer length {len} is not divisible by dim {dim}")]
    InvalidShape {
        /// Length of the flat buffer.
        len: usize,
        /// The row dimension the buffer was expected to be a multiple of.
        dim: usize,
    },

    /// `k` was zero or larger than the number of available vectors.
    #[error("invalid k: requested {k} neighbours from a corpus of {n} vectors")]
    InvalidK {
        /// Requested number of neighbours.
        k: usize,
        /// Number of vectors actually available.
        n: usize,
    },
}

/// Convenience result alias for the core kernels.
pub type Result<T> = core::result::Result<T, CoreError>;
