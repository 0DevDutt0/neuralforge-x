//! Borrow-only, row-major matrix view.
//!
//! A [`MatrixView`] is a thin `(data, rows, cols)` triple over a contiguous
//! `&[f32]` buffer. It owns nothing and allocates nothing, which makes it the
//! natural representation for zero-copy interop with NumPy arrays across the
//! Python FFI boundary: the Rust side simply borrows the buffer NumPy already
//! holds.

use crate::error::{CoreError, Result};

/// A borrowed, row-major (`C`-contiguous) view over a flat `f32` buffer,
/// interpreted as a `rows × cols` matrix.
#[derive(Debug, Clone, Copy)]
pub struct MatrixView<'a> {
    data: &'a [f32],
    rows: usize,
    cols: usize,
}

impl<'a> MatrixView<'a> {
    /// Creates a view from an explicit shape, validating `data.len() == rows * cols`.
    ///
    /// # Errors
    /// Returns [`CoreError::EmptyInput`] if either dimension is zero, or
    /// [`CoreError::InvalidShape`] if the buffer length does not match the shape.
    pub fn new(data: &'a [f32], rows: usize, cols: usize) -> Result<Self> {
        if rows == 0 || cols == 0 {
            return Err(CoreError::EmptyInput { what: "matrix" });
        }
        if data.len() != rows * cols {
            return Err(CoreError::InvalidShape {
                len: data.len(),
                dim: cols,
            });
        }
        Ok(Self { data, rows, cols })
    }

    /// Creates a view from a flat buffer and a column dimension, inferring `rows`.
    ///
    /// # Errors
    /// Returns [`CoreError::EmptyInput`] for an empty buffer or zero `cols`, and
    /// [`CoreError::InvalidShape`] if `data.len()` is not a multiple of `cols`.
    pub fn from_flat(data: &'a [f32], cols: usize) -> Result<Self> {
        if cols == 0 {
            return Err(CoreError::EmptyInput { what: "dim" });
        }
        if data.is_empty() {
            return Err(CoreError::EmptyInput { what: "matrix" });
        }
        if data.len() % cols != 0 {
            return Err(CoreError::InvalidShape {
                len: data.len(),
                dim: cols,
            });
        }
        Ok(Self {
            data,
            rows: data.len() / cols,
            cols,
        })
    }

    /// Number of rows (vectors).
    #[inline]
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns (vector dimensionality).
    #[inline]
    #[must_use]
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Borrows row `i` as a `cols`-length slice.
    ///
    /// # Panics
    /// Panics if `i >= rows`. Callers iterate over `0..rows`, so the bound is an
    /// invariant rather than a runtime concern; the slice indexing also enforces it.
    #[inline]
    #[must_use]
    pub fn row(&self, i: usize) -> &'a [f32] {
        let start = i * self.cols;
        &self.data[start..start + self.cols]
    }

    /// The underlying flat buffer.
    #[inline]
    #[must_use]
    pub const fn as_slice(&self) -> &'a [f32] {
        self.data
    }
}
