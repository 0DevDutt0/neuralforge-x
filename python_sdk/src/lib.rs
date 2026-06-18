//! PyO3 bindings for the NeuralForge-X core kernels.
//!
//! This crate is a thin, allocation-conscious adapter: it borrows the NumPy
//! buffers Python already owns (zero-copy via [`numpy::PyReadonlyArray`]),
//! releases the GIL around the Rust compute with [`Python::allow_threads`] so
//! other Python threads can run, and maps typed [`neuralforge_core::CoreError`]
//! values onto Python `ValueError`s.
//!
//! The functions here are intentionally low-level; the ergonomic, fully typed
//! and validated surface lives in the pure-Python `neuralforge` package that
//! wraps this `_native` extension module.

use neuralforge_core::{self as core, MatrixView, Metric};
use neuralforge_vector_db::{persistence, Filter, HnswConfig, Metadata, VectorStore};
use numpy::ndarray::Array2;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;

/// Translates a core error into a Python `ValueError`.
fn to_py_err(err: core::CoreError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// Translates a vector-database error into the most fitting Python exception:
/// id-lookup failures become `KeyError`, everything else `ValueError`.
fn vdb_err(err: neuralforge_vector_db::VectorDbError) -> PyErr {
    use neuralforge_vector_db::VectorDbError as E;
    match err {
        E::UnknownId { .. } | E::DuplicateId { .. } => PyKeyError::new_err(err.to_string()),
        _ => PyValueError::new_err(err.to_string()),
    }
}

/// Parses an optional JSON object into [`Metadata`] (an empty bag for `None`).
fn parse_metadata(json: Option<&str>) -> PyResult<Metadata> {
    match json {
        None => Ok(Metadata::new()),
        Some(s) => serde_json::from_str(s)
            .map_err(|e| PyValueError::new_err(format!("invalid metadata JSON: {e}"))),
    }
}

/// A `(indices-or-ids, scores)` pair of 1-D NumPy arrays, as returned by the
/// retrieval entry points. `I` is the index element type (`i64` row indices for
/// the kernels, `u64` ids for the vector store).
type ResultArrays<'py, I> = (Bound<'py, PyArray1<I>>, Bound<'py, PyArray1<f32>>);

/// Parses a metric name, raising a Python `ValueError` for unknown values.
fn parse_metric(name: &str) -> PyResult<Metric> {
    Metric::from_name(name).ok_or_else(|| {
        PyValueError::new_err(format!(
            "unknown metric '{name}'; expected one of 'cosine', 'dot', 'l2'"
        ))
    })
}

/// Cosine similarity between two 1-D float32 arrays.
#[pyfunction]
fn cosine_similarity(a: PyReadonlyArray1<'_, f32>, b: PyReadonlyArray1<'_, f32>) -> PyResult<f32> {
    core::cosine_similarity(a.as_slice()?, b.as_slice()?).map_err(to_py_err)
}

/// Inner product between two 1-D float32 arrays.
#[pyfunction]
fn dot_product(a: PyReadonlyArray1<'_, f32>, b: PyReadonlyArray1<'_, f32>) -> PyResult<f32> {
    core::dot_product(a.as_slice()?, b.as_slice()?).map_err(to_py_err)
}

/// Euclidean (L2) distance between two 1-D float32 arrays.
#[pyfunction]
fn l2_distance(a: PyReadonlyArray1<'_, f32>, b: PyReadonlyArray1<'_, f32>) -> PyResult<f32> {
    core::l2_distance(a.as_slice()?, b.as_slice()?).map_err(to_py_err)
}

/// Full `queries × corpus` similarity matrix.
///
/// `queries` is `(q, d)`, `corpus` is `(n, d)`; the result is `(q, n)`. Both
/// inputs must be C-contiguous float32 (the Python wrapper guarantees this).
#[pyfunction]
#[pyo3(signature = (queries, corpus, metric = "cosine"))]
fn batch_similarity<'py>(
    py: Python<'py>,
    queries: PyReadonlyArray2<'py, f32>,
    corpus: PyReadonlyArray2<'py, f32>,
    metric: &str,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let metric = parse_metric(metric)?;
    let (qr, qc) = (queries.shape()[0], queries.shape()[1]);
    let (cr, cc) = (corpus.shape()[0], corpus.shape()[1]);

    let q_slice = queries.as_slice()?;
    let c_slice = corpus.as_slice()?;
    let qv = MatrixView::new(q_slice, qr, qc).map_err(to_py_err)?;
    let cv = MatrixView::new(c_slice, cr, cc).map_err(to_py_err)?;

    // Release the GIL for the duration of the parallel compute.
    let flat = py
        .allow_threads(|| core::batch_similarity(qv, cv, metric))
        .map_err(to_py_err)?;

    let matrix =
        Array2::from_shape_vec((qr, cr), flat).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(matrix.into_pyarray(py))
}

/// Top-`k` nearest neighbours of `query` within `corpus`.
///
/// Returns `(indices, scores)` as a pair of 1-D arrays (`int64`, `float32`),
/// ordered best-match first.
#[pyfunction]
#[pyo3(signature = (query, corpus, k, metric = "cosine"))]
fn top_k_search<'py>(
    py: Python<'py>,
    query: PyReadonlyArray1<'py, f32>,
    corpus: PyReadonlyArray2<'py, f32>,
    k: usize,
    metric: &str,
) -> PyResult<ResultArrays<'py, i64>> {
    let metric = parse_metric(metric)?;
    let (cr, cc) = (corpus.shape()[0], corpus.shape()[1]);

    let q_slice = query.as_slice()?;
    let c_slice = corpus.as_slice()?;
    let cv = MatrixView::new(c_slice, cr, cc).map_err(to_py_err)?;

    let neighbors = py
        .allow_threads(|| core::top_k_search(q_slice, cv, k, metric))
        .map_err(to_py_err)?;

    let indices: Vec<i64> = neighbors.iter().map(|n| n.index as i64).collect();
    let scores: Vec<f32> = neighbors.iter().map(|n| n.score).collect();
    Ok((indices.into_pyarray(py), scores.into_pyarray(py)))
}

/// An HNSW vector index exposed to Python as `neuralforge._native.VectorIndex`.
///
/// Metadata and filters cross the FFI boundary as JSON strings (the typed
/// Python wrapper builds them from dicts), which keeps the native surface small
/// and the storage format identical to the persisted Parquet `metadata` column.
#[pyclass(name = "VectorIndex")]
struct PyVectorIndex {
    store: VectorStore,
}

#[pymethods]
impl PyVectorIndex {
    #[new]
    #[pyo3(signature = (dim, metric = "cosine", m = 16, ef_construction = 200, ef_search = 64))]
    fn new(
        dim: usize,
        metric: &str,
        m: usize,
        ef_construction: usize,
        ef_search: usize,
    ) -> PyResult<Self> {
        let metric = parse_metric(metric)?;
        let config = HnswConfig::new(metric)
            .with_m(m)
            .with_ef_construction(ef_construction)
            .with_ef_search(ef_search);
        Ok(Self {
            store: VectorStore::with_config(dim, config),
        })
    }

    /// Inserts a vector under a fresh id with optional JSON metadata.
    #[pyo3(signature = (id, vector, metadata = None))]
    fn insert(
        &mut self,
        id: u64,
        vector: PyReadonlyArray1<'_, f32>,
        metadata: Option<&str>,
    ) -> PyResult<()> {
        let md = parse_metadata(metadata)?;
        self.store
            .insert(id, vector.as_slice()?, md)
            .map_err(vdb_err)
    }

    /// Soft-deletes a live id.
    fn delete(&mut self, id: u64) -> PyResult<()> {
        self.store.delete(id).map_err(vdb_err)
    }

    /// Updates a live id's vector and/or metadata.
    #[pyo3(signature = (id, vector = None, metadata = None))]
    fn update(
        &mut self,
        id: u64,
        vector: Option<PyReadonlyArray1<'_, f32>>,
        metadata: Option<&str>,
    ) -> PyResult<()> {
        let md = match metadata {
            Some(s) => Some(parse_metadata(Some(s))?),
            None => None,
        };
        let vec_slice = vector.as_ref().map(|v| v.as_slice()).transpose()?;
        self.store.update(id, vec_slice, md).map_err(vdb_err)
    }

    /// Searches for the `k` nearest live vectors, optionally filtered by a JSON
    /// predicate. Returns `(ids, scores)` as `(uint64, float32)` arrays.
    #[pyo3(signature = (query, k, ef = 0, filter = None))]
    fn search<'py>(
        &self,
        py: Python<'py>,
        query: PyReadonlyArray1<'py, f32>,
        k: usize,
        ef: usize,
        filter: Option<&str>,
    ) -> PyResult<ResultArrays<'py, u64>> {
        let parsed: Option<Filter> = match filter {
            Some(s) => Some(
                serde_json::from_str(s)
                    .map_err(|e| PyValueError::new_err(format!("invalid filter JSON: {e}")))?,
            ),
            None => None,
        };
        let q = query.as_slice()?;
        let hits = py
            .allow_threads(|| self.store.search(q, k, ef, parsed.as_ref()))
            .map_err(vdb_err)?;

        let ids: Vec<u64> = hits.iter().map(|h| h.id).collect();
        let scores: Vec<f32> = hits.iter().map(|h| h.score).collect();
        Ok((ids.into_pyarray(py), scores.into_pyarray(py)))
    }

    /// Whether `id` is currently live.
    fn contains(&self, id: u64) -> bool {
        self.store.contains(id)
    }

    /// The JSON-encoded metadata for a live id, or `None` if absent.
    fn metadata_json(&self, id: u64) -> Option<String> {
        self.store
            .metadata(id)
            .and_then(|m| serde_json::to_string(m).ok())
    }

    /// Physically removes tombstones, rebuilding the graph.
    fn compact(&mut self) {
        self.store.compact();
    }

    /// Number of tombstoned (soft-deleted) vectors still occupying memory.
    fn tombstones(&self) -> usize {
        self.store.tombstones()
    }

    /// Writes a Parquet snapshot of the live vectors to `path`.
    fn save(&self, path: &str) -> PyResult<()> {
        persistence::save(&self.store, path).map_err(vdb_err)
    }

    /// Loads an index from a Parquet snapshot, rebuilding the graph.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        Ok(Self {
            store: persistence::load(path).map_err(vdb_err)?,
        })
    }

    /// Vector dimensionality.
    #[getter]
    fn dim(&self) -> usize {
        self.store.dim()
    }

    /// The ranking metric name.
    #[getter]
    fn metric(&self) -> String {
        self.store.metric().as_str().to_owned()
    }

    /// Number of live vectors.
    fn __len__(&self) -> usize {
        self.store.len()
    }
}

/// The native `neuralforge._native` extension module.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(dot_product, m)?)?;
    m.add_function(wrap_pyfunction!(l2_distance, m)?)?;
    m.add_function(wrap_pyfunction!(batch_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(top_k_search, m)?)?;
    m.add_class::<PyVectorIndex>()?;
    Ok(())
}
