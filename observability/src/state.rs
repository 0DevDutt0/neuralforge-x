//! Shared application state.
//!
//! The service owns a single in-memory [`VectorStore`] (the Phase-4 HNSW index)
//! behind an `RwLock`: searches take a read lock, mutations a write lock. The
//! readiness flag gates `/readyz` so an orchestrator only routes traffic once the
//! index is initialized.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Instant;

use metrics_exporter_prometheus::PrometheusHandle;
use neuralforge_core::Metric;
use neuralforge_vector_db::VectorStore;

/// Cloneable handle to the service's shared state (an axum `State`).
#[derive(Clone)]
pub struct AppState {
    /// The HNSW-backed vector store.
    pub store: Arc<RwLock<VectorStore>>,
    /// Readiness flag for `/readyz`.
    ready: Arc<AtomicBool>,
    /// Prometheus exposition handle, rendered by `/metrics`.
    pub prometheus: PrometheusHandle,
    /// Process start time, for the uptime readout.
    started: Instant,
    /// Configured vector dimensionality.
    pub dim: usize,
    /// Configured similarity metric.
    pub metric: Metric,
}

impl AppState {
    /// Creates fresh state with an empty index of the given shape.
    #[must_use]
    pub fn new(dim: usize, metric: Metric, prometheus: PrometheusHandle) -> Self {
        Self {
            store: Arc::new(RwLock::new(VectorStore::new(dim, metric))),
            ready: Arc::new(AtomicBool::new(true)),
            prometheus,
            started: Instant::now(),
            dim,
            metric,
        }
    }

    /// Whether the service is ready to serve traffic.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Relaxed)
    }

    /// Sets the readiness flag (e.g. flipped off during shutdown).
    pub fn set_ready(&self, ready: bool) {
        self.ready.store(ready, Ordering::Relaxed);
    }

    /// Seconds since the process started.
    #[must_use]
    pub fn uptime_secs(&self) -> u64 {
        self.started.elapsed().as_secs()
    }
}
