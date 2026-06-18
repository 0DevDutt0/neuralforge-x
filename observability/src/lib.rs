//! # neuralforge_service
//!
//! A production-posture HTTP service that exposes the **NeuralForge-X** vector
//! engine — the Phase-4 HNSW [`VectorStore`](neuralforge_vector_db::VectorStore)
//! served **in-process** (no FFI, no extra runtime) — with the observability a
//! real service needs:
//!
//! - **Metrics.** A Prometheus `/metrics` endpoint with RED signals (request
//!   rate, errors, latency histograms) plus index gauges.
//! - **Tracing.** Structured JSON logs via `tracing`, and optional OpenTelemetry
//!   OTLP span export to a collector (the `otel` feature).
//! - **Health.** `/healthz` (liveness) and `/readyz` (readiness) probes for
//!   orchestrators.
//!
//! The crate is split into a library (router + state, unit-testable with `tower`'s
//! `oneshot`) and a thin binary. Configuration is environment-driven
//! ([`Config::from_env`]).

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod config;
pub mod error;
pub mod metrics;
pub mod routes;
pub mod state;
pub mod telemetry;

pub use config::Config;
pub use error::{ApiError, ApiResult};
pub use routes::build_router;
pub use state::AppState;

/// Initializes telemetry and metrics, then serves the API until a shutdown
/// signal arrives.
///
/// # Errors
/// Returns an error if the listener cannot bind or the server loop fails.
pub async fn serve(config: Config) -> anyhow::Result<()> {
    let _guard = telemetry::init(&config);
    let prometheus = metrics::install_recorder();
    let state = AppState::new(config.dim, config.metric, prometheus);
    let app = build_router(state.clone());

    let listener = tokio::net::TcpListener::bind(config.bind).await?;
    tracing::info!(
        bind = %config.bind,
        dim = config.dim,
        metric = config.metric.as_str(),
        "neuralforge-service listening"
    );
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal(state))
        .await?;
    tracing::info!("shutdown complete");
    Ok(())
}

/// Resolves when SIGINT/SIGTERM arrives, flipping readiness off first so a load
/// balancer stops sending new traffic before the listener drains.
async fn shutdown_signal(state: AppState) {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("install Ctrl-C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("install SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => {},
        () = terminate => {},
    }
    tracing::info!("signal received; draining");
    state.set_ready(false);
}
