//! Service configuration, read from the environment.
//!
//! Twelve-factor style: every knob is an environment variable with a sensible
//! default, so the same binary runs locally and in the container unchanged.

use std::net::SocketAddr;

use neuralforge_core::Metric;

/// Runtime configuration for the service.
#[derive(Debug, Clone)]
pub struct Config {
    /// Address to bind the HTTP listener to (`NFX_BIND`, default `0.0.0.0:8080`).
    pub bind: SocketAddr,
    /// Dimensionality of the vector index (`NFX_DIM`, default `768`).
    pub dim: usize,
    /// Similarity metric (`NFX_METRIC`, default `cosine`).
    pub metric: Metric,
    /// OTLP collector endpoint for trace export (`NFX_OTEL_ENDPOINT`). When unset
    /// (or the `otel` feature is off) the service still logs and exposes metrics;
    /// only span export is disabled.
    pub otel_endpoint: Option<String>,
    /// Logical service name reported in traces/metrics (`NFX_SERVICE_NAME`).
    pub service_name: String,
}

impl Config {
    /// Builds a [`Config`] from the environment, falling back to defaults.
    ///
    /// # Errors
    /// Returns a message if `NFX_BIND` is not a valid socket address, `NFX_DIM`
    /// is not a positive integer, or `NFX_METRIC` is not a known metric.
    pub fn from_env() -> Result<Self, String> {
        let bind = env_or("NFX_BIND", "0.0.0.0:8080")
            .parse::<SocketAddr>()
            .map_err(|e| format!("invalid NFX_BIND: {e}"))?;
        let dim = env_or("NFX_DIM", "768")
            .parse::<usize>()
            .map_err(|e| format!("invalid NFX_DIM: {e}"))?;
        if dim == 0 {
            return Err("NFX_DIM must be positive".to_string());
        }
        let metric_name = env_or("NFX_METRIC", "cosine");
        let metric = Metric::from_name(&metric_name)
            .ok_or_else(|| format!("invalid NFX_METRIC '{metric_name}'"))?;
        let otel_endpoint = std::env::var("NFX_OTEL_ENDPOINT")
            .ok()
            .filter(|s| !s.is_empty());
        let service_name = env_or("NFX_SERVICE_NAME", "neuralforge-service");

        Ok(Self {
            bind,
            dim,
            metric,
            otel_endpoint,
            service_name,
        })
    }
}

/// Reads an environment variable or returns `default`.
fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}
