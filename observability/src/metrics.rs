//! Prometheus metrics: the recorder, metric descriptions, and the request
//! middleware that records RED-style signals (Rate, Errors, Duration) per route.

use std::time::Instant;

use axum::extract::{MatchedPath, Request};
use axum::middleware::Next;
use axum::response::Response;
use metrics::{counter, describe_counter, describe_gauge, describe_histogram, histogram, Unit};
use metrics_exporter_prometheus::{Matcher, PrometheusBuilder, PrometheusHandle};

/// HTTP request latency histogram buckets, in seconds (sub-ms to ~1s).
const LATENCY_BUCKETS: &[f64] = &[
    0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5,
];

/// Installs the global Prometheus recorder and registers metric metadata.
///
/// # Panics
/// Panics if a recorder is already installed (only called once at startup).
#[must_use]
pub fn install_recorder() -> PrometheusHandle {
    let handle = PrometheusBuilder::new()
        .set_buckets_for_metric(
            Matcher::Full("nfx_http_request_duration_seconds".to_string()),
            LATENCY_BUCKETS,
        )
        .expect("valid bucket configuration")
        .install_recorder()
        .expect("install prometheus recorder");

    describe_counter!(
        "nfx_http_requests_total",
        "Total HTTP requests by route and status"
    );
    describe_histogram!(
        "nfx_http_request_duration_seconds",
        Unit::Seconds,
        "HTTP request latency by route"
    );
    describe_gauge!("nfx_index_vectors", "Live vectors currently in the index");
    describe_histogram!("nfx_search_results", "Number of hits returned per search");

    handle
}

/// axum middleware: times each request and records its rate/latency, labelled by
/// the *matched* route path (not the raw URI) to bound metric cardinality.
pub async fn track_metrics(req: Request, next: Next) -> Response {
    let start = Instant::now();
    let path = req
        .extensions()
        .get::<MatchedPath>()
        .map(|p| p.as_str().to_owned())
        .unwrap_or_else(|| req.uri().path().to_owned());
    let method = req.method().clone();

    let response = next.run(req).await;

    let latency = start.elapsed().as_secs_f64();
    let status = response.status().as_u16().to_string();
    let labels = [
        ("method", method.to_string()),
        ("path", path),
        ("status", status),
    ];
    counter!("nfx_http_requests_total", &labels).increment(1);
    // Drop the status label from the latency series to keep its cardinality low.
    histogram!("nfx_http_request_duration_seconds", "method" => labels[0].1.clone(), "path" => labels[1].1.clone())
        .record(latency);

    response
}
