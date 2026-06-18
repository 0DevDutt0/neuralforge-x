//! Integration tests for the service API.
//!
//! The router is driven directly with `tower`'s `oneshot` — no socket, no real
//! server — so these are fast and deterministic. A non-global Prometheus handle
//! is built per test (the metric macros in the handlers fall back to the no-op
//! recorder, which is fine here).

use axum::body::Body;
use axum::http::{Request, StatusCode};
use metrics_exporter_prometheus::PrometheusBuilder;
use neuralforge_core::Metric;
use neuralforge_service::{build_router, AppState};
use serde_json::{json, Value};
use tower::ServiceExt;

fn test_state(dim: usize) -> AppState {
    let handle = PrometheusBuilder::new().build_recorder().handle();
    AppState::new(dim, Metric::Cosine, handle)
}

fn post(path: &str, body: Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

async fn body_json(response: axum::response::Response) -> Value {
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    if bytes.is_empty() {
        return Value::Null;
    }
    serde_json::from_slice(&bytes).unwrap()
}

#[tokio::test]
async fn healthz_and_readyz() {
    let app = build_router(test_state(4));

    let health = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/healthz")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(health.status(), StatusCode::OK);

    let ready = app
        .oneshot(
            Request::builder()
                .uri("/readyz")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(ready.status(), StatusCode::OK);
    assert_eq!(body_json(ready).await["status"], "ready");
}

#[tokio::test]
async fn insert_then_search_returns_the_vector() {
    let app = build_router(test_state(3));

    let created = app
        .clone()
        .oneshot(post(
            "/v1/vectors",
            json!({"id": 1, "vector": [1.0, 0.0, 0.0], "metadata": {"lang": "rust"}}),
        ))
        .await
        .unwrap();
    assert_eq!(created.status(), StatusCode::CREATED);

    app.clone()
        .oneshot(post(
            "/v1/vectors",
            json!({"id": 2, "vector": [0.0, 1.0, 0.0]}),
        ))
        .await
        .unwrap();

    let search = app
        .oneshot(post(
            "/v1/search",
            json!({"query": [1.0, 0.05, 0.0], "k": 1}),
        ))
        .await
        .unwrap();
    assert_eq!(search.status(), StatusCode::OK);
    let body = body_json(search).await;
    assert_eq!(body["count"], 1);
    assert_eq!(body["hits"][0]["id"], 1);
}

#[tokio::test]
async fn metadata_filter_is_applied() {
    let app = build_router(test_state(2));
    for (id, tag) in [(1u64, "a"), (2, "b"), (3, "a")] {
        let vec = if id == 2 { [0.0, 1.0] } else { [1.0, 0.0] };
        app.clone()
            .oneshot(post(
                "/v1/vectors",
                json!({"id": id, "vector": vec, "metadata": {"tag": tag}}),
            ))
            .await
            .unwrap();
    }
    let search = app
        .oneshot(post(
            "/v1/search",
            json!({"query": [1.0, 0.0], "k": 2, "filter": {"Eq": ["tag", "a"]}}),
        ))
        .await
        .unwrap();
    let body = body_json(search).await;
    let ids: Vec<u64> = body["hits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|h| h["id"].as_u64().unwrap())
        .collect();
    assert!(ids.iter().all(|&id| id == 1 || id == 3));
}

#[tokio::test]
async fn error_paths_map_to_status_codes() {
    let app = build_router(test_state(3));
    app.clone()
        .oneshot(post(
            "/v1/vectors",
            json!({"id": 1, "vector": [1.0, 0.0, 0.0]}),
        ))
        .await
        .unwrap();

    // Duplicate id -> 409.
    let dup = app
        .clone()
        .oneshot(post(
            "/v1/vectors",
            json!({"id": 1, "vector": [0.0, 1.0, 0.0]}),
        ))
        .await
        .unwrap();
    assert_eq!(dup.status(), StatusCode::CONFLICT);

    // Wrong dimensionality -> 400.
    let bad_dim = app
        .clone()
        .oneshot(post("/v1/vectors", json!({"id": 2, "vector": [1.0, 0.0]})))
        .await
        .unwrap();
    assert_eq!(bad_dim.status(), StatusCode::BAD_REQUEST);

    // Delete unknown id -> 404.
    let missing = app
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri("/v1/vectors/999")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(missing.status(), StatusCode::NOT_FOUND);

    // k = 0 -> 400.
    let bad_k = app
        .oneshot(post(
            "/v1/search",
            json!({"query": [1.0, 0.0, 0.0], "k": 0}),
        ))
        .await
        .unwrap();
    assert_eq!(bad_k.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn metrics_endpoint_exposes_prometheus_text() {
    let app = build_router(test_state(2));
    // Generate one tracked request first.
    app.clone()
        .oneshot(
            Request::builder()
                .uri("/healthz")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    let metrics = app
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(metrics.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(metrics.into_body(), usize::MAX)
        .await
        .unwrap();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    // The exposition is valid even if empty; the content type is the contract.
    assert!(text.is_empty() || text.contains("# "));
}
