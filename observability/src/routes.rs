//! HTTP routes: the vector-engine API plus health, readiness, and metrics.
//!
//! | Method & path | Purpose |
//! |---------------|---------|
//! | `GET /healthz` | liveness — always 200 while the process runs |
//! | `GET /readyz` | readiness — 200 when the index is initialized, else 503 |
//! | `GET /metrics` | Prometheus exposition |
//! | `GET /v1/stats` | index size / dimensionality / metric |
//! | `POST /v1/vectors` | insert `{id, vector, metadata?}` |
//! | `DELETE /v1/vectors/{id}` | soft-delete a vector |
//! | `POST /v1/search` | `{query, k, ef?, filter?}` → ranked hits |

use std::time::Duration;

use axum::extract::{Path, State};
use axum::http::{header, StatusCode};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{middleware, Json, Router};
use neuralforge_vector_db::{Filter, Metadata};
use serde::{Deserialize, Serialize};

use crate::error::{ApiError, ApiResult};
use crate::metrics::track_metrics;
use crate::state::AppState;

/// Builds the application router with all middleware layers attached.
pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route("/readyz", get(readyz))
        .route("/metrics", get(metrics_endpoint))
        .route("/v1/stats", get(stats))
        .route("/v1/vectors", post(insert_vector))
        .route("/v1/vectors/{id}", axum::routing::delete(delete_vector))
        .route("/v1/search", post(search))
        // Innermost first: trace the request, then record metrics, bound its time.
        .layer(middleware::from_fn(track_metrics))
        .layer(tower_http::trace::TraceLayer::new_for_http())
        .layer(tower_http::timeout::TimeoutLayer::with_status_code(
            StatusCode::REQUEST_TIMEOUT,
            Duration::from_secs(15),
        ))
        .layer(tower_http::catch_panic::CatchPanicLayer::new())
        .with_state(state)
}

/// Liveness probe — cheap and dependency-free.
async fn healthz() -> &'static str {
    "ok"
}

/// Readiness probe — gates traffic until the index is usable.
async fn readyz(State(state): State<AppState>) -> impl IntoResponse {
    if state.is_ready() {
        let body = Json(serde_json::json!({
            "status": "ready",
            "uptime_seconds": state.uptime_secs(),
        }));
        (StatusCode::OK, body)
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({ "status": "not_ready" })),
        )
    }
}

/// Prometheus exposition endpoint.
async fn metrics_endpoint(State(state): State<AppState>) -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/plain; version=0.0.4")],
        state.prometheus.render(),
    )
}

/// Index statistics.
#[derive(Serialize)]
struct Stats {
    vectors: usize,
    dim: usize,
    metric: String,
    tombstones: usize,
}

async fn stats(State(state): State<AppState>) -> ApiResult<Json<Stats>> {
    let store = state.store.read().map_err(lock_poisoned)?;
    Ok(Json(Stats {
        vectors: store.len(),
        dim: store.dim(),
        metric: store.metric().as_str().to_string(),
        tombstones: store.tombstones(),
    }))
}

/// Body of an insert request.
#[derive(Deserialize)]
struct InsertRequest {
    id: u64,
    vector: Vec<f32>,
    #[serde(default)]
    metadata: Metadata,
}

#[tracing::instrument(skip_all, fields(id = req.id, dim = req.vector.len()))]
async fn insert_vector(
    State(state): State<AppState>,
    Json(req): Json<InsertRequest>,
) -> ApiResult<impl IntoResponse> {
    let live = {
        let mut store = state.store.write().map_err(lock_poisoned)?;
        store.insert(req.id, &req.vector, req.metadata)?;
        store.len()
    };
    metrics::gauge!("nfx_index_vectors").set(live as f64);
    Ok((
        StatusCode::CREATED,
        Json(serde_json::json!({ "id": req.id, "vectors": live })),
    ))
}

async fn delete_vector(
    State(state): State<AppState>,
    Path(id): Path<u64>,
) -> ApiResult<impl IntoResponse> {
    let live = {
        let mut store = state.store.write().map_err(lock_poisoned)?;
        store.delete(id)?;
        store.len()
    };
    metrics::gauge!("nfx_index_vectors").set(live as f64);
    Ok(StatusCode::NO_CONTENT)
}

/// Body of a search request.
#[derive(Deserialize)]
struct SearchRequest {
    query: Vec<f32>,
    k: usize,
    /// Search beam width; `0` (default) uses the index default.
    #[serde(default)]
    ef: usize,
    /// Optional metadata predicate.
    #[serde(default)]
    filter: Option<Filter>,
}

/// A single ranked result.
#[derive(Serialize)]
struct Hit {
    id: u64,
    score: f32,
}

/// Search response payload.
#[derive(Serialize)]
struct SearchResponse {
    count: usize,
    hits: Vec<Hit>,
}

#[tracing::instrument(skip_all, fields(k = req.k, ef = req.ef, filtered = req.filter.is_some()))]
async fn search(
    State(state): State<AppState>,
    Json(req): Json<SearchRequest>,
) -> ApiResult<Json<SearchResponse>> {
    let hits = {
        let store = state.store.read().map_err(lock_poisoned)?;
        store.search(&req.query, req.k, req.ef, req.filter.as_ref())?
    };
    metrics::histogram!("nfx_search_results").record(hits.len() as f64);
    let hits: Vec<Hit> = hits
        .into_iter()
        .map(|h| Hit {
            id: h.id,
            score: h.score,
        })
        .collect();
    Ok(Json(SearchResponse {
        count: hits.len(),
        hits,
    }))
}

/// Maps a poisoned `RwLock` into a 500.
fn lock_poisoned<T>(_: T) -> ApiError {
    ApiError::Internal("internal lock poisoned".to_string())
}
