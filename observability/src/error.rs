//! API error type and its mapping to HTTP responses.
//!
//! Domain failures from the vector store are translated into the right status
//! code (409 for a duplicate id, 404 for a missing one, 400 for bad input, 500
//! for internal faults) with a small JSON body, so clients get actionable,
//! machine-readable errors instead of a bare 500.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use neuralforge_vector_db::VectorDbError;
use serde::Serialize;

/// An error surfaced by an API handler.
#[derive(Debug)]
pub enum ApiError {
    /// Malformed or invalid request input (400).
    BadRequest(String),
    /// A referenced id does not exist (404).
    NotFound(String),
    /// An id already exists (409).
    Conflict(String),
    /// An unexpected internal failure (500).
    Internal(String),
}

/// The JSON body returned for every error.
#[derive(Serialize)]
struct ErrorBody {
    error: String,
    message: String,
}

impl ApiError {
    fn parts(&self) -> (StatusCode, &'static str, &str) {
        match self {
            ApiError::BadRequest(m) => (StatusCode::BAD_REQUEST, "bad_request", m),
            ApiError::NotFound(m) => (StatusCode::NOT_FOUND, "not_found", m),
            ApiError::Conflict(m) => (StatusCode::CONFLICT, "conflict", m),
            ApiError::Internal(m) => (StatusCode::INTERNAL_SERVER_ERROR, "internal", m),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, code, message) = self.parts();
        if status == StatusCode::INTERNAL_SERVER_ERROR {
            tracing::error!(error = code, message, "request failed");
        } else {
            tracing::warn!(
                error = code,
                message,
                status = status.as_u16(),
                "request rejected"
            );
        }
        let body = ErrorBody {
            error: code.to_string(),
            message: message.to_string(),
        };
        (status, Json(body)).into_response()
    }
}

impl From<VectorDbError> for ApiError {
    fn from(err: VectorDbError) -> Self {
        match err {
            VectorDbError::DuplicateId { .. } => ApiError::Conflict(err.to_string()),
            VectorDbError::UnknownId { .. } => ApiError::NotFound(err.to_string()),
            VectorDbError::DimensionMismatch { .. }
            | VectorDbError::NonFinite { .. }
            | VectorDbError::InvalidK { .. }
            | VectorDbError::InvalidFilter(_) => ApiError::BadRequest(err.to_string()),
            VectorDbError::Persistence(_) | VectorDbError::Core(_) => {
                ApiError::Internal(err.to_string())
            }
        }
    }
}

/// Convenient result alias for handlers.
pub type ApiResult<T> = Result<T, ApiError>;
