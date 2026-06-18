# NeuralForge-X observability service — multi-stage Rust build.
# Build context is the repository root: `docker build -f observability/Dockerfile .`
# (or via observability/docker-compose.yml, which sets the context to `..`).

# ---- builder -------------------------------------------------------------
FROM rust:1-slim-bookworm AS builder

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential pkg-config \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY . .
# Build only the service (and its path deps); OTLP export on via the `otel` feature.
RUN cargo build --release -p neuralforge_service --features otel \
 && cp target/release/neuralforge_service /usr/local/bin/neuralforge_service

# ---- runtime -------------------------------------------------------------
FROM debian:bookworm-slim AS runtime

LABEL org.opencontainers.image.title="neuralforge-service" \
      org.opencontainers.image.source="https://github.com/0DevDutt0/neuralforge-x" \
      org.opencontainers.image.licenses="MIT OR Apache-2.0"

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --system --uid 10001 nfx

COPY --from=builder /usr/local/bin/neuralforge_service /usr/local/bin/neuralforge_service

USER nfx
ENV NFX_BIND=0.0.0.0:8080
EXPOSE 8080

# Liveness for orchestrators that honour HEALTHCHECK.
HEALTHCHECK --interval=10s --timeout=2s --retries=5 \
  CMD curl -fsS http://localhost:8080/healthz || exit 1

ENTRYPOINT ["neuralforge_service"]
