//! Telemetry setup: structured tracing, with optional OpenTelemetry export.
//!
//! Logs are emitted as JSON via `tracing-subscriber` (always on). When the
//! `otel` feature is built and `NFX_OTEL_ENDPOINT` is set, request spans are also
//! batch-exported over OTLP/gRPC to a collector; failure to connect is
//! non-fatal — the service degrades to logs + metrics rather than refusing to
//! start. The returned [`TelemetryGuard`] flushes the exporter on drop.

use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

use crate::config::Config;

/// Flushes the OpenTelemetry pipeline (if any) when dropped.
pub struct TelemetryGuard {
    #[cfg(feature = "otel")]
    otel_enabled: bool,
}

/// Initializes tracing/logging and (optionally) OTLP span export.
#[cfg_attr(not(feature = "otel"), allow(unused_variables))]
pub fn init(config: &Config) -> TelemetryGuard {
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,neuralforge_service=debug"));
    let fmt_layer = tracing_subscriber::fmt::layer()
        .json()
        .with_current_span(true)
        .with_span_list(false);

    #[cfg(feature = "otel")]
    if let Some(endpoint) = config.otel_endpoint.clone() {
        match build_tracer(config, &endpoint) {
            Ok(tracer) => {
                tracing_subscriber::registry()
                    .with(env_filter)
                    .with(fmt_layer)
                    .with(tracing_opentelemetry::layer().with_tracer(tracer))
                    .init();
                tracing::info!(endpoint, "OpenTelemetry OTLP span export enabled");
                return TelemetryGuard { otel_enabled: true };
            }
            Err(err) => {
                eprintln!("warning: OTLP init failed ({err}); continuing with logs + metrics only");
            }
        }
    }

    tracing_subscriber::registry()
        .with(env_filter)
        .with(fmt_layer)
        .init();

    TelemetryGuard {
        #[cfg(feature = "otel")]
        otel_enabled: false,
    }
}

/// Builds a batch OTLP/gRPC tracer and registers it as the global provider.
#[cfg(feature = "otel")]
fn build_tracer(
    config: &Config,
    endpoint: &str,
) -> anyhow::Result<opentelemetry_sdk::trace::Tracer> {
    use opentelemetry::trace::TracerProvider as _;
    use opentelemetry_otlp::WithExportConfig;

    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .with_endpoint(endpoint.to_string())
        .build()?;
    let provider = opentelemetry_sdk::trace::TracerProvider::builder()
        .with_batch_exporter(exporter, opentelemetry_sdk::runtime::Tokio)
        .with_resource(opentelemetry_sdk::Resource::new([
            opentelemetry::KeyValue::new("service.name", config.service_name.clone()),
        ]))
        .build();
    let tracer = provider.tracer("neuralforge_service");
    opentelemetry::global::set_tracer_provider(provider);
    Ok(tracer)
}

impl Drop for TelemetryGuard {
    fn drop(&mut self) {
        #[cfg(feature = "otel")]
        if self.otel_enabled {
            opentelemetry::global::shutdown_tracer_provider();
        }
    }
}
