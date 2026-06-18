//! Binary entrypoint for `neuralforge_service`.
//!
//! Reads [`Config`] from the environment and hands off to the library's
//! [`serve`](neuralforge_service::serve) loop. All wiring lives in the library so
//! the router is unit-testable without a live socket.

use neuralforge_service::Config;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = Config::from_env().map_err(|msg| anyhow::anyhow!(msg))?;
    neuralforge_service::serve(config).await
}
