use axum::routing::get;
use rag_gate::{calibrate::calibrate_handler, metrics::get_metrics_payload, proxy::create_router, ProxyConfig};
use tokio::net::TcpListener;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "rag_gate=debug,info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let config = ProxyConfig::load();
    let addr = config.listen_addr.clone();

    let app = create_router(config)
        .route("/v1/rag-gate/calibrate", axum::routing::post(calibrate_handler))
        .route("/metrics", get(|| async { get_metrics_payload() }));

    tracing::info!("Rag-Gate proxy listening on {}", addr);
    let listener = TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
