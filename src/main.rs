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
        .route("/metrics", get(|| async { get_metrics_payload() }))
        // Liveness/readiness probe for k8s, systemd, load balancers. No auth,
        // no upstream call — just confirms the process is accepting requests.
        .route("/healthz", get(|| async { "ok" }));

    tracing::info!("Rag-Gate proxy listening on {}", addr);
    let listener = TcpListener::bind(&addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

/// Resolves when a shutdown signal is received, letting Axum drain in-flight
/// requests before exiting. Handles Ctrl-C everywhere and SIGTERM on Unix
/// (the signal container orchestrators and systemd send on stop).
async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl-C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    tracing::info!("Shutdown signal received, draining connections");
}
