use axum::routing::get;
use clap::{Parser, Subcommand};
use rag_gate::{calibrate::calibrate_handler, eval, metrics::get_metrics_payload, proxy::create_router, ProxyConfig};
use std::path::{Path, PathBuf};
use tokio::net::TcpListener;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Parser)]
#[command(
    name = "rag-gate",
    about = "Confidence-gating proxy for LLM streams (default: serve). \
             Use `evaluate` to score an eval-results file offline."
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the proxy server (same as running `rag-gate` with no subcommand).
    Serve,
    /// Score an already-run eval-results file and print the offline report
    /// (accuracy, coverage, risk, AURC, false-accept / false-abstain rates).
    /// Does not start the server.
    ///
    /// Input shape: a JSON object with a "results" array — or a bare [...]
    /// array — where each record carries exactly two fields: "confidence"
    /// (f64, mean token logprob) and "correct" (bool). Every other field
    /// (question text, gold answers, logprobs, model metadata, ...) is
    /// ignored. Raw logprobs arrays are NOT accepted in v1: every benchmark
    /// file already ships a computed "confidence" per record. Thresholds are
    /// found with the exact same search as POST /v1/rag-gate/calibrate, so
    /// the two can never disagree.
    Evaluate {
        /// Path to the eval-results JSON file.
        #[arg(long, value_name = "PATH")]
        dataset: PathBuf,
        /// Fraction of highest-confidence samples to answer (0 < c <= 1).
        #[arg(long, default_value_t = 0.8)]
        target_coverage: f64,
        /// Print the report as a single JSON object instead of a table.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.command.unwrap_or(Commands::Serve) {
        Commands::Serve => serve().await,
        Commands::Evaluate {
            dataset,
            target_coverage,
            json,
        } => run_evaluate(&dataset, target_coverage, json),
    }
}

fn run_evaluate(
    dataset: &Path,
    target_coverage: f64,
    as_json: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if !(target_coverage > 0.0 && target_coverage <= 1.0) {
        return Err(format!(
            "invalid --target-coverage {target_coverage}: must satisfy 0 < c <= 1"
        )
        .into());
    }
    let samples =
        eval::load_dataset_file(dataset).map_err(|e| format!("evaluate: {e}"))?;
    let report = eval::build_report(&samples, target_coverage, &dataset.display().to_string());
    if as_json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!("{}", eval::format_human_report(&report));
    }
    Ok(())
}

async fn serve() -> Result<(), Box<dyn std::error::Error>> {
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
