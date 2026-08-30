//! Keyless mock OpenAI-compatible SSE upstream.
//!
//! Streams canned chat-completions frames with switchable confidence
//! profiles — the zero-API-key demo target for rag-gate (docker-compose
//! default upstream). Mirrors the Python port's
//! `examples/mock-upstream/app.py` frame-for-frame so both repos give the
//! same zero-key demo experience.
//!
//! Select a profile per request with the `x-mock-profile` header:
//! - `high` — mean logprob ~ -0.15: the gate ANSWERs, full passthrough.
//! - `low`  — mean logprob ~ -3.4: the gate cuts with ABSTAIN.
//!
//! Run standalone: `cargo run --release`

use axum::{
    body::Body,
    extract::Request,
    http::{header, StatusCode},
    response::Response,
    routing::{get, post},
    Router,
};
use bytes::Bytes;

const HIGH_PROFILE: &[f64] = &[-0.1, -0.2, -0.15, -0.1, -0.2, -0.1, -0.12, -0.18, -0.1, -0.2];
const LOW_PROFILE: &[f64] = &[-2.0, -2.5, -3.0, -3.5, -4.0, -4.5, -3.8, -4.2, -3.1, -2.9];

fn sse_frame(logprob: f64) -> Bytes {
    let payload = format!(
        r#"{{"choices":[{{"delta":{{"content":"x"}},"logprobs":{{"content":[{{"logprob":{logprob}}}]}}}}]}}"#
    );
    Bytes::from(format!("data: {payload}\n\n"))
}

async fn chat_completions(request: Request) -> Response {
    let profile = request
        .headers()
        .get("x-mock-profile")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("high")
        .to_string();

    let logprobs: &[f64] = if profile == "low" { LOW_PROFILE } else { HIGH_PROFILE };

    let mut body = Vec::new();
    for &lp in logprobs {
        body.extend_from_slice(&sse_frame(lp));
    }
    body.extend_from_slice(b"data: [DONE]\n\n");

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .body(Body::from(body))
        .unwrap()
}

async fn healthz() -> &'static str {
    "ok"
}

#[tokio::main]
async fn main() {
    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/healthz", get(healthz));

    let addr = std::env::var("MOCK_LISTEN_ADDR").unwrap_or_else(|_| "0.0.0.0:8080".to_string());
    println!("mock-upstream listening on {addr}");
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
