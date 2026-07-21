use axum::{
    body::Body,
    extract::{Request, State},
    response::{IntoResponse, Response},
    routing::post,
    Router,
};
use bytes::Bytes;
use reqwest::{Client, Method, header};
use tracing::error;
use futures_util::StreamExt;

use crate::config::ProxyConfig;
use crate::evaluator::ConfidenceEvaluator;
use crate::interceptor::{InterceptedStream, Protocol};
use crate::metrics::{PROXY_LATENCY_MS, REQUESTS_TOTAL};
use std::time::{Duration, Instant};

#[derive(Clone)]
pub struct AppState {
    pub config: ProxyConfig,
    pub http_client: Client,
}

pub fn create_router(config: ProxyConfig) -> Router {
    // A client with an explicit connect timeout so a dead/hung upstream can't
    // wedge a request forever. No overall/read timeout: streaming responses are
    // intentionally long-lived, so bounding total duration would truncate valid
    // long completions.
    let http_client = Client::builder()
        .connect_timeout(Duration::from_secs(config.connect_timeout_secs))
        .build()
        .unwrap_or_else(|_| Client::new());

    let state = AppState { config, http_client };

    Router::new()
        // OpenAI-compatible SSE endpoint.
        .route("/v1/chat/completions", post(openai_chat_handler))
        // Ollama native NDJSON endpoint. Ollama does not currently return
        // per-token logprobs (issue #16117, closed as not planned; #13638), so
        // this path proxies transparently and the confidence gate no-ops until
        // logprobs appear on the wire — see the interceptor's NDJSON parser.
        .route("/api/chat", post(ollama_chat_handler))
        .with_state(state)
}

/// Hop-by-hop headers that must NOT be forwarded to the upstream (RFC 7230 §6.1),
/// plus `host`/`content-length` which reqwest sets itself for the new request.
fn is_hop_by_hop(name: &header::HeaderName) -> bool {
    matches!(
        name.as_str(),
        "connection"
            | "keep-alive"
            | "proxy-authenticate"
            | "proxy-authorization"
            | "te"
            | "trailer"
            | "transfer-encoding"
            | "upgrade"
            | "host"
            | "content-length"
    )
}

/// Copies all end-to-end request headers through to the upstream. The previous
/// implementation forwarded only Authorization + Content-Type, which silently
/// dropped provider-specific auth headers (Azure `api-key`, Anthropic-compat
/// `x-api-key`, `OpenAI-Organization`, etc.) and produced spurious 401s.
fn forward_headers(src: &header::HeaderMap) -> header::HeaderMap {
    let mut headers = header::HeaderMap::new();
    for (name, value) in src.iter() {
        if !is_hop_by_hop(name) {
            headers.insert(name.clone(), value.clone());
        }
    }
    headers
}

async fn openai_chat_handler(State(state): State<AppState>, req: Request<Body>) -> Response {
    proxy_stream(state, req, "/v1/chat/completions", Protocol::Sse).await
}

async fn ollama_chat_handler(State(state): State<AppState>, req: Request<Body>) -> Response {
    proxy_stream(state, req, "/api/chat", Protocol::Ndjson).await
}

/// Shared proxy path for both upstream flavors. `path` is appended to the
/// configured `upstream_url`; `protocol` selects SSE vs NDJSON framing for the
/// interceptor and for the injected streaming flag.
async fn proxy_stream(
    state: AppState,
    req: Request<Body>,
    path: &str,
    protocol: Protocol,
) -> Response {
    let request_start = Instant::now();
    let upstream_url = format!("{}{}", state.config.upstream_url, path);

    let headers = forward_headers(req.headers());

    let body_bytes = match axum::body::to_bytes(req.into_body(), state.config.max_body_bytes).await {
        Ok(b) => b,
        Err(e) => {
            error!("Failed to read request body (limit {} bytes): {}", state.config.max_body_bytes, e);
            return axum::http::StatusCode::PAYLOAD_TOO_LARGE.into_response();
        }
    };

    // Force streaming on, and (for OpenAI-style upstreams, unless disabled)
    // `logprobs: true` so the interceptor always has token-level logprobs to
    // evaluate. Ollama's `/api/chat` has no `logprobs` field, so we only ensure
    // `stream: true` there.
    let body_bytes = match serde_json::from_slice::<serde_json::Value>(&body_bytes) {
        Ok(serde_json::Value::Object(mut map)) => {
            map.insert("stream".to_string(), serde_json::Value::Bool(true));
            if protocol == Protocol::Sse && state.config.inject_logprobs {
                map.insert("logprobs".to_string(), serde_json::Value::Bool(true));
            }
            match serde_json::to_vec(&serde_json::Value::Object(map)) {
                Ok(bytes) => Bytes::from(bytes),
                Err(_) => body_bytes,
            }
        }
        _ => body_bytes,
    };

    REQUESTS_TOTAL.inc();

    let res = match state
        .http_client
        .request(Method::POST, &upstream_url)
        .headers(headers)
        .body(body_bytes)
        .send()
        .await
    {
        Ok(res) => res,
        Err(e) => {
            error!("Upstream request failed: {}", e);
            return axum::http::StatusCode::BAD_GATEWAY.into_response();
        }
    };

    // Preserve the upstream status so client-visible errors (401, 429, 400 …)
    // aren't masked as a 200 stream of an error body.
    let status = res.status();

    // Time from receiving the client request to getting the first response
    // header back from upstream. Includes the real network round-trip to
    // upstream, so it is NOT an isolated measurement of rag-gate's own overhead.
    PROXY_LATENCY_MS.observe(request_start.elapsed().as_secs_f64() * 1000.0);

    let byte_stream = res.bytes_stream().map(|r| r.map_err(axum::Error::new));

    let evaluator = ConfidenceEvaluator::new(state.config.thresholds.clone());
    let intercepted = InterceptedStream::new_with_protocol(byte_stream, evaluator, 4, protocol);

    let content_type = match protocol {
        Protocol::Sse => "text/event-stream",
        Protocol::Ndjson => "application/x-ndjson",
    };

    let body = Body::from_stream(intercepted);
    let mut response = Response::new(body);
    *response.status_mut() = status;
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        header::HeaderValue::from_static(content_type),
    );
    response
}
