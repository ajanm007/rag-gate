use axum::{
    body::Body,
    extract::{Request, State},
    response::{IntoResponse, Response},
    routing::post,
    Router,
};
use bytes::Bytes;
use reqwest::{Client, Method, Url, header};
use tracing::error;
use futures_util::StreamExt;

use crate::config::ProxyConfig;
use crate::evaluator::ConfidenceEvaluator;
use crate::interceptor::{InterceptedStream, Protocol};
use crate::metrics::{PROXY_LATENCY_MS, REQUESTS_TOTAL};
use std::net::IpAddr;
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
        // Gemini native streaming endpoints, both surfaces. The patterns
        // capture the `{model}:streamGenerateContent` segment as one
        // parameter (the colon lives in the request path, not the pattern)
        // and the original path is forwarded verbatim, so both AI Studio
        // (`/v1beta/models/...`) and Vertex AI (`/v1beta1/projects/...`)
        // shapes work. As of 2026-08-22 logprobs are ENABLED on Vertex AI
        // and disabled on every AI Studio model (probed live — see
        // benchmarks/gemini_logprob_probe.py), so gating is live for Vertex
        // upstreams and auto-activates for AI Studio if Google flips it.
        .route("/v1beta/models/:target", post(gemini_stream_handler))
        .route(
            "/v1beta1/projects/:project/locations/:location/publishers/google/models/:target",
            post(gemini_stream_handler),
        )
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
    proxy_stream(state, req, "/v1/chat/completions", None, Protocol::Sse).await
}

async fn ollama_chat_handler(State(state): State<AppState>, req: Request<Body>) -> Response {
    proxy_stream(state, req, "/api/chat", None, Protocol::Ndjson).await
}

/// Gemini native streaming (AI Studio and Vertex path shapes). The original
/// request path is forwarded verbatim — both `/v1beta/models/{model}:…` and
/// Vertex's `/v1beta1/projects/…/models/{model}:…` are already correct Gemini
/// paths. Non-streaming `generateContent` is rejected: rag-gate is
/// stream-first and cannot gate a JSON-array response. Auth is the client's
/// job — `x-goog-api-key` for AI Studio, `Authorization: Bearer` (OAuth) for
/// Vertex — forwarded as-is like every other header.
async fn gemini_stream_handler(State(state): State<AppState>, req: Request<Body>) -> Response {
    let path = req.uri().path().to_string();
    let Some((_, action)) = path.rsplit_once(':') else {
        return gemini_error(
            axum::http::StatusCode::NOT_FOUND,
            "expected a Gemini path ending in :streamGenerateContent",
        );
    };
    if action != "streamGenerateContent" {
        return gemini_error(
            axum::http::StatusCode::NOT_IMPLEMENTED,
            &format!("rag-gate gates streams only; ':{}' is not supported", action),
        );
    }

    // Forward the client's query string, defaulting alt=sse — without it
    // Gemini returns one JSON array, not SSE frames, and there is nothing to
    // intercept.
    let mut query = req.uri().query().unwrap_or("").to_string();
    if !query.split('&').any(|p| p.starts_with("alt=")) {
        query = if query.is_empty() {
            "alt=sse".to_string()
        } else {
            format!("{}&alt=sse", query)
        };
    }

    proxy_stream(state, req, &path, Some(query), Protocol::Gemini).await
}

fn gemini_error(status: axum::http::StatusCode, msg: &str) -> Response {
    (status, msg.to_string()).into_response()
}

fn is_private_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            v4.is_private()
                || v4.is_loopback()
                || v4.is_link_local()
                || v4.is_multicast()
                || v4.is_broadcast()
                || v4.is_documentation()
                || v4.is_unspecified()
        }
        IpAddr::V6(v6) => {
            v6.is_loopback()
                || v6.is_unspecified()
                || v6.is_unique_local()
                || v6.is_multicast()
                || v6.is_unicast_link_local()
        }
    }
}

fn is_disallowed_host(host: &str) -> bool {
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }

    if let Ok(ip) = host.parse::<IpAddr>() {
        return is_private_ip(ip);
    }

    false
}

fn is_safe_upstream_base_url(url: &Url) -> bool {
    match url.scheme() {
        "http" | "https" => {}
        _ => return false,
    }

    if url.host_str().is_none() {
        return false;
    }

    if !url.username().is_empty() || url.password().is_some() {
        return false;
    }

    if let Some(host) = url.host_str() {
        if is_disallowed_host(host) {
            return false;
        }
    }

    true
}

/// Shared proxy path for all upstream flavors. `path` (plus optional `query`)
/// is appended to the configured `upstream_url`; `protocol` selects the
/// interceptor's framing/extraction and which streaming/logprobs fields get
/// injected.
async fn proxy_stream(
    state: AppState,
    req: Request<Body>,
    path: &str,
    query: Option<String>,
    protocol: Protocol,
) -> Response {
    let request_start = Instant::now();
    let base_url = match Url::parse(&state.config.upstream_url) {
        Ok(u) => u,
        Err(_) => {
            return (
                axum::http::StatusCode::BAD_GATEWAY,
                "invalid upstream_url configuration",
            )
                .into_response();
        }
    };

    if !is_safe_upstream_base_url(&base_url) {
        return (
            axum::http::StatusCode::BAD_GATEWAY,
            "unsafe upstream_url configuration",
        )
            .into_response();
    }

    let mut upstream_url = base_url.clone();
    let mut normalized = base_url.path().trim_end_matches('/').to_string();
    normalized.push('/');
    normalized.push_str(path.trim_start_matches('/'));
    upstream_url.set_path(&normalized);

    match query.as_deref() {
        Some(q) if !q.is_empty() => upstream_url.set_query(Some(q)),
        _ => upstream_url.set_query(None),
    }

    if upstream_url.scheme() != base_url.scheme()
        || upstream_url.host_str() != base_url.host_str()
        || upstream_url.port_or_known_default() != base_url.port_or_known_default()
    {
        return (
            axum::http::StatusCode::BAD_GATEWAY,
            "upstream URL validation failed",
        )
            .into_response();
    }

    let headers = forward_headers(req.headers());

    let body_bytes = match axum::body::to_bytes(req.into_body(), state.config.max_body_bytes).await {
        Ok(b) => b,
        Err(e) => {
            error!("Failed to read request body (limit {} bytes): {}", state.config.max_body_bytes, e);
            return axum::http::StatusCode::PAYLOAD_TOO_LARGE.into_response();
        }
    };

    // Force streaming on, and (unless disabled) the provider's logprobs flag,
    // so the interceptor always has token-level logprobs to evaluate when the
    // upstream can produce them. OpenAI-style: `stream` + `logprobs` top-level
    // fields (Ollama's `/api/chat` has no `logprobs` field, so only `stream`).
    // Gemini native: streaming is implied by the `:streamGenerateContent`
    // endpoint and the toggle is `generationConfig.responseLogprobs`.
    //
    // `logprobs_requested` tracks whether this request actually asked the
    // upstream for logprobs (via injection, or because the client's own body
    // already had the field) — passed to the interceptor so it can tell "we
    // asked and got nothing" (a silently unsupported upstream, worth warning
    // about) apart from "we never asked" (expected, e.g. `inject_logprobs =
    // false`, or Ollama which has no such field at all).
    let mut logprobs_requested = false;
    let body_bytes = match serde_json::from_slice::<serde_json::Value>(&body_bytes) {
        Ok(serde_json::Value::Object(mut map)) => {
            match protocol {
                Protocol::Gemini => {
                    if state.config.inject_logprobs {
                        let gc = map
                            .entry("generationConfig".to_string())
                            .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
                        if let Some(obj) = gc.as_object_mut() {
                            obj.insert(
                                "responseLogprobs".to_string(),
                                serde_json::Value::Bool(true),
                            );
                        }
                        logprobs_requested = true;
                    } else {
                        logprobs_requested = map
                            .get("generationConfig")
                            .and_then(|gc| gc.get("responseLogprobs"))
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false);
                    }
                }
                Protocol::Sse => {
                    map.insert("stream".to_string(), serde_json::Value::Bool(true));
                    if state.config.inject_logprobs {
                        map.insert("logprobs".to_string(), serde_json::Value::Bool(true));
                        logprobs_requested = true;
                    } else {
                        logprobs_requested = map
                            .get("logprobs")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false);
                    }
                }
                Protocol::Ndjson => {
                    map.insert("stream".to_string(), serde_json::Value::Bool(true));
                }
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
        .request(Method::POST, upstream_url)
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
    let intercepted = InterceptedStream::new_with_protocol_and_request(
        byte_stream,
        evaluator,
        4,
        protocol,
        logprobs_requested,
    );

    let content_type = match protocol {
        Protocol::Sse | Protocol::Gemini => "text/event-stream",
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
