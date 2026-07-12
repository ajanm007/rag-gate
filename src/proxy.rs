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
use crate::interceptor::InterceptedStream;
use crate::metrics::REQUESTS_TOTAL;

#[derive(Clone)]
pub struct AppState {
    pub config: ProxyConfig,
    pub http_client: Client,
}

pub fn create_router(config: ProxyConfig) -> Router {
    let state = AppState {
        config,
        http_client: Client::new(),
    };

    Router::new()
        .route("/v1/chat/completions", post(chat_completions_handler))
        .with_state(state)
}

async fn chat_completions_handler(
    State(state): State<AppState>,
    req: Request<Body>,
) -> Response {
    let upstream_url = format!("{}/v1/chat/completions", state.config.upstream_url);

    // Forward headers
    let mut headers = header::HeaderMap::new();
    if let Some(auth) = req.headers().get(header::AUTHORIZATION) {
        headers.insert(header::AUTHORIZATION, auth.clone());
    }
    if let Some(content_type) = req.headers().get(header::CONTENT_TYPE) {
        headers.insert(header::CONTENT_TYPE, content_type.clone());
    }
    
    let body_bytes = match axum::body::to_bytes(req.into_body(), usize::MAX).await {
        Ok(b) => b,
        Err(e) => {
            error!("Failed to read request body: {}", e);
            return axum::http::StatusCode::BAD_REQUEST.into_response();
        }
    };

    // Force `stream: true` and `logprobs: true` so the interceptor always has
    // token-level logprobs to evaluate, regardless of what the client sent.
    let body_bytes = match serde_json::from_slice::<serde_json::Value>(&body_bytes) {
        Ok(serde_json::Value::Object(mut map)) => {
            map.insert("stream".to_string(), serde_json::Value::Bool(true));
            map.insert("logprobs".to_string(), serde_json::Value::Bool(true));
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

    // Forward the response stream through our interceptor
    let byte_stream = res.bytes_stream().map(|r| r.map_err(|e| axum::Error::new(e)));
    
    let evaluator = ConfidenceEvaluator::new(state.config.thresholds.clone());
    let intercepted = InterceptedStream::new(byte_stream, evaluator, 4);

    let body = Body::from_stream(intercepted);
    
    let mut response = Response::new(body);
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        header::HeaderValue::from_static("text/event-stream"),
    );
    response
}
