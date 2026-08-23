//! Concurrent-stream throughput benchmark for the rag-gate proxy.
//!
//! Measures the one number the README admits was never measured: how the
//! proxy behaves under many simultaneous streams. A mock SSE upstream (fixed
//! inter-token delay, configurable logprobs) runs in-process; the actual
//! release binary runs as a subprocess pointed at it. For each concurrency
//! level we run a direct-to-upstream baseline phase and a proxied phase, then
//! report per-stream p50/p99/p999, wall-clock token throughput, and
//! (Windows only, best-effort) the proxy's resident memory.
//!
//! Usage:
//!   cargo build --release
//!   cargo run --release --example throughput_bench -- [--streams 10 50 200]
//!                                                   [--tokens 20]
//!                                                   [--token-delay-ms 15]

use axum::body::Body;
use axum::http::{header, StatusCode};
use axum::response::Response;
use axum::routing::post;
use axum::Router;
use bytes::Bytes;
use futures_util::StreamExt;
use std::sync::LazyLock;
use std::time::{Duration, Instant};
use tokio::process::{Child, Command};

static TOKENS_PER_STREAM: LazyLock<usize> = LazyLock::new(|| parse_flag("tokens", 20));
static TOKEN_DELAY_MS: LazyLock<u64> = LazyLock::new(|| parse_flag("token-delay-ms", 15));

fn parse_flag<T: std::str::FromStr>(name: &str, default: T) -> T {
    let args: Vec<String> = std::env::args().collect();
    if let Some(pos) = args.iter().position(|a| *a == format!("--{name}")) {
        if let Some(v) = args.get(pos + 1) {
            if let Ok(parsed) = v.parse() {
                return parsed;
            }
        }
    }
    default
}

fn parse_streams() -> Vec<usize> {
    let args: Vec<String> = std::env::args().collect();
    if let Some(pos) = args.iter().position(|a| a == "--streams") {
        let rest: Vec<usize> = args[pos + 1..]
            .iter()
            .map(|s| s.trim_end_matches("--").to_string())
            .take_while(|s| !s.starts_with("--"))
            .filter_map(|s| s.parse().ok())
            .collect();
        if !rest.is_empty() {
            return rest;
        }
    }
    vec![10, 50, 200]
}

/// Mock OpenAI-compatible SSE upstream: streams `tokens` frames at a fixed
/// inter-token delay with a constant, comfortably-confident logprob (-0.2),
/// then `[DONE]`. Delay is what dominates wall time; the proxy's own work is
/// what the added-latency columns isolate.
async fn mock_chat_completions() -> Response {
    let tokens = *TOKENS_PER_STREAM;
    let delay = Duration::from_millis(*TOKEN_DELAY_MS);
    let stream = async_stream::stream! {
        for _ in 0..tokens {
            yield Ok::<_, std::io::Error>(Bytes::from(
                "data: {\"choices\":[{\"delta\":{\"content\":\"token \"},\"logprobs\":{\"content\":[{\"logprob\":-0.2}]}}]}\n\n",
            ));
            tokio::time::sleep(delay).await;
        }
        yield Ok(Bytes::from_static(b"data: [DONE]\n\n"));
    };
    let mut response = Response::new(Body::from_stream(stream));
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        header::HeaderValue::from_static("text/event-stream"),
    );
    response
}

async fn spawn_mock_upstream() -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = Router::new().route("/v1/chat/completions", post(mock_chat_completions));
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

async fn spawn_proxy(upstream_url: &str) -> (String, Child) {
    let port = free_port();
    let addr = format!("127.0.0.1:{port}");
    let exe = if cfg!(windows) {
        "target\\release\\rag-gate.exe"
    } else {
        "target/release/rag-gate"
    };
    let child = Command::new(exe)
        .env("RAGGATE_LISTEN_ADDR", &addr)
        .env("RAGGATE_UPSTREAM_URL", upstream_url)
        .env("RAGGATE_ANSWER_ALPHA", "-0.5")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("failed to spawn proxy — run `cargo build --release` first");

    // Wait for /healthz so the load phase doesn't hit a not-yet-bound socket.
    let client = reqwest::Client::new();
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Ok(resp) = client
            .get(format!("http://{addr}/healthz"))
            .timeout(Duration::from_secs(1))
            .send()
            .await
        {
            if resp.status() == StatusCode::OK {
                break;
            }
        }
        assert!(Instant::now() < deadline, "proxy did not become healthy in 10s");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    (format!("http://{addr}"), child)
}

/// Runs `streams` concurrent full-stream requests against `base`, returning
/// (per-stream total-ms sorted ascending, wall-clock seconds for the phase).
async fn run_phase(client: &reqwest::Client, base: &str, streams: usize) -> (Vec<f64>, f64) {
    let body = r#"{"model":"mock","messages":[{"role":"user","content":"benchmark"}]}"#;
    let wall_start = Instant::now();
    let mut handles = Vec::with_capacity(streams);
    for _ in 0..streams {
        let client = client.clone();
        let url = format!("{base}/v1/chat/completions");
        let body = body.to_string();
        handles.push(tokio::spawn(async move {
            let t0 = Instant::now();
            let resp = client
                .post(&url)
                .header(header::CONTENT_TYPE, "application/json")
                .body(body)
                .send()
                .await
                .expect("request failed");
            assert_eq!(resp.status(), StatusCode::OK, "upstream/proxy error");
            let mut stream = resp.bytes_stream();
            while stream.next().await.transpose().expect("chunk error").is_some() {}
            t0.elapsed().as_secs_f64() * 1000.0
        }));
    }
    let mut times = Vec::with_capacity(streams);
    for handle in handles {
        times.push(handle.await.expect("task panicked"));
    }
    let wall = wall_start.elapsed().as_secs_f64();
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (times, wall)
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Best-effort RSS of the proxy subprocess, Windows-only (tasklist CSV).
/// Fields are quoted, and the memory value itself contains a thousands
/// separator comma, so split on the `","` delimiter rather than ','.
fn proxy_rss_kb(pid: u32) -> Option<u64> {
    if !cfg!(windows) {
        return None;
    }
    let out = std::process::Command::new("tasklist")
        .args(["/FI", &format!("PID eq {pid}"), "/FO", "CSV", "/NH"])
        .output()
        .ok()?;
    let line = String::from_utf8_lossy(&out.stdout);
    let mem_field = line.split("\",\"").nth(4)?.trim().trim_matches('"');
    let kb: u64 = mem_field
        .trim_end_matches("K")
        .trim()
        .replace(',', "")
        .parse()
        .ok()?;
    Some(kb)
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let levels = parse_streams();
    let tokens = *TOKENS_PER_STREAM;
    let delay_ms = *TOKEN_DELAY_MS;

    let upstream_url = spawn_mock_upstream().await;
    let (proxy_url, mut child) = spawn_proxy(&upstream_url).await;
    let client = reqwest::Client::builder()
        .pool_max_idle_per_host(256)
        .build()
        .unwrap();

    println!(
        "rag-gate throughput benchmark: {} tokens/stream, {} ms inter-token delay",
        tokens, delay_ms
    );
    println!(
        "{:>7} | {:>18} | {:>18} | {:>16} | {:>10} | {:>10} | {:>10}",
        "streams",
        "direct p50/p99 ms", "proxy p50/p99 ms", "added p50/p99 ms",
        "tok/s dir", "tok/s prx", "proxy RSS"
    );
    println!("{}", "-".repeat(105));

    for level in levels {
        // Warm up pools and the proxy's first-connection path before timing.
        run_phase(&client, &upstream_url, level.min(4)).await;
        run_phase(&client, &proxy_url, level.min(4)).await;

        // Three interleaved repetitions per level; keep each side's
        // best-wall rep so a stray scheduler hiccup in one phase doesn't
        // masquerade as (negative) proxy overhead. The direct phase runs in
        // this same process as the mock upstream, so the noise floor on a
        // busy machine is real — read the added columns as bounded-by-noise,
        // not exact.
        let mut best_direct: Option<(Vec<f64>, f64)> = None;
        let mut best_proxy: Option<(Vec<f64>, f64)> = None;
        for _ in 0..3 {
            let d = run_phase(&client, &upstream_url, level).await;
            if best_direct.as_ref().is_none_or(|(_, w)| d.1 < *w) {
                best_direct = Some(d);
            }
            let p = run_phase(&client, &proxy_url, level).await;
            if best_proxy.as_ref().is_none_or(|(_, w)| p.1 < *w) {
                best_proxy = Some(p);
            }
        }
        let (direct, direct_wall) = best_direct.unwrap();
        let (proxied, proxy_wall) = best_proxy.unwrap();

        let d_p50 = percentile(&direct, 0.50);
        let d_p99 = percentile(&direct, 0.99);
        let p_p50 = percentile(&proxied, 0.50);
        let p_p99 = percentile(&proxied, 0.99);
        let direct_tps = (level * tokens) as f64 / direct_wall;
        let proxy_tps = (level * tokens) as f64 / proxy_wall;
        let rss = proxy_rss_kb(child.id().unwrap_or(0))
            .map(|kb| format!("{:.1} MB", kb as f64 / 1024.0))
            .unwrap_or_else(|| "n/a".to_string());

        println!(
            "{:>7} | {:>8.1} / {:<8.1} | {:>8.1} / {:<8.1} | {:>7.1} / {:<7.1} | {:>10.0} | {:>10.0} | {:>10}",
            level, d_p50, d_p99, p_p50, p_p99,
            p_p50 - d_p50, p_p99 - d_p99,
            direct_tps, proxy_tps, rss,
        );
    }

    let _ = child.kill().await;
}
