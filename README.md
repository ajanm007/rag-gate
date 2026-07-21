# rag-gate

A lightweight, asynchronous Rust proxy that intercepts OpenAI-compatible LLM API streams, evaluates token-level logprob confidence in real time, and gates generation output into one of three decisions: **ANSWER**, **ABSTAIN**, or **ESCALATE**.

It uses a signal that's already computed for free during inference token logprobs as a real-time confidence gate, instead of a post-hoc LLM judge (expensive, slow) or a retrieval-score threshold (cheap, but empirically flat with model uncertainty).

It is not a model, and not a RAG framework. It's a wire-level proxy with one job: decide if the model is confident enough to trust.

## Status

Pre-1.0, but functionally verified. The core proxy, confidence evaluator, calibration endpoint, and stream chunk reassembly (see Known limitations) are implemented and tested including live end-to-end verification against both a mock upstream and a real OpenAI-compatible endpoint (xAI's Grok API). An Ollama `/api/chat` transport (NDJSON) is implemented and tested end-to-end against a mock Ollama upstream — but see Known limitations for why confidence gating is inert against Ollama today. Published on crates.io as `rag-gate`.

## Quick start

Install the binary from crates.io:

```bash
cargo install rag-gate
RAGGATE_UPSTREAM_URL=https://api.openai.com rag-gate
```

Or build from source:

```bash
cargo build --release
RAGGATE_UPSTREAM_URL=https://api.openai.com ./target/release/rag-gate
```

Point your client at `rag-gate` instead of the upstream directly:

```bash
curl -N -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "..."}]}'
```

`rag-gate` forwards the request upstream (auto-injecting `stream: true` and `logprobs: true`), streams the response back token by token, and — if confidence drops below threshold mid-stream — cuts the stream short and emits a decision frame instead of letting a low-confidence answer reach the client:

```json
{"rag_gate_decision": "ABSTAIN", "confidence_score": -1.47, "tokens_evaluated": 23, "threshold_used": -1.2}
```

## How it works

Confidence is the mean token logprob over the generated sequence so far:

```
confidence(tokens) = (1 / N) * Σ logprob(token_i)
```

```
confidence >= alpha        -> ANSWER    (stream continues normally)
beta <= confidence < alpha -> ESCALATE  (cut stream, emit decision frame)
confidence < beta          -> ABSTAIN   (cut stream, emit decision frame)
```

Evaluation happens incrementally per stream chunk, with a small lookahead buffer so tokens aren't forwarded before the first confidence check has a chance to fire.

A **warmup floor** (`min_tokens`, default 4) suppresses any ABSTAIN/ESCALATE until at least that many tokens have been evaluated: the mean over one or two tokens is too noisy to act on — a single low-probability opening token ("Well", "Hmm") shouldn't cut an answer that would have recovered. Set `min_tokens = 1` to disable.

Default thresholds: `alpha = -0.5`, `beta = -1.2`, `min_tokens = 4`.

## Configuration

Via `rag-gate.toml` in the working directory, or environment variables (env vars win):

```toml
[proxy]
listen_addr = "0.0.0.0:8080"
upstream_url = "https://api.openai.com"

inject_logprobs = true
max_body_bytes = 2097152
connect_timeout_secs = 10

[thresholds]
answer_alpha = -0.5
abstain_beta = -1.2
min_tokens = 4
```

| Env var | Overrides |
| --- | --- |
| `RAGGATE_CONFIG` | Path to the TOML config file (default: `rag-gate.toml`) |
| `RAGGATE_LISTEN_ADDR` | `proxy.listen_addr` |
| `RAGGATE_UPSTREAM_URL` | `proxy.upstream_url` |
| `RAGGATE_ANSWER_ALPHA` | `thresholds.answer_alpha` |
| `RAGGATE_ABSTAIN_BETA` | `thresholds.abstain_beta` |
| `RAGGATE_MIN_TOKENS` | `thresholds.min_tokens` |
| `RAGGATE_INJECT_LOGPROBS` | `inject_logprobs` — set `false` for upstreams that reject `logprobs: true` (e.g. Gemini) |
| `RAGGATE_MAX_BODY_BYTES` | `max_body_bytes` — client request body cap (default 2 MiB) |
| `RAGGATE_CONNECT_TIMEOUT_SECS` | `connect_timeout_secs` — upstream connect timeout (default 10s) |

All incoming request headers except hop-by-hop headers (`Connection`, `Transfer-Encoding`, `Host`, etc.) are forwarded to the upstream as-is — so provider-specific auth headers like Azure's `api-key`, Anthropic-compat's `x-api-key`, and `OpenAI-Organization` pass through, not just `Authorization`. `rag-gate` never stores or logs API keys.

## Calibration

`POST /v1/rag-gate/calibrate` takes labeled samples (mean logprobs + correctness) and a target coverage, and returns threshold values that satisfy that coverage at (approximately) minimum risk, computed from an actual risk-coverage sweep with trapezoidal AURC integration:

```bash
curl -X POST http://127.0.0.1:8080/v1/rag-gate/calibrate \
  -H "Content-Type: application/json" \
  -d '{
    "target_coverage": 0.8,
    "samples": [
      {"correct": true, "logprobs": [-0.1, -0.2, -0.1]},
      {"correct": false, "logprobs": [-1.5, -2.2, -1.8]}
    ]
  }'
```

```json
{
  "optimal_alpha": -0.48,
  "optimal_beta": -1.15,
  "aurc_at_target_coverage": 0.143,
  "abstain_rate": 0.18,
  "escalation_rate": 0.06
}
```

See `test_calibrate.py` for a runnable example.

## Performance

Measured added latency (rag-gate vs. calling the upstream directly), release build, local loopback against a 20-token streamed response, 200 requests:

| Percentile | Added overhead |
| --- | --- |
| p50 | \~0.2–0.4ms |
| p99 | \~3.6–4.0ms |

Both comfortably under the &lt;5ms p99 target. This measures rag-gate's own processing cost (SSE parsing, confidence evaluation, buffering) in isolation; real-world overhead will also include whatever network hop is introduced by routing through the proxy, which depends entirely on your deployment topology (same-host vs. remote).

## Metrics

`GET /metrics` exposes Prometheus-format metrics:

| Metric | Type | Description |
| --- | --- | --- |
| `raggate_requests_total` | Counter | Total requests proxied |
| `raggate_decisions_total{decision}` | Counter | ANSWER / ABSTAIN / ESCALATE counts |
| `raggate_confidence_score` | Histogram | Distribution of confidence scores |
| `raggate_tokens_evaluated` | Histogram | Tokens evaluated before decision |
| `raggate_token_savings_total` | Counter | Tokens saved by cutting the stream early on ABSTAIN/ESCALATE |
| `raggate_proxy_latency_ms` | Histogram | Added latency vs. direct API call |

`GET /healthz` returns `ok` for liveness/readiness probes (no auth, no upstream call). The server drains in-flight requests on `SIGINT`/`SIGTERM` before exiting.

## Known limitations

- **Ollama's native** `/api/chat` **(NDJSON) is supported as a transport** — `rag-gate` reassembles its newline-delimited stream correctly. Confidence gating, however, is inactive against Ollama: it returns no per-token logprobs on either `/api/chat` or its OpenAI-compat `/v1/chat/completions` layer — the compat request field is silently dropped, and the feature request to add it was [closed as not planned](https://github.com/ollama/ollama/issues/16117) (see also [#13638](https://github.com/ollama/ollama/issues/13638)). With no confidence signal on the wire, the gate no-ops and every Ollama response passes through as ANSWER. The transport is in place, so gating will activate automatically if Ollama ever emits logprobs in the OpenAI shape.
- Not every "OpenAI-compatible" API accepts the auto-injected `logprobs: true` field — Google's Gemini OpenAI-compat layer rejects it with a 400. Set `inject_logprobs = false` (or `RAGGATE_INJECT_LOGPROBS=false`) for those upstreams; gating then only fires if the client itself requests logprobs.
- **Confidence is temperature-dependent.** Mean token logprob rises toward 0 as sampling temperature drops (at temperature 0 the model always picks the argmax token, whose logprob is near 0). So a pipeline running the upstream at very low temperature will see near-perfect confidence on nearly everything and the gate will rarely fire — calibrate your thresholds at the temperature you actually serve at, and re-calibrate if you change it. (This also means "just retry at a lower temperature" is not a reliable recovery strategy — it inflates the confidence score without necessarily improving the answer; see `benchmarks/`.)
- Escalation routing (automatic retry/reroute to a fallback model on ESCALATE) is not yet implemented — the client currently has to handle that itself, and the evidence so far does not justify building it. Two recovery strategies were evaluated (see `benchmarks/`): a `lower_temperature` retry, which is *mechanistically* disqualified because low temperature inflates the confidence metric without necessarily improving the answer; and **rerouting to a stronger model** (`grok-3-mini`→`grok-4.5`, 45-question low-confidence band), which moved band accuracy 35.6%→42.2% but with **6 fixes against 3 regressions — net +3, McNemar p = 0.51, not significant**. Rerouting is not a free win (it also breaks answers), so neither strategy is currently shipped.
- The stream chunk parser reassembles both SSE events (`\n\n`) and NDJSON lines (`\n`) split across TCP/HTTP chunk boundaries rather than assuming one poll equals one complete frame; covered by dedicated tests, but real-world traffic patterns are inherently broader than any test suite.
- Added-latency overhead has been benchmarked (see Performance) but concurrent-stream throughput has not.

## Development

```bash
cargo build
cargo test
```

Requires a working Rust toolchain with a linker (MSVC or the MinGW-w64/GNU toolchain on Windows).