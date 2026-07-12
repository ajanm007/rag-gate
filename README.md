# rag-gate

A lightweight, asynchronous Rust proxy that intercepts OpenAI-compatible LLM API streams, evaluates token-level logprob confidence in real time, and gates generation output into one of three decisions: **ANSWER**, **ABSTAIN**, or **ESCALATE**.

It uses a signal that's already computed for free during inference — token logprobs — as a real-time confidence gate, instead of a post-hoc LLM judge (expensive, slow) or a retrieval-score threshold (cheap, but empirically flat with model uncertainty).

It is not a model, and not a RAG framework. It's a wire-level proxy with one job: decide if the model is confident enough to trust.

## Status

Early / pre-release. The core proxy, confidence evaluator, and calibration endpoint are implemented and tested (including live end-to-end verification against a mock upstream). It has **not** yet been tested against a real OpenAI or Ollama endpoint, and has no published crate. Treat it as a working prototype, not a production dependency.

## Quick start

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

Evaluation happens incrementally per SSE chunk, with a small lookahead buffer so tokens aren't forwarded before the first confidence check has a chance to fire.

Default thresholds: `alpha = -0.5`, `beta = -1.2`.

## Configuration

Via `rag-gate.toml` in the working directory, or environment variables (env vars win):

```toml
[proxy]
listen_addr = "0.0.0.0:8080"
upstream_url = "https://api.openai.com"

[thresholds]
answer_alpha = -0.5
abstain_beta = -1.2
```

| Env var | Overrides |
|---|---|
| `RAGGATE_CONFIG` | Path to the TOML config file (default: `rag-gate.toml`) |
| `RAGGATE_LISTEN_ADDR` | `proxy.listen_addr` |
| `RAGGATE_UPSTREAM_URL` | `proxy.upstream_url` |
| `RAGGATE_ANSWER_ALPHA` | `thresholds.answer_alpha` |
| `RAGGATE_ABSTAIN_BETA` | `thresholds.abstain_beta` |

The `Authorization` header on incoming requests is forwarded to the upstream as-is; `rag-gate` never stores or logs API keys.

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

## Metrics

`GET /metrics` exposes Prometheus-format metrics:

| Metric | Type | Description |
|---|---|---|
| `raggate_requests_total` | Counter | Total requests proxied |
| `raggate_decisions_total{decision}` | Counter | ANSWER / ABSTAIN / ESCALATE counts |
| `raggate_confidence_score` | Histogram | Distribution of confidence scores |
| `raggate_tokens_evaluated` | Histogram | Tokens evaluated before decision |
| `raggate_proxy_latency_ms` | Histogram | Added latency vs. direct API call |

## Known limitations

- Only OpenAI-compatible `/v1/chat/completions` is supported today. An Ollama `/api/chat` adapter is planned but not built.
- Escalation routing (automatic retry to a fallback model on ESCALATE) is not yet implemented — the client currently has to handle that itself.
- The SSE chunk parser assumes each stream poll yields one complete JSON frame; it hasn't been stress-tested against real-world TCP fragmentation.
- No published crate yet.

## Development

```bash
cargo build
cargo test
```

Requires a working Rust toolchain with a linker (MSVC or the MinGW-w64/GNU toolchain on Windows).
