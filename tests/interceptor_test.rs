use bytes::Bytes;
use futures_util::{stream, StreamExt};
use rag_gate::{ConfidenceEvaluator, GatingThresholds, InterceptedStream, Protocol};
use std::sync::{Mutex, OnceLock};

/// The Prometheus counters are process-global, so tests that assert on them
/// must not run concurrently with each other.
static METRIC_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

fn metric_lock() -> std::sync::MutexGuard<'static, ()> {
    METRIC_LOCK.get_or_init(|| Mutex::new(())).lock().unwrap_or_else(|p| p.into_inner())
}

fn thresholds() -> GatingThresholds {
    GatingThresholds {
        answer_alpha: -0.5,
        abstain_beta: -1.2,
        min_tokens: 1, // disable warmup floor so cut-behavior tests fire immediately
        degenerate_guard: true,
    }
}

fn sse_chunk(logprob: f64) -> Bytes {
    let json = format!(
        r#"{{"choices":[{{"logprobs":{{"content":[{{"logprob":{}}}]}}}}]}}"#,
        logprob
    );
    Bytes::from(format!("data: {}\n\n", json))
}

async fn run(chunks: Vec<Bytes>) -> Vec<Bytes> {
    let source = stream::iter(chunks.into_iter().map(Ok::<_, axum::Error>));
    let evaluator = ConfidenceEvaluator::new(thresholds());
    let intercepted = InterceptedStream::new(source, evaluator, 4);
    intercepted
        .map(|r| r.expect("stream item should not error"))
        .collect()
        .await
}

/// One Ollama `/api/chat` NDJSON line: bare JSON, no `data:` prefix, no
/// logprobs field (Ollama does not emit them), terminated by a single `\n`.
fn ndjson_chunk(content: &str, done: bool) -> Bytes {
    let json = format!(
        r#"{{"model":"llama3.2","message":{{"role":"assistant","content":"{}"}},"done":{}}}"#,
        content, done
    );
    Bytes::from(format!("{}\n", json))
}

async fn run_ndjson(chunks: Vec<Bytes>) -> Vec<Bytes> {
    let source = stream::iter(chunks.into_iter().map(Ok::<_, axum::Error>));
    let evaluator = ConfidenceEvaluator::new(thresholds());
    let intercepted = InterceptedStream::new_with_protocol(source, evaluator, 4, Protocol::Ndjson);
    intercepted
        .map(|r| r.expect("stream item should not error"))
        .collect()
        .await
}

/// One Gemini native `streamGenerateContent?alt=sse` SSE frame. With
/// logprobs enabled (Vertex AI today), the chosen token's log probability is
/// at `candidates[0].logprobsResult.chosenCandidates[].logProbability`.
fn gemini_chunk(logprob: Option<f64>) -> Bytes {
    let lp = match logprob {
        Some(v) => format!(
            r#","logprobsResult":{{"chosenCandidates":[{{"token":"a","logProbability":{}}}]}}"#,
            v
        ),
        None => String::new(),
    };
    let json = format!(
        r#"{{"candidates":[{{"content":{{"parts":[{{"text":"a"}}]}}{}}}]}}"#,
        lp
    );
    Bytes::from(format!("data: {}\n\n", json))
}

async fn run_gemini(chunks: Vec<Bytes>) -> Vec<Bytes> {
    let source = stream::iter(chunks.into_iter().map(Ok::<_, axum::Error>));
    let evaluator = ConfidenceEvaluator::new(thresholds());
    let intercepted = InterceptedStream::new_with_protocol(source, evaluator, 4, Protocol::Gemini);
    intercepted
        .map(|r| r.expect("stream item should not error"))
        .collect()
        .await
}

#[tokio::test]
async fn ndjson_without_logprobs_passes_through_ungated() {
    // Ollama emits no logprobs, so the gate must no-op and forward every frame
    // verbatim rather than cut the stream on a phantom zero-confidence.
    let chunks = vec![
        ndjson_chunk("The", false),
        ndjson_chunk(" capital", false),
        ndjson_chunk(" is Paris.", false),
        ndjson_chunk("", true),
    ];
    let expected = chunks.len();
    let output = run_ndjson(chunks).await;
    assert_eq!(output.len(), expected);
    // No rag-gate decision frame should be injected when ungated.
    for frame in &output {
        let text = String::from_utf8_lossy(frame);
        assert!(!text.contains("rag_gate_decision"));
    }
}

#[tokio::test]
async fn ndjson_line_split_across_chunks_is_reassembled() {
    // A TCP fragment lands mid-line; the `\n`-delimited frame must be
    // reassembled and forwarded as one intact line, not broken JSON.
    let whole = ndjson_chunk("hello", false);
    let bytes = whole.to_vec();
    let a = bytes[..8].to_vec();
    let b = bytes[8..20].to_vec();
    let c = bytes[20..].to_vec();

    let output = run_ndjson(vec![
        Bytes::from(a),
        Bytes::from(b),
        Bytes::from(c),
    ])
    .await;

    assert_eq!(output.len(), 1);
    let text = String::from_utf8_lossy(&output[0]);
    assert!(text.contains(r#""content":"hello""#));
    assert!(text.ends_with('\n'));
}

#[tokio::test]
async fn high_confidence_stream_passes_through_all_chunks() {
    let chunks = vec![
        sse_chunk(-0.1),
        sse_chunk(-0.2),
        sse_chunk(-0.1),
        sse_chunk(-0.15),
        sse_chunk(-0.1),
        Bytes::from_static(b"data: [DONE]\n\n"),
    ];
    let expected_len = chunks.len();

    let output = run(chunks).await;

    // All chunks should flow through since confidence stays above alpha.
    assert_eq!(output.len(), expected_len);
}

#[tokio::test]
async fn low_confidence_stream_cuts_and_emits_decision_frame() {
    let chunks = vec![
        sse_chunk(-2.0),
        sse_chunk(-2.5),
        sse_chunk(-3.0),
    ];

    let output = run(chunks).await;

    // Stream should be cut short with a decision frame, not all 3 chunks passed through.
    assert!(!output.is_empty());
    let last = output.last().unwrap();
    let text = String::from_utf8_lossy(last);
    assert!(text.contains("ABSTAIN") || text.contains("ESCALATE"));
}

#[tokio::test]
async fn sse_event_split_across_chunks_is_reassembled() {
    // Simulate a TCP/HTTP fragment boundary landing mid-JSON: one complete
    // event, split into three raw pieces that don't align with the event
    // boundary at all.
    let whole = sse_chunk(-0.1);
    let bytes = whole.to_vec();
    let split_a = bytes[..10].to_vec();
    let split_b = bytes[10..20].to_vec();
    let split_c = bytes[20..].to_vec();

    let chunks = vec![
        Bytes::from(split_a),
        Bytes::from(split_b),
        Bytes::from(split_c),
        Bytes::from_static(b"data: [DONE]\n\n"),
    ];

    let output = run(chunks).await;

    // The fragments should have been reassembled into exactly one forwarded
    // event plus [DONE], not silently dropped or forwarded as broken partial
    // JSON.
    assert_eq!(output.len(), 2);
    let first = String::from_utf8_lossy(&output[0]);
    assert!(first.contains("\"logprob\":-0.1"));
}

#[tokio::test]
async fn stream_ending_without_trailing_blank_line_flushes_leftover() {
    // Upstream closes the connection right after the JSON body, without the
    // usual trailing "\n\n" — the leftover bytes must still be flushed
    // rather than silently discarded.
    let whole = sse_chunk(-0.1);
    let mut bytes = whole.to_vec();
    // Strip the trailing "\n\n" to simulate a connection close mid-frame.
    bytes.truncate(bytes.len() - 2);

    let chunks = vec![Bytes::from(bytes)];

    let output = run(chunks).await;

    assert_eq!(output.len(), 1);
    let text = String::from_utf8_lossy(&output[0]);
    assert!(text.contains("\"logprob\":-0.1"));
}

#[tokio::test]
async fn degenerate_greedy_stream_disables_gating_and_counts_metric() {
    // Every logprob ≈ 0 is the temperature-0/greedy-decoding artifact. The
    // gate must (a) notice it, (b) count it in raggate_degenerate_signal_total,
    // and (c) pass the whole stream through — a dead signal passing every
    // frame as ANSWER must at least be visible, never silent.
    use rag_gate::metrics::DEGENERATE_SIGNAL_TOTAL;

    let _guard = metric_lock();
    let before = DEGENERATE_SIGNAL_TOTAL.get();
    let chunks: Vec<Bytes> = (0..30).map(|_| sse_chunk(-0.0001)).collect();
    let output = run(chunks).await;

    assert_eq!(DEGENERATE_SIGNAL_TOTAL.get(), before + 1.0);
    assert_eq!(output.len(), 30); // no cut, no injected decision frame
    for frame in &output {
        let text = String::from_utf8_lossy(frame);
        assert!(!text.contains("rag_gate_decision"));
    }
}

#[tokio::test]
async fn gemini_frames_without_logprobs_pass_through_ungated() {
    // AI Studio reality (probed 2026-08-22): no model emits logprobs, so the
    // frames carry no logprobsResult and the gate must no-op, forwarding
    // every frame verbatim — same stance as the Ollama transport.
    let chunks: Vec<Bytes> = (0..5).map(|_| gemini_chunk(None)).collect();
    let expected = chunks.len();
    let output = run_gemini(chunks).await;
    assert_eq!(output.len(), expected);
    for frame in &output {
        let text = String::from_utf8_lossy(frame);
        assert!(!text.contains("rag_gate_decision"));
    }
}

#[tokio::test]
async fn gemini_low_confidence_stream_cuts_and_emits_decision_frame() {
    // Vertex AI reality: logprobs ARE emitted. Low-confidence frames must cut
    // the stream and inject the decision frame, exactly like OpenAI SSE.
    let chunks = vec![
        gemini_chunk(Some(-2.0)),
        gemini_chunk(Some(-2.5)),
        gemini_chunk(Some(-3.0)),
    ];
    let output = run_gemini(chunks).await;
    assert!(!output.is_empty());
    let last = String::from_utf8_lossy(output.last().unwrap());
    assert!(last.contains("ABSTAIN") || last.contains("ESCALATE"));
    assert!(last.starts_with("data: ")); // SSE-wrapped, like the protocol's framing
}

#[tokio::test]
async fn gemini_degenerate_stream_disables_gating_and_counts_metric() {
    // The temp-0 guard is provider-agnostic: near-zero Gemini logProbability
    // values must trip the same degenerate detection as OpenAI frames.
    use rag_gate::metrics::DEGENERATE_SIGNAL_TOTAL;

    let _guard = metric_lock();
    let before = DEGENERATE_SIGNAL_TOTAL.get();
    let chunks: Vec<Bytes> = (0..30).map(|_| gemini_chunk(Some(-0.0001))).collect();
    let output = run_gemini(chunks).await;

    assert_eq!(DEGENERATE_SIGNAL_TOTAL.get(), before + 1.0);
    assert_eq!(output.len(), 30);
    for frame in &output {
        let text = String::from_utf8_lossy(frame);
        assert!(!text.contains("rag_gate_decision"));
    }
}

/// A frame that carries real content but no logprobs field at all — the
/// silently-unsupported-upstream shape (verified live 2026-08-23 against
/// xAI's grok-4.20+ line: `logprobs: true` requested, response has no
/// `logprobs` key whatsoever), distinct from the degenerate near-zero case.
fn sse_chunk_no_logprobs() -> Bytes {
    Bytes::from(r#"data: {"choices":[{"delta":{"content":"hi"}}]}"#.to_string() + "\n\n")
}

#[tokio::test]
async fn stream_with_zero_logprobs_despite_request_warns_and_counts_metric() {
    use rag_gate::metrics::NO_LOGPROB_SIGNAL_TOTAL;

    let _guard = metric_lock();
    let before = NO_LOGPROB_SIGNAL_TOTAL.get();

    let source = stream::iter(
        (0..5)
            .map(|_| sse_chunk_no_logprobs())
            .collect::<Vec<_>>()
            .into_iter()
            .map(Ok::<_, axum::Error>),
    );
    let evaluator = ConfidenceEvaluator::new(thresholds());
    let intercepted = InterceptedStream::new_with_protocol_and_request(
        source,
        evaluator,
        4,
        Protocol::Sse,
        true, // logprobs_requested
    );
    let output: Vec<Bytes> = intercepted
        .map(|r| r.expect("stream item should not error"))
        .collect()
        .await;

    assert_eq!(NO_LOGPROB_SIGNAL_TOTAL.get(), before + 1.0);
    // Every frame still passes through — the guard only warns/counts, it
    // does not cut the stream (there's no confidence signal to act on).
    assert_eq!(output.len(), 5);
}

#[tokio::test]
async fn stream_with_zero_logprobs_when_not_requested_does_not_warn() {
    use rag_gate::metrics::NO_LOGPROB_SIGNAL_TOTAL;

    let _guard = metric_lock();
    let before = NO_LOGPROB_SIGNAL_TOTAL.get();

    let source = stream::iter(
        (0..5)
            .map(|_| sse_chunk_no_logprobs())
            .collect::<Vec<_>>()
            .into_iter()
            .map(Ok::<_, axum::Error>),
    );
    let evaluator = ConfidenceEvaluator::new(thresholds());
    let intercepted = InterceptedStream::new_with_protocol_and_request(
        source,
        evaluator,
        4,
        Protocol::Sse,
        false, // logprobs_requested — e.g. inject_logprobs = false
    );
    let output: Vec<Bytes> = intercepted
        .map(|r| r.expect("stream item should not error"))
        .collect()
        .await;

    // Not requesting logprobs and not receiving them is expected, not a
    // silent failure — the metric must not fire.
    assert_eq!(NO_LOGPROB_SIGNAL_TOTAL.get(), before);
    assert_eq!(output.len(), 5);
}
