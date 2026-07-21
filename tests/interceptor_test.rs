use bytes::Bytes;
use futures_util::{stream, StreamExt};
use rag_gate::{ConfidenceEvaluator, GatingThresholds, InterceptedStream, Protocol};

fn thresholds() -> GatingThresholds {
    GatingThresholds {
        answer_alpha: -0.5,
        abstain_beta: -1.2,
        min_tokens: 1, // disable warmup floor so cut-behavior tests fire immediately
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
