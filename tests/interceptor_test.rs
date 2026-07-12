use bytes::Bytes;
use futures_util::{stream, StreamExt};
use rag_gate::{ConfidenceEvaluator, GatingThresholds, InterceptedStream};

fn thresholds() -> GatingThresholds {
    GatingThresholds {
        answer_alpha: -0.5,
        abstain_beta: -1.2,
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
