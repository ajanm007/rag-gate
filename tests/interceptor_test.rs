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
