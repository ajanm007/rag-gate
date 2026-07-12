use crate::evaluator::{ConfidenceEvaluator, Decision};
use crate::metrics::{CONFIDENCE_SCORE, DECISIONS_TOTAL, TOKENS_EVALUATED};
use futures_util::Stream;
use serde::Serialize;
use serde_json::Value;
use std::collections::VecDeque;
use std::pin::Pin;
use std::task::{Context, Poll};
use bytes::Bytes;

#[derive(Serialize)]
pub struct RagGateDecisionFrame {
    pub rag_gate_decision: String,
    pub confidence_score: f64,
    pub tokens_evaluated: usize,
    pub threshold_used: f64,
}

pub struct InterceptedStream<S> {
    inner: S,
    evaluator: ConfidenceEvaluator,
    lookahead_buffer: VecDeque<Bytes>,
    lookahead_size: usize,
    finished: bool,
}

impl<S> InterceptedStream<S> {
    pub fn new(inner: S, evaluator: ConfidenceEvaluator, lookahead_size: usize) -> Self {
        Self {
            inner,
            evaluator,
            lookahead_buffer: VecDeque::with_capacity(lookahead_size),
            lookahead_size,
            finished: false,
        }
    }

    fn generate_decision_frame(&self, decision: Decision) -> Bytes {
        let decision_str = match decision {
            Decision::Abstain => "ABSTAIN",
            Decision::Escalate => "ESCALATE",
            Decision::Answer => "ANSWER", // Should not emit for ANSWER unless requested
        };

        // Note: For beta threshold, we don't have direct access here if we don't store it, 
        // but we can just use 0.0 or pass it from eval. In a real app we'd get it from evaluator.
        let frame = RagGateDecisionFrame {
            rag_gate_decision: decision_str.to_string(),
            confidence_score: self.evaluator.current_confidence(),
            tokens_evaluated: self.evaluator.count(),
            threshold_used: 0.0, // Can be improved later
        };

        let json = serde_json::to_string(&frame).unwrap();
        Bytes::from(format!("data: {}\n\n", json))
    }
}

impl<S> Stream for InterceptedStream<S>
where
    S: Stream<Item = Result<Bytes, axum::Error>> + Unpin,
{
    type Item = Result<Bytes, axum::Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.finished {
            // If there's still data in the buffer, drain it
            if let Some(chunk) = self.lookahead_buffer.pop_front() {
                return Poll::Ready(Some(Ok(chunk)));
            }
            return Poll::Ready(None);
        }

        match Pin::new(&mut self.inner).poll_next(cx) {
            Poll::Ready(Some(Ok(chunk))) => {
                let text = String::from_utf8_lossy(&chunk);
                
                // Parse logprobs if this is a data chunk
                let mut logprob_found = false;
                if text.starts_with("data: ") && text.trim() != "data: [DONE]" {
                    let json_str = &text[6..];
                    if let Ok(value) = serde_json::from_str::<Value>(json_str) {
                        if let Some(choices) = value.get("choices").and_then(|c| c.as_array()) {
                            if let Some(choice) = choices.get(0) {
                                if let Some(logprobs) = choice.get("logprobs") {
                                    if let Some(content) = logprobs.get("content").and_then(|c| c.as_array()) {
                                        for token in content {
                                            if let Some(lp) = token.get("logprob").and_then(|v| v.as_f64()) {
                                                self.evaluator.add_logprob(lp);
                                                logprob_found = true;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // If we found a logprob, we evaluate
                if logprob_found {
                    let decision = self.evaluator.evaluate();
                    match decision {
                        Decision::Answer => {
                            // Enqueue chunk and maybe pop one if buffer is full
                            self.lookahead_buffer.push_back(chunk);
                            if self.lookahead_buffer.len() > self.lookahead_size {
                                if let Some(out_chunk) = self.lookahead_buffer.pop_front() {
                                    return Poll::Ready(Some(Ok(out_chunk)));
                                }
                            } else {
                                // Wait for more tokens to fill lookahead
                                cx.waker().wake_by_ref();
                                return Poll::Pending;
                            }
                        }
                        Decision::Abstain | Decision::Escalate => {
                            // We must cut the stream.
                            self.finished = true;
                            self.lookahead_buffer.clear(); // Do not emit buffered tokens

                            let label = if decision == Decision::Abstain { "ABSTAIN" } else { "ESCALATE" };
                            DECISIONS_TOTAL.with_label_values(&[label]).inc();
                            CONFIDENCE_SCORE.observe(self.evaluator.current_confidence());
                            TOKENS_EVALUATED.observe(self.evaluator.count() as f64);

                            let frame = self.generate_decision_frame(decision);
                            return Poll::Ready(Some(Ok(frame)));
                        }
                    }
                } else {
                    // Non-data chunk or no logprobs (e.g. initial chunk), just pass it through directly
                    // or push to buffer
                    self.lookahead_buffer.push_back(chunk);
                    if self.lookahead_buffer.len() > self.lookahead_size {
                        if let Some(out_chunk) = self.lookahead_buffer.pop_front() {
                            return Poll::Ready(Some(Ok(out_chunk)));
                        }
                    } else {
                        cx.waker().wake_by_ref();
                        return Poll::Pending;
                    }
                }
                
                // Should not reach here but just in case
                cx.waker().wake_by_ref();
                Poll::Pending
            }
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(e))),
            Poll::Ready(None) => {
                if !self.finished {
                    DECISIONS_TOTAL.with_label_values(&["ANSWER"]).inc();
                    CONFIDENCE_SCORE.observe(self.evaluator.current_confidence());
                    TOKENS_EVALUATED.observe(self.evaluator.count() as f64);
                }
                self.finished = true;
                if let Some(chunk) = self.lookahead_buffer.pop_front() {
                    Poll::Ready(Some(Ok(chunk)))
                } else {
                    Poll::Ready(None)
                }
            }
            Poll::Pending => Poll::Pending,
        }
    }
}
