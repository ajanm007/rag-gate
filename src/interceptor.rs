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
    /// Raw bytes carried over from a previous poll that didn't yet contain a
    /// complete SSE event (`...\n\n`). A TCP/HTTP chunk boundary can land
    /// mid-event, so events must be reassembled here before parsing rather
    /// than parsed off each raw poll result directly.
    partial_frame: Vec<u8>,
}

impl<S> InterceptedStream<S> {
    pub fn new(inner: S, evaluator: ConfidenceEvaluator, lookahead_size: usize) -> Self {
        Self {
            inner,
            evaluator,
            lookahead_buffer: VecDeque::with_capacity(lookahead_size),
            lookahead_size,
            finished: false,
            partial_frame: Vec::new(),
        }
    }

    /// Pulls one complete SSE event (terminated by a blank line) out of
    /// `partial_frame`, if one is available. Returns `None` if only a partial
    /// event is buffered so far.
    fn take_complete_frame(partial_frame: &mut Vec<u8>) -> Option<Bytes> {
        let pos = Self::find_frame_boundary(partial_frame)?;
        let frame: Vec<u8> = partial_frame.drain(..pos).collect();
        Some(Bytes::from(frame))
    }

    /// Byte offset just past the first `\n\n` in `buf`, if any, without
    /// consuming it.
    fn find_frame_boundary(buf: &[u8]) -> Option<usize> {
        let separator = b"\n\n";
        buf.windows(separator.len())
            .position(|w| w == separator)
            .map(|pos| pos + separator.len())
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
        loop {
            if self.finished {
                // If there's still data in the buffer, drain it
                if let Some(chunk) = self.lookahead_buffer.pop_front() {
                    return Poll::Ready(Some(Ok(chunk)));
                }
                return Poll::Ready(None);
            }

            // Drain any already-buffered complete frames before polling for
            // more bytes, so a single large read doesn't block on new I/O.
            if let Some(frame) = Self::take_complete_frame(&mut self.partial_frame) {
                match self.as_mut().process_frame(frame) {
                    Some(item) => return Poll::Ready(Some(item)),
                    // Frame consumed but nothing to emit yet (still filling
                    // the lookahead buffer) — loop to process the next
                    // buffered frame or poll for more bytes.
                    None => continue,
                }
            }

            match Pin::new(&mut self.inner).poll_next(cx) {
                Poll::Ready(Some(Ok(chunk))) => {
                    self.partial_frame.extend_from_slice(&chunk);
                    // Loop back around: take_complete_frame will either find
                    // a full frame now or we'll poll inner again.
                }
                Poll::Ready(Some(Err(e))) => return Poll::Ready(Some(Err(e))),
                Poll::Ready(None) => {
                    // Upstream closed without a trailing blank line; flush
                    // any leftover bytes as a final chunk rather than
                    // silently dropping a trailing logprob or partial payload.
                    if !self.partial_frame.is_empty() {
                        let leftover = Bytes::from(std::mem::take(&mut self.partial_frame));
                        self.lookahead_buffer.push_back(leftover);
                    }

                    if !self.finished {
                        DECISIONS_TOTAL.with_label_values(&["ANSWER"]).inc();
                        CONFIDENCE_SCORE.observe(self.evaluator.current_confidence());
                        TOKENS_EVALUATED.observe(self.evaluator.count() as f64);
                    }
                    self.finished = true;
                    return if let Some(chunk) = self.lookahead_buffer.pop_front() {
                        Poll::Ready(Some(Ok(chunk)))
                    } else {
                        Poll::Ready(None)
                    };
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

impl<S> InterceptedStream<S>
where
    S: Stream<Item = Result<Bytes, axum::Error>> + Unpin,
{
    /// Processes one complete, reassembled SSE event: extracts any logprobs,
    /// evaluates confidence, and decides whether to forward it (subject to
    /// the lookahead buffer) or cut the stream with a decision frame.
    /// Returns `None` if the frame was consumed but nothing is ready to
    /// emit yet (still filling the lookahead buffer) — the caller should
    /// keep processing rather than treat this as a real `Pending`.
    fn process_frame(
        self: Pin<&mut Self>,
        chunk: Bytes,
    ) -> Option<Result<Bytes, axum::Error>> {
        let this = self.get_mut();
        let text = String::from_utf8_lossy(&chunk);

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
                                        this.evaluator.add_logprob(lp);
                                        logprob_found = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if logprob_found {
            let decision = this.evaluator.evaluate();
            match decision {
                Decision::Answer => {
                    this.lookahead_buffer.push_back(chunk);
                    if this.lookahead_buffer.len() > this.lookahead_size {
                        if let Some(out_chunk) = this.lookahead_buffer.pop_front() {
                            return Some(Ok(out_chunk));
                        }
                    }
                }
                Decision::Abstain | Decision::Escalate => {
                    this.finished = true;
                    this.lookahead_buffer.clear(); // Do not emit buffered tokens

                    let label = if decision == Decision::Abstain { "ABSTAIN" } else { "ESCALATE" };
                    DECISIONS_TOTAL.with_label_values(&[label]).inc();
                    CONFIDENCE_SCORE.observe(this.evaluator.current_confidence());
                    TOKENS_EVALUATED.observe(this.evaluator.count() as f64);

                    let frame = this.generate_decision_frame(decision);
                    return Some(Ok(frame));
                }
            }
        } else {
            this.lookahead_buffer.push_back(chunk);
            if this.lookahead_buffer.len() > this.lookahead_size {
                if let Some(out_chunk) = this.lookahead_buffer.pop_front() {
                    return Some(Ok(out_chunk));
                }
            }
        }

        // Nothing to emit from this frame yet (still filling the lookahead
        // buffer) — caller will loop and process the next frame or poll for
        // more bytes.
        None
    }
}
