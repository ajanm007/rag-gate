use crate::evaluator::{ConfidenceEvaluator, Decision};
use crate::metrics::{
    CONFIDENCE_SCORE, DECISIONS_TOTAL, DEGENERATE_SIGNAL_TOTAL, NO_LOGPROB_SIGNAL_TOTAL,
    TOKENS_EVALUATED, TOKEN_SAVINGS_TOTAL,
};
use futures_util::Stream;
use serde::Serialize;
use serde_json::Value;
use std::collections::VecDeque;
use std::pin::Pin;
use std::task::{Context, Poll};
use bytes::Bytes;

/// Wire protocol of the upstream stream. OpenAI-compatible APIs use SSE
/// (`data: {json}\n\n`); Ollama's native endpoints use NDJSON (one bare JSON
/// object per `\n`-terminated line); Gemini's native
/// `streamGenerateContent?alt=sse` uses standard SSE framing with a different
/// JSON payload shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Protocol {
    /// Server-Sent Events: `data: {json}\n\n`, terminated by `data: [DONE]`.
    Sse,
    /// Newline-delimited JSON: one bare `{json}\n` object per line, as emitted
    /// by Ollama's `/api/chat` and `/api/generate`.
    Ndjson,
    /// Gemini native SSE (`streamGenerateContent?alt=sse`): standard SSE
    /// framing, but logprobs — when the model emits them — live at
    /// `candidates[0].logprobsResult.chosenCandidates[].logProbability`.
    /// As of 2026-08-22 no AI Studio model has logprobs enabled (probed all
    /// 50 — see `benchmarks/gemini_logprob_probe.py`), so this transport
    /// currently no-ops exactly like Ollama's and gating activates
    /// automatically if Google enables the server-side flag.
    Gemini,
}

impl Protocol {
    /// The byte sequence that terminates one complete frame on this protocol.
    fn frame_separator(self) -> &'static [u8] {
        match self {
            Protocol::Sse | Protocol::Gemini => b"\n\n",
            Protocol::Ndjson => b"\n",
        }
    }
}

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
    protocol: Protocol,
    /// Set once the evaluator's degeneracy check fires (temperature-0/greedy
    /// artifact: every logprob ≈ 0). For the rest of the stream the gate is
    /// disabled and frames pass through — a dead signal must not masquerade
    /// as "confident".
    signal_degenerate: bool,
    /// Whether `logprobs`/`responseLogprobs` was actually requested for this
    /// stream. Only meaningful for judging a zero-logprob stream at the end:
    /// if the field was never requested, an empty signal is expected, not a
    /// silent upstream failure.
    logprobs_requested: bool,
    /// True once at least one frame with a parseable body (not just SSE
    /// framing) has been processed, regardless of whether it carried a
    /// logprob. Distinguishes "upstream sent frames but never included the
    /// logprobs field" from "stream was empty/errored before any content."
    any_frame_processed: bool,
    /// Raw bytes carried over from a previous poll that didn't yet contain a
    /// complete frame. A TCP/HTTP chunk boundary can land mid-frame, so frames
    /// must be reassembled here before parsing rather than parsed off each raw
    /// poll result directly.
    partial_frame: Vec<u8>,
}

impl<S> InterceptedStream<S> {
    /// Constructs an interceptor for an OpenAI-compatible SSE upstream. Kept
    /// for backward compatibility — `new_with_protocol` is the general form.
    /// `logprobs_requested` defaults to `false` here (not `true`): callers
    /// using this shim haven't opted into the no-signal guard's distinction
    /// between "didn't ask" and "asked and got nothing", so it stays quiet
    /// rather than firing on every zero-logprob stream. `proxy.rs` calls
    /// `new_with_protocol_and_request` directly and sets this accurately.
    pub fn new(inner: S, evaluator: ConfidenceEvaluator, lookahead_size: usize) -> Self {
        Self::new_with_protocol(inner, evaluator, lookahead_size, Protocol::Sse)
    }

    /// Kept for backward compatibility (existing tests and any external
    /// caller not using the request-aware constructor). Defaults
    /// `logprobs_requested` to `false` — see `new`'s doc comment.
    pub fn new_with_protocol(
        inner: S,
        evaluator: ConfidenceEvaluator,
        lookahead_size: usize,
        protocol: Protocol,
    ) -> Self {
        Self::new_with_protocol_and_request(inner, evaluator, lookahead_size, protocol, false)
    }

    /// General constructor. `logprobs_requested` should reflect whether the
    /// outgoing request actually asked the upstream for logprobs (i.e.
    /// `inject_logprobs` was on, or the client's own body already requested
    /// them) — it gates the end-of-stream "upstream never sent any logprobs"
    /// warning so an intentional `inject_logprobs = false` setup doesn't
    /// trip it.
    pub fn new_with_protocol_and_request(
        inner: S,
        evaluator: ConfidenceEvaluator,
        lookahead_size: usize,
        protocol: Protocol,
        logprobs_requested: bool,
    ) -> Self {
        Self {
            inner,
            evaluator,
            lookahead_buffer: VecDeque::with_capacity(lookahead_size),
            lookahead_size,
            finished: false,
            protocol,
            signal_degenerate: false,
            logprobs_requested,
            any_frame_processed: false,
            partial_frame: Vec::new(),
        }
    }

    /// Pulls one complete frame (terminated by the protocol's separator) out of
    /// `partial_frame`, if one is available. Returns `None` if only a partial
    /// frame is buffered so far.
    fn take_complete_frame(&mut self) -> Option<Bytes> {
        let pos = Self::find_frame_boundary(self.protocol, &self.partial_frame)?;
        let frame: Vec<u8> = self.partial_frame.drain(..pos).collect();
        Some(Bytes::from(frame))
    }

    /// Byte offset just past the first frame separator in `buf`, if any,
    /// without consuming it.
    fn find_frame_boundary(protocol: Protocol, buf: &[u8]) -> Option<usize> {
        let separator = protocol.frame_separator();
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

        let threshold_used = match decision {
            Decision::Abstain => self.evaluator.thresholds().abstain_beta,
            Decision::Escalate | Decision::Answer => self.evaluator.thresholds().answer_alpha,
        };

        let frame = RagGateDecisionFrame {
            rag_gate_decision: decision_str.to_string(),
            confidence_score: self.evaluator.current_confidence(),
            tokens_evaluated: self.evaluator.count(),
            threshold_used,
        };

        let json = serde_json::to_string(&frame).unwrap();
        match self.protocol {
            Protocol::Sse | Protocol::Gemini => Bytes::from(format!("data: {}\n\n", json)),
            Protocol::Ndjson => Bytes::from(format!("{}\n", json)),
        }
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
            if let Some(frame) = self.take_complete_frame() {
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

                        // The stream completed, produced at least one parseable
                        // frame, logprobs were actually requested, and yet the
                        // evaluator never received a single one. Unlike the
                        // degenerate-guard case (logprobs present but stuck near
                        // zero), this is the field being absent from the wire
                        // entirely — a provider/model that silently ignores the
                        // logprobs request (e.g. xAI's grok-4.20+ line, verified
                        // 2026-08-23) rather than rejecting it outright. Gating
                        // was inert for the whole stream; the guard exists so
                        // that fact is visible instead of masquerading as a
                        // clean ANSWER.
                        if self.logprobs_requested
                            && self.any_frame_processed
                            && self.evaluator.count() == 0
                        {
                            NO_LOGPROB_SIGNAL_TOTAL.inc();
                            tracing::warn!(
                                "no logprobs received for this stream despite requesting them; \
                                 the upstream model/provider likely does not emit them (silently \
                                 ignored, not rejected) — confidence gating was inert for the \
                                 entire stream, see Known Limitations"
                            );
                        }
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
    /// Parses one SSE `data: {json}\n\n` frame for OpenAI-style
    /// `choices[0].logprobs.content[].logprob` values, feeding each into the
    /// evaluator. Returns true if at least one logprob was found.
    fn extract_sse_logprobs(text: &str, evaluator: &mut ConfidenceEvaluator) -> bool {
        if !text.starts_with("data: ") || text.trim() == "data: [DONE]" {
            return false;
        }
        let json_str = &text[6..];
        let Ok(value) = serde_json::from_str::<Value>(json_str) else {
            return false;
        };
        Self::feed_openai_logprobs(&value, evaluator)
    }

    /// Parses one `data: {json}` frame from Gemini's native
    /// `streamGenerateContent?alt=sse` stream. Logprobs, when present, sit at
    /// `candidates[0].logprobsResult.chosenCandidates[].logProbability`
    /// (natural-log units, same as OpenAI's `logprob`). As of 2026-08-22 no
    /// AI Studio model emits them, so in practice this returns false and the
    /// frame passes through ungated — the same forward-compatible stance as
    /// the Ollama transport. If Google enables them, gating activates with no
    /// further changes.
    fn extract_gemini_logprobs(text: &str, evaluator: &mut ConfidenceEvaluator) -> bool {
        if !text.starts_with("data: ") || text.trim() == "data: [DONE]" {
            return false;
        }
        let Ok(value) = serde_json::from_str::<Value>(&text[6..]) else {
            return false;
        };
        let Some(chosen) = value
            .get("candidates")
            .and_then(|c| c.as_array())
            .and_then(|arr| arr.first())
            .and_then(|cand| cand.get("logprobsResult"))
            .and_then(|lr| lr.get("chosenCandidates"))
            .and_then(|cc| cc.as_array())
        else {
            return false;
        };
        let mut found = false;
        for cand in chosen {
            if let Some(lp) = cand.get("logProbability").and_then(|v| v.as_f64()) {
                evaluator.add_logprob(lp);
                found = true;
            }
        }
        found
    }

    /// Parses one NDJSON line as emitted by Ollama's native endpoints. Ollama's
    /// `/api/chat` and `/api/generate` do NOT currently return per-token
    /// logprobs (see https://github.com/ollama/ollama/issues/16117, closed as
    /// not planned, and #13638), so in practice this returns false and the
    /// frame passes through ungated. If a future Ollama build adds a
    /// `logprobs`/`content` field in the OpenAI shape, gating activates
    /// automatically with no further changes.
    fn extract_ndjson_logprobs(text: &str, evaluator: &mut ConfidenceEvaluator) -> bool {
        let Ok(value) = serde_json::from_str::<Value>(text.trim()) else {
            return false;
        };
        // Accept either an OpenAI-shaped `choices[0].logprobs.content[]` (should
        // a compat build emit it) or a hypothetical top-level `logprobs.content`.
        if Self::feed_openai_logprobs(&value, evaluator) {
            return true;
        }
        if let Some(content) = value
            .get("logprobs")
            .and_then(|l| l.get("content"))
            .and_then(|c| c.as_array())
        {
            let mut found = false;
            for token in content {
                if let Some(lp) = token.get("logprob").and_then(|v| v.as_f64()) {
                    evaluator.add_logprob(lp);
                    found = true;
                }
            }
            return found;
        }
        false
    }

    /// Extracts `choices[0].logprobs.content[].logprob` (OpenAI shape) from a
    /// parsed JSON value, feeding each into the evaluator.
    fn feed_openai_logprobs(value: &Value, evaluator: &mut ConfidenceEvaluator) -> bool {
        let mut found = false;
        if let Some(content) = value
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|arr| arr.first())
            .and_then(|choice| choice.get("logprobs"))
            .and_then(|lp| lp.get("content"))
            .and_then(|c| c.as_array())
        {
            for token in content {
                if let Some(lp) = token.get("logprob").and_then(|v| v.as_f64()) {
                    evaluator.add_logprob(lp);
                    found = true;
                }
            }
        }
        found
    }

    fn process_frame(
        self: Pin<&mut Self>,
        chunk: Bytes,
    ) -> Option<Result<Bytes, axum::Error>> {
        let this = self.get_mut();
        let text = String::from_utf8_lossy(&chunk);
        this.any_frame_processed = true;

        let logprob_found = match this.protocol {
            Protocol::Sse => Self::extract_sse_logprobs(&text, &mut this.evaluator),
            Protocol::Ndjson => Self::extract_ndjson_logprobs(&text, &mut this.evaluator),
            Protocol::Gemini => Self::extract_gemini_logprobs(&text, &mut this.evaluator),
        };

        // Degenerate-signal guard: check once per frame after extraction. If
        // the running mean is still ≈ 0 past the warmup window, this stream is
        // almost certainly greedy-decoded and the confidence signal carries no
        // information — warn, count it, and stop gating this stream.
        if logprob_found && !this.signal_degenerate && this.evaluator.is_degenerate() {
            this.signal_degenerate = true;
            DEGENERATE_SIGNAL_TOTAL.inc();
            tracing::warn!(
                mean_logprob = this.evaluator.current_confidence(),
                tokens = this.evaluator.count(),
                "confidence signal degenerate (mean logprob ≈ 0 — likely temperature-0/greedy \
                 decoding); disabling gating for this stream, see Known Limitations"
            );
        }

        if logprob_found && !this.signal_degenerate {
            let decision = this.evaluator.evaluate();
            match decision {
                Decision::Answer => {
                    this.lookahead_buffer.push_back(chunk);
                    if this.lookahead_buffer.len() > this.lookahead_size
                        && let Some(out_chunk) = this.lookahead_buffer.pop_front() {
                            return Some(Ok(out_chunk));
                        }
                }
                Decision::Abstain | Decision::Escalate => {
                    this.finished = true;
                    // Every buffered-but-unsent frame is a token we generated
                    // upstream but never forwarded — plus the current one. This
                    // is the early-exit saving rag-gate exists to produce.
                    let saved = this.lookahead_buffer.len() + 1;
                    TOKEN_SAVINGS_TOTAL.inc_by(saved as f64);
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
            if this.lookahead_buffer.len() > this.lookahead_size
                && let Some(out_chunk) = this.lookahead_buffer.pop_front() {
                    return Some(Ok(out_chunk));
                }
        }

        // Nothing to emit from this frame yet (still filling the lookahead
        // buffer) — caller will loop and process the next frame or poll for
        // more bytes.
        None
    }
}
