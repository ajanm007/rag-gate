use crate::config::GatingThresholds;

/// Number of evaluated tokens before degeneracy detection can fire. Small
/// prefixes of near-perfect logprobs are common even in sampled streams.
const DEGENERATE_MIN_TOKENS: usize = 16;

/// A running mean above this after `DEGENERATE_MIN_TOKENS` tokens means every
/// token so far had probability > e^-0.01 ≈ 0.99 — the signature of greedy
/// (temperature-0) decoding, where the reported logprob of the argmax token
/// under the re-normalized distribution sits at ≈ 0.
const DEGENERATE_MEAN_EPSILON: f64 = -0.01;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    Answer,
    Abstain,
    Escalate,
}

#[derive(Debug)]
pub struct ConfidenceEvaluator {
    thresholds: GatingThresholds,
    sum_logprobs: f64,
    count: usize,
}

impl ConfidenceEvaluator {
    pub fn new(thresholds: GatingThresholds) -> Self {
        Self {
            thresholds,
            sum_logprobs: 0.0,
            count: 0,
        }
    }

    /// Adds a new logprob and returns the current confidence score
    pub fn add_logprob(&mut self, logprob: f64) -> f64 {
        self.sum_logprobs += logprob;
        self.count += 1;
        self.current_confidence()
    }

    pub fn current_confidence(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum_logprobs / (self.count as f64)
        }
    }

    pub fn evaluate(&self) -> Decision {
        // Warmup floor: the mean over very few tokens is too noisy to abstain
        // on. Until `min_tokens` have been evaluated, always pass through.
        if self.count < self.thresholds.min_tokens {
            return Decision::Answer;
        }
        let conf = self.current_confidence();
        if conf >= self.thresholds.answer_alpha {
            Decision::Answer
        } else if conf < self.thresholds.abstain_beta {
            Decision::Abstain
        } else {
            Decision::Escalate
        }
    }

    pub fn count(&self) -> usize {
        self.count
    }

    /// True when the accumulated signal is "too perfect to trust": the running
    /// mean is still above `DEGENERATE_MEAN_EPSILON` after
    /// `DEGENERATE_MIN_TOKENS` tokens. At temperature 0 the model always picks
    /// the argmax token, whose reported logprob sits at ≈ 0, so the mean never
    /// leaves the neighborhood of zero and the gate would silently pass every
    /// response as ANSWER. Real sampled streams at serving temperatures
    /// average well below this bound. Disabling gating on such a stream
    /// changes no decision (any sane `answer_alpha` is below the bound) — the
    /// point of detection is visibility: warn the operator, count it, and stop
    /// pretending the signal is live.
    pub fn is_degenerate(&self) -> bool {
        self.thresholds.degenerate_guard
            && self.count >= DEGENERATE_MIN_TOKENS
            && self.current_confidence() > DEGENERATE_MEAN_EPSILON
    }

    pub fn thresholds(&self) -> &GatingThresholds {
        &self.thresholds
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluator() {
        let thresholds = GatingThresholds {
            answer_alpha: -0.5,
            abstain_beta: -1.2,
            min_tokens: 1, // disable warmup floor for this threshold-logic test
            degenerate_guard: true,
        };
        let mut eval = ConfidenceEvaluator::new(thresholds);

        eval.add_logprob(-0.4);
        assert_eq!(eval.evaluate(), Decision::Answer);

        eval.add_logprob(-0.8);
        // sum = -1.2, count = 2, mean = -0.6
        assert_eq!(eval.evaluate(), Decision::Escalate);

        eval.add_logprob(-2.6);
        // sum = -3.8, count = 3, mean ≈ -1.267 (< -1.2 abstain threshold)
        assert_eq!(eval.evaluate(), Decision::Abstain);
    }

    #[test]
    fn warmup_floor_forces_answer_below_min_tokens() {
        let thresholds = GatingThresholds {
            answer_alpha: -0.5,
            abstain_beta: -1.2,
            min_tokens: 4,
            degenerate_guard: true,
        };
        let mut eval = ConfidenceEvaluator::new(thresholds);

        // Terrible logprobs, but below the 4-token warmup floor → ANSWER.
        eval.add_logprob(-5.0);
        assert_eq!(eval.evaluate(), Decision::Answer);
        eval.add_logprob(-5.0);
        eval.add_logprob(-5.0);
        assert_eq!(eval.evaluate(), Decision::Answer); // count = 3, still warming up

        // 4th token crosses the floor; mean -5.0 < abstain_beta → ABSTAIN.
        eval.add_logprob(-5.0);
        assert_eq!(eval.evaluate(), Decision::Abstain);
    }

    fn guard_thresholds(degenerate_guard: bool) -> GatingThresholds {
        GatingThresholds {
            answer_alpha: -0.5,
            abstain_beta: -1.2,
            min_tokens: 1,
            degenerate_guard,
        }
    }

    #[test]
    fn degenerate_signal_detected_only_after_warmup_window() {
        let mut eval = ConfidenceEvaluator::new(guard_thresholds(true));
        for _ in 0..15 {
            eval.add_logprob(-0.0001);
        }
        assert!(!eval.is_degenerate()); // 15 near-perfect tokens: still below the window
        eval.add_logprob(-0.0001);
        assert!(eval.is_degenerate()); // 16th near-perfect token fires detection
    }

    #[test]
    fn realistic_confidence_is_not_degenerate() {
        let mut eval = ConfidenceEvaluator::new(guard_thresholds(true));
        for lp in [-0.05, -0.3, -0.02, -0.6, -0.1, -0.45, -0.08, -0.2, -0.33, -0.15,
                   -0.05, -0.3, -0.02, -0.6, -0.1, -0.45, -0.08, -0.2, -0.33, -0.15] {
            eval.add_logprob(lp);
        }
        // mean ≈ -0.223, well below the epsilon bound
        assert!(!eval.is_degenerate());
    }

    #[test]
    fn degenerate_guard_can_be_disabled() {
        let mut eval = ConfidenceEvaluator::new(guard_thresholds(false));
        for _ in 0..30 {
            eval.add_logprob(-0.0001);
        }
        assert!(!eval.is_degenerate()); // detection opted out via config
    }
}
