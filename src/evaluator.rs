use crate::config::GatingThresholds;

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
}
