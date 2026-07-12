use rag_gate::{ConfidenceEvaluator, Decision, GatingThresholds};

fn thresholds() -> GatingThresholds {
    GatingThresholds {
        answer_alpha: -0.5,
        abstain_beta: -1.2,
    }
}

#[test]
fn answers_when_confidence_meets_alpha() {
    let mut eval = ConfidenceEvaluator::new(thresholds());
    eval.add_logprob(-0.1);
    eval.add_logprob(-0.2);
    assert_eq!(eval.evaluate(), Decision::Answer);
}

#[test]
fn escalates_between_thresholds() {
    let mut eval = ConfidenceEvaluator::new(thresholds());
    eval.add_logprob(-0.7);
    eval.add_logprob(-0.9);
    // mean = -0.8, between beta (-1.2) and alpha (-0.5)
    assert_eq!(eval.evaluate(), Decision::Escalate);
}

#[test]
fn abstains_below_beta() {
    let mut eval = ConfidenceEvaluator::new(thresholds());
    eval.add_logprob(-2.0);
    eval.add_logprob(-2.5);
    assert_eq!(eval.evaluate(), Decision::Abstain);
}

#[test]
fn boundary_alpha_is_answer() {
    let mut eval = ConfidenceEvaluator::new(thresholds());
    eval.add_logprob(-0.5);
    assert_eq!(eval.evaluate(), Decision::Answer);
}

#[test]
fn boundary_beta_is_escalate_not_abstain() {
    let mut eval = ConfidenceEvaluator::new(thresholds());
    eval.add_logprob(-1.2);
    // strict `<` for abstain means exactly -1.2 is still Escalate
    assert_eq!(eval.evaluate(), Decision::Escalate);
}

#[test]
fn confidence_is_running_mean() {
    let mut eval = ConfidenceEvaluator::new(thresholds());
    assert_eq!(eval.current_confidence(), 0.0);
    eval.add_logprob(-1.0);
    assert_eq!(eval.current_confidence(), -1.0);
    eval.add_logprob(-3.0);
    assert_eq!(eval.current_confidence(), -2.0);
    assert_eq!(eval.count(), 2);
}
