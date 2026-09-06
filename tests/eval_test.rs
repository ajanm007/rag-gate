use axum::Json;
use rag_gate::calibrate::{CalibrationRequest, CalibrationSample, calibrate_handler};
use rag_gate::eval::{
    ScoredSample, build_report, compute_accuracy, coverage_at_alpha, false_abstain_rate,
    false_accept_rate, parse_dataset_str, risk_at_coverage, select_optimal_alpha,
};

fn sample(confidence: f64, correct: bool) -> ScoredSample {
    ScoredSample {
        confidence,
        correct,
    }
}

#[test]
fn accuracy_all_correct_is_one() {
    let samples = vec![sample(-0.1, true), sample(-0.5, true)];
    assert!((compute_accuracy(&samples) - 1.0).abs() < 1e-12);
}

#[test]
fn accuracy_half_correct_is_half() {
    let samples = vec![
        sample(-0.1, true),
        sample(-0.5, true),
        sample(-1.0, false),
        sample(-2.0, false),
    ];
    assert!((compute_accuracy(&samples) - 0.5).abs() < 1e-12);
}

#[test]
fn accuracy_empty_is_zero() {
    assert!((compute_accuracy(&[]) - 0.0).abs() < 1e-12);
}

#[test]
fn false_accept_rate_counts_wrong_among_answered() {
    let samples = vec![
        sample(-0.1, true),
        sample(-0.2, false),
        sample(-1.0, true),
        sample(-2.0, false),
    ];
    // alpha = -0.5 answers the first two; one of them is wrong.
    assert!((false_accept_rate(&samples, -0.5) - 0.5).abs() < 1e-12);
}

#[test]
fn false_accept_rate_with_nothing_answered_is_zero() {
    let samples = vec![sample(-0.1, true), sample(-0.2, false)];
    // alpha above every confidence answers nothing.
    assert!((false_accept_rate(&samples, 0.0) - 0.0).abs() < 1e-12);
}

#[test]
fn false_abstain_rate_counts_correct_among_abstained() {
    let samples = vec![
        sample(-0.1, true),
        sample(-0.2, false),
        sample(-1.0, true),
        sample(-2.0, false),
    ];
    // beta = -0.5 abstains on the last two; one of them is actually correct.
    assert!((false_abstain_rate(&samples, -0.5) - 0.5).abs() < 1e-12);
}

#[test]
fn false_abstain_rate_with_nothing_abstained_is_zero() {
    let samples = vec![sample(-0.1, true), sample(-0.2, false)];
    // beta below every confidence abstains on nothing.
    assert!((false_abstain_rate(&samples, -100.0) - 0.0).abs() < 1e-12);
}

#[test]
fn coverage_at_alpha_is_answered_fraction() {
    let samples = vec![
        sample(-0.1, true),
        sample(-0.5, true),
        sample(-1.0, false),
        sample(-2.0, false),
    ];
    assert!((coverage_at_alpha(&samples, -0.5) - 0.5).abs() < 1e-12);
}

#[test]
fn risk_at_coverage_is_error_rate_among_top_fraction() {
    // Sorted descending: correct, correct, wrong, wrong.
    let samples = vec![
        sample(-0.1, true),
        sample(-0.5, true),
        sample(-1.0, false),
        sample(-2.0, false),
    ];
    assert!((risk_at_coverage(&samples, 0.5) - 0.0).abs() < 1e-12);
}

#[test]
fn risk_at_full_coverage_is_overall_error_rate() {
    let samples = vec![
        sample(-0.1, true),
        sample(-0.5, true),
        sample(-1.0, false),
        sample(-2.0, false),
    ];
    assert!((risk_at_coverage(&samples, 1.0) - 0.5).abs() < 1e-12);
}

#[test]
fn optimal_alpha_answers_target_fraction() {
    let samples = vec![
        sample(-0.1, true),
        sample(-0.5, true),
        sample(-1.0, false),
        sample(-2.0, false),
    ];
    // target 0.5 of 4 keeps 2; threshold is the 2nd-highest confidence.
    assert!((select_optimal_alpha(&samples, 0.5) - (-0.5)).abs() < 1e-12);
}

#[test]
fn parse_ignores_extra_fields() {
    let text = r#"{
        "model": "openai/gpt-4o-mini",
        "results": [
            {"id": "0", "question": "?", "gold": "18", "confidence": -0.1,
             "correct": true, "extracted": "18", "logprobs": [-0.1]},
            {"id": "1", "question": "?", "gold": "7", "confidence": -2.0,
             "correct": false, "extracted": "8", "logprobs": [-2.0]}
        ],
        "summary": {}
    }"#;
    let samples = parse_dataset_str(text).expect("should parse");
    assert_eq!(samples.len(), 2);
}

#[test]
fn parse_accepts_bare_array() {
    let text = r#"[{"confidence": -0.1, "correct": true}]"#;
    let samples = parse_dataset_str(text).expect("should parse");
    assert_eq!(samples.len(), 1);
}

#[test]
fn parse_rejects_record_missing_confidence() {
    let text = r#"{"results": [{"correct": true}]}"#;
    assert!(parse_dataset_str(text).is_err());
}

#[test]
fn build_report_end_to_end_on_known_samples() {
    // Deliberately unsorted: build_report must sort by confidence itself.
    // Sorted: (-0.1,T), (-0.5,T), (-1.0,F), (-2.0,F).
    let samples = vec![
        sample(-2.0, false),
        sample(-0.1, true),
        sample(-1.0, false),
        sample(-0.5, true),
    ];
    let report = build_report(&samples, 0.5, "test");
    assert_eq!(report.n, 4);
    assert_eq!(report.n_correct, 2);
    assert!((report.accuracy - 0.5).abs() < 1e-12);
    assert!((report.optimal_alpha - (-0.5)).abs() < 1e-12);
    // n = 4 < 5, so beta falls back to alpha - 0.7.
    assert!((report.optimal_beta - (-1.2)).abs() < 1e-12);
    assert!((report.coverage - 0.5).abs() < 1e-12);
    assert!((report.risk - 0.0).abs() < 1e-12);
    // Hand-computed trapezoidal AURC over the risk steps of [T,T,F,F] = 7/48.
    assert!((report.aurc - 7.0 / 48.0).abs() < 1e-9);
    assert!((report.false_accept_rate - 0.0).abs() < 1e-12);
    assert!((report.false_abstain_rate - 0.0).abs() < 1e-12);
}

#[tokio::test]
async fn calibrate_endpoint_and_cli_report_agree() {
    // Single-element logprobs, so the endpoint's mean == the CLI's confidence.
    // n = 6 exercises the real beta sweep (not the n < 5 fallback).
    let confidences = [-0.1, -0.5, -1.0, -2.0, -0.3, -1.5];
    let correct = [true, true, false, false, true, false];
    let scored: Vec<ScoredSample> = confidences
        .iter()
        .zip(correct.iter())
        .map(|(&confidence, &c)| sample(confidence, c))
        .collect();
    let request = CalibrationRequest {
        samples: confidences
            .iter()
            .zip(correct.iter())
            .map(|(&confidence, &c)| CalibrationSample {
                question: None,
                context: None,
                correct: c,
                logprobs: vec![confidence],
            })
            .collect(),
        target_coverage: 0.5,
    };
    let response = calibrate_handler(Json(request)).await.0;
    let report = build_report(&scored, 0.5, "parity");
    assert!((response.optimal_alpha - report.optimal_alpha).abs() < 1e-12);
    assert!((response.optimal_beta - report.optimal_beta).abs() < 1e-12);
    assert!((response.aurc_at_target_coverage - report.aurc).abs() < 1e-12);
    assert!((response.abstain_rate - report.abstain_rate).abs() < 1e-12);
    assert!((response.escalation_rate - report.escalation_rate).abs() < 1e-12);
}
