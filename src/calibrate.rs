use axum::Json;
use serde::{Deserialize, Serialize};

use crate::eval::{
    ScoredSample, abstain_rate_at_beta, compute_aurc, escalation_rate_between,
    select_optimal_alpha, select_optimal_beta, sort_by_confidence,
};

#[derive(Debug, Deserialize)]
pub struct CalibrationSample {
    pub question: Option<String>,
    pub context: Option<String>,
    pub correct: bool,
    pub logprobs: Vec<f64>,
}

#[derive(Debug, Deserialize)]
pub struct CalibrationRequest {
    pub samples: Vec<CalibrationSample>,
    pub target_coverage: f64,
}

#[derive(Debug, Serialize)]
pub struct CalibrationResponse {
    pub optimal_alpha: f64,
    pub optimal_beta: f64,
    pub aurc_at_target_coverage: f64,
    pub abstain_rate: f64,
    pub escalation_rate: f64,
}

pub async fn calibrate_handler(
    Json(payload): Json<CalibrationRequest>,
) -> Json<CalibrationResponse> {
    let mut scored: Vec<ScoredSample> = payload
        .samples
        .iter()
        .map(|s| {
            let count = s.logprobs.len();
            let confidence = if count == 0 {
                f64::NEG_INFINITY
            } else {
                s.logprobs.iter().sum::<f64>() / count as f64
            };
            ScoredSample {
                confidence,
                correct: s.correct,
            }
        })
        .collect();

    // Sort descending by confidence: highest-confidence samples are "answered" first.
    sort_by_confidence(&mut scored);

    let n = scored.len();
    let aurc_at_target_coverage = compute_aurc(&scored, 100.max(n));

    let optimal_alpha = select_optimal_alpha(&scored, payload.target_coverage);
    let optimal_beta = select_optimal_beta(&scored, payload.target_coverage, optimal_alpha);

    let abstain_rate = abstain_rate_at_beta(&scored, optimal_beta);
    let escalation_rate = escalation_rate_between(&scored, optimal_alpha, optimal_beta);

    Json(CalibrationResponse {
        optimal_alpha,
        optimal_beta,
        aurc_at_target_coverage,
        abstain_rate,
        escalation_rate,
    })
}
