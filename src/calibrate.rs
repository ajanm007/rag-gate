use axum::Json;
use serde::{Deserialize, Serialize};

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

/// A sample's mean logprob confidence paired with correctness.
struct ScoredSample {
    confidence: f64,
    correct: bool,
}

/// Risk at a given coverage: sort samples by confidence descending, keep the
/// top `coverage` fraction ("answered"), and measure the error rate among those.
fn risk_at_coverage(samples: &[ScoredSample], coverage: f64) -> f64 {
    let n = samples.len();
    if n == 0 {
        return 0.0;
    }
    let kept = ((coverage * n as f64).round() as usize).max(1).min(n);
    let errors = samples[..kept].iter().filter(|s| !s.correct).count();
    errors as f64 / kept as f64
}

/// Computes the area under the risk-coverage curve via trapezoidal integration,
/// sweeping coverage from 0 to 1, matching the RAG-Gate paper's AURC metric.
fn compute_aurc(samples: &[ScoredSample], steps: usize) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut area = 0.0;
    let mut prev_coverage = 0.0;
    let mut prev_risk = risk_at_coverage(samples, 1.0 / samples.len() as f64);
    for i in 1..=steps {
        let coverage = i as f64 / steps as f64;
        let risk = risk_at_coverage(samples, coverage);
        area += (risk + prev_risk) / 2.0 * (coverage - prev_coverage);
        prev_coverage = coverage;
        prev_risk = risk;
    }
    area
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
    scored.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));

    let n = scored.len();
    let aurc_at_target_coverage = compute_aurc(&scored, 100.max(n));

    // optimal_alpha: the confidence threshold that answers exactly `target_coverage`
    // of samples (highest-confidence fraction).
    let optimal_alpha = if n > 0 {
        let kept = ((payload.target_coverage * n as f64).round() as usize).max(1).min(n);
        scored[kept - 1].confidence
    } else {
        -0.5
    };

    // optimal_beta: sweep downward from alpha and pick the confidence level below
    // which risk exceeds 2x the risk at target coverage — everything below that is
    // abstained rather than escalated. Falls back to a fixed offset with too few samples.
    let risk_at_target = risk_at_coverage(&scored, payload.target_coverage);
    let optimal_beta = if n >= 5 {
        (0..n)
            .rev()
            .map(|idx| {
                let coverage = (idx + 1) as f64 / n as f64;
                (idx, risk_at_coverage(&scored, coverage))
            })
            .find(|&(_, risk)| risk <= risk_at_target * 2.0)
            .map(|(idx, _)| scored[idx].confidence)
            .unwrap_or(optimal_alpha - 0.7)
    } else {
        optimal_alpha - 0.7
    };

    let abstain_rate = if n > 0 {
        scored.iter().filter(|s| s.confidence < optimal_beta).count() as f64 / n as f64
    } else {
        0.0
    };

    let escalation_rate = if n > 0 {
        scored
            .iter()
            .filter(|s| s.confidence >= optimal_beta && s.confidence < optimal_alpha)
            .count() as f64
            / n as f64
    } else {
        0.0
    };

    Json(CalibrationResponse {
        optimal_alpha,
        optimal_beta,
        aurc_at_target_coverage,
        abstain_rate,
        escalation_rate,
    })
}
