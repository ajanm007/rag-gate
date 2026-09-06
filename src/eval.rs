//! Offline evaluation math shared by the `/calibrate` HTTP endpoint and the
//! `rag-gate evaluate` CLI, plus the CLI's dataset parsing and report rendering.
//!
//! The two entry points must never silently disagree, so all threshold search,
//! risk-coverage integration, and rate computation lives here as plain pure
//! functions over `&[ScoredSample]` (no I/O), with thin wrappers on each side.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

// ---------------------------------------------------------------------------
// Core types + shared math (also used by `crate::calibrate`)
// ---------------------------------------------------------------------------

/// A sample's mean logprob confidence paired with correctness.
#[derive(Debug, Clone, Copy)]
pub struct ScoredSample {
    pub confidence: f64,
    pub correct: bool,
}

/// Sort descending by confidence: highest-confidence samples are "answered" first.
/// Shared so the `/calibrate` endpoint and the `evaluate` CLI can never disagree
/// on ordering.
pub fn sort_by_confidence(samples: &mut [ScoredSample]) {
    samples.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// Risk at a given coverage: assumes `samples` is already sorted descending by
/// confidence (see [`sort_by_confidence`]). Keeps the top `coverage` fraction
/// ("answered") and measures the error rate among those.
pub fn risk_at_coverage(samples: &[ScoredSample], coverage: f64) -> f64 {
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
/// Assumes `samples` is already sorted descending by confidence.
pub fn compute_aurc(samples: &[ScoredSample], steps: usize) -> f64 {
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

/// Optimal alpha: the confidence threshold that answers exactly
/// `target_coverage` of samples (highest-confidence fraction).
/// Assumes `sorted` is already sorted descending by confidence.
pub fn select_optimal_alpha(sorted: &[ScoredSample], target_coverage: f64) -> f64 {
    let n = sorted.len();
    if n > 0 {
        let kept = ((target_coverage * n as f64).round() as usize).max(1).min(n);
        sorted[kept - 1].confidence
    } else {
        -0.5
    }
}

/// Optimal beta: sweep downward from alpha and pick the confidence level below
/// which risk exceeds 2x the risk at target coverage — everything below that is
/// abstained rather than escalated. Falls back to a fixed offset with too few samples.
/// Assumes `sorted` is already sorted descending by confidence.
pub fn select_optimal_beta(
    sorted: &[ScoredSample],
    target_coverage: f64,
    optimal_alpha: f64,
) -> f64 {
    let n = sorted.len();
    let risk_at_target = risk_at_coverage(sorted, target_coverage);
    if n >= 5 {
        (0..n)
            .rev()
            .map(|idx| {
                let coverage = (idx + 1) as f64 / n as f64;
                (idx, risk_at_coverage(sorted, coverage))
            })
            .find(|&(_, risk)| risk <= risk_at_target * 2.0)
            .map(|(idx, _)| sorted[idx].confidence)
            .unwrap_or(optimal_alpha - 0.7)
    } else {
        optimal_alpha - 0.7
    }
}

/// Overall accuracy: fraction of samples that were correct.
pub fn compute_accuracy(samples: &[ScoredSample]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.iter().filter(|s| s.correct).count() as f64 / samples.len() as f64
}

/// Coverage at a threshold: fraction of samples with confidence >= `alpha`
/// (the "answered" fraction).
pub fn coverage_at_alpha(samples: &[ScoredSample], alpha: f64) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.iter().filter(|s| s.confidence >= alpha).count() as f64 / samples.len() as f64
}

/// False-accept rate: fraction of *answered* samples (confidence >= `alpha`)
/// that were wrong. Returns 0.0 when nothing was answered.
pub fn false_accept_rate(samples: &[ScoredSample], alpha: f64) -> f64 {
    let answered: Vec<&ScoredSample> =
        samples.iter().filter(|s| s.confidence >= alpha).collect();
    if answered.is_empty() {
        return 0.0;
    }
    answered.iter().filter(|s| !s.correct).count() as f64 / answered.len() as f64
}

/// False-abstain rate: fraction of *abstained* samples (confidence < `beta`)
/// that were actually correct. Returns 0.0 when nothing was abstained.
pub fn false_abstain_rate(samples: &[ScoredSample], beta: f64) -> f64 {
    let abstained: Vec<&ScoredSample> =
        samples.iter().filter(|s| s.confidence < beta).collect();
    if abstained.is_empty() {
        return 0.0;
    }
    abstained.iter().filter(|s| s.correct).count() as f64 / abstained.len() as f64
}

/// Abstain rate: fraction of samples with confidence < `beta`.
pub fn abstain_rate_at_beta(samples: &[ScoredSample], beta: f64) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.iter().filter(|s| s.confidence < beta).count() as f64 / samples.len() as f64
}

/// Escalation rate: fraction of samples with `beta <= confidence < alpha`.
pub fn escalation_rate_between(samples: &[ScoredSample], alpha: f64, beta: f64) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    samples
        .iter()
        .filter(|s| s.confidence >= beta && s.confidence < alpha)
        .count() as f64
        / samples.len() as f64
}

// ---------------------------------------------------------------------------
// Dataset parsing
// ---------------------------------------------------------------------------

/// One record of an eval-results file. Only `confidence` + `correct` are read;
/// every other field (question text, gold answers, logprobs, model metadata…)
/// is ignored. Unknown fields are skipped by serde automatically.
#[derive(Debug, Deserialize)]
struct BenchmarkRecord {
    confidence: f64,
    correct: bool,
}

/// Parse dataset JSON text into scored samples.
///
/// Accepts either `{"results": [...]}` (the shape written by the
/// `benchmarks/*_pilot_eval.py` scripts) or a bare `[...]` array of records
/// (the shape of `benchmarks/hotpotqa_results.json`). Anything else — or a
/// record missing `confidence`/`correct` — is an error, not a silent skip.
pub fn parse_dataset_str(text: &str) -> Result<Vec<ScoredSample>, String> {
    let value: serde_json::Value =
        serde_json::from_str(text).map_err(|e| format!("invalid JSON: {e}"))?;
    let records_value = match &value {
        serde_json::Value::Object(map) => match map.get("results") {
            Some(r) => r,
            None => {
                return Err(
                    "expected an object with a \"results\" array or a bare [...] array of \
                     {\"confidence\": <f64>, \"correct\": <bool>} records"
                        .to_string(),
                );
            }
        },
        serde_json::Value::Array(_) => &value,
        _ => {
            return Err(
                "expected an object with a \"results\" array or a bare [...] array of \
                 {\"confidence\": <f64>, \"correct\": <bool>} records"
                    .to_string(),
            );
        }
    };
    let records: Vec<BenchmarkRecord> = serde_json::from_value(records_value.clone())
        .map_err(|e| {
            format!(
                "could not parse records as [{{\"confidence\": f64, \"correct\": bool}}, ...]: {e}"
            )
        })?;
    Ok(records
        .into_iter()
        .map(|r| ScoredSample {
            confidence: r.confidence,
            correct: r.correct,
        })
        .collect())
}

/// Read + parse a dataset file. Rejects empty files (zero usable records) so
/// the CLI prints an error instead of a meaningless all-zeros table.
pub fn load_dataset_file(path: &Path) -> Result<Vec<ScoredSample>, String> {
    let text =
        fs::read_to_string(path).map_err(|e| format!("could not read {}: {e}", path.display()))?;
    let samples = parse_dataset_str(&text)?;
    if samples.is_empty() {
        return Err(format!(
            "no usable records in {}: \"results\" array is empty",
            path.display()
        ));
    }
    Ok(samples)
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

/// Full offline evaluation report over one dataset at one target coverage.
#[derive(Debug, Clone, Serialize)]
pub struct EvalReport {
    pub dataset: String,
    pub target_coverage: f64,
    pub n: usize,
    pub n_correct: usize,
    pub accuracy: f64,
    pub optimal_alpha: f64,
    pub optimal_beta: f64,
    pub coverage: f64,
    pub risk: f64,
    pub aurc: f64,
    pub false_accept_rate: f64,
    pub false_abstain_rate: f64,
    pub abstain_rate: f64,
    pub escalation_rate: f64,
    pub n_answered: usize,
    pub n_abstained: usize,
    pub n_escalated: usize,
}

/// Build the full report from already-parsed samples. Pure computation, no I/O:
/// sorts a copy by confidence, derives alpha/beta with the exact same search
/// the `/calibrate` endpoint uses, then scores every rate from those thresholds.
pub fn build_report(
    samples: &[ScoredSample],
    target_coverage: f64,
    dataset: &str,
) -> EvalReport {
    let mut sorted = samples.to_vec();
    sort_by_confidence(&mut sorted);
    let n = sorted.len();
    let n_correct = sorted.iter().filter(|s| s.correct).count();
    let optimal_alpha = select_optimal_alpha(&sorted, target_coverage);
    let optimal_beta = select_optimal_beta(&sorted, target_coverage, optimal_alpha);
    let n_answered = sorted.iter().filter(|s| s.confidence >= optimal_alpha).count();
    let n_abstained = sorted.iter().filter(|s| s.confidence < optimal_beta).count();
    // Note: risk-at-achieved-coverage and false-accept rate coincide under
    // threshold gating (both = error rate among the answered set); both are
    // reported because the brief asks for both, computed once here.
    let far = false_accept_rate(&sorted, optimal_alpha);
    EvalReport {
        dataset: dataset.to_string(),
        target_coverage,
        n,
        n_correct,
        accuracy: compute_accuracy(&sorted),
        optimal_alpha,
        optimal_beta,
        coverage: coverage_at_alpha(&sorted, optimal_alpha),
        risk: far,
        aurc: compute_aurc(&sorted, 100.max(n)),
        false_accept_rate: far,
        false_abstain_rate: false_abstain_rate(&sorted, optimal_beta),
        abstain_rate: abstain_rate_at_beta(&sorted, optimal_beta),
        escalation_rate: escalation_rate_between(&sorted, optimal_alpha, optimal_beta),
        n_answered,
        n_abstained,
        n_escalated: n.saturating_sub(n_answered + n_abstained),
    }
}

/// Human-readable report table (default `evaluate` output).
pub fn format_human_report(r: &EvalReport) -> String {
    format!(
        "rag-gate evaluate: {dataset}\n\
         target coverage: {tc:.2}\n\
         +--------------------+----------+\n\
         | n                  | {n:>8} |\n\
         | n_correct          | {nc:>8} |\n\
         | accuracy           | {acc:>8.4} |\n\
         | optimal_alpha      | {alpha:>8.4} |\n\
         | optimal_beta       | {beta:>8.4} |\n\
         | coverage           | {cov:>8.4} |\n\
         | risk               | {risk:>8.4} |\n\
         | aurc               | {aurc:>8.4} |\n\
         | false_accept_rate  | {far:>8.4} |\n\
         | false_abstain_rate | {frr:>8.4} |\n\
         | abstain_rate       | {ar:>8.4} |\n\
         | escalation_rate    | {er:>8.4} |\n\
         +--------------------+----------+\n\
         answered {na}/{n}  abstained {nb}/{n}  escalated {ne}/{n}\n\
         (risk == false_accept_rate under threshold gating: both are the error\n\
         rate among the answered set.)",
        dataset = r.dataset,
        tc = r.target_coverage,
        n = r.n,
        nc = r.n_correct,
        acc = r.accuracy,
        alpha = r.optimal_alpha,
        beta = r.optimal_beta,
        cov = r.coverage,
        risk = r.risk,
        aurc = r.aurc,
        far = r.false_accept_rate,
        frr = r.false_abstain_rate,
        ar = r.abstain_rate,
        er = r.escalation_rate,
        na = r.n_answered,
        nb = r.n_abstained,
        ne = r.n_escalated,
    )
}
