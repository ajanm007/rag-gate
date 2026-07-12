use prometheus::{Counter, CounterVec, Histogram, Opts, Registry};
use std::sync::LazyLock;

// Ensure LazyLock is available (stable from 1.80.0, we are on 1.97.0).
pub static REGISTRY: LazyLock<Registry> = LazyLock::new(Registry::new);

pub static REQUESTS_TOTAL: LazyLock<Counter> = LazyLock::new(|| {
    let counter = Counter::new("raggate_requests_total", "Total requests proxied").unwrap();
    REGISTRY.register(Box::new(counter.clone())).unwrap();
    counter
});

pub static DECISIONS_TOTAL: LazyLock<CounterVec> = LazyLock::new(|| {
    let opts = Opts::new("raggate_decisions_total", "ANSWER / ABSTAIN / ESCALATE counts");
    let counter = CounterVec::new(opts, &["decision"]).unwrap();
    REGISTRY.register(Box::new(counter.clone())).unwrap();
    counter
});

pub static CONFIDENCE_SCORE: LazyLock<Histogram> = LazyLock::new(|| {
    // Mean logprob confidence is always <= 0. Buckets are centered around the
    // default answer/abstain thresholds (-0.5 / -1.2) so the interesting range
    // near the decision boundary gets resolution instead of collapsing into
    // Prometheus's default positive-latency-shaped buckets.
    let buckets = vec![
        -5.0, -3.0, -2.0, -1.5, -1.2, -1.0, -0.8, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0,
    ];
    let histogram = Histogram::with_opts(
        prometheus::HistogramOpts::new("raggate_confidence_score", "Distribution of confidence scores")
            .buckets(buckets),
    ).unwrap();
    REGISTRY.register(Box::new(histogram.clone())).unwrap();
    histogram
});

pub static TOKENS_EVALUATED: LazyLock<Histogram> = LazyLock::new(|| {
    // Token counts are small non-negative integers (often cut short by the
    // 4-token lookahead on early ABSTAIN/ESCALATE), so linear low buckets
    // resolve the interesting range better than exponential latency buckets.
    let buckets = vec![
        1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0,
    ];
    let histogram = Histogram::with_opts(
        prometheus::HistogramOpts::new("raggate_tokens_evaluated", "Tokens evaluated before decision")
            .buckets(buckets),
    ).unwrap();
    REGISTRY.register(Box::new(histogram.clone())).unwrap();
    histogram
});

pub static TOKEN_SAVINGS_TOTAL: LazyLock<Counter> = LazyLock::new(|| {
    let counter = Counter::new("raggate_token_savings_total", "Tokens saved by early stream termination").unwrap();
    REGISTRY.register(Box::new(counter.clone())).unwrap();
    counter
});

pub static PROXY_LATENCY_MS: LazyLock<Histogram> = LazyLock::new(|| {
    let histogram = Histogram::with_opts(
        prometheus::HistogramOpts::new("raggate_proxy_latency_ms", "Added latency vs. direct API call")
    ).unwrap();
    REGISTRY.register(Box::new(histogram.clone())).unwrap();
    histogram
});

pub fn get_metrics_payload() -> String {
    use prometheus::Encoder;
    let encoder = prometheus::TextEncoder::new();
    let mut buffer = vec![];
    encoder.encode(&REGISTRY.gather(), &mut buffer).unwrap();
    String::from_utf8(buffer).unwrap()
}
