use serde::{Deserialize, Serialize};
use std::env;
use std::fs;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatingThresholds {
    pub answer_alpha: f64,
    pub abstain_beta: f64,
    /// Minimum number of evaluated tokens before any ABSTAIN/ESCALATE decision
    /// can fire. The mean logprob over 1–2 tokens is extremely noisy — a single
    /// low-probability opening token ("Well", "Hmm") can drag the mean below
    /// `abstain_beta` and cut an answer that would have recovered. Below this
    /// floor the stream always passes through (ANSWER). Set to 1 to disable.
    #[serde(default = "default_min_tokens")]
    pub min_tokens: usize,
}

fn default_min_tokens() -> usize {
    4
}

impl Default for GatingThresholds {
    fn default() -> Self {
        Self {
            answer_alpha: -0.5,
            abstain_beta: -1.2,
            min_tokens: default_min_tokens(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProxyConfig {
    pub listen_addr: String,
    pub upstream_url: String,
    #[serde(default)]
    pub thresholds: GatingThresholds,
    /// Whether to force-inject `logprobs: true` into outgoing OpenAI-style
    /// requests. Some upstreams (notably Gemini's OpenAI-compat endpoint)
    /// reject the field with a 400. Disable for those, accepting that gating
    /// then only works if the client itself requests logprobs.
    #[serde(default = "default_inject_logprobs")]
    pub inject_logprobs: bool,
    /// Maximum request body size accepted from the client, in bytes. Guards
    /// against unbounded memory use from a malicious or buggy client.
    #[serde(default = "default_max_body_bytes")]
    pub max_body_bytes: usize,
    /// Upstream connect timeout in seconds. A hung upstream must not hang the
    /// proxy indefinitely.
    #[serde(default = "default_connect_timeout_secs")]
    pub connect_timeout_secs: u64,
}

fn default_inject_logprobs() -> bool {
    true
}

fn default_max_body_bytes() -> usize {
    2 * 1024 * 1024 // 2 MiB
}

fn default_connect_timeout_secs() -> u64 {
    10
}

impl Default for ProxyConfig {
    fn default() -> Self {
        Self {
            listen_addr: "0.0.0.0:8080".to_string(),
            upstream_url: "https://api.openai.com".to_string(),
            thresholds: GatingThresholds::default(),
            inject_logprobs: default_inject_logprobs(),
            max_body_bytes: default_max_body_bytes(),
            connect_timeout_secs: default_connect_timeout_secs(),
        }
    }
}

impl ProxyConfig {
    /// Loads config from `rag-gate.toml` (if present), then applies
    /// `RAGGATE_*` env var overrides on top.
    pub fn load() -> Self {
        let path = env::var("RAGGATE_CONFIG").unwrap_or_else(|_| "rag-gate.toml".to_string());

        let mut config = match fs::read_to_string(&path) {
            Ok(contents) => match toml::from_str::<TomlConfig>(&contents) {
                Ok(parsed) => parsed.into(),
                Err(e) => {
                    tracing::warn!("Failed to parse {}: {}, using defaults", path, e);
                    Self::default()
                }
            },
            Err(_) => Self::default(),
        };

        if let Ok(addr) = env::var("RAGGATE_LISTEN_ADDR") {
            config.listen_addr = addr;
        }
        if let Ok(url) = env::var("RAGGATE_UPSTREAM_URL") {
            config.upstream_url = url;
        }
        if let Ok(alpha) = env::var("RAGGATE_ANSWER_ALPHA")
            && let Ok(v) = alpha.parse() {
                config.thresholds.answer_alpha = v;
            }
        if let Ok(beta) = env::var("RAGGATE_ABSTAIN_BETA")
            && let Ok(v) = beta.parse() {
                config.thresholds.abstain_beta = v;
            }
        if let Ok(v) = env::var("RAGGATE_MIN_TOKENS")
            && let Ok(v) = v.parse() {
                config.thresholds.min_tokens = v;
            }
        if let Ok(v) = env::var("RAGGATE_INJECT_LOGPROBS")
            && let Ok(v) = v.parse() {
                config.inject_logprobs = v;
            }
        if let Ok(v) = env::var("RAGGATE_MAX_BODY_BYTES")
            && let Ok(v) = v.parse() {
                config.max_body_bytes = v;
            }
        if let Ok(v) = env::var("RAGGATE_CONNECT_TIMEOUT_SECS")
            && let Ok(v) = v.parse() {
                config.connect_timeout_secs = v;
            }

        config
    }
}

#[derive(Debug, Deserialize)]
struct TomlConfig {
    #[serde(default)]
    proxy: TomlProxy,
    #[serde(default)]
    thresholds: GatingThresholds,
    #[serde(default = "default_inject_logprobs")]
    inject_logprobs: bool,
    #[serde(default = "default_max_body_bytes")]
    max_body_bytes: usize,
    #[serde(default = "default_connect_timeout_secs")]
    connect_timeout_secs: u64,
}

#[derive(Debug, Deserialize)]
struct TomlProxy {
    #[serde(default = "default_listen_addr")]
    listen_addr: String,
    #[serde(default = "default_upstream_url")]
    upstream_url: String,
}

impl Default for TomlProxy {
    fn default() -> Self {
        Self {
            listen_addr: default_listen_addr(),
            upstream_url: default_upstream_url(),
        }
    }
}

fn default_listen_addr() -> String {
    "0.0.0.0:8080".to_string()
}

fn default_upstream_url() -> String {
    "https://api.openai.com".to_string()
}

impl From<TomlConfig> for ProxyConfig {
    fn from(toml: TomlConfig) -> Self {
        Self {
            listen_addr: toml.proxy.listen_addr,
            upstream_url: toml.proxy.upstream_url,
            thresholds: toml.thresholds,
            inject_logprobs: toml.inject_logprobs,
            max_body_bytes: toml.max_body_bytes,
            connect_timeout_secs: toml.connect_timeout_secs,
        }
    }
}
