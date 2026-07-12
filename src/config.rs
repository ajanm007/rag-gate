use serde::{Deserialize, Serialize};
use std::env;
use std::fs;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatingThresholds {
    pub answer_alpha: f64,
    pub abstain_beta: f64,
}

impl Default for GatingThresholds {
    fn default() -> Self {
        Self {
            answer_alpha: -0.5,
            abstain_beta: -1.2,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProxyConfig {
    pub listen_addr: String,
    pub upstream_url: String,
    #[serde(default)]
    pub thresholds: GatingThresholds,
}

impl Default for ProxyConfig {
    fn default() -> Self {
        Self {
            listen_addr: "0.0.0.0:8080".to_string(),
            upstream_url: "https://api.openai.com".to_string(),
            thresholds: GatingThresholds::default(),
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
        if let Ok(alpha) = env::var("RAGGATE_ANSWER_ALPHA") {
            if let Ok(v) = alpha.parse() {
                config.thresholds.answer_alpha = v;
            }
        }
        if let Ok(beta) = env::var("RAGGATE_ABSTAIN_BETA") {
            if let Ok(v) = beta.parse() {
                config.thresholds.abstain_beta = v;
            }
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
        }
    }
}
