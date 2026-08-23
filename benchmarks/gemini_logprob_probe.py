"""Probe which Gemini models currently emit logprobs via the native API.

As of 2026-08-22 every model on AI Studio (generativelanguage.googleapis.com)
rejects `generationConfig.responseLogprobs` with "Logprobs is not enabled for
this model" — while Vertex AI documents full support. rag-gate's Gemini
transport is therefore gating-live for Vertex upstreams and inert (forward-
compatible) for AI Studio. Re-run this when Google changes either surface:

    python gemini_logprob_probe.py
"""

import json
import os
from pathlib import Path

import requests

ENV_FALLBACK = Path(__file__).parent.parent / "src" / ".env"
BASE = "https://generativelanguage.googleapis.com/v1beta"


def load_key():
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key and ENV_FALLBACK.exists():
        for line in ENV_FALLBACK.read_text(encoding="utf-8").splitlines():
            if line.startswith(("GEMINI_API_KEY=", "GOOGLE_API_KEY=")):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                break
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY (or put it in src/.env) first.")
    return key


def main():
    key = load_key()
    headers = {"x-goog-api-key": key, "Content-Type": "application/json"}

    resp = requests.get(f"{BASE}/models", headers=headers, timeout=15)
    resp.raise_for_status()
    names = [
        m["name"].split("/")[-1]
        for m in resp.json().get("models", [])
        if not any(
            x in m["name"]
            for x in ("embedding", "image", "tts", "aqa", "veo", "imagen", "live", "banana")
        )
    ]
    print(f"{len(names)} generative models visible on AI Studio; probing responseLogprobs...\n")

    enabled, disabled = [], 0
    for name in names:
        r = requests.post(
            f"{BASE}/models/{name}:generateContent",
            headers=headers,
            json={
                "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
                "generationConfig": {"responseLogprobs": True, "logprobs": 1, "maxOutputTokens": 8},
            },
            timeout=30,
        )
        if r.status_code == 200:
            body = json.dumps(r.json().get("candidates", [{}])[0])
            has = "logprobsResult" in body
            if has:
                enabled.append(name)
            print(f"  {name:45s} logprobs={'YES' if has else 'no (200 but empty)'}")
        else:
            disabled += 1
            print(f"  {name:45s} {r.status_code} {r.json().get('error', {}).get('message', '')[:55]}")

    print(f"\nlogprobs-enabled models: {len(enabled)}")
    for name in enabled:
        print(f"  {name}")
    if not enabled:
        print("  (none — AI Studio still has the server-side flag off; Vertex AI remains the gating-live surface)")


if __name__ == "__main__":
    main()
