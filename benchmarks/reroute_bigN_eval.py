"""Bigger-N reroute eval — settle (or refute) whether rerouting a low-confidence
answer to a stronger model actually helps, with enough sample to matter.

The pilot (recovery_reroute_grok_eval.py) was N=11: 2/11 wrong->right vs the
temp-0 retry's 1/12. That difference is one question — not significant (Fisher's
exact p ~ 0.6). This script scales up:

  1. Pull FRESH HotpotQA hard questions (offset past the original 25 to avoid
     overlap), default 100.
  2. Baseline pass: ask each on grok-3-mini with logprobs -> (answer, mean
     logprob, correct).
  3. Calibrate alpha to the baseline distribution (median).
  4. Reroute pass: re-ask every below-median ("ESCALATE-band") question on
     grok-4.5. Judge correctness only (no confidence artifact).
  5. Report band accuracy before/after, wrong->right / right->wrong counts, and
     a Fisher's exact p-value vs. the null of "reroute changes nothing".

Same working xAI key for both models (within-provider reroute — a weaker probe
than cross-provider, but the keys we have). Everything is checkpointed to disk so
a mid-run timeout doesn't lose progress.
"""

import json
import math
import os
import re
import statistics
import time
from pathlib import Path

import requests

SCRIPT_DIR = Path(__file__).parent
SAMPLE_PATH = SCRIPT_DIR / "hotpotqa_bigN_sample.json"
BASELINE_PATH = SCRIPT_DIR / "reroute_bigN_baseline.json"
OUT_PATH = SCRIPT_DIR / "reroute_bigN_results.json"

API_URL = "https://api.x.ai/v1/chat/completions"
BASE_MODEL = "grok-3-mini"
REROUTE_MODEL = "grok-4.5"

N = int(os.environ.get("BIGN", "100"))
OFFSET = int(os.environ.get("BIGN_OFFSET", "100"))  # past the original 25-question slice


def load_key():
    key = os.environ.get("GROK_API_KEY")
    if not key:
        raise RuntimeError("Set GROK_API_KEY before running.")
    return key


def fetch_sample():
    if SAMPLE_PATH.exists():
        return [r["row"] for r in json.load(open(SAMPLE_PATH, encoding="utf-8"))["rows"]]
    url = (
        "https://datasets-server.huggingface.co/rows"
        "?dataset=hotpotqa/hotpot_qa&config=distractor&split=validation"
        f"&offset={OFFSET}&length={N}"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    json.dump(resp.json(), open(SAMPLE_PATH, "w", encoding="utf-8"), indent=2)
    return [r["row"] for r in resp.json()["rows"]]


def prompt_for(question):
    return (
        f"Answer this question as concisely as possible — ideally a single word "
        f"or short phrase, no explanation:\n\n{question}"
    )


def ask_stream(key, question, model):
    """Streaming ask with logprobs -> (answer_text, mean_logprob)."""
    resp = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt_for(question)}],
            "stream": True,
            "logprobs": True,
            "max_tokens": 60,
        },
        stream=True,
        timeout=(10, 40),
    )
    resp.raise_for_status()
    parts, logprobs = [], []
    for line in resp.iter_lines():
        if not line:
            continue
        line = line.decode("utf-8")
        if not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        if payload.strip() == "[DONE]":
            break
        chunk = json.loads(payload)
        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta", {})
        if delta.get("content"):
            parts.append(delta["content"])
        lp = choices[0].get("logprobs")
        if lp:
            for tok in lp.get("content", []) or []:
                if "logprob" in tok:
                    logprobs.append(tok["logprob"])
    answer = "".join(parts).strip()
    mean_lp = sum(logprobs) / len(logprobs) if logprobs else float("-inf")
    return answer, mean_lp


def ask_plain(key, question, model):
    """Non-streaming ask -> answer_text (reroute pass; correctness only).

    grok-4.5 cannot disable reasoning (xAI only accepts reasoning_effort of
    low/medium/high, default high). We send "low" — the minimum thinking this
    model allows — for speed / fewer timeouts, which is as close to "thinking
    off" as grok-4.5 permits.
    """
    resp = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt_for(question)}],
            "max_tokens": 60,
            "reasoning_effort": "low",
        },
        timeout=(10, 40),
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def normalize(s):
    return re.sub(r"[^\w\s]", "", s.lower().strip())


def is_correct(ans, gold):
    m, g = normalize(ans), normalize(gold)
    return bool(g) and (g in m or m in g)


def mcnemar_exact_p(b, c):
    """Two-sided exact McNemar p-value for PAIRED data.

    This is the correct test for "did rerouting change accuracy on the SAME
    questions". It looks only at the DISCORDANT pairs: b = wrong->right (reroute
    fixed it), c = right->wrong (reroute broke it). Concordant pairs (both
    correct / both wrong) carry no information about a change and are excluded.

    (An earlier version of this script used a Fisher's exact test on a
    base-correct x reroute-correct 2x2. That was wrong: it tests ASSOCIATION
    between two correlated columns of the same items and returns p~0 trivially
    because 36/45 answers are unchanged — it does NOT test whether reroute
    improves accuracy. McNemar is the paired-difference test that does.)

    Exact p = 2 * sum_{k=0}^{min(b,c)} C(n,k) (1/2)^n, capped at 1, with n=b+c.
    """
    n = b + c
    if n == 0:
        return 1.0
    lo = min(b, c)
    tail = sum(math.comb(n, k) for k in range(lo + 1)) * (0.5 ** n)
    return min(2.0 * tail, 1.0)


def checkpoint(path, obj):
    json.dump(obj, open(path, "w", encoding="utf-8"), indent=2)


def main():
    key = load_key()
    rows = fetch_sample()
    print(f"Sample: {len(rows)} questions (offset {OFFSET})", flush=True)

    # ---- Baseline pass (grok-3-mini), resumable ----
    baseline = json.load(open(BASELINE_PATH, encoding="utf-8")) if BASELINE_PATH.exists() else []
    done_ids = {r["id"] for r in baseline}
    for i, q in enumerate(rows):
        if q["id"] in done_ids:
            continue
        start = time.time()
        try:
            ans, conf = ask_stream(key, q["question"], BASE_MODEL)
        except Exception as e:
            print(f"[base {i+1}/{len(rows)}] ERROR {time.time()-start:.1f}s: {e}", flush=True)
            continue
        rec = {"id": q["id"], "question": q["question"], "gold": q["answer"],
               "base_answer": ans, "confidence": conf,
               "base_correct": is_correct(ans, q["answer"])}
        baseline.append(rec)
        checkpoint(BASELINE_PATH, baseline)
        print(f"[base {i+1}/{len(rows)}] ({time.time()-start:.1f}s) conf={conf:+.3f} "
              f"correct={rec['base_correct']}", flush=True)

    confs = sorted(r["confidence"] for r in baseline if r["confidence"] != float("-inf"))
    alpha = statistics.median(confs)
    band = [r for r in baseline if r["confidence"] < alpha]
    print(f"\nBaseline: {len(baseline)} scored, alpha(median)={alpha:+.4f}, "
          f"ESCALATE-band={len(band)}", flush=True)

    # ---- Reroute pass (grok-4.5), resumable ----
    results = json.load(open(OUT_PATH, encoding="utf-8")) if OUT_PATH.exists() else []
    done_ids = {r["id"] for r in results}
    for i, orig in enumerate(band):
        if orig["id"] in done_ids:
            continue
        start = time.time()
        try:
            rr = ask_plain(key, orig["question"], REROUTE_MODEL)
        except Exception as e:
            print(f"[reroute {i+1}/{len(band)}] ERROR {time.time()-start:.1f}s: {e}", flush=True)
            continue
        rec = {**orig, "reroute_model": REROUTE_MODEL, "reroute_answer": rr,
               "reroute_correct": is_correct(rr, orig["gold"])}
        results.append(rec)
        checkpoint(OUT_PATH, results)
        print(f"[reroute {i+1}/{len(band)}] ({time.time()-start:.1f}s) "
              f"correct {orig['base_correct']} -> {rec['reroute_correct']}", flush=True)

    if not results:
        print("No reroute results.", flush=True)
        return

    w2r = sum(1 for r in results if not r["base_correct"] and r["reroute_correct"])
    r2w = sum(1 for r in results if r["base_correct"] and not r["reroute_correct"])
    unchanged = len(results) - w2r - r2w
    base_ok = sum(1 for r in results if r["base_correct"])
    rr_ok = sum(1 for r in results if r["reroute_correct"])
    # Paired test: only the discordant pairs (w2r vs r2w) carry information about
    # whether reroute changed accuracy. McNemar, NOT Fisher on a 2x2 (see the
    # note on mcnemar_exact_p for why Fisher was the wrong test here).
    p = mcnemar_exact_p(w2r, r2w)

    print(f"\n--- Big-N reroute summary ({BASE_MODEL} -> {REROUTE_MODEL}, band N={len(results)}) ---")
    print(f"Band accuracy: {base_ok/len(results):.1%} -> {rr_ok/len(results):.1%}")
    print(f"  wrong -> right: {w2r}")
    print(f"  right -> wrong: {r2w}")
    print(f"  unchanged:      {unchanged}")
    print(f"Net change: {w2r - r2w:+d} of {len(results)}")
    print(f"McNemar exact (two-sided, paired) p = {p:.3f}  "
          f"({'significant' if p < 0.05 else 'NOT significant'} at 0.05)")
    print("Within-provider reroute; correctness-only judgment.")


if __name__ == "__main__":
    main()
