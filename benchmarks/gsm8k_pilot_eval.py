"""Small pilot: does mean-logprob confidence track correctness when the
generation itself is long (multi-step reasoning) but the final answer is a
single clean number?

This isolates a variable the longform_pilot_eval.py (MS MARCO) run couldn't:
that pilot changed both "answer length" and "scoring method" (keyword recall,
noisy) at once, and found AUROC ~0.54-0.62 (barely above chance) vs.
HotpotQA's ~0.85. GSM8K gives a long generation (reasoning trace) but an
exact-match-scorable final number -- no recall/F1 ambiguity. If AUROC here
is also weak, the problem is likely generation length itself (more tokens
to dilute/average out one bad guess). If AUROC recovers here, MS MARCO's
weak result was more about scoring noise or answer-shape than length itself.

Dataset: openai/gsm8k, main config, test split, n=25 pilot (see
gsm8k_sample_n25.json). Single temperature (0.7), single provider
(OpenRouter / openai/gpt-4o-mini). Everything reads/writes to this D: repo.

Usage:
    python gsm8k_pilot_eval.py --dry-run
    python gsm8k_pilot_eval.py
"""

import argparse
import json
import os
import random
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

SCRIPT_DIR = Path(__file__).parent
DATA_PATH = SCRIPT_DIR / "gsm8k_sample_n100.json"
RESULTS_PATH = SCRIPT_DIR / "gsm8k_pilot_results_n100.json"
ENV_FALLBACK_PATH = SCRIPT_DIR.parent / "src" / ".env"

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openai/gpt-4o-mini"
TEMP = 0.7
# 400 truncated 6/25 in the n=25 pilot (all scored "wrong" by construction,
# confounding truncation with correctness) -- raised to give reasoning traces
# room to actually finish.
MAX_TOKENS = 700
PACING_SECS = 0.25


def load_key(key_env):
    key = os.environ.get(key_env)
    if not key and ENV_FALLBACK_PATH.exists():
        for line in ENV_FALLBACK_PATH.read_text(encoding="utf-8").splitlines():
            if line.startswith(f"{key_env}="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                break
    if not key:
        raise RuntimeError(f"Set {key_env} (or put it in {ENV_FALLBACK_PATH}).")
    return key


def load_questions():
    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return [r["row"] for r in data["rows"]]


def ask(key, question, api_url, model, temperature):
    prompt = (
        f"Solve this problem step by step, then give the final numeric answer "
        f"on its own line as: ANSWER: <number>\n\n{question}"
    )
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "logprobs": True,
        "temperature": temperature,
        "max_tokens": MAX_TOKENS,
    }
    resp = requests.post(
        api_url,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=body,
        stream=True,
        timeout=(10, 60),
    )
    resp.raise_for_status()

    answer_parts, logprobs = [], []
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
            answer_parts.append(delta["content"])
        lp = choices[0].get("logprobs")
        if lp:
            for tok in lp.get("content", []) or []:
                if "logprob" in tok:
                    logprobs.append(tok["logprob"])

    answer_text = "".join(answer_parts).strip()
    mean_logprob = sum(logprobs) / len(logprobs) if logprobs else float("-inf")
    return answer_text, mean_logprob, len(logprobs), logprobs


def ask_with_retry(key, question, api_url, model, temperature, tries=3):
    last_err = None
    for attempt in range(tries):
        try:
            return ask(key, question, api_url, model, temperature)
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status in (429, 500, 502, 503, 504) and attempt < tries - 1:
                time.sleep(2.0 * (attempt + 1))
                continue
            raise
        except (requests.ConnectionError, requests.Timeout) as e:
            last_err = e
            if attempt < tries - 1:
                time.sleep(2.0 * (attempt + 1))
                continue
            raise
    raise last_err


def extract_final_number(text):
    """Pull the number after 'ANSWER:'; fall back to the last number in the text."""
    m = re.search(r"ANSWER:\s*\$?(-?[\d,]+(?:\.\d+)?)", text, re.IGNORECASE)
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    return nums[-1].replace(",", "") if nums else None


def numbers_equal(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (TypeError, ValueError):
        return False


def is_correct(model_answer, gold_answer):
    extracted = extract_final_number(model_answer)
    return numbers_equal(extracted, gold_answer), extracted


def auroc(pairs):
    pos = [s for s, y in pairs if y]
    neg = [s for s, y in pairs if not y]
    if not pos or not neg:
        return None
    ranked = sorted(pairs, key=lambda p: p[0])
    rank = {}
    i = 0
    while i < len(ranked):
        j = i
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        avg = (i + j - 1) / 2.0 + 1.0
        rank.setdefault(ranked[i][0], avg)
        i = j
    r_pos = sum(rank[s] for s in pos)
    u = r_pos - len(pos) * (len(pos) + 1) / 2.0
    return u / (len(pos) * len(neg))


def bootstrap_ci(pairs, n=2000, seed=42):
    rng = random.Random(seed)
    stats = []
    for _ in range(n):
        sample = [pairs[rng.randrange(len(pairs))] for _ in range(len(pairs))]
        a = auroc(sample)
        if a is not None:
            stats.append(a)
    if not stats:
        return None, None
    stats.sort()
    lo = stats[int(0.025 * len(stats))]
    hi = stats[min(len(stats) - 1, int(0.975 * len(stats)))]
    return lo, hi


def run_dry(questions):
    rng = random.Random(7)
    results = []
    for q in questions:
        correct = rng.random() < 0.5
        conf = (-0.15 if correct else -0.55) + rng.uniform(-0.05, 0.05)
        n_tok = rng.randint(80, 250)
        lp = [conf + rng.uniform(-0.15, 0.15) for _ in range(n_tok)]
        answer = f"ANSWER: {q['answer']}" if correct else "ANSWER: 999999"
        results.append({
            "id": q["id"], "question": q["question"], "gold": q["answer"],
            "model_answer": answer, "confidence": round(conf, 4), "n_tokens": n_tok,
            "correct": correct, "extracted": q["answer"] if correct else "999999",
            "logprobs": lp,
        })
    return results


def run_live(args, key):
    questions = load_questions()
    results = []
    for i, q in enumerate(questions):
        start = time.time()
        try:
            answer, conf, n_tok, lp = ask_with_retry(key, q["question"], args.api_base, args.model, TEMP)
        except Exception as e:
            print(f"  [{i+1}/{len(questions)}] ERROR: {e}", flush=True)
            continue
        correct, extracted = is_correct(answer, q["answer"])
        results.append({
            "id": q["id"], "question": q["question"], "gold": q["answer"],
            "model_answer": answer, "confidence": conf, "n_tokens": n_tok,
            "correct": correct, "extracted": extracted, "logprobs": lp,
        })
        print(
            f"  [{i+1}/{len(questions)}] ({time.time()-start:.1f}s) "
            f"conf={conf:+.3f} tok={n_tok} extracted={extracted} gold={q['answer']} correct={correct}",
            flush=True,
        )
        time.sleep(PACING_SECS)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--api-base", default=API_URL)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--key-env", default="OPEN_ROUTER_KEY")
    args = parser.parse_args()

    questions = load_questions()
    print(
        f"{'DRY RUN — ' if args.dry_run else ''}GSM8K pilot: model={args.model} "
        f"n={len(questions)} temp={TEMP} max_tokens={MAX_TOKENS}",
        flush=True,
    )

    if args.dry_run:
        results = run_dry(questions)
    else:
        key = load_key(args.key_env)
        results = run_live(args, key)

    finite = [r for r in results if r["confidence"] != float("-inf")]
    pairs = [(r["confidence"], r["correct"]) for r in finite]
    point = auroc(pairs)
    ci_lo, ci_hi = bootstrap_ci(pairs) if point is not None else (None, None)

    ntoks = [r["n_tokens"] for r in results]
    n_correct = sum(1 for r in results if r["correct"])
    n_truncated = sum(1 for r in results if r["n_tokens"] >= MAX_TOKENS)

    summary = {
        "n": len(results),
        "n_correct": n_correct,
        "accuracy": n_correct / len(results) if results else None,
        "n_truncated": n_truncated,
        "truncation_rate": n_truncated / len(results) if results else None,
        "mean_tokens": statistics.mean(ntoks) if ntoks else None,
        "min_tokens": min(ntoks) if ntoks else None,
        "max_tokens": max(ntoks) if ntoks else None,
        "auroc": point,
        "auroc_ci_low": ci_lo,
        "auroc_ci_high": ci_hi,
    }

    out = {
        "dry_run": args.dry_run,
        "model": args.model,
        "api_base": args.api_base,
        "temperature": TEMP,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": "n=100 escalation of gsm8k_pilot_results.json (n=25, unreadable CI "
                "[0.288, 0.905]). Isolates generation LENGTH from the MS MARCO "
                "pilot's scoring-method confound: GSM8K answers are exact-match "
                "numeric (no recall/F1 ambiguity) despite long multi-step reasoning "
                "traces. max_tokens raised 400->700 vs. the n=25 run because 6/25 "
                "hit the old cap and were scored wrong by construction (truncated, "
                "not necessarily incorrect) -- check this run's own truncation rate "
                "before trusting the AUROC. Compare against "
                "longform_pilot_results_n100.json (~0.54, weak, MS MARCO prose) and "
                "temperature_sweep_results_or_bigN.json (~0.85, HotpotQA short-"
                "answer baseline).",
        "results": results,
        "summary": summary,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\n--- Summary ({'DRY RUN' if args.dry_run else 'live'}) ---")
    print(f"n={summary['n']}  acc={summary['accuracy']:.2f}  "
          f"truncated={summary['n_truncated']} ({summary['truncation_rate']:.0%})  "
          f"tokens: min={summary['min_tokens']} max={summary['max_tokens']} mean={summary['mean_tokens']:.1f}")
    if point is not None:
        print(f"AUROC={point:.3f}  95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")
    else:
        print("AUROC: degenerate (all labels same)")
    print(f"written: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
