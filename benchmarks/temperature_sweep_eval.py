"""Temperature-stability sweep for the mean-logprob confidence signal.

Implements the measurement side of signal_decision.md (rules pre-registered there):
for each temperature, ask a fixed HotpotQA sample to grok-3-mini with logprobs,
then report per-temperature AUROC (bootstrap 95% CI), confidence distribution
stats, and the ~80%-coverage threshold tau80. Decision rules are applied by
reading the summary against signal_decision.md — this script only measures.

Usage:
    python temperature_sweep_eval.py            # live run, 25 questions x 5 temps
    python temperature_sweep_eval.py --dry-run  # fabricated data, plumbing check only
    python temperature_sweep_eval.py --limit 100 --sample bigN   # escalation run
    python temperature_sweep_eval.py --api-base <url> --key-env OPENAI_API_KEY \
        --model gpt-4o-mini                     # cross-family replication
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
DATA_PATH = SCRIPT_DIR / "hotpotqa_sample.json"
BIG_DATA_PATH = SCRIPT_DIR / "hotpotqa_bigN_sample.json"
RESULTS_PATH = SCRIPT_DIR / "temperature_sweep_results.json"
DRY_RESULTS_PATH = SCRIPT_DIR / "temperature_sweep_dryrun.json"
ENV_FALLBACK_PATH = SCRIPT_DIR.parent / "src" / ".env"

API_URL = "https://api.x.ai/v1/chat/completions"
MODEL = "grok-3-mini"
TEMPS = [0.0, 0.2, 0.5, 0.7, 1.0]
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42
MAX_TOKENS = 60
PACING_SECS = 0.25


def load_key(key_env):
    key = os.environ.get(key_env)
    if not key and ENV_FALLBACK_PATH.exists():
        for line in ENV_FALLBACK_PATH.read_text(encoding="utf-8").splitlines():
            if line.startswith(f"{key_env}="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                break
    if not key:
        raise RuntimeError(
            f"Set the {key_env} environment variable (or put it in {ENV_FALLBACK_PATH}) "
            f"before running this script."
        )
    return key


def load_questions(sample):
    path = BIG_DATA_PATH if sample == "bigN" else DATA_PATH
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [r["row"] for r in data["rows"]]


def ask(key, question, temperature, api_url, model, reasoning_effort=None):
    prompt = (
        f"Answer this question as concisely as possible — ideally a single word "
        f"or short phrase, no explanation:\n\n{question}"
    )
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "logprobs": True,
        "temperature": temperature,
        "max_tokens": MAX_TOKENS,
    }
    # grok-3-mini accepts "none" (disables reasoning entirely); grok-4.5 does
    # not (low/medium/high only). Without this, reasoning tokens carry logprobs
    # too and dominate the running mean — a different signal from answer-token
    # confidence (see signal_decision.md, protocol amendment 2026-08-22).
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    resp = requests.post(
        api_url,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=body,
        stream=True,
        timeout=(10, 40),
    )
    resp.raise_for_status()

    answer_parts = []
    logprobs = []
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
        content = delta.get("content")
        if content:
            answer_parts.append(content)
        lp = choices[0].get("logprobs")
        if lp:
            for tok in lp.get("content", []) or []:
                if "logprob" in tok:
                    logprobs.append(tok["logprob"])

    answer_text = "".join(answer_parts).strip()
    mean_logprob = sum(logprobs) / len(logprobs) if logprobs else float("-inf")
    return answer_text, mean_logprob, len(logprobs)


def ask_with_retry(key, question, temperature, api_url, model, reasoning_effort=None, tries=3):
    last_err = None
    for attempt in range(tries):
        try:
            return ask(key, question, temperature, api_url, model, reasoning_effort)
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status in (429, 500, 502, 503, 504) and attempt < tries - 1:
                time.sleep(2.0 * (attempt + 1))
                continue
            raise
        except (requests.ConnectionError, requests.Timeout):
            last_err = e
            if attempt < tries - 1:
                time.sleep(2.0 * (attempt + 1))
                continue
            raise
    raise last_err


def normalize(s):
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    return s


def is_correct(model_answer, gold_answer):
    m = normalize(model_answer)
    g = normalize(gold_answer)
    if not g:
        return False
    return g in m or m in g


def auroc(pairs):
    """AUROC of score -> label via average-rank Mann-Whitney U. pairs: (score, label)."""
    pos = [s for s, y in pairs if y]
    neg = [s for s, y in pairs if not y]
    if not pos or not neg:
        return None
    ranked = sorted(pairs, key=lambda p: p[0])
    # average ranks for ties
    rank_sum = {}
    i = 0
    while i < len(ranked):
        j = i
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0  # 1-based
        rank_sum.setdefault(ranked[i][0], avg_rank)
        i = j
    r_pos = sum(rank_sum[s] for s in pos)
    u = r_pos - len(pos) * (len(pos) + 1) / 2.0
    return u / (len(pos) * len(neg))


def bootstrap_ci(pairs, n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    rng = random.Random(seed)
    stats = []
    for _ in range(n):
        sample = [pairs[rng.randrange(len(pairs))] for _ in range(len(pairs))]
        a = auroc(sample)
        if a is not None:
            stats.append(a)
    if not stats:
        return None, None, None
    stats.sort()
    lo = stats[int(0.025 * len(stats))]
    hi = stats[min(len(stats) - 1, int(0.975 * len(stats)))]
    return lo, hi, (hi - lo) / 2.0


def percentile(sorted_values, p):
    """Value at percentile p (0..100) of an already ascending-sorted list, nearest-rank."""
    if not sorted_values:
        return None
    idx = int(round((p / 100.0) * (len(sorted_values) - 1)))
    return sorted_values[idx]


def summarize(answers):
    """answers: list of result dicts with confidence + correct fields."""
    finite = [a for a in answers if a["confidence"] != float("-inf")]
    pairs = [(a["confidence"], a["correct"]) for a in finite]
    point = auroc(pairs)
    ci_lo, ci_hi, ci_half = bootstrap_ci(pairs) if point is not None else (None, None, None)
    confs_correct = [a["confidence"] for a in finite if a["correct"]]
    confs_wrong = [a["confidence"] for a in finite if not a["correct"]]
    confs_sorted = sorted(a["confidence"] for a in finite)
    max_abs = max((abs(a["confidence"]) for a in finite), default=None)
    return {
        "n": len(answers),
        "n_scored": len(finite),
        "n_correct": sum(1 for a in answers if a["correct"]),
        "auroc": point,
        "auroc_ci_low": ci_lo,
        "auroc_ci_high": ci_hi,
        "auroc_ci_halfwidth": ci_half,
        "mean_conf_correct": statistics.mean(confs_correct) if confs_correct else None,
        "mean_conf_wrong": statistics.mean(confs_wrong) if confs_wrong else None,
        "tau80": percentile(confs_sorted, 20),  # keep top-80% most confident
        "max_abs_logprob": max_abs,
        "readable": ci_half is not None and ci_half <= 0.15,
    }


def run_live(args, key):
    questions = load_questions(args.sample)[: args.limit]
    all_results = {}
    total = len(questions) * len(args.temps)
    done = 0
    for temp in args.temps:
        results = []
        print(f"\n=== temperature {temp} ===", flush=True)
        for i, q in enumerate(questions):
            start = time.time()
            try:
                answer, conf, n_tok = ask_with_retry(
                    key, q["question"], temp, args.api_base, args.model, args.reasoning_effort
                )
            except Exception as e:
                done += 1
                print(f"  [{i+1}/{len(questions)}] ERROR: {e}", flush=True)
                continue
            done += 1
            correct = is_correct(answer, q["answer"])
            results.append(
                {
                    "id": q["id"],
                    "question": q["question"],
                    "gold": q["answer"],
                    "model_answer": answer,
                    "confidence": conf,
                    "n_tokens": n_tok,
                    "correct": correct,
                }
            )
            print(
                f"  [{i+1}/{len(questions)}] ({done}/{total}) ({time.time()-start:.1f}s) "
                f"conf={conf:+.3f} tok={n_tok} correct={correct}",
                flush=True,
            )
            time.sleep(PACING_SECS)
        all_results[str(temp)] = results
    return all_results


def run_dry(args):
    """Fabricate deterministic pseudo-data to verify plumbing. Not science."""
    questions = load_questions(args.sample)[: args.limit]
    rng = random.Random(7)
    all_results = {}
    for temp in args.temps:
        results = []
        for q in questions:
            correct = rng.random() < 0.44
            base = -0.0005 if temp == 0.0 else -0.10 - 0.05 * temp
            spread = 0.0 if temp == 0.0 else 0.06
            conf = base + rng.uniform(-spread, spread) + (0.05 if correct else -0.05)
            conf = min(conf, -0.0002) if temp == 0.0 else conf
            answer = q["answer"] if correct else "wrong"
            results.append(
                {
                    "id": q["id"],
                    "question": q["question"],
                    "gold": q["answer"],
                    "model_answer": answer,
                    "confidence": round(conf, 4),
                    "n_tokens": 30,
                    "correct": correct,
                }
            )
        all_results[str(temp)] = results
    return all_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="fabricated data, no API calls")
    parser.add_argument("--limit", type=int, default=25, help="number of questions (default 25)")
    parser.add_argument("--sample", choices=["default", "bigN"], default="default")
    parser.add_argument("--temps", type=float, nargs="+", default=TEMPS)
    parser.add_argument("--api-base", default=API_URL)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--key-env", default="GROK_API_KEY")
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        help='pass e.g. "none" or "low" to control reasoning tokens (grok-3-mini accepts "none")',
    )
    parser.add_argument("--out", default=None, help="override the results output path")
    args = parser.parse_args()

    dry = args.dry_run
    key = None if dry else load_key(args.key_env)

    print(
        f"{'DRY RUN — ' if dry else ''}temperature sweep: model={args.model} "
        f"api={args.api_base} temps={args.temps} questions={args.limit} sample={args.sample}",
        flush=True,
    )

    all_results = run_dry(args) if dry else run_live(args, key)

    summary = {}
    for temp, answers in all_results.items():
        summary[temp] = summarize(answers)

    # pre-registered statistics (signal_decision.md): stability set excludes 0.0
    s_set = [t for t in args.temps if t != 0.0]
    aurocs = [summary[str(t)]["auroc"] for t in s_set if summary[str(t)]["auroc"] is not None]
    tau80s = [summary[str(t)]["tau80"] for t in s_set if summary[str(t)]["tau80"] is not None]
    derived = {}
    if aurocs:
        derived["auroc_range_S"] = round(max(aurocs) - min(aurocs), 4)
    if tau80s:
        derived["tau80_drift_S"] = round(max(tau80s) - min(tau80s), 4)
    t0 = summary.get("0.0")
    if t0 and t0["max_abs_logprob"] is not None:
        derived["temp0_max_abs_logprob"] = round(t0["max_abs_logprob"], 4)

    out = {
        "dry_run": dry,
        "model": args.model,
        "api_base": args.api_base,
        "reasoning_effort": args.reasoning_effort,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rules_version": "signal_decision.md pre-registered 2026-08-22",
        "per_question": all_results,
        "summary": summary,
        "derived": derived,
    }
    out_path = (
        args.out
        if args.out
        else (DRY_RESULTS_PATH if dry else RESULTS_PATH)
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\n--- Summary ({'DRY RUN' if dry else 'live'}) ---", flush=True)
    print(f"{'temp':>5} {'n':>3} {'acc':>5} {'AUROC':>6} {'95% CI':>17} {'tau80':>7} {'max|lp|':>8}")
    for temp in args.temps:
        s = summary[str(temp)]
        if s["auroc"] is None:
            print(f"{temp:>5} {s['n']:>3} {s['n_correct']/max(s['n'],1):>5.2f}  (degenerate labels)")
            continue
        print(
            f"{temp:>5} {s['n']:>3} {s['n_correct']/max(s['n'],1):>5.2f} {s['auroc']:>6.3f} "
            f"[{s['auroc_ci_low']:.3f}, {s['auroc_ci_high']:.3f}] "
            f"{s['tau80'] if s['tau80'] is not None else float('nan'):>7.3f} "
            f"{s['max_abs_logprob'] if s['max_abs_logprob'] is not None else float('nan'):>8.4f}"
        )
    print(f"\nderived: {json.dumps(derived)}")
    print(f"written: {out_path}")
    if dry:
        print("NOTE: dry-run data is fabricated; do not interpret; results file is separate.")


if __name__ == "__main__":
    main()
