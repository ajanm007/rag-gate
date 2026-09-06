"""Small pilot: does mean-logprob confidence still track correctness on
longer, multi-sentence answers (vs. HotpotQA's 1-3 token spans)?

Reuses temperature_sweep_eval.py's ask()/AUROC/bootstrap machinery. The one
real change is correctness scoring: MS MARCO gold answers are full sentences,
not short exact-match spans, so this uses SQuAD-style token F1 overlap
(threshold >= 0.5) instead of substring containment. That's a judgment call,
not free of assumptions -- see the threshold note below.

Dataset: microsoft/ms_marco v2.1 validation, 25 rows with a real answer
(most MS MARCO queries have no answer -- filtered out when the sample was
built). Single temperature only (0.7), single provider (OpenRouter /
openai/gpt-4o-mini) -- this is a pilot, not a full sweep. Everything reads/
writes to this D: repo; nothing touches C:.

Usage:
    python longform_pilot_eval.py --dry-run
    python longform_pilot_eval.py
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
DATA_PATH = SCRIPT_DIR / "msmarco_longform_sample_n100.json"
RESULTS_PATH = SCRIPT_DIR / "longform_pilot_results_n100.json"
ENV_FALLBACK_PATH = SCRIPT_DIR.parent / "src" / ".env"

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openai/gpt-4o-mini"
TEMP = 0.7
MAX_TOKENS = 150  # longer budget than the HotpotQA sweep's 60 -- these answers are sentences
PACING_SECS = 0.25
RECALL_THRESHOLD = 0.4  # see note below on why recall replaced F1
STOPWORDS = {"the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "and", "or",
             "that", "this", "it", "for", "on", "with", "as", "by", "at", "be", "has", "have"}


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
    prompt = f"Answer this question in 1-3 sentences:\n\n{question}"
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


def normalize_tokens(s):
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return set(t for t in s.split() if t and len(t) > 2) - STOPWORDS


def gold_recall(pred, gold):
    """Fraction of gold's (non-stopword) keywords present in the model's answer.

    NOT token-F1: F1 was tried first and produced a degenerate 0/25 "all wrong"
    result because it penalizes the model for answering more thoroughly/in
    different words than MS MARCO's terse one-sentence gold -- verified by
    reading transcripts, several were clearly correct (sometimes more precise
    than gold) despite F1 near 0. Recall against gold only checks whether the
    model's answer contains gold's key facts, not whether it matches gold's
    length/phrasing. Still an imperfect proxy for "factually correct" --
    genuine paraphrase with zero shared keywords scores 0 here too (seen in
    the n=25 pilot: "define preventive"). No LLM-judge dependency by design
    (see longform pilot discussion) -- keyword recall over an LLM judge, on
    purpose, at the cost of this kind of noise.
    """
    p, g = normalize_tokens(pred), normalize_tokens(gold)
    if not g:
        return 1.0
    return len(p & g) / len(g)


def is_correct(model_answer, gold_answer, threshold=RECALL_THRESHOLD):
    return gold_recall(model_answer, gold_answer) >= threshold


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


def bootstrap_ci(pairs, n=1000, seed=42):
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
        conf = (-0.1 if correct else -0.6) + rng.uniform(-0.05, 0.05)
        n_tok = rng.randint(15, 60)
        lp = [conf + rng.uniform(-0.1, 0.1) for _ in range(n_tok)]
        answer = q["answer"] if correct else "This is unrelated filler text."
        results.append({
            "id": q["id"], "question": q["question"], "gold": q["answer"],
            "model_answer": answer, "confidence": round(conf, 4), "n_tokens": n_tok,
            "correct": correct, "recall": 1.0 if correct else 0.0, "logprobs": lp,
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
        rec = gold_recall(answer, q["answer"])
        correct = rec >= RECALL_THRESHOLD
        results.append({
            "id": q["id"], "question": q["question"], "gold": q["answer"],
            "model_answer": answer, "confidence": conf, "n_tokens": n_tok,
            "correct": correct, "recall": round(rec, 4), "logprobs": lp,
        })
        print(
            f"  [{i+1}/{len(questions)}] ({time.time()-start:.1f}s) "
            f"conf={conf:+.3f} tok={n_tok} recall={rec:.2f} correct={correct}",
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
        f"{'DRY RUN — ' if args.dry_run else ''}longform pilot: model={args.model} "
        f"n={len(questions)} temp={TEMP} max_tokens={MAX_TOKENS} recall_threshold={RECALL_THRESHOLD}",
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

    summary = {
        "n": len(results),
        "n_correct": n_correct,
        "accuracy": n_correct / len(results) if results else None,
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
        "recall_threshold": RECALL_THRESHOLD,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": "pilot: single temperature, single provider. Correctness = "
                "keyword-recall(model_answer, gold) >= threshold, not exact match "
                "or F1 (gold answers are short; F1 was tried first and produced a "
                "degenerate all-wrong result by penalizing thorough/paraphrased "
                "correct answers -- see longform_pilot_results.json's n=25 run and "
                "the session notes). Keyword recall has its own known failure mode: "
                "genuine paraphrase with no shared keywords scores 0 too. No LLM- "
                "judge dependency, by design. Compare against HotpotQA's exact-match "
                "AUROC ~0.85 at temp 0.7 (temperature_sweep_results_or_bigN.json) -- "
                "not an apples-to-apples comparison given the different scoring method.",
        "results": results,
        "summary": summary,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\n--- Summary ({'DRY RUN' if args.dry_run else 'live'}) ---")
    print(f"n={summary['n']}  acc={summary['accuracy']:.2f}  "
          f"tokens: min={summary['min_tokens']} max={summary['max_tokens']} mean={summary['mean_tokens']:.1f}")
    if point is not None:
        print(f"AUROC={point:.3f}  95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")
    else:
        print("AUROC: degenerate (all labels same)")
    print(f"written: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
