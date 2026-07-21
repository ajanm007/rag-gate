"""Recovery-strategy eval #2: does REROUTING a low-confidence answer to a
stronger model actually fix it?

This is the honest counterpart to recovery_retry_eval.py. That script showed
`lower_temperature` retry inflates the confidence metric (temp 0 -> argmax token
-> logprob ~0) without improving answers. Rerouting to a *stronger model* is the
other option-D strategy, and the one the previous eval pointed to as promising.

Because cross-model confidence isn't comparable and low temperature inflates it
anyway, success here is judged ONLY by correctness transitions (wrong->right),
not by any confidence number.

Method:
  1. Load hotpotqa_results.json (25 questions asked on grok-3-mini).
  2. Take the same ESCALATE-band as recovery_retry_eval: confidence below the
     median (the default -0.5 alpha fires on nothing on this distribution).
  3. Re-ask each on a stronger OpenAI model (gpt-4o) at its default temperature.
  4. Report correctness transitions vs. the original grok answers.

Two variables move at once here (model AND provider), so this measures "reroute
to gpt-4o" as a whole, not "gpt-4o vs grok" in isolation — which is exactly the
operational question a reroute strategy answers.
"""

import json
import os
import re
import statistics
import time
from pathlib import Path

import requests

SCRIPT_DIR = Path(__file__).parent
ORIGINAL_RESULTS = SCRIPT_DIR / "hotpotqa_results.json"
OUT_PATH = SCRIPT_DIR / "recovery_reroute_results.json"
API_URL = "https://api.openai.com/v1/chat/completions"
REROUTE_MODEL = "gpt-4o"


def load_key():
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "Set the OPENAI_API_KEY environment variable before running this script."
        )
    return key


def ask(key, question):
    """Ask the reroute model one question at its default temperature. Returns
    the answer text. Non-streaming — we only need the final answer, not tokens."""
    prompt = (
        f"Answer this question as concisely as possible — ideally a single word "
        f"or short phrase, no explanation:\n\n{question}"
    )
    resp = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": REROUTE_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 60,
        },
        timeout=(10, 30),
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


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


def main():
    key = load_key()
    original = json.load(open(ORIGINAL_RESULTS, encoding="utf-8"))

    confs = sorted(r["confidence"] for r in original)
    alpha = statistics.median(confs)
    escalate_band = [r for r in original if r["confidence"] < alpha]

    print(f"Calibrated alpha (median): {alpha:+.4f}", flush=True)
    print(
        f"ESCALATE-band questions to reroute to {REROUTE_MODEL}: "
        f"{len(escalate_band)}/{len(original)}",
        flush=True,
    )

    results = []
    for i, orig in enumerate(escalate_band):
        q = orig["question"]
        gold = orig["gold"]
        start = time.time()
        try:
            reroute_answer = ask(key, q)
        except Exception as e:
            print(f"[{i+1}/{len(escalate_band)}] ERROR after {time.time()-start:.1f}s: {e}", flush=True)
            continue

        reroute_correct = is_correct(reroute_answer, gold)
        results.append(
            {
                "id": orig["id"],
                "question": q,
                "gold": gold,
                "orig_answer": orig["model_answer"],
                "orig_correct": orig["correct"],
                "reroute_model": REROUTE_MODEL,
                "reroute_answer": reroute_answer,
                "reroute_correct": reroute_correct,
            }
        )
        print(
            f"[{i+1}/{len(escalate_band)}] ({time.time()-start:.1f}s) "
            f"correct {orig['correct']} -> {reroute_correct}  "
            f"gold='{gold}' got='{reroute_answer[:40]}'",
            flush=True,
        )

    if not results:
        print("No results — nothing to summarize.", flush=True)
        return

    wrong_to_right = sum(1 for r in results if not r["orig_correct"] and r["reroute_correct"])
    right_to_wrong = sum(1 for r in results if r["orig_correct"] and not r["reroute_correct"])
    unchanged = len(results) - wrong_to_right - right_to_wrong
    orig_acc = sum(1 for r in results if r["orig_correct"]) / len(results)
    reroute_acc = sum(1 for r in results if r["reroute_correct"]) / len(results)

    print(f"\n--- Reroute summary (ESCALATE-band -> {REROUTE_MODEL}) ---")
    print(f"Rerouted: {len(results)}")
    print(f"Accuracy on this band: {orig_acc:.1%} (grok) -> {reroute_acc:.1%} ({REROUTE_MODEL})")
    print(f"  wrong -> right: {wrong_to_right}")
    print(f"  right -> wrong: {right_to_wrong}")
    print(f"  unchanged:      {unchanged}")
    print(
        "\nNote: small N on hard multi-hop questions, and provider+model both "
        "change — treat as directional. Judged on correctness only (no "
        "cross-model confidence comparison)."
    )

    json.dump(results, open(OUT_PATH, "w", encoding="utf-8"), indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
