"""Recovery-strategy eval #2c: does REROUTING a low-confidence answer to a
stronger model in the SAME family (grok-3-mini -> grok-4.5) actually fix it?

The cross-provider reroute targets (gpt-4o, claude-sonnet-5) were blocked on
invalid keys, so this uses the working xAI key to reroute within the Grok family
instead. It's a within-provider test — a weaker probe of "a genuinely different
model helps" than cross-provider would be — but grok-4.5 is a real capability
jump over grok-3-mini, and it's the one we can actually run.

Judged ONLY by correctness transitions (wrong->right); no cross-model confidence
comparison (which would be inflated/incomparable — see recovery_retry_eval.py).

Method:
  1. Load hotpotqa_results.json (25 questions asked on grok-3-mini).
  2. Take the ESCALATE-band (confidence below the median).
  3. Re-ask each on grok-4.5 at default settings.
  4. Report correctness transitions vs. the original grok-3-mini answers.
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
OUT_PATH = SCRIPT_DIR / "recovery_reroute_grok_results.json"
API_URL = "https://api.x.ai/v1/chat/completions"
REROUTE_MODEL = "grok-4.5"
ORIGINAL_MODEL = "grok-3-mini"


def load_key():
    key = os.environ.get("GROK_API_KEY")
    if not key:
        raise RuntimeError("Set the GROK_API_KEY environment variable before running this script.")
    return key


def ask(key, question):
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
        timeout=(10, 40),
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
        f"ESCALATE-band questions to reroute {ORIGINAL_MODEL} -> {REROUTE_MODEL}: "
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
                "orig_model": ORIGINAL_MODEL,
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

    print(f"\n--- Reroute summary (ESCALATE-band: {ORIGINAL_MODEL} -> {REROUTE_MODEL}) ---")
    print(f"Rerouted: {len(results)}")
    print(f"Accuracy on this band: {orig_acc:.1%} -> {reroute_acc:.1%}")
    print(f"  wrong -> right: {wrong_to_right}")
    print(f"  right -> wrong: {right_to_wrong}")
    print(f"  unchanged:      {unchanged}")
    print(
        "\nNote: small N on hard multi-hop questions, within-provider reroute — "
        "treat as directional. Judged on correctness only."
    )

    json.dump(results, open(OUT_PATH, "w", encoding="utf-8"), indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
