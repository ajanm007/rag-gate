"""Recovery-strategy eval: does a lower-temperature RETRY on a low-confidence
answer actually raise confidence and/or fix wrong answers?

This is the upstream-side counterpart to hotpotqa_confidence_eval.py. That
script measured whether the raw logprob confidence signal separates correct from
wrong answers (the *gating* premise). This one measures the *recovery* premise
behind rag-gate's ESCALATE path: when confidence is low, is re-generating at
temperature 0 (the `lower_temperature` strategy, for users with no stronger
model to reroute to) worth doing?

Method:
  1. Load the existing hotpotqa_results.json (25 questions already asked at the
     model's default temperature, with per-answer mean logprob + correctness).
  2. Calibrate an alpha threshold to THIS distribution — the default -0.5 never
     fires here (observed range ~ -0.29..-0.06). We use the median confidence,
     so the low-confidence (wrong-heavy) tail falls into the ESCALATE band.
  3. Re-ask each ESCALATE-band question at temperature 0, same model.
  4. Report confidence lift and correctness transitions (wrong->right etc.).

Honest caveat baked into the output: N is small and these are hard multi-hop
questions, so treat the correctness deltas as directional, not significant.
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
OUT_PATH = SCRIPT_DIR / "recovery_retry_results.json"
API_URL = "https://api.x.ai/v1/chat/completions"
MODEL = "grok-3-mini"
RETRY_TEMPERATURE = 0.0


def load_key():
    key = os.environ.get("GROK_API_KEY")
    if not key:
        raise RuntimeError(
            "Set the GROK_API_KEY environment variable before running this script "
            "(e.g. an xAI API key from https://console.x.ai)."
        )
    return key


def ask(key, question, temperature=None):
    """Ask one question with logprobs on. Returns (answer_text, mean_logprob).

    Mirrors hotpotqa_confidence_eval.ask so original and retry are apples-to-
    apples; the only intended variable is `temperature`.
    """
    prompt = (
        f"Answer this question as concisely as possible — ideally a single word "
        f"or short phrase, no explanation:\n\n{question}"
    )
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "logprobs": True,
        "max_tokens": 60,
    }
    if temperature is not None:
        payload["temperature"] = temperature

    resp = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload,
        stream=True,
        timeout=(10, 20),
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
        chunk_payload = line[len("data: "):]
        if chunk_payload.strip() == "[DONE]":
            break
        chunk = json.loads(chunk_payload)
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
    return answer_text, mean_logprob


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
    # Calibrate alpha to this distribution. The default -0.5 fires on nothing
    # here; the median puts the low-confidence half into the ESCALATE band.
    alpha = statistics.median(confs)
    escalate_band = [r for r in original if r["confidence"] < alpha]

    print(f"Calibrated alpha (median): {alpha:+.4f}", flush=True)
    print(
        f"ESCALATE-band questions to retry at temp {RETRY_TEMPERATURE}: "
        f"{len(escalate_band)}/{len(original)}",
        flush=True,
    )

    results = []
    for i, orig in enumerate(escalate_band):
        q = orig["question"]
        gold = orig["gold"]
        start = time.time()
        try:
            retry_answer, retry_conf = ask(key, q, temperature=RETRY_TEMPERATURE)
        except Exception as e:
            print(f"[{i+1}/{len(escalate_band)}] ERROR after {time.time()-start:.1f}s: {e}", flush=True)
            continue

        retry_correct = is_correct(retry_answer, gold)
        results.append(
            {
                "id": orig["id"],
                "question": q,
                "gold": gold,
                "orig_answer": orig["model_answer"],
                "orig_confidence": orig["confidence"],
                "orig_correct": orig["correct"],
                "retry_answer": retry_answer,
                "retry_confidence": retry_conf,
                "retry_correct": retry_correct,
                "confidence_lift": retry_conf - orig["confidence"],
            }
        )
        print(
            f"[{i+1}/{len(escalate_band)}] ({time.time()-start:.1f}s) "
            f"conf {orig['confidence']:+.3f} -> {retry_conf:+.3f} "
            f"(lift {retry_conf - orig['confidence']:+.3f})  "
            f"correct {orig['correct']} -> {retry_correct}",
            flush=True,
        )

    if not results:
        print("No results — nothing to summarize.", flush=True)
        return

    lifts = [r["confidence_lift"] for r in results]
    raised = sum(1 for r in results if r["confidence_lift"] > 0)
    wrong_to_right = sum(1 for r in results if not r["orig_correct"] and r["retry_correct"])
    right_to_wrong = sum(1 for r in results if r["orig_correct"] and not r["retry_correct"])
    unchanged = len(results) - wrong_to_right - right_to_wrong

    orig_acc = sum(1 for r in results if r["orig_correct"]) / len(results)
    retry_acc = sum(1 for r in results if r["retry_correct"]) / len(results)

    print("\n--- Recovery summary (ESCALATE-band, temp-0 retry) ---")
    print(f"Retried: {len(results)}")
    print(f"Confidence raised on retry: {raised}/{len(results)}")
    print(f"Mean confidence lift: {statistics.mean(lifts):+.4f}")
    print(f"Accuracy on this band:  {orig_acc:.1%} -> {retry_acc:.1%}")
    print(f"  wrong -> right: {wrong_to_right}")
    print(f"  right -> wrong: {right_to_wrong}")
    print(f"  unchanged:      {unchanged}")
    print(
        "\nNote: small N on hard multi-hop questions — treat correctness deltas "
        "as directional, not statistically significant."
    )

    json.dump(results, open(OUT_PATH, "w", encoding="utf-8"), indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
