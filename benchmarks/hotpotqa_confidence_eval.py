import json
import os
import re
import statistics
import time
from pathlib import Path

import requests

SCRIPT_DIR = Path(__file__).parent
DATA_PATH = SCRIPT_DIR / "hotpotqa_sample.json"
RESULTS_PATH = SCRIPT_DIR / "hotpotqa_results.json"
API_URL = "https://api.x.ai/v1/chat/completions"
MODEL = "grok-3-mini"


def load_key():
    key = os.environ.get("GROK_API_KEY")
    if not key:
        raise RuntimeError(
            "Set the GROK_API_KEY environment variable before running this script "
            "(e.g. an xAI API key from https://console.x.ai)."
        )
    return key


def load_questions():
    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return [r["row"] for r in data["rows"]]


def fetch_hotpotqa_sample(n=25, offset=0):
    """Pulls a fresh sample from the HuggingFace Datasets Server (no auth needed)
    and writes it to DATA_PATH in the same shape this script expects."""
    url = (
        "https://datasets-server.huggingface.co/rows"
        f"?dataset=hotpotqa/hotpot_qa&config=distractor&split=validation"
        f"&offset={offset}&length={n}"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(resp.json(), f, indent=2)


def ask(key, question):
    prompt = (
        f"Answer this question as concisely as possible — ideally a single word "
        f"or short phrase, no explanation:\n\n{question}"
    )
    resp = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "logprobs": True,
            "max_tokens": 60,
        },
        stream=True,
        timeout=(10, 20),  # (connect timeout, read timeout) — read applies per chunk
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
    questions = load_questions()
    results = []

    print(f"Starting eval of {len(questions)} questions...", flush=True)
    for i, q in enumerate(questions):
        question = q["question"]
        gold = q["answer"]
        start = time.time()
        try:
            answer, conf = ask(key, question)
        except Exception as e:
            print(f"[{i+1}/{len(questions)}] ERROR after {time.time()-start:.1f}s: {e}", flush=True)
            continue
        correct = is_correct(answer, gold)
        results.append(
            {
                "id": q["id"],
                "level": q["level"],
                "type": q["type"],
                "question": question,
                "gold": gold,
                "model_answer": answer,
                "confidence": conf,
                "correct": correct,
            }
        )
        print(
            f"[{i+1}/{len(questions)}] ({time.time()-start:.1f}s) conf={conf:+.3f} correct={correct} "
            f"gold='{gold}' got='{answer[:40]}'",
            flush=True,
        )

    correct_confs = [r["confidence"] for r in results if r["correct"]]
    wrong_confs = [r["confidence"] for r in results if not r["correct"]]

    print("\n--- Summary ---")
    print(f"Total: {len(results)}  Correct: {len(correct_confs)}  Wrong: {len(wrong_confs)}")
    if correct_confs:
        print(f"Mean confidence when CORRECT: {statistics.mean(correct_confs):+.4f}")
    if wrong_confs:
        print(f"Mean confidence when WRONG:   {statistics.mean(wrong_confs):+.4f}")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
