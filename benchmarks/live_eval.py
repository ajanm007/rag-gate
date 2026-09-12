"""Live-eval harness: point rag-gate at your own model + questions, in one command.

Takes a JSON file of {question, gold_answer} pairs, calls a real
OpenAI-compatible chat-completions API for each question (streaming, with
logprobs), scores each answer by exact match (trim + lowercase both sides),
writes a results file in the exact shape `rag-gate evaluate` already reads,
then calls the already-built `rag-gate evaluate --dataset <file> --json`
binary as a subprocess and prints its report.

Architectural rule: this script NEVER re-implements scoring math (no AURC,
no threshold search, no coverage/risk computation). All of that lives in
exactly one place -- the Rust binary. This script only produces
{confidence, correct} records, calls the binary, and relays what it says.

Correctness is exact match only, deliberately. Fuzzier scorers (keyword
recall etc., as used by this repo's internal research benchmarks on
open-ended prose) would make the verdict arguable; a user-facing
credibility tool needs a deterministic, undebatable correctness function.

Usage:
    python live_eval.py \
      --dataset questions.json \
      --upstream https://openrouter.ai/api \
      --model openai/gpt-4o-mini \
      --output results.json \
      --rag-gate-bin ../target/release/rag-gate.exe \
      [--temperature 0.7] \
      [--target-coverage 0.8]

The streaming/logprob-extraction logic is ported from
benchmarks/temperature_sweep_eval.py's ask() (same SSE parsing approach:
accumulate delta.content, collect choices[0].logprobs.content[].logprob).
Simplified: no reasoning_effort option, no retries -- a failed request
aborts the whole run with a clear error naming the question.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
# Reference scripts fall back to this file for API keys; mirror that so a
# user with a working benchmarks/ setup needs no new configuration.
ENV_FALLBACK_PATH = SCRIPT_DIR.parent / "src" / ".env"
if not ENV_FALLBACK_PATH.exists():
    ENV_FALLBACK_PATH = SCRIPT_DIR / "src" / ".env"

MAX_TOKENS = 60

# An empty logprob list means "no confidence signal at all". The Rust side
# (src/eval.rs) uses f64::NEG_INFINITY for this case. Python's json module
# cannot emit -Infinity in a way serde_json will parse (it writes the bare
# token `-Infinity`, which is invalid JSON and serde_json rejects it), so we
# write a very large negative finite number instead. It sorts below any real
# mean logprob (which live in roughly [-20, 0]) and round-trips cleanly
# through serde_json as f64. Verified in gate G5.
EMPTY_LOGPROB_CONFIDENCE = -1e18


def load_key(key_env, explicit_key=None):
    """Resolve the API key: explicit flag first, then env, then src/.env."""
    if explicit_key:
        return explicit_key
    key = os.environ.get(key_env)
    if not key and ENV_FALLBACK_PATH.exists():
        for line in ENV_FALLBACK_PATH.read_text(encoding="utf-8").splitlines():
            if line.startswith(f"{key_env}="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                break
    if not key:
        raise RuntimeError(
            f"Set the {key_env} environment variable (or put it in "
            f"{ENV_FALLBACK_PATH}), or pass --api-key explicitly."
        )
    return key


def ask(key, question, temperature, upstream, model):
    """Ask one question; return (answer_text, mean_logprob).

    Ported from benchmarks/temperature_sweep_eval.py's ask(): POST with
    stream + logprobs, iterate the SSE lines, accumulate delta.content into
    the answer and choices[0].logprobs.content[].logprob into a list.
    Raises on any transport/API failure -- the caller aborts the run.
    """
    prompt = (
        "Answer this question as concisely as possible — ideally a single word "
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
    url = upstream.rstrip("/") + "/v1/chat/completions"
    resp = requests.post(
        url,
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
    if logprobs:
        mean_logprob = sum(logprobs) / len(logprobs)
    else:
        mean_logprob = EMPTY_LOGPROB_CONFIDENCE
    return answer_text, mean_logprob


def is_correct(model_answer, gold_answer):
    """Exact match after trimming whitespace and lowercasing both sides."""
    return model_answer.strip().lower() == gold_answer.strip().lower()


def load_dataset(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(
            f"dataset {path}: expected a JSON array of "
            '{"question": ..., "gold_answer": ...} objects'
        )
    for i, row in enumerate(data):
        if not isinstance(row, dict) or "question" not in row or "gold_answer" not in row:
            raise ValueError(
                f"dataset {path}: record {i} must be an object with "
                '"question" and "gold_answer" fields'
            )
    return data


def default_rag_gate_bin():
    """Sane default relative to this script's own location, overridable.

    Prefers a native binary next to the .exe-named one when both exist;
    falls back to the .exe name (this repo's checked-in build artifact).
    """
    native = SCRIPT_DIR / ".." / "target" / "release" / "rag-gate"
    exe = SCRIPT_DIR / ".." / "target" / "release" / "rag-gate.exe"
    if native.exists():
        return str(native)
    return str(exe)


def path_for_subprocess(path, rag_gate_bin):
    """Translate a path for the rag-gate binary subprocess if needed.

    The repo's checked-in artifact is a Windows .exe, which -- when invoked
    from WSL via interop -- cannot resolve WSL-absolute paths such as
    /tmp/x.json or /mnt/d/... (it sees the Windows filesystem). Relative
    paths work because the child inherits the working directory, so this
    only rewrites absolute /mnt/<drive>/... paths to <Drive>:\\... form,
    and only when the binary is a Windows .exe. Everything else passes
    through untouched.
    """
    p = str(path)
    if not rag_gate_bin.lower().endswith(".exe"):
        return p
    if len(p) > 6 and p.startswith("/mnt/") and p[6] == "/":
        drive = p[5].upper()
        rest = p[7:].replace("/", "\\")
        return f"{drive}:\\{rest}"
    return p


def run_evaluate_subprocess(rag_gate_bin, dataset_path, target_coverage):
    """Call `rag-gate evaluate --dataset ... --json` and return its report.

    Returns the parsed JSON dict. Raises RuntimeError (with the binary's
    own stderr surfaced) on any failure -- never synthesizes numbers here.
    """
    if not Path(rag_gate_bin).exists():
        raise RuntimeError(
            f"rag-gate binary not found at {rag_gate_bin} "
            "(pass --rag-gate-bin with the path to the built binary)"
        )
    cmd = [
        rag_gate_bin,
        "evaluate",
        "--dataset",
        path_for_subprocess(dataset_path, rag_gate_bin),
        "--json",
        "--target-coverage",
        str(target_coverage),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except OSError as e:
        raise RuntimeError(f"could not launch rag-gate binary ({e})")
    if proc.returncode != 0:
        raise RuntimeError(
            f"rag-gate evaluate failed (exit {proc.returncode}):\n{proc.stderr.strip()}"
        )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"rag-gate evaluate printed non-JSON stdout ({e}):\n{proc.stdout[:2000]}"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="JSON file of {question, gold_answer} pairs")
    parser.add_argument("--upstream", required=True, help="base URL, e.g. https://openrouter.ai/api")
    parser.add_argument("--model", required=True, help="model id, e.g. openai/gpt-4o-mini")
    parser.add_argument("--output", required=True, help="where to write the results JSON file")
    parser.add_argument(
        "--rag-gate-bin",
        default=None,
        help="path to the built rag-gate binary (default: relative to this script)",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--target-coverage", type=float, default=0.8)
    parser.add_argument(
        "--key-env",
        default="OPEN_ROUTER_KEY",
        help="env var holding the upstream API key (also read from src/.env)",
    )
    parser.add_argument("--api-key", default=None, help="upstream API key (overrides --key-env)")
    args = parser.parse_args(argv)

    rag_gate_bin = args.rag_gate_bin or default_rag_gate_bin()

    try:
        questions = load_dataset(args.dataset)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        print(f"live_eval: error: {e}", file=sys.stderr)
        return 2
    if not questions:
        print(f"live_eval: error: dataset {args.dataset} contains no questions", file=sys.stderr)
        return 2

    try:
        key = load_key(args.key_env, args.api_key)
    except RuntimeError as e:
        print(f"live_eval: error: {e}", file=sys.stderr)
        return 2

    # Collect everything in memory; the output file is written only after
    # ALL questions succeed, so a failure never leaves a partial file.
    results = []
    for i, row in enumerate(questions):
        question, gold = row["question"], row["gold_answer"]
        try:
            answer, confidence = ask(key, question, args.temperature, args.upstream, args.model)
        except Exception as e:
            print(
                f"live_eval: error: question {i} ({question!r}) failed: {e}\n"
                "live_eval: aborting run; no results file written. "
                "Fix the problem and re-run the whole command.",
                file=sys.stderr,
            )
            return 1
        correct = is_correct(answer, gold)
        results.append(
            {
                "confidence": confidence,
                "correct": correct,
                "question": question,
                "gold_answer": gold,
                "model_answer": answer,
            }
        )
        print(
            f"[{i + 1}/{len(questions)}] conf={confidence:+.4f} "
            f"correct={correct} q={question!r} a={answer!r}",
            file=sys.stderr,
        )

    try:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"results": results}, f, indent=2)
    except OSError as e:
        print(f"live_eval: error: could not write {args.output}: {e}", file=sys.stderr)
        return 1
    print(f"live_eval: wrote {len(results)} results to {args.output}", file=sys.stderr)

    try:
        report = run_evaluate_subprocess(rag_gate_bin, args.output, args.target_coverage)
    except RuntimeError as e:
        print(f"live_eval: error: {e}", file=sys.stderr)
        return 1

    # The final report is the Rust binary's own output, relayed -- nothing
    # is computed from the results in Python.
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
