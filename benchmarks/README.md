# Confidence-vs-correctness benchmark

Answers a specific question: does mean token logprob confidence actually correlate
with answer correctness on hard, multi-hop questions — independent of rag-gate's
own gating logic, just checking the raw signal?

## Method

25 "hard" difficulty questions from the [HotpotQA](https://hotpotqa.github.io/)
distractor validation split (bridge + comparison reasoning types — the standard
multi-hop QA benchmark), pulled live via the HuggingFace Datasets Server API (no
auth required). Each question was sent to `grok-3-mini` (xAI) with `logprobs: true`
and streaming enabled, no RAG context provided — a pure closed-book test of the
model's own knowledge and confidence calibration. Mean token logprob was computed
over the full streamed answer. Answer correctness was checked via simple substring
match against the gold answer (not exact match, since models often wrap the answer
in a short explanation).

## Results (as run 2026-07-18)

| | |
|---|---|
| Total questions | 25 |
| Correct | 11 (44%) |
| Wrong | 14 (56%) |
| Mean confidence, correct answers | -0.104 |
| Mean confidence, wrong answers | -0.149 |
| Best single-threshold accuracy | 72% (vs. 44% baseline) |

Confidence separates correct from incorrect answers in the expected direction,
but the distributions overlap substantially (correct range: -0.160 to -0.058;
wrong range: -0.293 to -0.057) — this is not a clean separator on its own. Using
the empirically best single threshold on this sample as an ANSWER/ABSTAIN cutoff
would have classified 72% of cases correctly, versus 44% if every answer were
accepted at face value.

## Caveats

- **n=25** — a directional signal, not a statistically robust estimate. Don't
  read the exact percentages as precise; read the ordering (wrong answers have
  lower confidence on average) as the finding.
- **One model, one dataset slice.** Different models calibrate differently;
  results will vary by model family and by how "hard" a given question sample
  actually is.
- **Not an independent replication of the RAG-Gate paper's AURC figures** — this
  is a quick sanity check of the same underlying idea, not a rerun of the paper's
  methodology or dataset.
- **No RAG context was provided** — this tests the model's own parametric
  knowledge and confidence, not a real retrieval-augmented pipeline. rag-gate
  itself is upstream-agnostic and would behave the same either way, but this
  benchmark doesn't test retrieval-augmented accuracy specifically.

## Recovery-retry benchmark (does re-generating a low-confidence answer help?)

`recovery_retry_eval.py` tests the premise behind rag-gate's **ESCALATE** path:
when confidence is low, is re-generating at temperature 0 (the `lower_temperature`
recovery strategy, intended for users with no stronger model to reroute to) worth
doing? It reuses the 25 answers above, calibrates an alpha threshold to *this*
distribution (the default -0.5 fires on nothing here — observed range ~-0.29..-0.06
— so it uses the median, -0.119), and re-asks the 12 lowest-confidence
("ESCALATE-band") questions at temperature 0.

### Results (as run 2026-07-21)

| | |
|---|---|
| ESCALATE-band questions retried | 12 of 25 (9 originally wrong, 3 correct) |
| Confidence raised on retry | **12 / 12** |
| Mean confidence lift | **+0.171** |
| Band accuracy | 25% → 33% (**1** wrong→right, 0 right→wrong, 11 unchanged) |

### The finding — read this before trusting the confidence number

Every retry came back at confidence ≈ **-0.000**. This is **not** "temp 0 makes
the model reliably confident" — it is a **measurement artifact of the confidence
metric itself**. At temperature 0 the model greedily selects the argmax token at
every step, and the logprob *of the token you selected* is near 0 precisely
because you always pick the most-likely one. So **lowering temperature
mechanically inflates mean-logprob confidence regardless of whether the answer
actually improved.**

Consequences:

- The **confidence-lift metric is not a valid success signal for this strategy** —
  temp-0 guarantees a lift by construction. Only the **correctness delta** is
  trustworthy here, and it is +1 of 12 (directional, tiny N, not significant).
- More broadly, this exposes a real property of rag-gate's core signal:
  **mean token logprob is temperature-dependent.** A pipeline running the upstream
  at temperature 0 will see near-perfect confidence on everything, and rag-gate's
  gate will effectively never fire. This affects *gating*, not just recovery — see
  the crate's Known Limitations.
- A naive `lower_temperature` recovery feature would therefore *look* successful on
  every dashboard (confidence histograms, token-savings) while barely moving answer
  quality. The candidate honest strategy — rerouting to a stronger model — is tested
  next, and it doesn't hold up cleanly either.

## Reroute benchmark (does sending a low-confidence answer to a STRONGER model fix it?)

`recovery_reroute_grok_eval.py` tests the other option-D strategy: on ESCALATE,
re-ask the question on a stronger model instead of the same one. (Cross-provider
targets `gpt-4o` and `claude-sonnet-5` were written too — `recovery_reroute_eval.py`,
`recovery_reroute_anthropic_eval.py` — but the OpenAI and Anthropic keys on hand
returned 401, so the live run used the working xAI key to reroute *within* the
Grok family: `grok-3-mini` → `grok-4.5`.) Judged on **correctness only** — no
cross-model confidence comparison, which sidesteps the temperature artifact above.

### Results (as run 2026-07-21)

| | temp-0 retry | reroute → grok-4.5 |
|---|---|---|
| ESCALATE-band retried | 12 | 11 (1 request timed out) |
| Band accuracy | 25% → 33% | 27% → 45% |
| wrong → right | 1 | 2 |
| right → wrong | 0 | 0 |
| unchanged | 11 | 9 |

(2 vs 1 wrong→right is one question; Fisher's exact p ≈ 0.6 — see below.)

In the pilot, rerouting fixed answers the smaller model was unsure about (e.g.
"Animorphs", "keyboard function keys") with zero regressions — but that
zero-regression result did not survive a bigger sample (see below).

**Read the two conclusions differently — they rest on different kinds of evidence:**

- **`lower_temperature` is disqualified *mechanistically*.** The temp-0 confidence
  lift is a known artifact (argmax token ⇒ logprob ≈ 0), so the strategy is
  guaranteed to "clear the gate" without necessarily improving the answer. That
  holds at any sample size — it's an argument about the metric, not about these 12
  data points.
- **`reroute` is *directionally positive but not statistically significant*, even
  at bigger N.** The pilot (2/11 wrong→right) was one question — noise. So we ran
  `reroute_bigN_eval.py`: 100 fresh HotpotQA questions on grok-3-mini, then the
  50-question below-median band rerouted to grok-4.5 (`reasoning_effort=low`; 45
  completed, 5 timed out). Result: band accuracy **35.6% → 42.2%**, but that is
  **6 wrong→right against 3 right→wrong — net +3 of 45.** Rerouting is *not* a
  free win: the stronger model fixes some answers and breaks others. The correct
  paired test (McNemar exact, two-sided) gives **p = 0.508 — not significant.**

  So across both the pilot and the 45-question run, the honest conclusion is:
  rerouting to a stronger model shows a *small positive lean with real downside
  (regressions)*, and we do **not** have evidence it reliably improves accuracy.
  It is not the clean win the first pilot suggested.

  (Methodology note / correction: the big-N script originally reported a Fisher's
  exact p ≈ 0.000 "significant" — that was the **wrong test**. Fisher on a
  base-correct × reroute-correct 2×2 measures association between two correlated
  columns of the *same* items and returns p≈0 trivially because 36/45 answers are
  unchanged; it says nothing about whether accuracy *changed*. McNemar on the
  discordant pairs (6 vs 3) is the right paired test, and it is not significant.
  The script now uses McNemar.)

Other caveats: the reroute here is *within-provider* (grok→grok), a weaker probe
than a fully independent stronger model (the cross-provider variants were blocked
on invalid OpenAI/Anthropic keys); and grok-4.5 is noticeably slower (one request
exceeded a 40s read timeout) — a real consideration for a latency-sensitive proxy,
and an argument for keeping any recovery feature opt-in, capped, and time-bounded.

## Reproducing

```bash
pip install requests
export GROK_API_KEY=your-xai-key   # or set it however your shell prefers
python hotpotqa_confidence_eval.py       # original confidence-vs-correctness eval
python recovery_retry_eval.py            # temp-0 recovery-retry eval (reads the first's results)
python recovery_reroute_grok_eval.py     # reroute grok-3-mini -> grok-4.5 (needs GROK_API_KEY)
# cross-provider reroute variants (need OPENAI_API_KEY / ANTHROPIC_API_KEY):
#   python recovery_reroute_eval.py            # -> gpt-4o
#   python recovery_reroute_anthropic_eval.py  # -> claude-sonnet-5
```

`hotpotqa_sample.json` is checked in so the exact same 25 questions are used by
default. To pull a fresh/different sample, call `fetch_hotpotqa_sample(n=..., offset=...)`
from a Python shell before running — it hits the HuggingFace Datasets Server API
directly, no key needed for that part.

Raw per-question results (question, gold answer, model's full answer, confidence,
correct/wrong) are in `hotpotqa_results.json`; the recovery-retry per-question
results (original vs. temp-0 retry answer, confidence, correctness) are in
`recovery_retry_results.json`.
