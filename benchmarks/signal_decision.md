# Signal decision: what does the runtime trust, and why

No sweep results existed in this repo at the time the rules below were fixed. The sweep's job is to fill the one open cell in the decision tree; the rules decide how the runtime (Rust data plane today, whatever serves it tomorrow) configures its signal strategy.

## The decision tree

```
logprob ──► strong + stable?          ──► calibrate it (plain thresholds)
        ──► temperature-sensitive?    ──► temperature-aware calibration
        ──► weak?                     ──► move to hidden-state signals
retrieval + logprob ──► better?       ──► (that was the V1 result)
hidden state ──► incremental over both? ──► (that is the V2 result)
```

## What is already answered (evidence map)

| Question | Answer | Source |
| --- | --- | --- |
| Does logprob carry information about correctness at all? | Yes — recovers 45–55% of the random→oracle AURC gap in clean multi-hop RAG, where retrieval scores stay flat at 8–12% | V1 paper ("The Cost of Confidence", CODS-COMAD 2026 submission); recap in `../rag-gate-prd.md` §1 |
| Practical separation on the crate's own 25-question Grok sample? | Directional only — mean conf −0.104 (correct) vs −0.149 (wrong), heavily overlapping ranges; best single threshold 72% vs 44% accept-all baseline | `README.md` (benchmarks section), `hotpotqa_results.json` |
| Stable across datasets / adversarial pressure? | **No** — collapses under adversarial distractors (MuSiQue) | V2 PRD §1 recap of V1 findings |
| Stable across model scales? | **No** — thresholds do not transfer across scales | V2 PRD §1 |
| Does combining retrieval + logprob close the gap? | **No** — linear combiners leave 0.10–0.16 AURC of oracle headroom unrealized (the V2 motivation) | V2 PRD §1 |
| Temperature-sensitive? | **Qualitatively yes at the extreme** — every temp-0 retry returns confidence ≈ −0.000 (12/12), an argmax measurement artifact, mechanistically argued to hold at any N | `README.md` (recovery-retry benchmark) |
| Hidden state incremental over logprob? | **In flight** — probe AUROC 0.7269 vs logprob 0.6752 on HotpotQA (n=701 test, 95% CI \[−0.019, +0.123\], includes 0); MuSiQue go/no-go running | research notebooks 28–31 (Kaggle), D:\\Research\\RAG-Gate-V2 |
| Is escalation (reroute to stronger model) a free win? | **No** — net +3/45, McNemar p = 0.508 | `README.md` (reroute big-N benchmark) |

## The open cell

Everything above pins behavior at temperature extremes (the artifact) or ignores temperature entirely. Missing: a **quantitative sweep** of how much signal mean-logprob retains as temperature moves through realistic serving values, and how far the operating threshold drifts. That is what `temperature_sweep_eval.py` measures.

## Pre-registered decision rules

Fixed 2026-08-22, before the sweep was run. Metric per temperature T: AUROC of mean-logprob as a discriminator of correctness, with a bootstrap 95% CI (1000 resamples, seed 42). Stability set **S = {0.2, 0.5, 0.7, 1.0}**; temperature 0.0 is analyzed separately as the artifact cell and never mixed into stability statistics.

- **R1 — ranking stability.** `range = max−min` of point AUROCs over S.
  - range &lt; 0.05 → ranking stable
  - 0.05 ≤ range &lt; 0.10 → marginal
  - range ≥ 0.10 → unstable
- **R2 — operating-point drift.** τ80(T) = the mean-logprob threshold that keeps \~80% coverage at T (20th percentile of confidence). `drift = max−min` of τ80 over S.
  - drift &lt; 0.10 nats → thresholds portable across temperature
  - 0.10 ≤ drift ≤ 0.25 → per-temperature recalibration justified
  - drift &gt; 0.25 nats → thresholds non-portable; temperature normalization required
- **R3 — strength floor.** If point AUROC &lt; 0.60 at two or more temperatures in S → logprob is demoted as a default signal; hidden-state work accelerates.
- **R4 — temp-0 verdict.** m0 = largest |logprob| observed across all temp-0 answers.
  - m0 ≤ 0.05 → hard floor confirmed → the runtime guard at temp≈0 should be a hard refuse-to-gate
  - m0 &gt; 0.05 → gradation exists → warn-only plus temperature-aware calibration
- **R5 — readability gate.** If any cell's CI half-width &gt; 0.15, that cell is marked *unreadable (n=25)* and no conclusion is drawn from it until the 100-question sample is run.
- **R6 — scope.** Closed-book (no retrieval context), single model (grok-3-mini), single provider; interaction with retrieval quality is out of scope for this sweep and remains bounded by the V1 combiner result above.

## Outcome → runtime action

| Sweep verdict (by the rules above) | Runtime action |
| --- | --- |
| Ranking stable + thresholds portable (R1 &lt;0.05, R2 &lt;0.10) | Plain calibrated thresholds; today's `/calibrate` endpoint suffices; add the temp≈0 guard |
| Ranking stable + drift (R1 &lt;0.05, R2 ≥0.10) | Temperature-aware calibration: normalize confidence by serving temperature before thresholding |
| Unstable or below strength floor (R1 ≥0.10 or R3) | Demote mean-logprob from default signal; prioritize hidden-state/probe signal work |
| R4 hard floor | Ship the degenerate-stream guard as hard refuse-to-gate |
| R4 gradation | Ship the guard as warn-only |

## Interpretation discipline

n=25 per cell is a directional instrument, not an effect-size instrument: the sweep's job is ordering and stability, mirroring the caveats the original 25-question eval already carries. Conclusions are tagged *directional (n=25)* unless the 100-question escalation has confirmed them.

## Protocol amendment (2026-08-22, mid-run, before any results were read)

The first live run revealed grok-3-mini's default reasoning stream dominates the token counts (60–2,493 logprob-bearing tokens per answer vs `max_tokens=60` — reasoning tokens are not counted against that cap on xAI). The running mean was therefore averaging mostly over reasoning chatter, not answer tokens — a different, much noisier signal than the one V1's answer-token evidence describes, and one whose per-question token-count bimodality (60 vs 2,500) adds variance unrelated to the question under test.

Amendment, decided on observing token counts only (no AUROC or summary output had been produced or read at this point):

- **Primary condition:** `reasoning_effort: "none"` (verified accepted by grok-3-mini; drops generation to answer-only tokens). All pre-registered rules R1–R6 apply to this condition.
- **Secondary condition:** the in-flight default-reasoning run was allowed to complete as a robustness comparison (`temperature_sweep_results.json`), reported alongside but not used for rule verdicts.
- The temp-0.0 artifact cell is unaffected by the amendment concern (its predicted signature — every logprob ≈ 0 — is about the sampling distribution, not which tokens carry it) and was confirmed live before the amendment.

## Results and verdicts (2026-08-22)

Primary condition — grok-3-mini, `reasoning_effort: "none"`, 25 questions (`temperature_sweep_results_nothinking.json`):

| temp | acc | AUROC | 95% CI | τ80 | max abs logprob |
| --- | --- | --- | --- | --- | --- |
| 0.0 | 0.56 | 0.494 | \[0.256, 0.746\] | −0.000 | **0.0000** |
| 0.2 | 0.52 | 0.612 | \[0.390, 0.851\] | −0.039 | 0.4287 |
| 0.5 | 0.56 | 0.649 | \[0.417, 0.865\] | −0.354 | 0.6968 |
| 0.7 | 0.52 | 0.558 | \[0.325, 0.792\] | −0.683 | 1.7280 |
| 1.0 | 0.32 | 0.772 | \[0.522, 0.956\] | −1.023 | 2.8148 |

Rule-by-rule verdicts:

- **R4 — FIRED, decisively.** m0 = 0.0000 ≤ 0.05 (not a single temp-0 logprob deviated from zero at all). Hard floor confirmed: the runtime guard at temp≈0 is a **hard refuse-to-gate** (this is what the shipped guard does — warn, count, stop gating the stream).
- **R2 — FIRED, decisively.** τ80 drift over S = 0.983 nats &gt; 0.25. The threshold that holds 80% coverage moves from −0.04 to −1.02 across the temperature range. **Thresholds are non-portable across temperature; temperature normalization is required.** This rests on a distribution-location statistic, not the AUROC ranking, and a \~1-nat systematic shift is far beyond n=25 sampling noise.
- **R1 — fired, directional only.** AUROC range 0.214 ≥ 0.10 → unstable, but see R5.
- **R3 — did not fire.** Only one temperature in S had point AUROC &lt; 0.60 (0.7: 0.558). Logprob retains directional discrimination, and is strongest at the highest temperature (1.0: 0.772, the only CI excluding 0.5) — consistent with low temperature compressing the distribution toward the argmax artifact.
- **R5 — fired for every AUROC cell.** All CI half-widths ≈ 0.22–0.25 &gt; 0.15. Every ranking-level conclusion above is tagged **directional (n=25)**; the τ80-drift and temp-0-floor verdicts stand on their own statistics.

Secondary condition — default reasoning ON (robustness comparison, `temperature_sweep_results.json`): with grok-3-mini's reasoning tokens in the stream (60–2,493 tokens per answer), the signal is effectively dead at serving temperatures — AUROC 0.487–0.617, every CI covering 0.5, every cell unreadable under R5. Operational implication: against reasoning models that stream reasoning tokens with logprobs, mean-over-all-tokens confidence must not be trusted as currently computed.

### Escalation run — n=100 (`temperature_sweep_results_nothinking_bigN.json`)

R5 fired for every n=25 AUROC cell, so per protocol the same reasoning-off condition was re-run on the 100-question `bigN` sample (500 calls; one temp-0.5 response unscored, n=99 there):

| temp | n | acc | AUROC | 95% CI | τ80 | max abs mean logprob |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0 | 100 | 0.33 | 0.582 | \[0.451, 0.706\] | −0.000 | **0.0000** |
| 0.2 | 100 | 0.32 | 0.671 | \[0.563, 0.770\] | −0.196 | 3.8765 |
| 0.5 | 99 | 0.31 | **0.824** | \[0.729, 0.909\] | −0.590 | 3.2549 |
| 0.7 | 100 | 0.31 | **0.844** | \[0.760, 0.919\] | −0.793 | 3.8447 |
| 1.0 | 100 | 0.30 | **0.820** | \[0.717, 0.905\] | −1.261 | 4.1598 |

Final rule verdicts (all cells now readable — CI half-widths 0.08–0.13 ≤ 0.15):

- **R5 — cleared at n=100.** Every cell readable; ranking-level verdicts below are statistically licensed, not directional.
- **R3 — does not fire.** No serving temperature has AUROC &lt; 0.60. Mean-logprob is a **strong ranker** at temps ≥ 0.5: AUROC 0.824/0.844/0.820, all CIs far above 0.5. (The n=25 escalation was worth it: it estimated these at 0.558–0.772 with unreadable CIs.)
- **R1 — fires by the letter (0.174 ≥ 0.10), but the structure is not noise**:the point estimates form a stable plateau across 0.5–1.0 (range 0.024, heavily overlapping CIs) with a single low-temp outlier at 0.2 (0.671). Read as: *ranking is stable across the standard serving range and degrades toward low temperature*, hitting the dead floor at 0.0.
- **R2 — fires decisively (drift 1.064 nats &gt; 0.25).** τ80 walks monotonically −0.196 → −0.590 → −0.793 → −1.261 across 0.2 → 1.0. Confirmed at n=100 (0.98 at n=25). **Temperature normalization is mandatory.** The τ80(T) curve itself is a ready-made empirical normalization function.
- **R4 — fires with 100 samples of it: m0 = 0.0000.** Hard floor; hard refuse-to-gate guard is correct behavior (shipped).

### Final composite verdict

**Mean-logprob is a strong, statistically confirmed answer-correctness ranker (AUROC ≈ 0.82–0.84) across the standard serving range 0.5–1.0 — but its operating point drifts \~1.1 nats across that range and dies entirely at temperature 0.** The runtime action, per the outcome table: keep logprob as the default signal behind (a) the temperature-aware normalization/ calibration now backed by the measured τ80(T) curve, (b) the hard temp-0 guard (shipped), and (c) documented degradation below temp 0.5 — the regime where the V2 hidden-state signal has the clearest opening, since it does not inherit the sampling- temperature artifact.

## Reference upstream retired: grok-3-mini is gone (2026-08-23)

Everything above was measured against `grok-3-mini` via xAI. As of this writing that model **no longer exists** — confirmed live against the account's own key: it is absent from `GET /v1/models`, not merely redirected. xAI deprecated the grok-3/grok-4-0709/grok-4/grok-4-1-fast family on 2026-05-15, retired 2026-08-15; requests now resolve to `grok-4.3`. Worse, the failure mode is silent: a live request to `grok-4.3` with `logprobs: true` set returns 200 with a complete answer and **no** `logprobs` **field in the response at all** — not empty, not zero, absent. rag-gate's own degenerate-guard (built for the near-zero-but-present case) does not catch this; the crate gained a new guard for exactly this shape the same day (`raggate_no_logprob_signal_total`, see README Known Limitations) so this failure is visible rather than silently masquerading as ungated-but-fine.

The same pattern independently hit the paper's Groq/Qwen evidence trail: `qwen/qwen3-32b` (used in `scripts/make_02_groq.py` et al., where logprobs were required for the 72B baseline) was deprecated by Groq 2026-06-17, fully retired by August 2026. Its recommended successors — `openai/gpt-oss-120b` and `qwen/qwen3.6-27b` — both hard-reject the `logprobs` parameter with a 400 (confirmed live, all current Groq chat models tested: `openai/gpt-oss-20b`, `qwen/qwen3.6-27b`, `allam-2-7b`, `groq/compound` all reject it identically). Two independent research/evidence trails, two independent providers, both reference models deprecated within roughly the same month, both replacement lineups dropping the signal entirely — this reads as an industry-wide contraction in inference-time logprob availability, not one provider's isolated call.

Full live-verified state as of 2026-08-23, every option actually tried against a real key (not read from docs):

| Upstream | Free tier? | Logprobs? | Verified |
| --- | --- | --- | --- |
| OpenAI direct (GPT-3.5 free tier) | Yes | No logprobs access | Docs + free-tier model list |
| OpenAI direct (GPT-4o, paid) | No | Supported | Docs (key on hand was invalid, not live-tested direct) |
| OpenAI direct (GPT-5 line) | No | Deprecated even paid | Docs |
| Anthropic (any endpoint) | — | Never supported | Docs; no valid key on hand to double check |
| Gemini AI Studio | Yes | Hard-disabled, 0/50 models | Live probe, `gemini_logprob_probe.py` |
| Gemini Vertex AI | No (GCP billing) | Documented, live-unverified | Docs only, no OAuth cred |
| Ollama (any endpoint) | Yes (local) | Never supported | Upstream issue tracker |
| xAI grok-4.3 (current default) | No | **Silently absent** | Live request, this session |
| Groq (all current chat models) | Yes | **Hard-rejected, 400** | Live, 4 models tested |
| Together AI | — | Documented support | Docs only, key on hand was invalid |
| OpenRouter `stealth/ox-alpha` | Yes | Field present, always `null` | Live, this session |
| **OpenRouter** `openai/gpt-4o-mini` | **Effectively yes** | **Confirmed working** | **Live, 9/9 calls, real per-token logprobs, $0 balance, zero incremental cost registered** |

## New reference upstream: `openai/gpt-4o-mini` via OpenRouter

Confirmed live 2026-08-23: `POST https://openrouter.ai/api/v1/chat/completions`, `model: "openai/gpt-4o-mini"`, `logprobs: true` returns the exact OpenAI shape (`choices[0].logprobs.content[].logprob`) rag-gate already parses — **no code changes needed**, `Protocol::Sse` handles it as-is. 9 consecutive test calls all returned real per-token logprobs; `total_usage` did not increase across the last 8 of them despite a $0 credit balance, suggesting OpenRouter is fronting a small free-trial allowance on this low-cost model rather than hard-billing per call. This is not guaranteed to hold at production volume (a genuinely $0-balance account will eventually hit a paywall), but it is very likely sufficient to re-run the n=25/n=100 temperature-sweep protocol this file's evidence depends on.

**Next steps, in order:**

1. **Re-point** `temperature_sweep_eval.py` **at OpenRouter/**`gpt-4o-mini` instead of xAI. Same script shape (fixed sample, temperature loop, logprobs + streaming, substring-match correctness) — swap the base URL, model id, and auth header (`OPENROUTER_API_KEY` via `OPEN_ROUTER_KEY` in `src/.env`, `Authorization: Bearer`, no other request-shape changes since it's already OpenAI-compatible). Watch `total_usage` via `GET /api/v1/credits` between batches to catch a paywall before it burns the whole run.
2. **Re-run the pre-registered decision rules (R1–R6)** against fresh results before reading them, exactly as before — the rules themselves don't change, only the upstream does. Do not assume the old grok-3-mini verdicts transfer to a different model family; GPT-4o-mini's logprob behavior at temperature 0 and its AUROC/temperature curve are unmeasured and could differ materially.
3. **Live-verify rag-gate itself against this upstream end-to-end** — point the actual `rag-gate` binary at `https://openrouter.ai/api/v1` with `RAGGATE_UPSTREAM_URL`, confirm a real streamed request through the proxy produces logprobs, a correct decision frame, and that `raggate_no_logprob_signal_total` stays at zero (proving the new guard doesn't false-positive on a working upstream).
4. **Decide whether to also verify Together AI / a fresh OpenAI key** once valid credentials exist — both are docs-confirmed but untested live this session; a second confirmed-working upstream would de-risk relying on a single stealth-tier grace allowance.
5. **Update README's Known Limitations and Status** once a fresh sweep exists, replacing "no verified-working free-tier path" with the OpenRouter/gpt-4o-mini finding and its caveats (small-scale free-trial grace, not a guaranteed production-free path).
6. **Leave the historical grok-3-mini numbers in place, dated and marked unreproducible-as-of-2026-08** — they remain evidence of what was measured at the time; do not delete or silently overwrite them, only append.

## Reference-upstream re-run: GPT-4o-mini via OpenRouter (2026-08-23)

Executes the addendum's next steps 1–4. Rules R1–R6 unchanged (fixed 2026-08-22; the addendum mandated re-running them fresh against the new upstream, not editing them). 650 calls total, $0.0063 against the $0-credit grace allowance, polled between batches, never blocked.

n=25 (`temperature_sweep_results_or.json`) — every AUROC cell unreadable under R5 (CI half-widths 0.17–0.26); escalated per protocol.

n=100 (`temperature_sweep_results_or_bigN.json`) — every cell readable (CI half-widths 0.09–0.11):

| temp | n | acc | AUROC | 95% CI | τ80 |
| --- | --- | --- | --- | --- | --- |
| 0.0 | 100 | 0.35 | 0.822 | \[0.718, 0.908\] | −0.471 |
| 0.2 | 100 | 0.36 | 0.814 | \[0.718, 0.904\] | −0.463 |
| 0.5 | 100 | 0.37 | 0.728 | \[0.616, 0.831\] | −0.532 |
| 0.7 | 100 | 0.37 | 0.804 | \[0.710, 0.889\] | −0.778 |
| 1.0 | 100 | 0.34 | 0.795 | \[0.692, 0.890\] | −1.180 |

Fresh verdicts (GPT-4o-mini, n=100):

- **R5 — cleared.** All cells readable.
- **R1 — STABLE.** AUROC range over S = 0.086 &lt; 0.10. (The n=25 estimate of 0.130 was noise; escalation flipped it under the threshold — exactly what the grok escalation did for its 0.174.)
- **R2 — FIRES.** τ80 drift = 0.717 nats &gt; 0.25, concentrated between temps 0.7 and 1.0 (−0.778 → −1.180), the same shape as grok's curve. **Temperature normalization remains mandatory — now confirmed on a second model family.**
- **R3 — no fire.** Point AUROCs 0.73–0.82 at every temperature including 0.0.
- **R4 — FLIPS vs grok.** Max |mean logprob| at temp 0 is 1.96 ≫ 0.05: gradation, not a hard floor — and temp 0 is GPT-4o-mini's *most*informative temperature (AUROC 0.822, the best cell). The grok collapse was a provider-specific post-temperature-scaling artifact, not a property of the signal. Consequence: "hard refuse-to-gate at temp 0" would be wrong on this family. The shipped degenerate guard's design — detect the ≈-0.00 signature only when present — is exactly right: silent on OpenAI-family streams, firing on xAI-family ones.

**Cross-family synthesis** (grok-3-mini, retired 2026-08; gpt-4o-mini, live): the ranking signal is real and roughly temperature-stable on both families; the operating threshold is never temperature-portable (0.72–1.06 nats of drift); the temp-0 artifact is provider-specific. **Robust-aggregation note:** pathological single-token logprobs (−2001, −3333 — numerically clamped tokens) appear on this stack at temps ≥ 0.7; they drag the mean but not rank-based metrics or percentiles like τ80 — an argument for median/clipped aggregation in future signal work.

**Live E2E of the proxy against this upstream** (addendum step 3): a real streamed answer carried per-token logprobs through the proxy; `raggate_no_logprob_signal_total` stayed silent on the working path (and correctly **fired** during a deliberately misconfigured double-`/v1` URL pass, catching OpenRouter's HTML 404 page as a zero-logprob stream — incidental live validation of the new guard); and an ESCALATE decision frame fired on real logprobs under a strict alpha (confidence −0.0015, 4 tokens evaluated, 5 tokens withheld as savings). Note for operators: `RAGGATE_UPSTREAM_URL` for OpenRouter is `https://openrouter.ai/api` — the proxy appends `/v1/chat/completions` itself.