# Cross-Family Replication — findings (3 of 4 models)

Testing whether the iteration-1 findings (Qwen2.5) generalize across **tokenizer families**. Same pipeline,
byte-identical science cells; only the model swaps. Compare **sign / shape / significance, NOT raw cosine
magnitude** across models (each has its own geometry) — except token counts, which *are* directly comparable.

| model | tokenizer | precision | layers | status |
|-------|-----------|-----------|--------|--------|
| Qwen2.5-3B (baseline) | BPE 152k | fp16 | 36 | done |
| Gemma-2-9B | SentencePiece 256k | 4-bit | 42 | done |
| Mistral-7B-v0.3 | SentencePiece 32k | 4-bit | 32 | done |
| Llama-3.1-8B | tiktoken 128k | — | — | PENDING (Meta gate) |

## 1. H1 (surface recognition) — REPLICATES in all three
Peak 0.887 Latin / 0.920 Greek (Qwen) · 0.947 (Gemma) · 0.960 (Mistral), early-layer, beats surprisal-only by ~+0.4. Holds even where
voces run *more* surprising than controls → recognition is architectural, not exposure. (Mistral's *Greek* peak
shifts late, L19 — its 32k vocab shreds Greek, so the distinction is assembled deeper.) **The one robust positive.**

## 2. The decider (voces-specific deep representation) — NULL across families → spell is dead
| decider p | Greek | Latin |
|---|---|---|
| Qwen-3B | 0.224 | 0.886 |
| Gemma-9B | 0.094 | **0.007** |
| Mistral-7B | 0.266 | 0.146 |

Null in Qwen & Mistral. **Gemma's lone p=0.007 (Latin) is NOT replicated** by the other two and is run-variable
under 4-bit (→ p=0.032 on a re-run) → held as **provisional pending fp16**, not a settled artifact. **No
*detectable* voces-specific deep representation in any family at this power** (absence of detection, not proof of absence).

## 3. The deep-Greek effect — a BYTE-FRAGMENTATION ARTIFACT (the real mechanistic finding)
Magnitude scales **monotonically with Greek tokens-per-vox** (direct measure, n=49, each model's own tokenizer):

| model | vocab | **Greek tok/vox** | deep-Greek gap (auth−Latin) |
|-------|-------|-------------------|------------------------------|
| Mistral-7B | 32k | **10.76** | **+0.051** (strongest) |
| Qwen-3B | 152k | **9.63** | +0.016 |
| Gemma-9B | 256k | **7.69** | ~0.000 (none) |

More Greek fragmentation → stronger deep-Greek persistence. **Not explained by precision** (the two 4-bit models
are at opposite ends) **or size**. It is script-general (decider null → not voces-specific), encoding-invariant
(same-strings control), and vanishes when Greek tokenizes cleanly. → **The "depth" was the tokenizer fragmenting
Greek, not the words.** (Data: `greek_fragmentation.json`. Gemma tokens via the non-gated `unsloth/gemma-2-9b`
mirror.)

## 3b. The non-name-Greek falsifier (RUN on Mistral) — UNINFORMATIVE (a confound, caught)
The "killer" falsifier: does *non-name* Greek (numerals/function/common words — ΚΑΙ, ΔΥΟ, ΛΟΓΟΣ) also persist
deep in the most-fragmenting model? Mistral result (`nonname_falsifier` in the results JSON):

| cohort | n | Greek tok/str | deep-Greek gap |
|---|---|---|---|
| voces (low-T, asemic) | 49 | **10.73** | **+0.041** |
| non-name Greek (lexical) | 26 | **5.15** | **−0.005** |

The notebook's binary rule printed *"name-specific"* — **but the verdict is confounded and we do not report it.**
The non-name words are short/common, so Mistral shreds them **~2× less** than the asemic voces (5.15 vs 10.73
Greek tok/str — reproduce with `src/check_nonname_fragmentation.py`). The cohorts differ on **three axes at
once** — fragmentation, lexicality (the non-name words are real, in-vocabulary Greek; the voces are asemic), and
namehood — so the verdict cannot be attributed to any one of them, **in either direction**. The result is
**uninformative on namehood.** (We caught this only after the falsifier returned the inconvenient answer — a
scrutiny asymmetry we disclose; the confound itself is direction-independent.)

§2's token-matched controls are the closest available proxy (asemic, fragmentation-matched to the voces; decider
null) but they fix fragmentation only by being *maximally non-lexical*, so "controls persist deep too" is
consistent with both a fragmentation reading **and** an asemic-ness reading — a meaningful-Greek (lexicality)
reading is **not excluded.** The purpose-built arm that would settle it — non-name Greek matched to the voces on
**both** fragmentation and lexicality (an asemic-matched arm AND a lexical-matched arm at ~10.7 Greek tok) — is
in `notebooks/voces_falsifier_v2_lexicality.ipynb`, ready to fire. Until it runs, §3.4's question
(fragmentation vs. lexicality vs. namehood) stays **open.**

## 4. Synthesis (provisional, 3 of 4)
Surface recognition generalizes (H1, robust); no *detectable* voces-specific deep representation (decider null
at this power); the deep "script" effect's *magnitude* **tracks** Greek byte-fragmentation across 3 (confounded)
models. Whether the residual depth is fragmentation, lexicality, or namehood is **unresolved** — the falsifier
that would separate them was confounded (§3b); the lexicality-matched arm is owed
(`notebooks/voces_falsifier_v2_lexicality.ipynb`).
*"It's the script, not the spell — and the depth tracks the tokenizer."*

## Prediction scorecard (logged before the runs)
- H1 replicates — **correct** (3/3).
- Decider stays null — **correct** for Qwen & Mistral; Gemma's p=0.007 was a false positive, killed by non-replication.
- Deep-Greek replicates, *stronger* in big-Greek-vocab Gemma — **WRONG, and backwards**: the effect is strongest
  in the *worst*-Greek-coverage model (Mistral) and absent in the best (Gemma). The most informative miss — it
  flipped the mechanism from "distributional namehood" to "fragmentation artifact."
- "Boring and robust most likely" — **wrong**: a clean mechanism emerged.
- **Non-name Greek persists deep too (ratio>0.5), sealing pure fragmentation-clustering** — **WRONG, and the
  test was confounded**: the chosen non-name cohort was under-fragmented (5.15 vs 10.73 Greek tok/str) *and*
  lexical, so it could not adjudicate either way. Caught by *measuring* the cohorts' tokenization — but only
  after it returned the inconvenient answer (a scrutiny asymmetry, disclosed). A miss about the study's own
  *instrument*, not just its hypothesis.

## Next
1. **Purpose-built falsifier (v2) — `notebooks/voces_falsifier_v2_lexicality.ipynb`, ready to fire.** Holds
   fragmentation *and* lexicality fixed: voces vs an **asemic frag-matched** non-name arm (isolates namehood) vs
   a **lexical frag-matched** real-Greek arm (isolates lexicality) vs the original low-frag baseline (isolates
   fragmentation). Bootstrap CIs on every gap. This is what the v1 falsifier should have been.
2. **Llama-3.1-8B** (tiktoken 128k) — 4th point. Predicted: Greek ≈ 9–10 tok/vox, deep gap ≈ Qwen's +0.016.
3. **fp16 Gemma** — confirm the p=0.007 dissolves un-quantized (and resolves the 4-bit re-run p-value anomaly,
   below). The v2 notebook re-runs the Gemma decider if pointed at `google/gemma-2-9b`.


## Addendum — the 4-bit Gemma decider is unreliable (run-variance + a value anomaly); fp16 owed
A second seed-0 Gemma run (`voces_xfam-gemma-2-9b_rerun_results.json`) gave a *different* Latin decider
(p=0.032 vs the first run's p=0.007) and a different H1-peak layer (L4 vs L7) — despite identical seed and model
(4-bit non-determinism; bitsandbytes/cuDNN are not bit-reproducible). Note both values are still significant at
α=0.05, so this is *wobble within the significant range*, not a vanishing effect — we therefore hold Gemma's
p=0.007 as **provisional pending an fp16 pass**, not as a settled artifact (we cannot treat run-variance as
disqualifying for Gemma while our own single-seed nulls are equally unbounded by seed-variance).

**Disclosed data anomaly (do not rely on the Gemma re-run's absolute p-values).** The Gemma re-run's *Greek*
decider p is **byte-identical to Mistral's** — `0.2659000459593744` — despite completely different underlying
means (Gemma vox_deep 0.0068 vs Mistral 0.0598). Identical analytic p-values from different data are impossible
by independent computation; this is a **value-bleed/assembly bug** in the 4-bit re-run output. It does not touch
the paper's headline cells (Mistral's own p comes from the Mistral file; the Gemma column uses the *original*
Gemma run, p=0.094), but it confirms the 4-bit Gemma re-run artifact is not trustworthy. fp16 Gemma (via the v2
notebook pointed at `google/gemma-2-9b`) resolves both the run-variance and this anomaly.

The non-name-Greek falsifier (Cell 10g) ran on the Gemma re-run too but is uninformative there (Gemma has no
deep-Greek effect to falsify: voces +0.001 ≈ non-name 0.000). The Mistral falsifier (§3b) is also uninformative,
for the cohort-confound reason — not a within-model "confirmation."
