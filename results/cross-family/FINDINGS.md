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

## 3b. The matched factorial (RUN on Mistral) — fragmentation AND meaninglessness, but NOT names
The v1 falsifier (`nonname_falsifier`) compared the voces to short common Greek words and printed
*"name-specific"* — confounded (its cohort fragmented ~2× less, 5.15 vs 10.73 Greek tok/str, AND was lexical).
We rebuilt it as a **factorial holding Greek-token fragmentation fixed at ~11 tokens** with bootstrap 95% CIs
(`notebooks/voces_falsifier_v2_lexicality.ipynb`; results
`voces_falsifier-v2-lexicality_Mistral-7B-v0.3_results.json`). All four cohorts share one model + one pair of
centroids → magnitudes directly comparable.

| cohort | Greek tok/str | n | deep-Greek gap [95% CI] |
|---|---|---|---|
| voces (low-T, asemic) | 10.73 | 48 | **+0.040** [+0.027, +0.051] |
| non-name asemic, MATCHED | 11.04 | 28 | **+0.073** [+0.053, +0.094] |
| non-name lexical, MATCHED | 11.03 | 29 | **−0.003** [−0.015, +0.007] |
| non-name lexical, low-frag (v1) | 5.15 | 26 | **−0.005** [−0.019, +0.010] |

Three single-axis contrasts, **all significant** (bootstrap CI on the diff excludes 0):
- **FRAGMENTATION** (asemic-matched − low-frag) = **+0.078** [+0.053, +0.101] → fragmentation is a real driver.
- **LEXICALITY** (asemic-matched − lexical-matched) = **+0.076** [+0.053, +0.099] → meaninglessness is a co-equal
  driver (meaningful Greek at the SAME fragmentation does NOT persist deep).
- **NAMEHOOD** (voces − asemic-matched) = **−0.033** [−0.057, −0.010] → no namehood; the voces persist deep
  *less* than matched random asemic strings. **Reversed, not just null.**

**Mechanism:** the deep region holds Greek that is *both* heavily fragmented *and* meaning-less ("fragmented
nonsense") — neither condition alone suffices. The voces sit just *below* pure noise because they are a
recognizable genre (exactly what H1 reads), so they are marginally less alien than true random. The v2 run also
**replicates v1 exactly** (voces +0.040≈+0.041; low-frag −0.005).

**Replicated on Gemma-2-9B** (the least-Greek-fragmenting model): same structure, all three significant —
FRAGMENTATION +0.031 [0.016, 0.046], LEXICALITY +0.039 [0.028, 0.051], NAMEHOOD −0.029 [−0.044, −0.015]
(smaller magnitudes, Gemma cohorts at ~8 vs Mistral's ~11 Greek tok — consistent with fragmentation mattering).
So the decomposition is **not single-model**. Caveat that survives: the lexical arm bundles meaning with
frequency/familiarity (v3 notebook splits it by the model's own surprisal); non-name arms n≈28–34.

**v3 splits the bundle → the second factor is MEANING, not familiarity (Gemma; `..._v3-separation_...json`).**
Splitting the real-Greek pool by the model's own surprisal into familiar (2.4 nats) vs unfamiliar (4.0 nats):
FRAGMENTATION +0.018 [0.005,0.031] sig · **FAMILIARITY −0.002 [−0.016,0.013] n.s.** · **MEANING +0.020
[0.009,0.031] sig** · NAMEHOOD −0.023 [−0.036,−0.011] sig. Familiar and unfamiliar *meaningful* Greek both fail
to persist (≈+0.008); only *meaningless* asemic Greek persists (+0.027). So frequency/familiarity is ruled out —
"fragmented nonsense" is **earned**. Bonus: asemic deep gap across 5 seeds sd=0.0024 (single-seed concern dead);
and *absolute* Greek name-likeness shows meaningful Greek IS deep-name-like in BOTH scripts (so its Greek−Latin
gap is ~0 via recognition, not un-name-likeness) — the gap is a script-asymmetry recognition collapses. Mistral
v3 (a one-cell re-run) is the natural confirmation.

## 4. Synthesis (3 of 4 models; factorial single-model)
Surface recognition generalizes (H1, robust); no voces-specific deep representation (decider null, and §3b's
NAMEHOOD arm slightly *negative*); the deep "script" effect decomposes (Mistral factorial, §3b) into **two
co-equal significant drivers — fragmentation AND meaninglessness — with namehood ruled out.** The deep region
holds *fragmented nonsense*; the voces are too recognizable (H1) to live all the way down there.
*"It's the script, not the spell — and the depth is fragmented nonsense."*

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
   below). NB: resolving the *decider* anomaly needs re-running the cross-family **decider** notebook
   (`notebooks/voces_crossfamily.ipynb`) on Gemma — the falsifier notebooks compute the *factorial*, not
   `voces_specificity`.


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
Gemma run, p=0.094), but it confirms the 4-bit Gemma re-run artifact is not trustworthy. Resolving it needs an
**fp16 re-run of the cross-family decider notebook** (`notebooks/voces_crossfamily.ipynb`) on Gemma — the
falsifier notebooks do not compute `voces_specificity`, so they cannot resolve this anomaly.

The original v1 non-name falsifier (Cell 10g) ran on the Gemma re-run too but was uninformative there (Gemma has
no deep-Greek voces effect: voces +0.001). The **matched factorial (§3b), by contrast, is informative on Gemma**
and replicates Mistral's structure (all three contrasts significant) — because it builds its own
fragmentation-matched asemic cohort rather than relying on the voces' weak Gemma signal.
