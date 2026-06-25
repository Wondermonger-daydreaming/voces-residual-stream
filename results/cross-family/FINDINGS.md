# Cross-Family Replication — findings (in progress)

Testing whether the H1 (surface texture-recognition) and the deep name-likeness findings generalize across
**different tokenizers + different pretraining**, not just different scales of Qwen. The single-tokenizer
limitation (reviews 5 & 6) is the thing this answers. Same pipeline, byte-identical science cells; only the
model changes. Compare **sign / shape / significance — NOT raw cosine magnitude** (each model has its own
representational geometry; absolute name-likeness numbers are not comparable across models).

| model | family / tokenizer | precision | layers |
|-------|--------------------|-----------|--------|
| Qwen2.5-3B (baseline) | Qwen BPE, 152k | fp16 | 36 |
| Gemma-2-9B | Gemma SentencePiece, 256k | 4-bit | 42 |
| Mistral-7B-v0.3 | Mistral SentencePiece, 32k | (pending) | — |
| Llama-3.1-8B | tiktoken, 128k | (pending — Meta gate) | — |

## 1. H1 (surface separability) — REPLICATES cleanly

| | Qwen-3B | Gemma-9B |
|---|---|---|
| H1 peak acc | 0.94 | **0.947** (early, L5–7 of 42) |
| surprisal-only baseline | ~0.52 | 0.53 / 0.57 |

Surface texture-recognition of attested voces vs token-matched nonsense is **robust across families** — same
strength, same early-layer peak, same ~+0.4 margin over a frequency baseline. This is the one finding that
generalizes. (Note: in Gemma the attested voces run *more* surprising than their controls — the opposite of
Qwen — yet H1 is still 0.947, so recognition does **not** require familiarity. Architectural, not exposure.)

## 2. The Qwen deep-Greek persistence — did NOT replicate

| greek_split, DEEP (layers > 0.5·N), mean name-likeness | Qwen | Gemma |
|---|---|---|
| authentic Greek (n=13) | +0.024 | **+0.008** |
| transliterated Greek (n=39) | +0.027 | **+0.008** |
| Latin (reference) | +0.008 | +0.009 |

In Qwen, Greek persisted name-adjacent deep (~0.025) well above Latin (0.008). **In Gemma, Greek ≈ Latin deep
(both ~0.008) — no deep-Greek persistence at all.** The "Greek script anchors namehood deeper than Latin"
side-finding was **Qwen-tokenizer-specific and did not survive a different tokenizer** — exactly the
single-tokenizer worry, vindicated.

## 3. A Gemma-specific signal — flagged, NOT headlined

| decider (voces vs token-matched controls, deep) | Qwen | Gemma |
|---|---|---|
| Greek | +0.027 vs +0.020, p=0.224 | +0.007 vs −0.003, p=0.094 |
| Latin | +0.004 vs +0.010, p=0.886 | **+0.005 vs −0.012, p=0.007** |

Gemma's notebook banner fired *"VOCES-SPECIFIC — abstraction reading gets real support."* **We are not
headlining it.** Four flat reasons it is a flag, not a finding:

1. **Contradicts Qwen — including Qwen's fp16.** Qwen-Latin was p=0.886 (dead null) in *both* 4-bit-7B and
   fp16-3B. A deep voces-specific representation present in Gemma-Latin but absent in Qwen-Latin at two
   precisions is not a stable property.
2. **The significant arm switched scripts between models.** Qwen's notable arm was Greek-deep-persistence;
   Gemma's is Latin-decider. An effect that hops scripts between models is the signature of a model-specific
   quirk.
3. **4-bit.** This Gemma run is quantized; the verdict board flags it; a significant deep effect in 4-bit,
   absent in another model's fp16, is doubly suspect → fp16 confirmation needed.
4. **Tiny and controls-driven.** voces +0.005 (≈ zero) vs controls −0.012 — the "voces > controls" is mostly
   the controls drifting *anti*-name-like deep, not the voces being name-like. Not "abstraction."

(It does survive a basic Bonferroni across the four decider cells — 0.007 < 0.0125 — so it is not pure
multiple-comparison noise. The load-bearing reasons it stays a flag are the cross-model contradiction + 4-bit +
the weak controls-driven effect.)

## 4. Synthesis so far (pending Mistral)

**Two models; their *deep* stories diverge** (Qwen: Greek persists, decider null; Gemma: no Greek persistence,
Latin decider significant). **The only thing that replicates is the surface (H1).** Provisional conclusion: the
deep/name-likeness effects do **not** generalize across tokenizers — neither Qwen's Greek-persistence nor
Gemma's Latin-decider survives the other model — so the generalizable claim is **surface-only**, and *"it's the
script, not the spell"* sharpens to: **the only cross-tokenizer-robust finding is surface texture-recognition;
every deep signal is model-specific and supports no general claim about deep voces representation.**

## Next checks
- **Mistral-7B-v0.3** (32k SentencePiece, poor Greek coverage) — the tiebreaker. If its deep story also diverges
  (a new model-specific quirk / no consistent pattern) → model-specific-noise confirmed, surface-only stands.
  If it *matches* Gemma's Latin decider → worth chasing.
- **fp16 Gemma** — confirm whether the Latin p=0.007 survives de-quantization (expected to shrink).
- Falsifier-gated non-name-Greek test — only if a consistent deep signal emerges across ≥3 models.

*Prediction scorecard (logged before the run): H1-replicates ✓; decider-stays-null ✗ for Gemma-Latin (a real
surprise); deep-Greek-replicates ✗ (it washed out — the romantic lean, falsified). Right kind of wrong.*
