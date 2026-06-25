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
Peak 0.94 (Qwen) / 0.947 (Gemma) / 0.96 (Mistral), early-layer, beats surprisal-only by ~+0.4. Holds even where
voces run *more* surprising than controls → recognition is architectural, not exposure. (Mistral's *Greek* peak
shifts late, L19 — its 32k vocab shreds Greek, so the distinction is assembled deeper.) **The one robust positive.**

## 2. The decider (voces-specific deep representation) — NULL across families → spell is dead
| decider p | Greek | Latin |
|---|---|---|
| Qwen-3B | 0.224 | 0.886 |
| Gemma-9B | 0.094 | **0.007** |
| Mistral-7B | 0.266 | 0.146 |

Null in Qwen & Mistral. **Gemma's lone p=0.007 (Latin) is NOT replicated** by the other two → classified as a
Gemma-specific artifact (unreplicated, 4-bit, control-driven). **No voces-specific deep representation in any family.**

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

## 4. Synthesis (provisional, 3 of 4)
Surface recognition generalizes; the spell (deep voces representation) is dead; the deep "script" effect is a
measured fragmentation artifact with a dose-response. *"It's the script, not the spell — and the depth is the tokenizer."*

## Prediction scorecard (logged before the runs)
- H1 replicates — **correct** (3/3).
- Decider stays null — **correct** for Qwen & Mistral; Gemma's p=0.007 was a false positive, killed by non-replication.
- Deep-Greek replicates, *stronger* in big-Greek-vocab Gemma — **WRONG, and backwards**: the effect is strongest
  in the *worst*-Greek-coverage model (Mistral) and absent in the best (Gemma). The most informative miss — it
  flipped the mechanism from "distributional namehood" to "fragmentation artifact."
- "Boring and robust most likely" — **wrong**: a clean mechanism emerged.

## Next
1. **Non-name-Greek falsifier** (the killer): do non-name Greek strings also persist deep in fragmenting models?
   Yes → pure fragmentation-clustering, not namehood. Most decisive next test.
2. **Llama-3.1-8B** (tiktoken 128k) — 4th point. Predicted: Greek ≈ 9–10 tok/vox, deep gap ≈ Qwen's +0.016.
3. **fp16 Gemma** — confirm the p=0.007 dissolves un-quantized.
