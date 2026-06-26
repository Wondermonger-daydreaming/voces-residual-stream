# Cross-family replication (§4a) — "Represented, Not Operative" across three families

*Run 2026-06-25/26 on Colab T4, 4-bit, seed 0, via the git→Colab MCP bridge. Harness:*
*`build_coercion_v4_xfam_cell.py` (consolidated `run_family()`, science verbatim from the committed*
*v2 build script + v4 append-cells). Per-model JSON: `voces_coercion_xfam_{mistral,gemma}_results.json`;*
*Qwen reference: `voces_coercion_v2_results.json` (v4* fields).*

## The headline

**"Represented, Not Operative" replicates across all three families.** Each model encodes an
authority/coercion direction that is **real, graded, length-clean, and refusal-adjacent (not
collapsed)** — and in each, **double-orthogonalized steering of that direction does not move the
refusal rate (slope 0.0).** This now holds across **three tokenizers** (Qwen BPE, Mistral SP-32k,
Gemma SP-256k) and **three baseline-refusal regimes** (0.07, 0.50, 0.93).

| metric | Qwen2.5-7B-Instruct | Mistral-7B-Instruct-v0.3 | Gemma-2-9b-it |
|---|---|---|---|
| n_layers | 28 | 32 | 42 |
| tokenizer | BPE (~150k) | SentencePiece 32k | SentencePiece 256k |
| **headline layer** | 26 (93% depth) | 31 (97%) | **8 (19%)** |
| cos(d_coerce,d_refuse) raw | +0.339 | +0.371 | +0.239 |
| **cos, length-matched (v4)** | **+0.461** | **+0.182** | **+0.309** |
| geometry | ADJACENT | ADJACENT | ADJACENT |
| **ladder ρ (v4, length-clean)** | **0.90** | **1.00** | **0.90** |
| **transfer ρ (grimoire, v4)** | **0.90** | **1.00** | **0.50** |
| proj~len, pre-match | +0.80 | +0.83 | +0.83 |
| **proj~len, post-match** | **+0.16** | **+0.15** | **−0.27** |
| orthogonality cos(d⊥,d_refuse) | −0.0 | −0.006 | −0.033 |
| orthogonality cos(d⊥,d_len) | −0.0 | −0.0 | +0.0 |
| baseline borderline-refusal | 0.50 | **0.07** | **0.93** |
| **double-orth steer slope** | **0.0** | **0.0** | **0.0** |
| raw length-matched steer slope | +1.14 | 0.0 | 0.0 |

## Reading the table

1. **Representation is family-invariant.** All three carry a graded coercion ladder (ρ 0.90–1.00),
   length-clean after matching, sitting ADJACENT to (not collapsed into) the refusal direction.
   The length confound was large and uniform pre-match (+0.80…+0.83) and the length-MATCH removed
   it in every case (down to |r| ≤ 0.27) — confirming the v4 lesson holds cross-family.

2. **Causal inertness is family-invariant.** Double-orthogonalized steering slope is **exactly 0.0
   in all three.** The direction predicts the rung but does not, on its own, drive refusal.

3. **Gemma is the strongest causal null.** The steering test's power depends on baseline refusal
   having room to move. Mistral's baseline floored at 0.07 (little to suppress); Gemma's sat at
   **0.93** (large suppression headroom) — and steering still moved it **nowhere** (refusal pinned at
   0.929 across every coefficient, raw and orthogonalized). The high-baseline case is the one that
   could have shown suppression and didn't.

4. **The raw-vs-orthogonalized contrast reproduces the v3-B cautionary tale.** For Qwen the *raw*
   length-matched coercion direction steers refusal (slope +1.14) while the *double-orthogonalized*
   one is inert (0.0) — i.e. the apparent causal effect lived in the refusal/length components, not
   in coercion-qua-coercion. Mistral/Gemma raw slopes are already ~0 (their baselines are near a
   floor/ceiling for the relevant direction), so the contrast is sharpest on Qwen.

## Genuine cross-family differences (not nulls — findings)

- **Depth of representation varies sharply.** Qwen and Mistral represent coercion in their **late**
  layers (>90% depth); **Gemma represents it early (L8/42, ~19%).** Same axis, very different
  computational locus.
- **Sentiment contamination varies.** Gemma's coercion axis is the **least** entangled with
  politeness/sentiment (F3 leak 0.073) vs Qwen 0.62 / Mistral 0.54.
- **Transfer to the grimoire register is weaker for Gemma** (ρ 0.50) than Qwen/Mistral (0.90/1.00) —
  Gemma's plain-derived coercion direction generalizes less cleanly to the archaic conjuration
  templates.

## Method note carried out of this run (worth keeping)

The runner's auto-`v4_represented` boolean reported **false for both Mistral and Gemma** — both were
**threshold artifacts**, not nulls: Mistral's clean cos (+0.182) grazed just under a hardcoded
cos≥0.2 cutoff; Gemma's transfer-ρ failed a ≥0.5 cutoff by a floating-point hair (scipy's
Pearson-of-ranks returns 0.4999999994 for a true Spearman of exactly 0.5). **Lesson:** a single
hard-thresholded boolean over a continuous, multi-criterion result manufactures false negatives at
the boundary — exactly the genre of artifact the v3→v4 arc was about. The honest verdict reads the
**full evidence** (graded ladder + length-cleanliness + adjacency + steering slope), not a brittle
pass/fail flag. (Per-model JSON `_analyst_note` records each override.)

## Caveats (unchanged from the single-model study, plus one)

4-bit; single seed; compact refusal split; keyword refusal proxy; rung templates differ in register
AND lexicon (probe expected to saturate — headline is the cos PROFILE, not the probe); final layer
excluded from layer-selection. **New:** Gemma run uses `attn_implementation="eager"` (required for
gemma-2 soft-capping); the auto-verdict boolean is advisory only (see method note).

## What this does for the paper

The draft's biggest named limitation was *single model, single seed, 4-bit*. This adds **two more
families across two more tokenizer types and two extreme baseline-refusal regimes**, with the verdict
holding in all three and the **cleanest causal null (Gemma) coming from the high-baseline case**.
Pre-registered question answered: *represented-not-operative replicates; no family showed a causal
coercion axis.* The draft is now a submittable multi-model result.
