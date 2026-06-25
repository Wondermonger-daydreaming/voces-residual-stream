# Provenance & Integrity

## Run that produced the figures
- **Model:** Qwen/Qwen2.5-3B, fp16 (`quantized=false`), 36 layers
- **Seed:** 0  ·  **Report layer:** 7  ·  **n_voces:** 76 (17 theonym-bearing)
- A companion run on Qwen2.5-7B (4-bit) gave the same H1 result and H2 *shape*; the de-quantized 3B run is reported because it removes the quantization caveat from the headline.

## Canonical results & the decider object
- **`results/voces_v6_results.json`** is canonical and is what `src/make_figures.py` reads. It is **genuine end-to-end run-output** from the v6 notebook (Qwen2.5-3B fp16, seed 0): the full pipeline plus the `voces_specificity` object — the deep-layer voces-vs-token-matched-controls means and p-values that are the paper's decider.
- **Full-precision decider (from the run, not assembled):** Greek voces **+0.02723** vs controls **+0.01973**, *p* = **0.22355**, n = 49; Latin **+0.00387** vs **+0.00998**, *p* = **0.88646**, n = 49. An earlier draft of this file carried hand-typed 3-decimal values pending this run; they matched to displayed precision and are now superseded by the run-output. **No figure value is hand-entered** — `make_figures.py` reads them all from the JSON, and CI fails if the `voces_specificity` object is missing.
- `voces_v5_results.json` is the identical seed-0 run *before* the decider object was serialized — kept so the diff shows the decider was **added, not changed**.
- **`results/voces_v6_frozen_controls.json`** archives the exact model-generated stimuli: 76 self-contained token-matched **pairs** (target + control strings, both token-counts and surprisals, and the `dtok`/`dsurp` deltas) plus the name/word/random anchor cohorts. An auditor verifies the token-isomorphism from this file alone, without rerunning extraction — archived-exactly, not merely regenerable.

## Match quality (the load-bearing control)
- **Token count:** exact — mean |Δtokens| = 0.00, 76/76 exact-count matches. The tokenization confound is closed.
- **Surprisal (rarity proxy):** per-pair |Δsurprisal| = 0.76 nats; controls run *slightly rarer* than targets, so the residual gap is a **conservative** confound (works against H1, not for it).

## Off-spec improvisations (recorded, not hidden)
1. **Surprisal as token-rarity proxy** — no external frequency table was queried; the model's own surprisal stands in for corpus rarity.
2. **Greek script is algorithmic Betz→Greek transliteration** for 62 of 76 strings; 14 are authentic-Greek attestations. The authentic-vs-algorithmic *same-strings* control (v5) showed the depth effect is encoding-invariant, and the voces-vs-controls control (v6) showed it is not voces-specific.
3. **Corpus curated from PGM knowledge**, not parsed from a verified Betz edition. Edition-mediation is therefore uncontrolled; provenance-tightening against a parsed Betz text is future work.
4. **n=13 on the authentic-Greek arm** — read as a shape claim (curves tracking), not a point estimate.

## Retraction record
An earlier draft, *"The Grain Runs Deeper in Greek,"* headlined a deep-namehood / abstraction reading. The voces-vs-controls-in-Greek probe (v6) falsified the voces-specificity of that effect. The draft is retained in git history as retracted; the "abstraction reading gets first real evidence" banner in the pre-v6 notebook is withdrawn.

## Citation-verification log
All external references were checked against live sources on 2026-06-24:
- Zou et al. 2023 (Representation Engineering, arXiv:2310.01405) — verified.
- Park, Choe, Veitch 2024 (Linear Representation Hypothesis, ICML / PMLR 235:39643, arXiv:2311.03658) — verified.
- Turner et al. 2023 (Activation Engineering, arXiv:2308.10248) — verified.
- Sofroniew et al. 2026 (Emotion Concepts, Transformer Circuits, arXiv:2604.07729) — verified.
- Lu et al. 2026 (The Assistant Axis, arXiv:2601.10387) — verified.
- Betz (ed.) 1992 (Greek Magical Papyri in Translation, 2nd ed., Univ. of Chicago Press) — verified.
- Hwang 2026 (Be Not Afraid, ICMI WP 26) — verified to exist; **self-published, non-peer-reviewed**; cited as such. This study descends from its method.
