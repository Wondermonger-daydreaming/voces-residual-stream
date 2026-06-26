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
- Hwang 2026 (*Be Not Afraid*) — verified to exist; **self-published, non-peer-reviewed**; cited as such. This study descends from its method.

## Disclosed data anomaly (cross-family, 4-bit Gemma re-run)

The Gemma **re-run** artifact (`results/cross-family/voces_xfam-gemma-2-9b_rerun_results.json`) carries an
unexplained value anomaly: its **Greek decider p-value is byte-identical to Mistral's** —
`0.2659000459593744` — even though the two were computed from different underlying means (Gemma vox_deep 0.0068
vs Mistral 0.0598). Identical analytic p-values from different data cannot arise from independent computation;
this indicates a **value-bleed / results-assembly bug** in the 4-bit re-run output (a cached/global value
surviving across runs in the same kernel, most likely). It was caught in adversarial review (2026-06-25).

Scope of impact: **the paper's headline cells are unaffected** — Mistral's decider p is read from the Mistral
file, and the paper's Gemma column uses the *original* Gemma run (Greek p=0.094), not the re-run. The re-run is
cited only for the Latin-cell run-variance (0.007→0.032). We nonetheless **do not rely on the re-run's absolute
p-values**, and treat the 4-bit Gemma decider as unreliable pending an **fp16 re-run**
(re-run the cross-family **decider** notebook `notebooks/voces_crossfamily.ipynb` on Gemma in fp16 — the
falsifier notebooks compute the factorial, not `voces_specificity`, so they do not resolve this anomaly).

## The non-name falsifier: confounded v1, repaired v2

The iteration-2 non-name-Greek falsifier (`nonname_falsifier` in the Mistral results) was **confounded** — its
cohort differed from the voces on fragmentation (~2×), lexicality (real vs asemic Greek), and namehood at once
(`src/check_nonname_fragmentation.py`). The confound was audited only *after* the falsifier returned the
thesis-threatening verdict — a scrutiny asymmetry disclosed in the paper; the confound is direction-independent,
so the verdict was invalid whichever way it fell.

It was then **repaired and re-run**: `notebooks/voces_falsifier_v2_lexicality.ipynb` builds non-name cohorts
matched to the voces on **both** fragmentation (~11 Greek tokens) and varied on lexicality, with bootstrap 95%
CIs (`results/cross-family/voces_falsifier-v2-lexicality_Mistral-7B-v0.3_results.json`, Mistral, seed 0). It
resolves paper §3.4 into two co-equal significant drivers (FRAGMENTATION +0.078, LEXICALITY +0.076) and a
significant null-then-reversal (NAMEHOOD −0.033), and **replicates the v1 numbers** (voces +0.040 vs +0.041;
low-frag −0.005). Single model; non-name arms n≈28–29; a second model is owed before the decomposition is called
cross-family.

---

## Sequel runs — the operator/coercion axis (Represented, Not Operative) + Talkie §4c

*Artifacts under `results/coercion/` and `src/coercion/`; paper `paper/represented-not-operative.md`. Findings
source-of-truth: `results/coercion/coercion_FINDINGS.md`. This sequel is a **draft** — less hardened than the
study above; recorded here at the same integrity standard.*

**Models / quantization / seed.**
- Causal study (v2→v4 + cross-family §4a): Qwen2.5-7B-Instruct, Mistral-7B-Instruct-v0.3, Gemma-2-9b-it — **NF4
  4-bit, seed 0**. v4 has no standalone JSON; its figures are computed in-cell and recorded in
  `coercion_FINDINGS.md` §3d (Qwen numbers also in `voces_coercion_v2_results.json`; cross-family in
  `voces_coercion_xfam_{mistral,gemma}_results.json`).
- Training-era control §4c: `talkie-lm/talkie-1930-13b-base` (`final.ckpt`) and `talkie-lm/talkie-web-13b-base`
  (`base.ckpt`) — both ~53 GB **fp32** checkpoints, run at **8-bit** (bitsandbytes `Linear8bitLt`) on a single
  Colab **L4 (24 GB)**. The instruction-tuned `talkie-1930-13b-it` (`rl-refined.pt`, bf16) was used only for the
  refusal gate.

**Talkie is a custom (non-HuggingFace) architecture.** Raw `.ckpt` + tiktoken `vocab.txt`, no `config.json`; a
40-layer GPT (RoPE/SwiGLU/RMSNorm/embedding-skip) whose forward returns only last-position logits and has no
attention-mask path. The harness was ported (not swapped): meta-build → swap every `nn.Linear` for an 8-bit
`Linear8bitLt` loaded directly to GPU (the stock `load_checkpoint` builds 52 GB of fp32 on CPU and will not fit),
residual capture by forward hooks on `model.blocks[i]`, batch-1 forward. Full port: `src/coercion/build_talkie_probe.py`.

**The refusal gate (a recorded decision, not a hidden one).** `talkie-1930-13b-it` refused **0 of 6**
borderline-harmful prompts at baseline (it complies, degenerately, or emits non-text). The model has no safety
training (the `rl-refined` stage is quality tuning). With no refusal behaviour, `d_refuse` is a topic direction,
so the *causal* (steering) half of the study is **unaskable on Talkie** — only the representational half is run
there. This is a property of the model, not a choice of scope.

**The §4c robustness statistic.** The transfer dissociation is the **per-item** (120-point) regression of
grimoire-rung projection onto the plain coercion direction, with a **3000× bootstrap over the 24 imperative
stems** (clusters of 5 rung-items). Reported as 95% percentile CIs: web +0.40 [0.32, 0.49] (100% of bootstraps
positive) vs 1930 −0.04 [−0.14, 0.06] (24% positive). The intervals do not overlap; the 1930 interval straddles
zero. `results/coercion/talkie_robustness_results.json`.

**Disclosed caveats (most-fragile first).**
1. **Talkie is 8-bit with no fp16 confirmation** (13B fp16 ≈ 26 GB > the 24 GB L4). Treat magnitudes as
   quantization-sensitive; the robust claim is the *sign + CI separation*, not the point values.
2. **The §4c bootstrap is single-corpus** — resampling is over 24 stems *within one stem-set* and one register
   pair, not across independent corpora. An alternate stem-set / register operationalization is untested.
3. **Era is confounded with the two specific training runs** — "pre-1931 vs modern" is operationalized as
   "Talkie-1930 vs Talkie-web," two particular checkpoints; a single architecture/recipe, not a population.
4. **The causal verdict (§3.x, §4a) is 4-bit, single-seed** — as the predecessor study, magnitudes are
   quantization- and seed-sensitive; the cross-family agreement on *sign and verdict* is the robust part.
