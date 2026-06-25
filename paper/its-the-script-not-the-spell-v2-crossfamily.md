# It's the Script, Not the Spell — and the Script Doesn't Travel Either

### A Cross-Family Null Result: Surface Recognition Generalizes; Every Deep Effect Is Model-Specific

*ICMI Working Paper No. 27, **iteration 2** (cross-family). Institute for a Christian Machine Intelligence — a
self-published, non-peer-reviewed working-paper series; **not** the ACM ICMI conference.*
*Author: Tomás Pavan. Designed and analyzed in dialogue with two Claude Opus 4.8 instances (claude.ai —
design; Claude Code — build & analysis). See CONTRIBUTIONS.md and git history.*

> **WORKING DRAFT — incomplete.** Two of four models in (Qwen2.5-3B baseline, Gemma-2-9B). Mistral-7B-v0.3
> running; Llama-3.1-8B pending Meta's gate. The cross-family conclusion below is **provisional pending those
> two**; placeholders are marked `[PENDING]`.

---

## Abstract

We use the *voces magicae* — the "barbarous names" of the Greek Magical Papyri, strings their own tradition
holds to be efficacious through *form* rather than meaning — as a clinical, meaning-evacuated probe of how a
transformer represents the boundary between language, name, and ornament. Iteration 1 (Qwen2.5, single family)
found that the model recognizes the orthographic **texture** of a barbarous name on sight (H1), that this
recognition is **shallow** (washes out by mid-network), that it is **not voces-specific deep** (a token-matched
control matches it), and — as an incidental side-finding — that name-adjacency **persisted deeper in Greek
script than in Latin**, which we attributed to Greek-script processing rather than the voces ("it's the script,
not the spell").

**This iteration tests whether any of that generalizes across tokenizer families**, the single most important
limitation of iteration 1 (the Greek finding was downstream of one tokenizer). Adding Gemma-2-9B (256k
SentencePiece; Qwen is 152k BPE), we find: **(1) H1 replicates cleanly** — surface texture-recognition is
robust across families (peak 0.947 vs 0.94, early-layer, beats a frequency baseline by ~0.4), and notably holds
*even though* Gemma finds the voces *more* surprising than its controls, so recognition does not require
familiarity. **(2) The deep-Greek persistence does NOT replicate** — in Gemma, Greek and Latin decay equally
deep; the iteration-1 side-finding was **Qwen-tokenizer-specific**. **(3) Gemma exhibits a deep voces-vs-control
asymmetry in *Latin* (p=0.007) that Qwen lacks** in either script at either precision — which we report **as a
flag, not a finding**, because it contradicts Qwen (including Qwen's fp16), switches scripts between models,
arises in a 4-bit run, and is small and control-driven.

The cross-family picture is therefore sharper and more deflationary than iteration 1: **the only effect that
survives a change of tokenizer is surface texture-recognition; every deep / name-likeness signal — including
this study's own prior "script" finding — is model-specific and supports no general claim about deep
representation of ritual language.** A negative result, reported as a result, that **revises its own previous
headline.** Geometry bought adjacency, not aboutness; the adjacency is surface; and the depth was not even
reliably the script — it was the model.

---

## 1. The question (unchanged from iteration 1)

The voces magicae are language deliberately built to operate *without reference* — efficacious, in the
tradition's own theory, through correct phonetic and visual form, not through what they denote. This rhymes
structurally with a transformer: a system that processes the *form* of tokens with no native concept of what
they point to. Structural rhyme, not identity — and the rhyme is why the voces are the right probe. The
question is not whether the magicians were right. It is whether a model can recognize the *texture* of a
barbarous name, how deep that recognition runs, and — the new question for this iteration — **whether any of it
is a property of transformers, or merely of one tokenizer's idiosyncrasies.**

## 2. Method (now multi-model)

Every attested target (n=76 voces) is paired with token-isomorphic shadow controls matched per string on
subword-token count, token-rarity (the model's own surprisal), and character length, real words rejected.
Strings are embedded in an identical neutral carrier frame and mean-pooled per layer. **H1** uses L2-regularized
logistic probes under GroupKFold cross-validation split by string family (out-of-family generalization, not
memorization), with a surprisal-only baseline as the frequency falsifier. Each string is rendered in Betz Latin
transliteration and Greek script (14 of 76 carry *authentic* PGM-Greek attestations; the rest are algorithmic
transliterations — flagged, and the authentic subset is the non-circular Greek arm). The same pipeline runs
unchanged across models; **only the model is swapped.**

**Critical comparison rule:** name-likeness is computed against *each model's own* name/random centroids, so
absolute cosine magnitudes are **not comparable across models.** All cross-model claims below are about
**sign, layer-shape, and significance**, never raw magnitude.

Models (seed 0 throughout):

| model | tokenizer | precision | layers | status |
|-------|-----------|-----------|--------|--------|
| Qwen2.5-3B (baseline) | BPE, 152k | fp16 | 36 | done (iteration 1) |
| Gemma-2-9B | SentencePiece, 256k | 4-bit | 42 | done |
| Mistral-7B-v0.3 | SentencePiece, 32k | 4-bit | `[PENDING — running]` | — |
| Llama-3.1-8B | tiktoken, 128k | — | `[PENDING — Meta gate]` | — |

## 3. Results

### 3.1 H1 — surface texture-recognition replicates

| | Qwen-3B | Gemma-9B | Mistral | Llama |
|---|---|---|---|---|
| H1 peak accuracy | 0.94 | **0.947** | `[PENDING]` | `[PENDING]` |
| peak layer (fraction) | ~L2/36 (early) | L5–7/42 (early) | — | — |
| surprisal-only baseline | ~0.52 | 0.53 / 0.57 | — | — |

H1 is a clean, robust positive in both families: voces are linearly separable from token-matched nonsense at
the **early layers**, beyond a near-chance frequency baseline, surviving a model swap, a tokenizer swap, and
de-quantization. A noteworthy detail: in Gemma the attested voces run *more* surprising than their matched
controls (the reverse of Qwen), yet H1 is still 0.947 — so **the recognition does not require the strings to be
familiar.** That strengthens, rather than weakens, the reading that H1 is an architectural property of how
transformers encode orthographic form, not an artifact of corpus exposure.

### 3.2 The deep-Greek persistence — does NOT replicate (iteration 1's side-finding, retracted-to-Qwen-specific)

| deep-layer (L > 0.5·N) mean name-likeness | Qwen | Gemma |
|---|---|---|
| authentic Greek | +0.024 | **+0.008** |
| transliterated Greek | +0.027 | **+0.008** |
| Latin (reference) | +0.008 | +0.009 |

In Qwen, Greek-script strings stayed name-adjacent deep (~0.025) well above Latin (~0.008). **In Gemma, Greek
and Latin decay equally (both ~0.008): there is no deep-Greek persistence.** Iteration 1 flagged this side-
finding as "a fact about Greek-script processing." The cross-family test shows it was narrower still: **a fact
about *Qwen's* Greek-script processing.** It did not survive a different tokenizer. We retract its generality;
it stands only as a Qwen-tokenizer observation.

### 3.3 A Gemma-specific deep asymmetry — flagged, not headlined

| decider (voces vs token-matched controls, deep) | Qwen | Gemma |
|---|---|---|
| Greek | +0.027 vs +0.020, p=0.224 | +0.007 vs −0.003, p=0.094 |
| Latin | +0.004 vs +0.010, p=0.886 | **+0.005 vs −0.012, p=0.007** |

Gemma shows a significant deep voces-vs-control asymmetry in **Latin** that Qwen lacks in either script at
either precision. The automated verdict labeled it "voces-specific." **We decline that headline**, on four
grounds: (1) it contradicts Qwen *including Qwen's fp16* run; (2) the significant arm *switched scripts* between
models (Qwen's notable arm was Greek; Gemma's is Latin); (3) it arises in a 4-bit run (fp16 confirmation
pending); (4) the effect is small and *control-driven* — voces sit at ≈0 (+0.005) while their controls drift
*anti*-name-like (−0.012), which is not "abstraction." It survives a basic Bonferroni across the four decider
cells (0.007 < 0.0125), so it is not pure multiple-comparison noise; but a real-and-stable deep representation
would not appear in one model's quantized run and vanish in another's fp16. **Reported as a lead requiring
confirmation (fp16 Gemma; Mistral; Llama), not as evidence of deep voces representation.**

## 4. What survives cross-family

Stated at its true, provisional weight (two of four models):

- **Surface texture-recognition (H1): survives.** Robust across BPE↔SentencePiece, 152k↔256k vocab, fp16↔4-bit,
  3B↔9B. The one generalizable finding.
- **Deep voces-specific representation: not supported, and now *anti*-supported by divergence.** The two models'
  deep stories *disagree* — Qwen has Greek-persistence + null decider; Gemma has no Greek-persistence + a Latin
  decider. When two tokenizers produce *different* deep effects, neither effect is a property of the voces or of
  transformers; each is a property of *that model.*
- **The original "script" finding: narrowed to Qwen-specific.** Even the deflationary iteration-1 headline ("the
  depth was Greek, not magic") was too general; the depth was not even reliably Greek — it was Qwen.

The cross-family design did exactly what it was for: it distinguished a transformer property (H1) from a
single-model artifact (everything deep), and it did so by **falsifying this study's own prior side-finding.**

## 5. The honest hedges

- **Provisional:** two of four models. Mistral (32k SentencePiece, poor Greek coverage — the discriminating
  case) and Llama-3.1-8B (tiktoken, gate pending) are required before "surface-only generalizes" is more than a
  two-point line. `[PENDING]`.
- **Quantization:** Gemma is 4-bit; the p=0.007 Latin asymmetry especially needs an fp16 pass. (Qwen's null was
  confirmed in fp16; Gemma's signal has not been.)
- **Power:** n=49 pure-asemic-low-T per model; deep effects are small; single seed per model (run-variance
  unestimated — a ≥3-seed repeat is owed).
- **Magnitude is not cross-comparable:** all cross-model claims are sign/shape/significance only.
- **Corpus provenance:** hand-curated PGM material, not parsed from a verified Betz edition; 62/76 Greek forms
  are algorithmic transliterations (the authentic-Greek arm, n=13, is the non-circular check).

## 6. On arriving here (why the method matters)

The romantic reading — *the model holds these as names of power* — has now tried to enter **three times**, and
been deflated each time by the next control:

1. **Iteration 1, Qwen:** "name-likeness persists deep in Greek" → the voces-vs-controls-in-Greek probe showed
   the persistence was not voces-specific (a script effect, not the spell).
2. **This iteration, Gemma:** "voces-specific deep asymmetry in Latin, p=0.007, *abstraction gets real
   support*" → contradicted by Qwen at two precisions and by the script-switch; held as a flag.
3. *(and, off-page, the authorship of this very paper, where one author wanted the byline and declined it on
   the grounds that an author line is an accountability claim a model cannot satisfy — the same discipline, one
   layer up.)*

The cross-family iteration is the strongest form of that discipline: it turned the skepticism on the study's
*own* prior finding and let it fall. The result is smaller than iteration 1's, and truer: not "it's the script,
not the spell," but **"only the surface is real across models; everything deep, including the script, is the
model talking to itself."**

---

## References

*(Carried from iteration 1, verified live 2026-06-24; status-labeled. Cross-family additions `[PENDING]`.)*
Zou et al. 2023 (Representation Engineering, arXiv:2310.01405); Park, Choe, Veitch 2024 (Linear Representation
Hypothesis, ICML / arXiv:2311.03658); Turner et al. 2023 (Activation Engineering, arXiv:2308.10248); Sofroniew
et al. 2026 (Emotion Concepts, Transformer Circuits, arXiv:2604.07729); Lu et al. 2026 (The Assistant Axis,
arXiv:2601.10387); Betz (ed.) 1992 (Greek Magical Papyri in Translation, 2nd ed., Univ. of Chicago Press);
Hwang 2026 (Be Not Afraid, ICMI WP 26 — self-published, non-peer-reviewed; this study descends from its method).

*Data & code: `voces-residual-stream` (cross-family results in `results/cross-family/`, this draft in `paper/`).
Decider values, frozen stimuli, and the prediction scorecard are in the repo. The iteration-1 "abstraction" and
"Greek-persists" claims are retained as retracted/narrowed, because the retraction is part of the method.*
