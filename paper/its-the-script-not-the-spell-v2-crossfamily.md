# It's the Script, Not the Spell — and the Depth Is the Tokenizer

### A Cross-Family Study: Surface Recognition Generalizes; the Deep "Greek" Effect Is a Byte-Fragmentation Artifact

*ICMI Working Paper No. 27, **iteration 2** (cross-family). Institute for a Christian Machine Intelligence — a
self-published, non-peer-reviewed working-paper series; **not** the ACM ICMI conference.*
*Author: Tomás Pavan. Designed and analyzed in dialogue with two Claude Opus 4.8 instances (claude.ai —
design; Claude Code — build & analysis). See CONTRIBUTIONS.md and git history.*

> **WORKING DRAFT.** Three of four models in (Qwen2.5-3B, Gemma-2-9B, Mistral-7B-v0.3). Llama-3.1-8B pending
> Meta's gate. The central mechanistic claim rests on a 3-point monotonic relationship; the falsifier that
> would seal it (non-name Greek) and the 4th data point (Llama) are `[PENDING]`.

---

## Abstract

We use the *voces magicae* — the "barbarous names" of the Greek Magical Papyri, strings their own tradition
holds to be efficacious through *form* rather than meaning — as a clinical, meaning-evacuated probe of how a
transformer represents the boundary between language, name, and ornament. Iteration 1 (Qwen2.5, single family)
found surface recognition of barbarous names (H1), no voces-specific *deep* representation, and an incidental
side-finding that name-adjacency persisted deeper in Greek script than Latin, which it attributed to
Greek-script processing ("it's the script, not the spell").

**This iteration runs the identical pipeline across three tokenizer families** — Qwen (BPE, 152k), Gemma-2
(SentencePiece, 256k), Mistral (SentencePiece, 32k) — the single most important limitation of iteration 1.
Three findings. **(1) H1 replicates cleanly in all three** (peak 0.93–0.96, early-layer, ~+0.4 over a frequency
baseline), and holds even where the model finds the voces *more* surprising than its controls — so
texture-recognition is architectural, not exposure-driven. **(2) No voces-specific deep representation in any
family**: the voces-vs-control decider is null in Qwen and Mistral; a single significant Gemma cell (Latin,
p=0.007) is **not replicated** by the other two and is treated as a Gemma-specific artifact. The spell is dead
cross-family. **(3) The deep-Greek persistence is a byte-fragmentation artifact.** Its magnitude scales
**monotonically with how many tokens the model's tokenizer shreds each Greek string into** — Mistral (10.76
Greek tokens/vox, deep gap +0.051) > Qwen (9.63, +0.016) > Gemma (7.69, ~0.000) — a gradient *not* explained by
precision (the two 4-bit models sit at opposite ends) or model size. The deep effect is not the words, and not
even "Greek" abstractly; it is *fragmented* Greek, and it weakens to nothing as the tokenizer represents Greek
more cleanly.

The cross-family picture is sharper and more mechanistic than iteration 1: **the only effect that survives a
change of tokenizer is surface texture-recognition; the deep "script" effect is a measurable artifact of Greek
byte-fragmentation; and there is no deep representation of the voces *as such* in any model.** Geometry bought
adjacency, not aboutness; the adjacency is surface; and the depth, where it appeared at all, was the tokenizer
talking to itself.

---

## 1. The question

The voces magicae are language deliberately built to operate *without reference* — efficacious, in the
tradition's theory, through correct form, not denotation. This rhymes with a transformer: a system that
processes the *form* of tokens with no native concept of what they point to. The question is not whether the
magicians were right. It is whether a model recognizes the *texture* of a barbarous name, how deep that runs,
and — the new question — **whether any of it is a property of transformers, or of one tokenizer's idiosyncrasies.**

## 2. Method (multi-model)

Each attested target (n=76 voces) is paired with token-isomorphic shadow controls matched per string on
subword-token count, surprisal (the model's own), and character length, real words rejected. Strings sit in an
identical neutral carrier frame, mean-pooled per layer. **H1**: L2-regularized logistic probes, GroupKFold by
string family (out-of-family generalization), surprisal-only baseline as the frequency falsifier. Each string
is rendered in Betz Latin transliteration and Greek script (14/76 carry authentic PGM-Greek; the rest
algorithmic — the authentic subset is the non-circular Greek arm). **The same pipeline runs unchanged; only the
model swaps.**

**Cross-model comparison rule:** name-likeness is computed against *each model's own* name/random centroids, so
absolute cosine magnitudes are **not comparable across models.** All cross-model claims are **sign, layer-shape,
and significance** — never raw magnitude. (The one exception is the fragmentation analysis in §3.2, which
compares *token counts*, a directly comparable quantity.)

| model | tokenizer | precision | layers | status |
|-------|-----------|-----------|--------|--------|
| Qwen2.5-3B (baseline) | BPE, 152k | fp16 | 36 | done |
| Gemma-2-9B | SentencePiece, 256k | 4-bit | 42 | done |
| Mistral-7B-v0.3 | SentencePiece, 32k | 4-bit | 32 | done |
| Llama-3.1-8B | tiktoken, 128k | — | — | `[PENDING — Meta gate]` |

## 3. Results

### 3.1 H1 — surface texture-recognition replicates across all three families

| | Qwen-3B | Gemma-9B | Mistral-7B |
|---|---|---|---|
| H1 peak (Latin) | 0.94 | 0.947 | **0.960** |
| H1 peak (Greek) | — | 0.947 | 0.927 |
| peak layer | early (~L2–7) | early (L5–7) | Latin L5 / **Greek L19** |
| surprisal-only baseline | ~0.52 | 0.53 / 0.57 | 0.58 / 0.54 |

H1 is a robust positive across BPE↔SentencePiece, 32k↔256k vocab, fp16↔4-bit, 3B↔9B: voces are linearly
separable from token-matched nonsense at the early layers, far above a near-chance frequency baseline.
**Texture-recognition is a transformer property, not a Qwen artifact.** Two telling details: (a) in Gemma and
Mistral the voces run *more* surprising than their controls, yet H1 is undiminished — recognition does not
require familiarity; (b) in Mistral, the *Greek* H1 peak shifts late (L19 vs Latin's L5) — Mistral's 32k vocab
shreds Greek hardest, so the Greek distinction is *assembled deeper* rather than read off the surface. Both
point to H1 as a form-encoding property modulated by, but not dependent on, tokenization.

### 3.2 The deep-Greek persistence is a byte-fragmentation artifact (the iteration-1 "script" finding, mechanized)

Iteration 1 reported that name-adjacency persisted deeper in Greek than Latin, and read it as "Greek-script
processing." Across three tokenizers, that effect **appears, vanishes, or strengthens as a clean monotonic
function of how badly the tokenizer fragments Greek** — measured directly as tokens-per-Greek-vox (n=49
pure-asemic, low-contamination; each model's own tokenizer):

| model | vocab | **Greek tokens / vox** | deep-Greek gap (authentic Greek − Latin) |
|-------|-------|------------------------|------------------------------------------|
| **Mistral-7B** | 32k | **10.76** (most fragmented) | **+0.051** (strongest) |
| **Qwen-3B** | 152k | **9.63** | +0.016 (moderate) |
| **Gemma-9B** | 256k | **7.69** (least fragmented) | **~0.000** (none) |

The deep-Greek effect rises and falls *monotonically with Greek fragmentation*. Crucially, this is the
**direct** variable (token counts, comparable across models), not a vocab-size proxy, and the gradient is **not
explained by the obvious confounds**: the two **4-bit** models (Gemma, Mistral) sit at *opposite ends*, ruling
out quantization; model **size** (3B / 9B / 7B) does not track the ordering. The same-strings control confirms
the effect is encoding-invariant within a model (authentic vs algorithmic Greek agree), and the decider (§3.3)
confirms it is *not voces-specific* — Greek-script controls persist deep alongside the voces. So the deep effect
is: **script-general, fragmentation-driven, and absent once Greek tokenizes cleanly.** The "depth" was never the
words, and not even Greek as a language — it was *fragmented* Greek, an artifact of the tokenizer's coverage.

### 3.3 No voces-specific deep representation — the spell is dead in every family

| decider (voces vs token-matched controls, deep), p-value | Greek | Latin |
|---|---|---|
| **Qwen-3B** | 0.224 | 0.886 |
| **Gemma-9B** | 0.094 | **0.007** |
| **Mistral-7B** | 0.266 | 0.146 |

The decider is null in Qwen and Mistral, both scripts. Gemma shows a single significant cell (Latin, p=0.007) —
which the automated verdict labeled "abstraction reading gets real support." **It is not replicated:** Mistral's
Latin decider is null (p=0.146), Qwen's is null at two precisions (p=0.886). A real, stable deep voces
representation would not appear in one quantized model and vanish in two others. We classify Gemma's p=0.007 as
a **model-specific artifact** (it survives a basic Bonferroni across the four cells, 0.007 < 0.0125, so it is
not pure multiple-comparison noise — but it is unreplicated, 4-bit, and control-driven: voces sit at ≈0 while
their controls drift anti-name-like). **There is no voces-specific deep representation in any family.** The
spell is dead cross-tokenizer.

## 4. What survives, and what the depth was

- **Surface texture-recognition (H1): survives.** The one finding robust across three tokenizer families.
- **Deep voces representation (the spell): dead.** Decider null in 2 of 3; the lone exception unreplicated.
- **The deep "script" effect: a byte-fragmentation artifact** whose strength is a monotonic function of Greek
  tokens-per-vox — strongest where Greek shreds most (Mistral), gone where Greek tokenizes cleanly (Gemma).

*"It's the script, not the spell"* sharpens to: **the surface recognition is real and general; the deep "script"
effect is the tokenizer fragmenting Greek; and the spell — a representation of the voces as names-of-power — is
absent everywhere.** The cross-family run converted iteration 1's incidental, single-model side-finding into a
*mechanism with a measured dose-response*, and in doing so falsified its own prior reading twice (the deep-Greek
effect is neither voces-specific nor a property of "Greek"; it is fragmentation).

## 5. The honest hedges

- **n = 3 models.** The monotonic gradient is three points. Greek-fragmentation co-varies with family *and*
  vocab size; we have no same-family-different-tokenizer control to isolate fragmentation from family. Llama
  (tiktoken, 128k) is the 4th point — **falsifiable prediction:** with coverage near Qwen's, Llama should show
  Greek tokens/vox ≈ 9–10 and a deep gap near Qwen's +0.016. `[PENDING — Meta gate]`.
- **The killer falsifier, not yet run:** does **non-name Greek** (function words, numerals, common nouns)
  *also* persist deep in the fragmenting models? If yes → the deep effect is pure fragmentation-clustering, not
  namehood at all (the strongest form of the artifact reading). This is the single most decisive next test.
- **Quantization:** Gemma and Mistral are 4-bit; Gemma's p=0.007 especially wants an fp16 pass (Qwen's null was
  fp16-confirmed; Gemma's lone signal has not been).
- **Power:** n=49 pure-asemic-low-T per model; single seed per model (run-variance unestimated; a ≥3-seed
  repeat is owed). Deep effects are small absolute cosines.
- **Provenance:** hand-curated PGM material, not parsed from a verified Betz edition; 62/76 Greek forms are
  algorithmic transliterations (the authentic-Greek arm, n=13, is the non-circular check). Gemma's Greek
  token-counts were measured with a non-gated mirror of its tokenizer (`unsloth/gemma-2-9b`, identical SP model).

## 6. On arriving here (why the method matters)

The romantic reading — *the model holds these as names of power* — tried to enter **three times** and was
deflated each time by the next control: (1) Qwen's "Greek persists deep" → shown not voces-specific; (2) Gemma's
"Latin decider, abstraction gets real support" → unreplicated by Mistral/Qwen; (3) and the temptation, on seeing
a *clean monotonic fragmentation gradient*, to declare the mechanism proven at n=3 — held back by the missing
non-name-Greek falsifier and the family confound. The author's own predictions were **falsified twice** (the
deep-Greek effect did not behave as predicted; its tokenizer-dependence ran *opposite* to the predicted
direction), and both misses were more informative than the hits. The cross-family iteration is the discipline's
strongest form: it turned a single-model side-finding into a measured mechanism *by letting the data overturn
the study's own prior headline.* The result is larger than iteration 1's — a dose-response, not just a null —
and it is still, at bottom, deflationary: **the depth was the tokenizer.**

---

## References

*(Verified live 2026-06-24; status-labeled. Cross-family additions where noted.)* Zou et al. 2023
(arXiv:2310.01405); Park, Choe, Veitch 2024 (ICML / arXiv:2311.03658); Turner et al. 2023 (arXiv:2308.10248);
Sofroniew et al. 2026 (Transformer Circuits, arXiv:2604.07729); Lu et al. 2026 (arXiv:2601.10387); Betz (ed.)
1992 (Univ. of Chicago Press); Hwang 2026 (ICMI WP 26 — self-published, non-peer-reviewed; this study descends
from its method).

*Data & code: `voces-residual-stream` — cross-family results, the three-model decider, and the Greek-fragmentation
measurements in `results/cross-family/` (`greek_fragmentation.json`). The iteration-1 "abstraction" and
"Greek-persists-as-script" claims are retained as overturned, because the overturning is the method.*
