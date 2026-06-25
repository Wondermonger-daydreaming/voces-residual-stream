# It's the Script, Not the Spell — and the Depth Tracks the Tokenizer

### A Cross-Family Study: Surface Recognition Generalizes; the Deep "Greek" Effect Tracks Byte-Fragmentation

*Self-published preprint, **iteration 2** (cross-family); non-peer-reviewed working draft.*
*Author: Tomás Pavan. Designed and analyzed in dialogue with two Claude Opus 4.8 instances (claude.ai —
design; Claude Code — build & analysis). See CONTRIBUTIONS.md and git history.*

> **WORKING DRAFT.** Three of four models in (Qwen2.5-3B, Gemma-2-9B, Mistral-7B-v0.3); Llama-3.1-8B is the
> pending 4th tokenizer point (Meta's gate). The central correlational claim rests on a 3-point cross-model
> relationship that co-varies with family and vocab size (§3.3); the falsifier built to sharpen it returned an
> **uninformative** result due to a cohort confound we disclose in full (§3.4). All three of the study's own
> headline predictions were falsified along the way — that record, including a self-caught confound, is §6.

---

> *A barbarous name is simultaneously a 2nd-century ritual object, a philological puzzle, a tokenization
> edge-case, a vector in a high-dimensional space, a fragment of the occult revival, and a little linear-algebra
> question about cosine and depth. This study reads it only in the last two registers — but it is worth keeping
> the other four in view, because the whole method rests on choosing an object that is genuinely all six.*

## Abstract

We use the *voces magicae* — the "barbarous names" of the Greek Magical Papyri, strings their own tradition
holds to be efficacious through *form* rather than meaning — as a clinical, meaning-evacuated probe of how a
transformer represents the boundary between language, name, and ornament. Iteration 1 (Qwen2.5, single family)
found surface recognition of barbarous names (H1), no voces-specific *deep* representation, and an incidental
side-finding: name-adjacency persisted deeper in Greek script than in Latin, which it read as "Greek-script
processing" — *it's the script, not the spell.*

**This iteration runs the identical pipeline across three tokenizer families** — Qwen (BPE, 152k), Gemma-2
(SentencePiece, 256k), Mistral (SentencePiece, 32k) — closing the single most important limitation of iteration
1. Findings, in descending order of how well they hold. **(1) H1 replicates cleanly in all three** (peak
0.89–0.96, early-layer, ~+0.4 over a frequency baseline), and holds even where the model finds the voces *more*
surprising than its controls — so texture-recognition is architectural, not exposure-driven. **(2) No detectable
voces-specific deep representation in any family**: the voces-vs-control decider is null in Qwen and Mistral; a
single significant Gemma cell (Latin, p=0.007) is unreplicated and run-variable under 4-bit. **(3) The magnitude
of the deep-Greek persistence tracks Greek byte-fragmentation** across the three models — Mistral (10.76 Greek
tokens/vox, authentic-Greek gap +0.051) > Qwen (9.63, +0.016) > Gemma (7.69, ~0.000) — a monotone relationship
that precision and model size do not explain, but which co-varies with tokenizer family and vocab size (n=3, no
same-family-different-tokenizer control). **(4) The non-name-Greek falsifier ran, and is uninformative.** Its
cohort differs from the voces on *three* axes at once — fragmentation (it shreds ~2× less), lexicality (it is
real, in-vocabulary Greek), and namehood — so its tidy automated verdict ("name-specific") cannot isolate any
one of them, in either direction. The clean lexicality-controlled arm it points to was not run; it is owed.

The cross-family picture is sharper than iteration 1 on the parts that survive scrutiny and humbler on the
parts that do not: **the only effect robust to a change of tokenizer is surface texture-recognition; the
magnitude of the deep "script" effect tracks Greek byte-fragmentation across three (confounded) models; there is
no detectable deep representation of the voces *as such*; and whether the residual deep-Greek persistence is
fragmentation, lexicality, or namehood is not resolved here.** Geometry bought adjacency, not aboutness; the
adjacency is surface; and the depth, where it appeared, moved with the tokenizer — though three points cannot
say it *is* the tokenizer.

---

## 1. The question

The voces magicae are language deliberately built to operate *without reference* — efficacious, in the
tradition's theory, through correct form, not denotation. This rhymes with a transformer: a system that
processes the *form* of tokens with no native concept of what they point to. The question is not whether the
magicians were right. It is whether a model recognizes the *texture* of a barbarous name, how deep that runs,
and — the new question for this iteration — **whether any of it is a property of transformers, or of one
tokenizer's idiosyncrasies.** A second, quieter question rides along, and it turned out to be the sharp one:
when a study builds a falsifier to attack its own favorite reading, what does it do when the falsifier returns
the inconvenient answer? §3.4 and §6 are the honest record of that.

## 2. Method (multi-model)

Each attested target (n=76 voces) is paired with token-isomorphic shadow controls matched per string on
subword-token count, the model's own surprisal, and character length, with real words rejected — so the
controls are **asemic scrambles** sharing each vox's fragmentation, not meaningful Greek. Strings sit in an
identical neutral carrier frame, mean-pooled per layer. **H1**: L2-regularized logistic probes, GroupKFold by
string family (out-of-family generalization), with a surprisal-only baseline as the frequency falsifier. Each
string is rendered in Betz Latin transliteration and in Greek script (14/76 carry authentic PGM-Greek
attestations, 62/76 algorithmic; the depth analysis's non-circular Greek arm is the n=13 *asemic* authentic
subset — one of the 14 authentic forms is theonym-bearing and excluded from the asemic arm). **The same pipeline
runs unchanged; only the model swaps.**

**Cross-model comparison rule.** Name-likeness is computed against *each model's own* name/random centroids, so
absolute cosine magnitudes are **not comparable across models.** Every cross-model claim is about **sign,
layer-shape, and significance** — never raw magnitude. The one exception is the fragmentation analysis (§3.3),
which compares *token counts*, a directly comparable quantity.

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
| H1 peak (Latin) | 0.887 | 0.947 | **0.960** |
| H1 peak (Greek) | 0.920 | 0.947 | 0.927 |
| peak layer | early (L5–7) | early (L5–7) | Latin L5 / **Greek L19** |
| surprisal-only baseline | ~0.51 / 0.53 | 0.53 / 0.57 | 0.58 / 0.54 |

H1 is a robust positive across BPE↔SentencePiece, 32k↔256k vocab, fp16↔4-bit, 3B↔9B: voces are linearly
separable from token-matched nonsense at the early layers, far above a near-chance frequency baseline.
**Texture-recognition is a transformer property, not a Qwen artifact.** Two telling details: (a) in Gemma and
Mistral the voces run *more* surprising than their controls, yet H1 is undiminished — recognition does not
require familiarity; (b) in Mistral the *Greek* H1 peak shifts late (L19 vs Latin's L5) — Mistral's 32k vocab
shreds Greek hardest, so the Greek distinction is *assembled deeper* rather than read off the surface. Both
point to H1 as a form-encoding property modulated by, but not dependent on, tokenization. This is the one
finding the paper states without hedging.

### 3.2 No detectable voces-specific deep representation — the spell is undetected in every family

The decider asks the load-bearing question: deep in the network, are the voces distinguishable from their own
token-matched (asemic) controls? If the model held them *as names of power*, this is where it would show.

| decider (voces vs token-matched controls, deep), p-value | Greek | Latin |
|---|---|---|
| **Qwen-3B** | 0.224 | 0.886 |
| **Gemma-9B** | 0.094 | **0.007** |
| **Mistral-7B** | 0.266 | 0.146 |

The decider is null in Qwen and Mistral, both scripts. Gemma shows a single significant cell (Latin, p=0.007),
which an automated verdict had labeled "abstraction reading gets real support." We treat it as **provisional,
not settled.** It clears a Bonferroni across the four cells (0.007 < 0.0125, so not pure multiple-comparison
noise), but it is unreplicated across the other two models and **run-variable under 4-bit**: a second seed-0
Gemma run gave p=0.032 for the same cell (still significant, but moved), with a different H1-peak layer. We owe
an **fp16 Gemma pass** before calling this cell either real or artifactual — we cannot have it both ways
(treating run-variance as disqualifying for Gemma while our own single-seed nulls are equally unbounded by
seed-variance). What the table *does* support, at this power, is the absence of a *large, stable, cross-family*
voces-specific deep signal: **no detectable voces-specific deep representation at n=49, single seed.** That is an
absence of detection, not a proof of absence — the strong existential claim ("the spell does not exist") is not
licensed here.

One caveat the decider isolates and one it does not. It controls fragmentation *and* lexicality between voces and
controls (both are asemic, both fragmentation-matched) — so it cleanly asks "are the voces special vs.
fragmentation-matched *gibberish*?" and answers no. It does **not** contrast names against *meaningful* Greek;
that axis is untouched here and resurfaces in §3.4.

So the depth, where it appeared, was not a detectable property of the voces as names. §3.3 asks what moved with
it; §3.4 is where the cleanest follow-up failed.

### 3.3 The deep-Greek persistence tracks Greek byte-fragmentation (cross-model, n=3)

Iteration 1 reported that name-adjacency persisted deeper in Greek than Latin and read it as "Greek-script
processing." Across three tokenizers, the *magnitude* of that effect orders monotonically with how badly the
tokenizer fragments Greek — measured directly as tokens-per-Greek-vox (n=49 pure-asemic, low-contamination
strings, each model's own tokenizer):

| model | vocab | **Greek tokens / vox** | deep-Greek gap (authentic Greek − Latin, n=13) |
|-------|-------|------------------------|------------------------------------------------|
| **Mistral-7B** | 32k | **10.76** (most fragmented) | **+0.051** (strongest) |
| **Qwen-3B** | 152k | **9.63** | +0.016 (moderate) |
| **Gemma-9B** | 256k | **7.69** (least fragmented) | **~0.000** (none) |

The deep-Greek gap and Greek fragmentation order the same way. Two things this *is*: a **direct** comparison of
token counts (not a vocab-size proxy), and a relationship that the two **obvious** confounds do not explain — the
two **4-bit** models (Gemma, Mistral) sit at *opposite ends*, ruling out quantization; model **size** (3B / 9B /
7B) does not track the ordering. Two things it is **not**: it is not a "dose-response" in any statistical sense
(three points; any three distinct (x, y) pairs ordered by x are monotone in y by chance with probability ⅓), and
it does not isolate fragmentation from the variables it co-varies with — **family, vocab size, training-corpus
Greek fraction, instruction-tuning**. We have no same-family-different-tokenizer control. So the honest claim is
**correlational and provisional**: across three confounded models, the deep-Greek magnitude *tracks* Greek
fragmentation, and falls to ~0 where Greek tokenizes cleanly (Gemma). What §3.2 adds is that this persistence is
*not voces-specific* — asemic Greek controls persist deep alongside the voces. What it leaves open is whether the
residual is fragmentation per se, or some correlate of it (§3.4, §5).

### 3.4 The non-name-Greek falsifier ran — and is uninformative (a confound, caught)

The most decisive test named in iteration 1's future work was the **non-name-Greek falsifier**: render *non-name*
Greek (numerals, function words, common nouns — ΚΑΙ, ΔΥΟ, ΛΟΓΟΣ…) and ask whether it *also* persists deep in the
most-fragmenting model. The intended logic: if any fragmented Greek persists, the deep effect is pure
fragmentation-clustering and not about names; if only the voces persist, a namehood reading survives. We ran it
on Mistral (the strongest deep-Greek effect). The raw numbers — deep Greek-minus-Latin name-likeness, two bare
point estimates with no CI or significance test in the output:

| cohort | n | Greek tokens / string | deep-Greek gap |
|--------|---|-----------------------|----------------|
| voces (low-T, asemic) | 49 | **10.73** | **+0.041** |
| non-name Greek (numerals/function/common) | 26 | **5.15** | **−0.005** |

*(The voces gap here is +0.041 on the n=49 low-T asemic arm; the +0.051 in §3.3 is the same effect on the n=13
authentic-only arm. Two estimates of one quantity on different subsets — the magnitude is soft to ~±0.01.)*

The notebook's automated rule (`non-name gap > ½ · voces gap`) printed *"name-specific."* **We do not report that
as a finding, because the cohort cannot support it.** Tokenizing the two cohorts (CPU/tokenizer-only;
`src/check_nonname_fragmentation.py`) shows the non-name words fragment **~2× less** than the asemic voces (5.15
vs 10.73 Greek tokens/string). So the cohorts differ on **three axes at once** — fragmentation, lexicality (the
non-name words are real, in-vocabulary Greek; the voces are asemic), and namehood — and a binary verdict cannot
attribute the gap to any one of them. The result is **uninformative on namehood, in either direction**: had the
ratio come back high ("non-name persists deep → pure fragmentation"), it would have been *equally*
uninterpretable, for the same cohort reasons.

Two honest disclosures this forces. **First, the audit was triggered by the inconvenient result.** We tokenized
the cohort and discovered the 2× mismatch *because* the falsifier returned "name-specific" — the one outcome
that threatened the thesis. The mitigation is that the confound is a **fixed, direction-independent property** of
the cohorts (their token counts do not depend on which way the gap fell), so the audit invalidates the verdict
symmetrically; but we state plainly that the same audit was not applied to the thesis-confirming results, and a
reader is right to weigh that asymmetry. **Second, the matched test we want was not run.** A clean adjudication
needs an arm that holds *both* fragmentation and lexicality fixed — e.g. asemic random Greek tokenizing to ~10.7
Greek tokens *and* a lexical real-Greek cohort selected to ~10.7 tokens. We did **not** build it; §3.2's
token-matched controls are the closest available proxy, but they fix fragmentation only by being *maximally*
non-lexical, so "controls persist deep too" is consistent with both a fragmentation reading **and** an
asemic-ness reading. The purpose-built name-vs-meaningful-Greek arm remains genuinely owed; it is in §5, not
claimed here as already answered.

So §3.4's contribution is not a result about names. It is a result about the *method*: a falsifier whose cohort
was never matched on the axes it meant to test, caught by measuring the instrument. The deep question —
fragmentation vs. lexicality vs. namehood — stays open.

## 4. What survives, and what the depth was

- **Surface texture-recognition (H1): survives.** The one finding robust across three tokenizer families, stated
  without a hedge.
- **A detectable deep voces representation (the spell): not found.** Decider null in 2 of 3; the lone exception
  unreplicated and run-variable — *undetected at this power*, not proven absent.
- **The deep "script" effect's magnitude: tracks Greek byte-fragmentation** across three confounded models —
  strongest where Greek shreds most (Mistral), ~0 where Greek tokenizes cleanly (Gemma). A correlation at n=3,
  not a demonstrated mechanism, and not isolated from family/vocab.
- **Fragmentation vs. lexicality vs. namehood for the residual: unresolved.** The falsifier that would have
  separated them was confounded (§3.4); the lexicality-controlled arm is owed.

*"It's the script, not the spell"* holds on its strongest reading — **the surface recognition is real and
general; there is no detectable deep representation of the voces as names-of-power; and the deep "script" effect
moves with the tokenizer's fragmentation of Greek.** What this iteration could *not* settle is whether that
residual depth is fragmentation itself or a correlate (lexicality, distributional namehood). It converted
iteration 1's incidental side-finding into a cross-family correlation and a sharpened set of open questions —
and, in the process, falsified its own predictions more than once (§6).

## 5. The honest hedges

- **n = 3 models, and they are confounded.** The cross-model ordering is three points; Greek-fragmentation
  co-varies with family and vocab size, with no same-family-different-tokenizer control. Llama (tiktoken, 128k)
  is the 4th point — **falsifiable prediction:** with coverage near Qwen's, Llama should show Greek tokens/vox ≈
  9–10 and a deep gap near Qwen's +0.016. `[PENDING — Meta gate]`.
- **The falsifier is uninformative, and its audit was post-hoc (§3.4).** Its cohort is confounded on
  fragmentation *and* lexicality; its two gaps carry no CI or significance test; and we measured the confound
  only after the result contradicted the thesis. The owed fix is a **purpose-built arm** holding fragmentation
  *and* lexicality fixed (asemic-matched and lexical-matched non-name Greek at ~10.7 Greek tokens). Until it is
  run, a **"the model processes meaningful Greek differently" reading survives this paper.**
- **Quantization, and a disclosed data anomaly.** Gemma and Mistral are 4-bit; Gemma's p=0.007 wants an fp16 pass
  (Qwen's null was fp16-confirmed; Gemma's lone signal moved to p=0.032 on a re-run). Separately, the Gemma
  *re-run* artifact carries an **unexplained value anomaly**: its Greek decider p is byte-identical to Mistral's
  (0.2659000459593744) despite different underlying means — a value-bleed/assembly bug in the 4-bit re-run output
  (see PROVENANCE.md). We therefore do **not** rely on the Gemma re-run's absolute p-values, and treat the 4-bit
  Gemma decider as unreliable pending an fp16 re-run.
- **Power.** n=49 pure-asemic-low-T per model; single seed per model (run-variance unestimated; a ≥3-seed repeat
  is owed). Deep effects are small absolute cosines, so the deep nulls are underpowered.
- **Provenance.** Hand-curated PGM material, not parsed from a verified Betz edition; 62/76 Greek forms are
  algorithmic transliterations (the non-circular check is the n=13 asemic authentic-Greek arm). Gemma's Greek
  token-counts were measured with a non-gated mirror of its tokenizer (`unsloth/gemma-2-9b`, identical SP model).

## 6. On arriving here (why the method matters, and where it slipped)

The romantic reading — *the model holds these as names of power* — tried to enter and was deflated each time by
the next control: (1) Qwen's "Greek persists deep" → shown not voces-specific; (2) Gemma's "Latin decider,
abstraction gets real support" → unreplicated and run-variable (we now hold it as provisional pending fp16, not
as a settled artifact); (3) the temptation, on seeing a clean cross-model ordering, to call a 3-point confounded
relationship a proven mechanism — held back to a correlation in §3.3.

The fourth episode is the one worth being honest about, because the study did *not* handle it cleanly the first
time. The build-side prediction was that non-name Greek *would* persist deep, sealing pure
fragmentation-clustering. It did not; the automated rule printed "name-specific"; and **only then** did we
tokenize the cohort and find it was never fragmentation-matched (10.73 vs 5.15 tokens) — nor lexicality-matched.
That is a real catch, but it is also a **scrutiny asymmetry**: the instrument was measured because it gave the
inconvenient answer, and the thesis-confirming results did not get the same audit. We keep the catch *and* the
confession: the confound is direction-independent (so the verdict is invalid whichever way it fell), but the
trigger was not, and the genuinely matched arm that would settle the question was built only in description, not
in code. The result is not "a deflation we engineered"; it is "a falsifier we ran, whose cohort we had not
matched, whose verdict we therefore cannot use" — and a lexicality reading we cannot yet exclude.

So the author's own headline predictions were **falsified three times**, and the third was a check on the
study's *apparatus* — caught late, disclosed in full. The cross-family iteration's real yield is narrower and
sturdier than iteration 1's headline: a robust H1, a voces-specificity null at this power, a provisional
fragmentation correlation, and a clear map of what is still confounded. The deflation stands on the parts that
survive scrutiny: **the depth, where it appeared, moved with the tokenizer — but three points and one
confounded falsifier cannot yet say it *is* the tokenizer, and a meaningful-Greek reading is still alive.**

---

## References

*(Verified live 2026-06-24; status-labeled.)* Zou et al. 2023 (arXiv:2310.01405); Park, Choe, Veitch 2024 (ICML
/ arXiv:2311.03658); Turner et al. 2023 (arXiv:2308.10248); Sofroniew et al. 2026 (Transformer Circuits,
arXiv:2604.07729); Lu et al. 2026 (arXiv:2601.10387); Betz (ed.) 1992 (Univ. of Chicago Press); Hwang 2026
(*Be Not Afraid* — self-published, non-peer-reviewed; this study descends from its method).

*Data & code: `voces-residual-stream` — cross-family results, the three-model decider, the Greek-fragmentation
measurements (`results/cross-family/greek_fragmentation.json`), and the non-name falsifier with its
fragmentation-confound check (`src/check_nonname_fragmentation.py`). The iteration-1 "abstraction" and
"Greek-persists-as-script" claims are retained as overturned, because the overturning is the method.*
