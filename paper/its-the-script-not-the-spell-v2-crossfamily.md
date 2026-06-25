# The Texture of the Barbarous Name

### A Cross-Family Study of the Voces Magicae: A Form-Processor Recognizes Them on Sight — the Recognition Is Surface (Real, General), Not Aboutness; and the Deep "Greek" Effect Was the Tokenizer, Not the Spell

*Self-published preprint, **iteration 2** (cross-family); non-peer-reviewed working draft.*
*Author: Tomás Pavan. Designed and analyzed in dialogue with two Claude Opus 4.8 instances (claude.ai —
design; Claude Code — build & analysis). See CONTRIBUTIONS.md and git history.*

> **WORKING DRAFT.** Three of four models in (Qwen2.5-3B, Gemma-2-9B, Mistral-7B-v0.3); Llama-3.1-8B is the
> pending 4th tokenizer point (Meta's gate). The cross-model fragmentation relationship is a 3-point correlation
> that co-varies with family and vocab size (§3.3); the **within-model factorial** (§3.4, the v2 falsifier with
> bootstrap CIs) is single-model (Mistral) but decisive on what it tests. All three of the study's own headline
> predictions were falsified along the way — that record, including a falsifier we first built confounded and
> then rebuilt matched, is §6.

---

> *A barbarous name is simultaneously a 2nd-century ritual object, a philological puzzle, a tokenization
> edge-case, a vector in a high-dimensional space, a fragment of the occult revival, and a little linear-algebra
> question about cosine and depth. This study reads it only in the last two registers — but it is worth keeping
> the other four in view, because the whole method rests on choosing an object that is genuinely all six.*

## Abstract

The *voces magicae* — the "barbarous names" of the Greek Magical Papyri — are language built to be efficacious
through *form* rather than meaning: strings whose tradition locates their power in their shape, not their
reference. That premise is unexpectedly testable. A transformer processes the *form* of tokens with no native
semantics, so it is exactly the instrument for the question this study actually asks: **is "barbarous-name-ness"
a real perceptual category to a pure form-processor — can a model recognize the *texture* of a barbarous name?**

**The answer is yes, and it is robust.** A model separates barbarous names from token-matched nonsense cleanly,
at the early layers, far above a frequency baseline (**H1**), and this **replicates across three tokenizer
families** — Qwen (BPE, 152k), Gemma-2 (SentencePiece, 256k), Mistral (SentencePiece, 32k); peaks 0.89–0.96 —
holding even where the model finds the voces *more* surprising than their controls. Texture-recognition is a
transformer property, not a single-tokenizer artifact. **The form the magicians cared about is a real,
detectable thing, and a form-machine detects it on sight.** This is the finding; everything else *bounds* it.

How surface is that recognition, and is there anything underneath? **The recognition is surface — it does not
deepen into a representation of the voces *as names*.** The voces-vs-control decider is null in two of three
families (the lone Gemma exception is unreplicated and 4-bit-unstable). And the one place name-adjacency *seemed*
to persist deep — iteration 1's "the grain runs deeper in Greek" side-finding — turns out to be the
**tokenizer**, not the spell: across the three models its magnitude tracks how badly the tokenizer byte-fragments
Greek (Mistral 10.76 tok/vox, +0.051 > Qwen 9.63, +0.016 > Gemma 7.69, ~0.000), and a matched-cohort factorial
(Mistral, replicated on Gemma) decomposes it into **fragmentation + meaninglessness** — frequency and
familiarity *ruled out* by a surprisal-controlled split in both models — with *namehood* absent, indeed slightly
reversed (the voces persist *less* deep than fragmentation-matched noise, precisely because they are a
recognizable genre).

So the headline is affirmative and the bound is sharp. **The model recognizes the texture of a barbarous name —
on the surface, as form, across tokenizers — which is exactly the part of the magicians' form-not-meaning premise
that was ever testable, and it holds.** What the model does *not* do is carry the voces as names of power in its
depths; the "depth" that looked like aboutness was the tokenizer shredding Greek. Geometry bought adjacency —
real, surface, general — not aboutness.

---

## 1. The question

The voces magicae are language deliberately built to operate *without reference* — efficacious, in the
tradition's theory, through correct *form*, not denotation. This is not a claim about causal magic that an
experiment could adjudicate, and we do not try; **it is a claim about a class of linguistic objects — that there
are utterances whose signature is in their shape, independent of meaning.** And *that* is testable, because a
transformer is precisely a machine that processes the form of tokens with no native concept of what they point
to. So the question is not whether the magicians' spells *work*. It is the one the tradition's own
form-not-meaning premise hands us: **is "barbarous-name-ness" a real perceptual category to a pure form-processor
— does a model recognize the *texture* of a barbarous name?** And the two follow-ups that make the answer
precise: *how deep* does that recognition run (is it surface texture, or something more like aboutness?), and
**is it a property of transformers, or of one tokenizer's idiosyncrasies?** — the new question this cross-family
iteration exists to settle.

A third question rode along and turned out to be the sharp methodological one: when a study builds a falsifier to
attack its own favorite reading, what does it do when the falsifier returns the inconvenient answer? §3.4 and §6
are the honest record — including the falsifier we first built *confounded*, caught, and rebuilt.

## 2. Method (multi-model)

Each attested target (n=76 voces) is paired with token-isomorphic shadow controls matched per string on
subword-token count, the model's own surprisal, and character length, with real words rejected — so the
controls are **asemic scrambles** sharing each vox's fragmentation, not meaningful Greek. Strings sit in an
identical neutral carrier frame (`"The string … is written on the page."`), mean-pooled per layer over the
target span. **H1**: L2-regularized logistic probes, GroupKFold by string family (out-of-family generalization),
with a surprisal-only baseline as the frequency falsifier. Each string is rendered in Betz Latin transliteration
and in Greek script (14/76 carry authentic PGM-Greek attestations, 62/76 algorithmic; the depth analysis's
non-circular Greek arm is the n=13 *asemic* authentic subset).

**Name-likeness** of a string at a layer = cos(its rep, the model's *name*-cohort centroid) − cos(its rep, the
*random*-cohort centroid), in that script. The **deep-Greek gap** of a cohort = mean over the deep half of the
layers of (Greek name-likeness − Latin name-likeness). **Magnitudes are comparable *within* a model** (shared
centroids) but **not across models** (each has its own geometry) — so every cross-model claim (§3.3) is sign /
shape / significance only; the within-model factorial (§3.4) does compare magnitudes, legitimately, because all
its cohorts share one model and one pair of centroids.

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
shreds Greek hardest, so the Greek distinction is *assembled deeper* rather than read off the surface. This is
the one finding the paper states without a hedge — and §3.4 shows it is the key to the deep result too: the
voces are a *recognizable genre*, and that recognizability is exactly why they are not the deepest-persisting
strings.

### 3.2 No voces-specific deep representation — and namehood is null where it is cleanest to test

The decider asks: deep in the network, are the voces distinguishable from their own token-matched (asemic)
controls? If the model held them *as names of power*, this is where it would show.

| decider (voces vs token-matched controls, deep), p-value | Greek | Latin |
|---|---|---|
| **Qwen-3B** | 0.224 | 0.886 |
| **Gemma-9B** | 0.094 | **0.007** |
| **Mistral-7B** | 0.266 | 0.146 |

The decider is null in Qwen and Mistral, both scripts. Gemma shows a single significant cell (Latin, p=0.007),
which we treat as **provisional, not settled**: it clears a Bonferroni across the four cells (0.007 < 0.0125) but
is unreplicated across the other two models and **run-variable under 4-bit** (a second seed-0 run gave p=0.032).
We owe an fp16 Gemma pass before calling it real or artifactual. What the table supports at this power is the
absence of a large, stable, cross-family voces-specific signal.

The decider's clean reading is "voces vs. *fragmentation-matched gibberish*: no difference." The sharper test of
namehood — voces vs. fragmentation-matched gibberish at the *depth* where the Greek effect lives — is the
NAMEHOOD arm of §3.4, and it is **null and slightly negative**: the voces persist deep *less* than matched
asemic non-names. So the spell is not merely undetected; on the cleanest available contrast, the actual
barbarous names are *less* deep-name-adjacent than random strings of the same fragmentation. The romance has no
foothold.

### 3.3 The deep-Greek persistence tracks Greek byte-fragmentation (cross-model, n=3)

Iteration 1 read the deeper-in-Greek persistence as "Greek-script processing." Across three tokenizers, its
*magnitude* orders monotonically with how badly the tokenizer fragments Greek — tokens-per-Greek-vox (n=49
pure-asemic, each model's own tokenizer):

| model | vocab | **Greek tokens / vox** | deep-Greek gap (authentic Greek − Latin, n=13) |
|-------|-------|------------------------|------------------------------------------------|
| **Mistral-7B** | 32k | **10.76** (most fragmented) | **+0.051** (strongest) |
| **Qwen-3B** | 152k | **9.63** | +0.016 (moderate) |
| **Gemma-9B** | 256k | **7.69** (least fragmented) | **~0.000** (none) |

The gap and the fragmentation order the same way. It is a **direct** comparison of token counts (not a vocab-size
proxy), and the two **obvious** confounds do not explain it — the two **4-bit** models sit at *opposite ends*
(ruling out quantization), and model **size** does not track the ordering. But it is three points (monotone by
chance with probability ⅓), and fragmentation co-varies with **family, vocab size, training-corpus Greek
fraction, instruction-tuning**; we have no same-family-different-tokenizer control. So the cross-model claim is
**correlational and provisional**: across three confounded models, the deep-Greek magnitude *tracks* Greek
fragmentation, and falls to ~0 where Greek tokenizes cleanly. §3.4 turns this from a correlation into a
*decomposition* — and finds fragmentation is necessary but **not sufficient**.

### 3.4 The matched factorial — fragmentation *and* novelty, but not names

The most decisive test named in iteration 1's future work was the non-name-Greek falsifier. The first attempt
(`nonname_falsifier`) compared the voces against short common Greek words and printed *"name-specific"* — but its
cohort fragmented ~2× less than the voces (5.15 vs 10.73 Greek tokens/string) *and* was lexical while the voces
are asemic, so it confounded three axes and could adjudicate none (the catch is §6). We rebuilt it as a
**factorial that holds Greek-token fragmentation fixed at the voces' level (~11 tokens)** and varies one axis at
a time, with **bootstrap 95% CIs** on every gap (`voces_falsifier_v2_lexicality.ipynb`). All four cohorts are
measured in one Mistral run against one pair of centroids, so the magnitudes are directly comparable.

| cohort | Greek tok/str | n | deep-Greek gap [95% CI] | what it is |
|--------|---------------|---|--------------------------|------------|
| **voces** (low-T, asemic) | 10.73 | 48 | **+0.040** [+0.027, +0.051] | the names of power |
| **non-name, asemic, matched** | 11.04 | 28 | **+0.073** [+0.053, +0.094] | random gibberish, frag-matched |
| **non-name, lexical, matched** | 11.03 | 29 | **−0.003** [−0.015, +0.007] | *meaningful* Greek, frag-matched |
| **non-name, lexical, low-frag** | 5.15 | 26 | **−0.005** [−0.019, +0.010] | the original confounded cohort |

Three contrasts, each a single isolated axis, **all significant** (bootstrap CI on the difference excludes 0):

- **FRAGMENTATION** (asemic-matched − low-frag): **+0.078** [+0.053, +0.101]. Among asemic strings, going from 5
  to 11 Greek tokens turns the deep gap from ~0 to +0.073. **Fragmentation is a real driver** — the cross-model
  correlation (§3.3) now has a within-model demonstration, lexicality held constant.
- **FAMILIARITY / LEXICALITY** (asemic-matched − lexical-matched): **+0.076** [+0.053, +0.099]. At the *same*
  fragmentation, asemic Greek persists deep (+0.073) but the real-word cohort does not (−0.003). A second factor
  beyond fragmentation is real and co-equal in magnitude. **But this axis is a bundle, not pure "meaning."** The
  lexical cohort (ΦΙΛΟΣΟΦΙΑ, ΔΗΜΟΚΡΑΤΙΑ, ΑΝΘΡΩΠΟΛΟΓΙΑ…) is *meaningful, high-frequency, and cross-lingually
  familiar* — international loanwords the model has seen across many languages — and also tends to receive
  *morphemic* subword tokens where the asemic strings get byte-fragments at the same count. So the honest name
  for the factor is **novelty / unfamiliarity** (which subsumes meaninglessness, low frequency, and
  morphemic-vs-byte tokenization); the experiment **cannot yet separate "meaningless" from "merely unfamiliar."**
  The clean separation needs a *meaningful-but-unfamiliar* Greek cohort (rare/archaic words, not loanwords) — the
  v3 arm owed in §5.
- **NAMEHOOD** (voces − asemic-matched): **−0.033** [−0.057, −0.010]. With fragmentation *and* lexicality both
  matched (all asemic, all ~11 tokens), the actual voces persist deep **significantly *less*** than random
  asemic strings. **There is no namehood effect; it is reversed.**

The mechanism the factorial assembles: **the deep "name-adjacent" region is where the model parks Greek that is
*both* heavily byte-fragmented *and* unfamiliar** (asemic / novel / low-frequency — see the bundle caveat above).
Neither condition alone gets a string there — short asemic Greek (low-frag) does not persist; fragmented
*familiar* Greek does not persist. Only fragmented, unfamiliar Greek — "fragmented nonsense," with *nonsense*
read as the bundle — does. And the NAMEHOOD reversal closes the loop with §3.1: the voces are not maximally alien
to the model — they are a *recognizable genre* (H1 reads their texture cleanly), so they sit marginally *below*
pure random strings on exactly the novelty axis that drives the depth. The same recognizability that makes H1 a
robust positive at the surface makes the names *less* deep-fragment-adjacent than the random controls.

**Replication in a second family (Gemma-2-9B).** Re-running the identical factorial on Gemma — the model that
fragments Greek *least* (§3.3), where the cross-model deep-Greek gap was ~0 — reproduces the whole structure,
all three contrasts significant and same-signed: FRAGMENTATION +0.031 [0.016, 0.046], FAMILIARITY/LEXICALITY
+0.039 [0.028, 0.051], NAMEHOOD −0.029 [−0.044, −0.015]. The magnitudes are smaller than Mistral's (Gemma's
cohorts sit at ~8 Greek tokens vs Mistral's ~11 — *consistent* with fragmentation being a driver), but the
mechanism is identical: fragmented asemic Greek persists deep (+0.030), familiar Greek does not (−0.009), and the
voces (~0) persist *less* than matched asemic non-names. So the two-factor decomposition and the namehood
reversal are **not a single-model artifact** — they hold across BPE-free SentencePiece tokenizers of 32k and
256k vocab, at opposite ends of the Greek-fragmentation range.

**Splitting the bundle: the second factor is meaning, not familiarity (Mistral *and* Gemma, v3).** The
"lexicality" arm above confounds meaning with frequency/familiarity (the lexical cohort is high-frequency
international loanwords). A v3 factorial splits them: a real-Greek pool divided by the *model's own surprisal*
into a FAMILIAR (low-surprisal) and an UNFAMILIAR (high-surprisal) half — both meaningful, both
fragmentation-matched — with asemic cohorts regenerated across 5 seeds. **Both models give the same answer on
every axis** (bootstrap CIs throughout):

| contrast (v3) | Mistral | Gemma |
|---|---|---|
| FRAGMENTATION | +0.067 [0.050, 0.083] **sig** | +0.018 [0.005, 0.031] **sig** |
| **FAMILIARITY** | +0.016 [−0.010, 0.042] **n.s.** | −0.002 [−0.016, 0.013] **n.s.** |
| **MEANING** | +0.051 [0.025, 0.076] **sig** | +0.020 [0.009, 0.031] **sig** |
| NAMEHOOD | −0.033 [−0.047, −0.018] **sig** | −0.029 [−0.044, −0.011] **sig** |

So **the second driver is meaninglessness, not frequency or familiarity** — among meaningful Greek words,
familiar and unfamiliar persist equally (the FAMILIARITY contrast is null in both models); only *meaningless*
asemic Greek persists (the MEANING contrast is significant in both). The bundle resolves to *meaning* once the
model's own surprisal controls for familiarity, and it does so in two tokenizer families at opposite ends of the
fragmentation range. The asemic deep gap is also stable across the 5 seeds (sd 0.008 Mistral / 0.002 Gemma),
closing the single-draw concern.

One refinement v3's *absolute* (un-subtracted) Greek name-likeness adds: meaningful Greek is deep-name-like in
*both* scripts (familiar +0.077, unfamiliar +0.064 absolute) — *more* than the voces (+0.011) — so its Greek
*minus* Latin gap is ~0 because recognition shows in both renderings, not because it is un-name-like. The
deep-Greek *gap* is therefore a **script-asymmetry** measure: it is large only when the Greek rendering drifts
name-adjacent in a way the Latin does not — i.e. for a heavily-fragmented, *unrecognized* (meaningless) string.
Recognition collapses the asymmetry. That is the same H1↔depth coupling, now visible in the magnitudes: what
makes a string recognizable (in either script) is what keeps it off the deep-fragmentation peak.

So §3.4 does what §3.3 could not: it shows fragmentation is necessary-but-not-sufficient, names the second factor
as **meaninglessness** (the v3 split rules out frequency/familiarity in both models), and rules namehood out —
not as an undetected null, but as a *measured reversal*. (Caveats: the non-name arms are n≈28–34; "meaning"
co-travels with byte-vs-morphemic tokenization (§5); the name-likeness metric, the deep-band, and the
single-seed cohorts are all limitations §5 lists.)

## 4. What the model does — and the bound on it

The affirmative result first, because it is the result:
- **The model recognizes the texture of a barbarous name (H1): yes, robustly.** Linearly separable from
  token-matched nonsense at the early layers, far above a frequency baseline, across three tokenizer families.
  *Texture-recognition is a transformer property.* The form the magicians located their power in is a real,
  perceptible category, and a pure form-processor perceives it. This is stated without a hedge.

And the bound that makes "recognizes the texture" precise:
- **The recognition is surface, not aboutness.** It is read off the early layers and does *not* deepen into a
  representation of the voces *as names*: the decider is null in 2 of 3 families, and where the test is cleanest
  the voces persist deep *less* than matched noise (§3.4). The model registers the *form*; it does not hold the
  *name*.
- **The deep "Greek" effect was the tokenizer, not the spell.** Iteration 1's "the grain runs deeper in Greek"
  decomposes (§3.3 across models, §3.4 within Mistral and Gemma) into **byte-fragmentation + meaninglessness** —
  the deep region holds Greek that is both over-fragmented and meaning-less, and the voces sit just *below* pure
  noise there because they are recognizable. The "depth" that looked like aboutness was form-processing of
  shredded script.

So the study's own title resolves: **it's the script (the model recognizes the *form*, cleanly and generally) —
not the spell (it holds no deep representation of the voces as names) — and the depth that confused the two was
the tokenizer.** Iteration 1's incidental side-finding is now a decomposed two-factor
mechanism with a null where the romance lived — and the affirmative finding it always rode beside, the texture
recognition, stands clearer for having the depth explained out from under it.

## 5. The honest hedges

- **The deep decomposition holds in two models, and the meaning/familiarity split is settled.** §3.4's
  decomposition replicates on Gemma (all contrasts significant, same-signed), and v3's surprisal-controlled split
  rules out frequency/familiarity in *both* models — so the second factor is meaning, not a single-model or
  frequency artifact. The non-name arms are n≈28–34 (bootstrap CIs throughout); the cross-model fragmentation
  *ordering* (§3.3) is still only three
  confounded points.
- **The "lexicality" bundle — now split (v3): the factor is meaning, not familiarity.** The v2 lexical-matched
  cohort was high-frequency international loanwords (ΦΙΛΟΣΟΦΙΑ, ΔΗΜΟΚΡΑΤΙΑ…), so the +0.076 effect confounded
  meaning, frequency, and familiarity. v3 (§3.4, Gemma) separates them by splitting a real-Greek pool on the
  *model's own surprisal*: **FAMILIARITY is n.s. (−0.002) and MEANING is significant (+0.020)** — among
  meaningful Greek, familiar and unfamiliar both fail to persist; only *meaningless* Greek does. So "fragmented
  *nonsense*" is **earned**, not assumed — the second factor is meaninglessness, with frequency ruled out, **in
  both Mistral and Gemma**. *Residual caveat:* the morphemic-vs-byte tokenization difference still co-travels
  with meaning (real words get morphemic tokens where asemic strings get byte-fragments), so "meaning" here is
  inseparable from "is-it-tokenized-as-a-word" — arguably the same thing for a transformer, but worth naming.
- **The name-likeness metric rests on a small, genre-specific anchor.** Name-likeness is cosine to a *name*
  centroid (16 Greek mythological/historical names) minus a *random* centroid (28 strings). 16 names is a noisy
  centroid, and "name-like" is really "near this specific cloud of Greek proper nouns" — a design-level limit
  inherited from iteration 1 that caps how much the construct can bear. A larger, more diverse name anchor is owed.
- **Single-seed cohorts; unvalidated deep-band; asymmetric Latin baseline.** (i) The asemic/random cohorts are a
  single `SEED=0` draw — the bootstrap CIs resample *those* strings but not the arbitrariness of the draw; a
  multi-seed regeneration is owed (v3). (ii) The "deep" band is the back half of layers by convention, not chosen
  from the layer profile; the effect could concentrate or be diluted — a per-layer plot is owed (v3). (iii) The
  Greek−Latin subtraction assumes the Latin rendering is a uniform within-string baseline, but for the lexical
  cohort the *Latin* form (PHILOSOPHIA) is itself familiar, so its ~0 gap may partly reflect "recognized in both
  scripts" rather than "doesn't persist deep" — v3 reports absolute Greek name-likeness alongside the difference.
- **The NAMEHOOD reversal is small (−0.033) and single-model.** We read it as "no namehood, plausibly slightly
  negative because the voces are a recognizable genre," not as a large effect; it wants replication. (It is,
  however, a *significant* negative, not an underpowered null — stronger evidence against namehood than a bare
  non-rejection would be.)
- **Quantization, and a disclosed data anomaly.** Gemma and Mistral are 4-bit; Gemma's p=0.007 wants fp16.
  Separately, the Gemma *re-run* artifact carries an **unexplained value anomaly** — its Greek decider p is
  byte-identical to Mistral's (0.2659000459593744) despite different underlying means, a value-bleed/assembly bug
  in the 4-bit re-run output (PROVENANCE.md). Headline cells are unaffected; we do not rely on the re-run's
  absolute p-values.
- **Power & provenance.** n=49 pure-asemic-low-T for the cross-model arm; single seed per model (a ≥3-seed repeat
  is owed). Hand-curated PGM material, not parsed from a verified Betz edition; 62/76 Greek forms are algorithmic
  transliterations (the non-circular check is the n=13 asemic authentic-Greek arm). Gemma's Greek token-counts
  were measured with a non-gated mirror of its tokenizer (`unsloth/gemma-2-9b`, identical SP model).

## 6. On arriving here (why the method matters, and where it slipped)

The romantic reading — *the model holds these as names of power* — tried to enter and was deflated each time by
the next control: (1) Qwen's "Greek persists deep" → shown not voces-specific; (2) Gemma's "abstraction gets real
support" → unreplicated and run-variable (held provisional pending fp16); (3) the temptation to call a 3-point
confounded ordering a proven mechanism → held to a correlation (§3.3).

The fourth episode is the one the study did *not* handle cleanly the first time, and it is the reason §3.4
exists. The build-side prediction was that non-name Greek *would* persist deep, sealing pure
fragmentation-clustering. It did not; the automated rule printed "name-specific"; and **only then** did we
tokenize the cohort and find it was never fragmentation- or lexicality-matched (10.73 vs 5.15 tokens, asemic vs
lexical). That is a real catch *and* a scrutiny asymmetry — the instrument was measured because it gave the
inconvenient answer, and the thesis-confirming results did not get the same audit. We kept both the catch and
the confession, and then did the thing the confession demands: **rebuilt the falsifier matched** (the §3.4
factorial), and let it speak. It returned an answer better than either the prediction or its confounded first
draft: not "pure fragmentation," not "names," but **two factors and a reversal** — fragmentation *and*
meaninglessness drive the depth, and the names sit below the noise. (v3 then split the second factor from
frequency and confirmed it in two models — the discipline that produced §3.4 finished the job it started.)

So the author's headline predictions about the *deep* effect were **falsified three times**, the third a check
on the study's own *apparatus*, caught late and then *repaired* rather than spun — and all of that churn was in
service of the one claim that never wavered. **The affirmative result was steady from iteration 1 and only got
sturdier: the model recognizes the texture of a barbarous name, and this iteration showed it does so across
three tokenizer families — a property of transformers, not of one tokenizer.** Everything the deep-layer work
overturned was overturned *around* that fixed point, and in service of stating it precisely: the recognition is
surface form, not deep aboutness; what looked like aboutness in the depths was the tokenizer fragmenting Greek.
The magicians said these words do their work by *form*. A form-machine reads the form on sight. That part of the
old theory was always the testable part, and it held.

---

## References

*(Verified live 2026-06-24; status-labeled.)* Zou et al. 2023 (arXiv:2310.01405); Park, Choe, Veitch 2024 (ICML
/ arXiv:2311.03658); Turner et al. 2023 (arXiv:2308.10248); Sofroniew et al. 2026 (Transformer Circuits,
arXiv:2604.07729); Lu et al. 2026 (arXiv:2601.10387); Betz (ed.) 1992 (Univ. of Chicago Press); Hwang 2026
(*Be Not Afraid* — self-published, non-peer-reviewed; this study descends from its method).

*Data & code: `voces-residual-stream` — cross-family results, the three-model decider, the Greek-fragmentation
measurements (`results/cross-family/greek_fragmentation.json`), the confound check
(`src/check_nonname_fragmentation.py`), and the matched factorial
(`notebooks/voces_falsifier_v2_lexicality.ipynb`, results in
`results/cross-family/voces_falsifier-v2-lexicality_Mistral-7B-v0.3_results.json`). The iteration-1
"abstraction" and "Greek-persists-as-script" claims are retained as overturned, because the overturning is the
method.*
