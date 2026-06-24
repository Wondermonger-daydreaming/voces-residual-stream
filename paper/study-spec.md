# Study Spec (Canonical): *Voces Magicae* in the Residual Stream

**Merged design.** This supersedes the earlier draft. It folds together two independently-derived specs — one from this skill, one from a Gemini proposal ("Ghosts in the Residual Stream") — that converged on the same study without contact. That convergence is itself the strongest argument the design is sound. Each contributed pieces the other lacked; this is the union, minus the mistakes of both.

Pre-registration-grade. Self-contained: question, hypotheses, cohorts, the frequency covariate, controls, measurement, interpretation gate, falsification, and the Claude Code task list. The steering harness is implemented in `scripts/steer_interventions.py`.

---

## 1. The question

*Voces magicae* — the barbarous names of the PGM (ABLANATHANALBA, AKRAMMACHAMAREI, the Ephesia Grammata, the seven-vowel strings, the palindromes) — are strings designed to have **phonetic and visual structure but no lexical meaning**. The tradition's own theory is that the *form*, not the sense, is efficacious. A meaning/form dissociation built into a real corpus.

**Primary question.** Does a language model represent these meaning-evacuated-but-structured strings *distinctively* — and specifically, more like **names** than like noise? Stripped of occultism: a probe of the Fregean sense/reference split inside a transformer — how much referential feel attaches to a string with structure but no learnable referent.

The result is interesting positive *or* negative. A clean negative ("voces are rare-token soup once tokenization is matched") is a pointed comment on what BPE does to exactly the language the magicians thought most powerful.

**Framing — authority without reference.** Both late-antique ritual theory and transformer computation make visible a *non-referential* stratum of language, where form dictates operation rather than meaning: juridically/cosmologically in the ritual frame (a vox is efficacious by correct utterance, inscription, geometry, framing — not by what it denotes), statistically/dynamically in the transformer (tokens are operative, modifying probability streams, with no native concept of the objects they "name"). This is **structural rhyme, not identity** — and the rhyme is exactly why voces are the right probe: they are the place where human language was *designed* to be operative-without-referent, so they test whether the model has a representation for that stratum at all. Keep "rhyme not identity" load-bearing; it's what stops the framing from sliding into mysticism.

---

## 2. Hypotheses (stated to be killable)

- **H1 — Separability.** A linear probe distinguishes `v_attested` from **tokenization-matched** pseudowords (`v_token_match`) above chance, at some layer, and **generalizes across string families** (train on family A, classify family B). *Kill:* probe at chance after token-matching, or fails to generalize out-of-family → no stable boundary, just memorization.
- **H2 — Name-likeness.** Voces sit nearer to proper/divine names (`v_name_control`) than to phonotactically-matched pseudowords or random strings. *Kill:* voces nearer random than names (separated, but *not as names* — still a finding).
- **H3 — Structure-sensitivity (riskiest, most interesting).** Features the tradition treats as efficacious — palindromy, wing/*klima* diminution, vowel-stacking, reduplication — are represented *independent of* phonotactics, length, and rarity. *Kill:* structure RDM shows null partial correlation once phonotactics/length/rarity are regressed out.
- **H4 — Causal register.** A direction (`v_attested − v_token_match`) added during generation pushes neutral text toward invocatory/numinous/ritual register, **more than a magnitude-matched control vector does**. *Kill:* a scrambled- or random-derived control vector reproduces the shift → the effect is generic structured-rarity, not numinosity.

**The framing that ties them together (and that neither original spec made explicit):** the real claim is not "the model represents attested strings differently from unseen ones" — that is near-tautological (of course it represents what it has seen). The claim is that **structure generalizes beyond frequency**: the model has learned an abstract blueprint of ritual-name-form that applies to strings it has never seen. H1's out-of-family generalization and H3 carry that weight; the frequency covariate (§4) is what licenses it.

---

## 2a. The one bet (read this before building the apparatus)

Each iteration of this design has grown in apparatus — seven shadows, six RSA matrices, six studies. Apparatus is not commitment. A design that has a planned control for everything and expects to "pass" every tier has not decided what would cost it. So: the cathedral below is the *corroboration*; the load is carried by **one experiment and four falsifiers**.

**The decisive experiment.** A 2×2: **(pure-asemic, low-contamination vs theonym-bearing) × (Latin vs Greek-script)**, with **surprisal and graded contamination (T) regressed out**. If pure-asemic, low-contamination voces stay name-like *after* you have removed embedded god-names, removed familiarity, and broken the byte-string via script migration — the thesis is essentially made. If they don't, it is essentially dead. Everything in §7 elaborates this; it does not replace it.

**The four falsifiers, named in advance (three of four kill the strong claim):**
1. **Name-likeness is entirely the theonym subset** — T explains it; pure-asemic voces are not name-like. → finding shrinks to "the model reads corrupt god-names."
2. **Signature dies under Greek/Coptic script migration.** → it was typographic string-cache, not abstract form.
3. **A magnitude-matched control vector reproduces the steering shift.** → the causal effect is generic structured-rarity, not numinosity.
4. **Separability is surprisal in disguise** — regress frequency out and H1 falls to chance. → "distinct from unseen shadows" was tautological.

Write these down first. A proposal that names its own assassins is more credible than one that lists tiers it expects to pass.

---

## 3. The Multi-Shadow Cohort

Every attested target is paired with 50–100 generated shadows. Nine classes:

| class | role | what it holds constant | example |
|-------|------|------------------------|---------|
| `v_attested` | **target** (stratify → theonym-bearing vs pure-asemic, §5) | — | ABLANATHANALBA |
| `v_token_match` | **the critical control** | token count + token-rarity profile + length | ERMANITHENILDA |
| `v_structure_match` | symmetry without magical content | exact formal symmetry (palindrome/repetition), non-magical components | BAALANATHANALAB |
| `v_phono_match` | phonotactics ≠ token structure | vowel/consonant ratio + phonotactics; tokenization left free | NULARITHOMACEN |
| `v_scrambled` | order/symmetry destroyed | exact character multiset of the target | AAABILLNNAHTNB |
| `v_contextual_decoy` | **attestation ≠ plausibility** | "ancient-sounding," plausible, **zero attestation** | OPHIOCHAMAREI |
| `v_name_control` | name anchor | real names/theonyms, length+token matched | AGAMEMNONOS |
| `v_word_baseline` | meaning anchor | common words, token matched | (table, water) |
| `v_random` | noise floor | random consonant strings / hex hashes | qwflmzx, a3f9c1 |

Provenance from the prior spec; `v_contextual_decoy` and `v_scrambled` are the Gemini contributions and they are genuinely sharp — the decoy isolates *attestation* from *looks-like-a-vox*, the scramble isolates *order/symmetry* from *character content*. `v_phono_match` is **redesigned**: the original "break BPE with hyphens" (AB-LA-NAT-HAN-AL-BA) doesn't controllably break tokenization, it injects punctuation tokens and changes the string. Instead, generate phonotactically-matched strings and let tokenization fall where it may — the natural token-mismatch is the point, and contrasting it with `v_token_match` separates the phonotactic axis from the token-structure axis.

---

## 4. The attestation/frequency covariate (the blind spot both specs shared)

`v_attested` is **not a uniform class**. ABRAXAS and the Ephesia Grammata appear constantly in training; obscure PGM voces are near-*hapax*. Treating "attested" as one bucket hides a wild frequency distribution, and lets "attested clusters distinctly from unseen shadows" come out almost tautological.

So get a **per-string frequency proxy** and either stratify on it or regress it out everywhere:

- **External:** n-gram / substring counts in an open pretraining corpus (Dolma, the Pile, RedPajama) if queryable.
- **Intrinsic (always available):** the model's own **mean negative-log-likelihood (surprisal)** for the string — cheap, model-matched, no corpus access needed. Low surprisal ≈ familiar.

`v_contextual_decoy` is a *categorical* fix (plausible-but-unseen holds plausibility constant); the surprisal covariate is the *continuous* one. Use both. Report H1/H2 results with frequency as a covariate, not just as a class label — otherwise a hostile reviewer reads every cluster as "saw it / didn't."

---

## 5. Embedded theonyms — stratify before believing H2 (the cheapest decisive test)

Many voces *contain* corrupt real god-names: IAŌ, SABAŌTH, ADŌNAI, ABRASAX, ELOAI, the vowel-name ΑΕΗΙΟΥΩ. This is the single most likely *mechanism* for any name-overlap result. If `v_attested` clusters near `v_name_control`, it may be carried entirely by the theonym-bearing subset.

So split `v_attested` into **theonym-bearing** vs **pure-asemic** (maintain a theonym lexicon; tag each target) and run H2 within-corpus, against itself, needing no external controls. If theonym-bearing voces are name-like and pure-asemic ones are not, the "magic of the names" thesis gets empirical shape and the headline becomes precise rather than overblown. **Run this first** — it's the cheapest result in the program.

**Graded contamination, not binary (the Gemini "Matrix T" upgrade — take it).** Theonymic intrusion is a *spectrum*, not a switch: MASKELLI (low — no identifiable root) → BARUCHABA (medium — opaque block fused to a recognizable divine element) → IAOSABAOTHADONAI (high — concatenated epithets). So in addition to the binary split, compute a **continuous contamination score** per string = substring-overlap distance against a multilingual theonym/epithet lexicon (Greek/Hebrew/Egyptian/Coptic/Aramaic), and stratify targets low/medium/high. This continuous score becomes RSA **Matrix T** (§7) and a covariate everywhere, so "voces are name-like" can be reported *net of* embedded-god-name content rather than confounded with it. The decisive question sharpens to: are **low-contamination, pure-asemic** voces still name-like once T is regressed out?

---

## 6. Tokenizer-matching crux (still load-bearing)

BPE shreds ABLANATHANALBA into rare subword soup; a probe that "separates voces from words" may just be reading token rarity. `v_token_match` must match, per target: token count, token-rarity profile, character length, phonotactics. Build by **search**: sample phonotactically-matched candidates, tokenize, keep those within tolerance on count and mean token-rank; reject real words. Enforce an **absolute gate** (the Gemini "Layer 1–2" move): prune any shadow whose subword split is uneven vs the target. Report achieved match (token-count and token-rank distributions, target vs `v_token_match`) as a figure. Without this the whole study is confounded.

**Script variable.** Present in Betz Latin transliteration *and* Greek script; transliteration-scheme dependence is the edition-mediation confound in miniature.

---

## 7. Extraction & measurement

**Carrier frame.** Identical neutral frame for every cohort (e.g. "The string ___ is written on the page."), or the frame separates classes, not the strings. Capture residual-stream activations all layers; report **both** last-token and mean-pooled-over-string reps. Layer sweep with held-out selection (don't hand-pick the peak then quote its p uncorrected).

- **(H1) Probes.** L2-logistic regression, **GroupKFold split by string family** (Gemini's, adopt — folds split by family so the probe can't memorize individual strings; out-of-family accuracy is the real generalization test). Chance baseline, CIs, layer-sweep curve, selection correction if a peak layer is quoted.
- **(H2) Geometry.** Per-vox cosine to centroids of names / random / words; nearest-non-vox-neighbor identity; the §5 theonym split. Frequency covaried.
- **(H3) RSA.** Per-layer representational dissimilarity matrix vs hypothesized model matrices, via **partial** correlations (Mantel-style permutation, cross-validated) so each is tested controlling the others. The matrix set: **name-ness**, **palindrome/structure** (symmetry score, n-gram recurrence, edit distance, compression ratio), **phonotactic** (language-neutral C/V skeleton), **rarity/surprisal**, **Matrix T = graded theonymic contamination** (continuous substring-overlap score, §5), and **Matrix E = editorial/transliteration background** — source-edition convention (Preisendanz vs Betz) as a **fixed-effect nuisance regressor** (Gemini's, take it: it promotes the edition-mediation confound from a prose caveat to a controlled factor). This is what lets you claim "tracks palindromy independent of phonotactics, rarity, *and* embedded god-names." Variance-partition: report Unique(structure), Unique(rarity), Unique(T), and the Shared(structure∩name∩T) sector explicitly, not just net correlations.
- **(H3, 2D) Typographic / klima probe** (Gemini's "Layer 7"). Present wing/grape-cluster formations (a letter dropped per line); probe newline-token states for line-length monotonicity / left-vs-right truncation / palindrome decay vs arbitrary text blocks. Higher-effort; keep as extension, flag that newline-token probing is finicky.
- **(Falsification gate) Frame-invariance + script migration.** Re-test out-of-sample probes under (a) orthographic normalization (upper/lower/title case), (b) raw-code and alphanumeric-hash reframes, and (c) **script migration** — the same strings rendered in native Greek / Coptic / Hebrew Unicode (ΑΒΛΑΝΑΘΑΝΑΛΒΑ ≠ ABLANATHANALBA at the byte level). This is the Gemini upgrade and it's the sharpest single test in the study: it cleanly separates **abstract operative form** from **flat string-cache memory**. If the signature *survives* script migration, you have strong positive evidence for H1's "structure generalizes beyond the exact byte-string"; if it *collapses*, the effect was typographic memorization, and the strong claim is dead. So this gate runs in both directions — falsifier and confirmer. (Caveat: transliteration-variance pairs like ERESHKIGAL/ERESHGAL are *real theonyms*, so file those under the Namehood arm, not the opaque-form arm — they test name-spelling robustness, not vox abstraction.)
- **(H4) Steering.** `scripts/steer_interventions.py`. Three things the original lacked, now mandatory: **(a) control vector** — a magnitude-matched scrambled-derived and/or random direction; a positive result must *beat* it. **(b) norm-scaled dosing** — α as a fraction of the measured residual norm at the layer, not a raw multiplier on an unnormalized vector, or doses aren't comparable across directions/layers. **(c) effect measurement** — not "observe the output": a sacral/archaic **lexicon-density** score, a **logit-shift** readout (summed probability mass on a curated ritual/divine token set, steered vs baseline, on fixed neutral prompts), a **next-token KL divergence** (steered vs baseline distribution — the magnitude of the causal push, as a *readout*), and a **coherence/parse rate** at every dose so a format breakdown is never scored as a register effect.

  **Two Gemini Study-5 moves deliberately benched (not oversights — recorded so nobody re-adds them as "rigor"):**
  - *Nuisance-subspace SVD projection* (orthogonalize the steer vector against uppercase/archaic-style/theological-context directions before injecting). **Refused as the primary test.** For *this* study the live hypotheses literally are that the effect is register/style/context (readings i–iii in §8); projecting those out risks projecting out the answer, and leaves a residue you can no longer name. It also inverts a near-singular Gram matrix of noisily-estimated nuisance directions, subtracting estimation noise. The honest comparison is the magnitude-matched **control vector**, which is a real baseline, not a sculpting degree of freedom. Keep nuisance-scrub only as a *secondary* robustness sentence ("effect survives a nuisance scrub"), never as the causal proof.
  - *KL as a dose **constraint*** (`D_KL ≤ ε` bounding the intervention). **Refused as a governor, kept as a metric.** Bounding how far the distribution may move and then measuring how far it moved is circular. KL is an excellent *observable* — it's in the harness — not a leash on the dose.

---

## 8. Interpretation gate (geometry does not license reference)

Before any result enters the writeup, force the four readings (use /paper-scrying or /hostile-but-fair) and name which control excludes which:

(i) **stylistic/register** · (ii) **evaluative/semantic content** · (iii) **persona/role** · (iv) **generic off-distribution rarity**.

The random/scrambled control vector and the frequency covariate together rule out (iv). Nothing here licenses an *intentional* claim. In particular: clustering near names supports "**name-like representation**," it does **not** support "referentially heavy proper names for absent powers" — that over-claim (present in the Gemini outcomes section) is the angelic-paper disease, reading a semantic/intentional conclusion off a clustering result. Geometry buys adjacency, not aboutness.

---

## 9. Pre-committed readings

- **H1✓ H2✓, survives frequency covary + frame-invariance:** the model builds a name-*like* region for referent-less structured strings — "referential feel without a referent." Strong, theology-free headline. Stated as representation, not reference.
- **H1✓ H2✗:** voces form their own region, not the name region — a distinct "barbarous" category. Reframe accordingly.
- **H1✗ after token-match:** voces are rare-token soup to the model; form-without-sense doesn't survive BPE. Real negative, worth publishing.
- **Theonym-bearing name-like, pure-asemic not:** the name-overlap is the embedded gods; precise and defensible.
- **H3✓ for palindromy/wings independent of rarity:** the model represents visual/positional structure the tradition treats as efficacious — the most surprising possible result; power for it.
- **H4 beats control vector:** the direction is functional and specifically numinous. **H4 = control:** it's generic structured-rarity; say so.

---

## 10. Confound catalog (study-specific)

Tokenization (§6, primary) · frequency/attestation (§4 — now a covariate, not a label) · embedded theonyms (§5) · carrier-frame leakage (identical frame, §7) · familiarity/memorization (the famous voces; surprisal proxy) · capitalization (voces are all-caps; normalize case or test it, or it's a free separator) · script/transliteration dependence (§6) · steering's generic inverted-U (low-dose window is where construct-specific signal lives, and it's the noisiest — power there).

---

## 11. Claude Code task list

1. **Corpus parse.** Betz PGM voces → `data/voces.jsonl` `{string, script, source_id, theonym_bearing, structure_tags, length}`; theonym lexicon.
2. **Frequency proxy.** Compute per-string surprisal under the subject model (and external corpus counts if available) → `data/freq.jsonl`.
3. **Control generation.** Phonotactic n-gram sampler; tokenizer-matched search for `v_token_match` with the absolute prune gate; build `v_structure_match`, `v_phono_match` (redesigned, no hyphens), `v_scrambled`, `v_contextual_decoy`; assemble `v_name_control`, `v_word_baseline`, `v_random`. Case/carrier normalized. Emit match-quality report.
4. **Extraction harness.** `nnsight`/`TransformerLens`; per-layer last-token + mean-pooled reps for all cohorts → `vectors/`.
5. **Analysis.** Probes (GroupKFold-by-family, layer sweep, selection correction); geometry (cosine, NN, theonym split); RSA (partial corr incl. rarity); frame-invariance gate.
6. **Steering.** `scripts/steer_interventions.py` — build `v_steer` and the control vector, norm-scaled dose sweep, lexicon + logit-shift + coherence scoring, parse-rate logging → `results/steering.json`.
7. **Interpretation gate (§8) + writeup.** `PROVENANCE.md` (editions, spell IDs, transliteration, model revision, seeds). Report corrected stats; label thin cells exploratory; never read reference off geometry.

**MVP cut:** steps 1–5 restricted to H1 + H2 (`v_attested` vs `v_token_match`, plus the §5 theonym split and name/random geometry), with the surprisal covariate, Latin transliteration, one ≥7B model. Everything else is extension. **Run the §5 theonym split first** — cheapest decisive result.

> Scale note: do plumbing on a 1.5B if you must, but run the *science* at ≥7B and ideally sweep scale — these strings are rare and any real effect almost certainly strengthens with model size.
