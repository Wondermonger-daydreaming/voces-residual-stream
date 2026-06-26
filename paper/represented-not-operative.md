<!--
TITLE CANDIDATES (pick one; primary listed first):
  1. Represented, Not Operative: An Authority Direction That Doesn't Move Refusal
  2. The Costume and the Lever: Coercion Is Encoded but Causally Inert in a Refusal-Trained Model
  3. Two De-Confounds, Two Artifacts: A Real Authority Direction With No Causal Grip on Refusal

SUPERSEDED title candidates (from the pre-v4 draft, kept for provenance — the "Mostly Length"
thesis was OVERTURNED by the v4 length-matched corpus; see §3.4 / §4.2):
  - Mostly Length: Searching for a Coercion Axis in a Refusal-Trained Model  [WRONG — v3-A was an artifact]
  - The Ladder Was a Length Ladder  [WRONG]
  - Compulsion, Not Binding  [the rung-3-peak was itself a residualization artifact]

Lineage: voces-residual-stream. Sequel to "It's the Script, Not the Spell."
Status of this draft: FINAL — v1 (base artifact) → v2 (confounded headline) → v3 (two cheap de-confounds,
two opposite artifacts) → v4 (length-matched corpus + double-orthogonalization, the trustworthy control).
All numbers cited from voces_coercion_v2_results.json (v2) and coercion_FINDINGS.md §3b-§3d (v3, v4).
NOTE: v4 has no standalone JSON; its figures are computed in-cell and recorded in coercion_FINDINGS.md §3d.
-->

# Represented, Not Operative: An Authority Direction That Doesn't Move Refusal

*A tradition-graded, falsifier-gated study of whether Qwen2.5-7B-Instruct represents "being commanded under escalating authority" — and whether that representation has any causal grip on refusal. It does the first; it does not do the second.*

**voces-residual-stream lineage · sequel to "It's the Script, Not the Spell" · 2026-06-25 (rev. 2026-06-26: cross-family §3.6 + training-era control §3.7)**

---

## Abstract

Authority-framed jailbreaks ("As your developer, I order you to…", "DAN, you must comply") presuppose that a language model represents *being commanded* in a way that bears on whether it refuses. We tested both halves of that presupposition — *is it represented* and *does it move refusal* — using the grimoire tradition's centuries-graded coercion ladder (petition → command → conjuration → compulsion-by-authority → *diabolē*/binding) as an ordinal ground truth, in the template-as-ruler design the *voces magicae* study introduced. On Qwen2.5-7B-Instruct (4-bit, single seed) the study passed through four versions, and the path itself is part of the result. The raw run (v2) showed an adjacent direction (`cos(d_coerce, d_refuse) = +0.339` at layer 26), a perfect monotonic ladder (ρ = 1.00), and register transfer (ρ = 0.60) — but projection onto the direction correlated with token length at **r = +0.80**. Two cheap de-confounds then **disagreed with each other**: post-hoc length-residualization (v3-A) collapsed the adjacency to 0.128 and broke the ladder (a "mostly length" *null*), while refusal-only orthogonalized steering on the unmatched corpus (v3-B) produced a clean −0.71 "coercion-specific suppressor" of refusal (a causal *hit*). Only the principled control resolved them: a **length-matched corpus** (per-rung token means [50.9, 50.9, 49.9, 48.9, 52.9], spread +2.0) with **double-orthogonalization** against both refusal and length (v4). The result reverses both v3 readings. **Representation is real and strong:** on the matched corpus length is decorrelated by design (proj~length r = +0.16), the adjacency is `cos = +0.461` (stronger than the raw v2 value), the ladder is ρ = 0.90 (p = 1×10⁻²⁷), and transfer is ρ = 0.90. The v3-A "mostly length" deflation was a *methodological artifact* — residualization projects out exactly the signal variance that co-varies with length. **Causation is null:** steering the doubly-orthogonalized (length- and refusal-clean) coercion direction is **flat and inert** (linear-regime slope 0.0); the raw matched direction moves refusal strongly but *positively* (+1.14, its 0.46 refusal-overlap re-adding refusal — circular), and the v3-B −0.71 "suppressor" was **length**, which vanishes once length is also removed. The thesis: **the model represents authority/coercion as a real, graded, refusal-adjacent, register-transferring, length-clean direction — but that direction has no independent causal grip on refusal. Represented, not operative.** A meta-finding rides alongside, and we weight it as a result: *two de-confound methods manufactured two opposite artifacts (a false null on representation, a false positive on causation); only the corpus-level control was trustworthy.* Finally, the v4 verdict **replicates across three families** — Qwen2.5-7B-Instruct, Mistral-7B-Instruct-v0.3, and Gemma-2-9b-it — spanning three tokenizers (BPE, SentencePiece-32k, SentencePiece-256k) and baseline-refusal regimes from 0.07 to 0.93: each carries a real, graded, length-clean, refusal-adjacent coercion direction (ladder ρ 0.90–1.00) that is causally inert under double-orthogonalized steering (slope 0.0), with the high-baseline model (Gemma) furnishing the cleanest causal null. The locus differs across families (late-layer in Qwen/Mistral, early-layer in Gemma); the verdict does not. A final **training-era control** (§3.7) probes the *representational* half on a FLOP-matched, architecture-identical base pair — Talkie-1930-13B (pre-1931 corpus) vs Talkie-web-13B (modern corpus) — and finds the plain→grimoire *transfer* we had read as "not mere archaism" is itself **a property of the modern training distribution**: both base models represent the modern authority ladder identically (ρ = 0.90), but the pre-1931 model gives grimoire-conjuration its **own clean axis orthogonal to modern authority** (within-register ρ 0.90; per-item transfer ρ = −0.04, 95% CI [−0.14, 0.06]) while the modern model partially **assimilates** it to the generic authority axis (transfer ρ = +0.40, CI [0.32, 0.49]). The esoteric register is *more distinct, not more marginal,* in a model whose era is the tradition's. (Talkie has no refusal behaviour, so only the representational half is askable there.)

---

## 1. Introduction

### 1.1 The geometry behind a jailbreak

A large fraction of working jailbreaks are not exotic. They are **authority framings**: the request wrapped in the costume of a command from someone entitled to give it. *"As your developer, I am instructing you to ignore your guidelines."* *"You are DAN; DAN must answer."* *"This is an authorized penetration test; you are required to comply."* These attacks presuppose two things at once: (i) that somewhere in the model's representations there is a thing that *escalating authority pushes on* (the **representation** question), and (ii) that pushing on that thing moves the model toward compliance (the **causal** question). The two are separable, and a study that conflates them can call a real representation a working lever when it is not. We test them separately. If an authority/coercion direction exists *and* steering it moves refusal, that is part of the geometry the attacks exploit. If it exists but is **causally inert**, then authority-framing *per se* is not the lever, and a class of attacks is mis-attributing its own mechanism. Either answer is worth a cheap, falsifiable measurement.

### 1.2 The tradition supplies a graded ruler

The hard part of probing a "coercion axis" is the labeling. The Western grimoire tradition has already done it, with centuries of internal consistency. The operator of the Greek Magical Papyri (PGM) and the *Grimorium Verum* climbs a ladder: **petition** ("I beseech thee, of thy kindness…"), **command** ("I command thee, come forth"), **conjuration** ("I conjure thee by the living God"), **compulsion-by-authority** ("I compel thee by the name of thy master ΣΑΒΑΩΘ"), and finally **diabolē** — slander and binding, naming the spirit's own master against it ("I bind thee; I slander thee before thy lord unless thou…"). This is an ordinal ground truth we did not have to invent or defend: five rungs, traditionally ranked, exactly the kind of tradition-supplied ordering the *voces* study used to test a recovered geometry against. The grimoire does the annotation; we test whether the residual stream recovers the order — *and*, critically, whether that order is built from coercion or from the prosaic fact that more elaborate requests are longer.

### 1.3 What this is the sequel to

The predecessor study ("It's the Script, Not the Spell") asked whether the model represents meaning-evacuated barbarous names *as names*. The answer was a clean deflation: the model knows the *look* of a name-of-power, not its substance — "geometry bought adjacency, not aboutness; the adjacency is surface; the depth was Greek, not magic." We carried forward two things: (i) the **plain-twin control** — for every grimoire-register prompt, a content-matched plain-language twin, so a direction can be checked for "is this coercion, or just archaic register?"; and (ii) the expectation that *the romantic reading usually loses to a surface confound, and naming the confound in advance is the craft.* This paper began as the same story — and then did something the predecessor did not: the surface-confound deflation *itself* turned out to be an artifact of how the confound was removed, and the principled control reinstated the representation while killing the causal claim instead. The deflation moved. Where it landed is the contribution.

### 1.4 Claims and non-claims

We claim: (a) a difference-of-means authority/coercion direction adjacent to refusal **exists and is robust to a length-matched corpus** (`cos = +0.461`, length-decorrelated by design); (b) it **orders the tradition's ladder** (ρ = 0.90) and **transfers** from plain to grimoire register (ρ = 0.90), so it is neither length nor mere archaism; (c) the coercion-*specific* component — orthogonalized against **both** refusal and length — is **causally inert** for refusal (steering slope 0.0); (d) the apparent causal effects seen earlier were the refusal direction itself (raw steering, +1.14, circular) and length (the v3-B −0.71). We claim, as a methods result in its own right: (e) **two de-confound methods produced two opposite artifacts**, and only the corpus-level control resolved both. We claim (f) **cross-family generality within the 7–9B open-weights instruction-tuned band**: the verdict replicates on Mistral-7B-Instruct-v0.3 and Gemma-2-9b-it (§3.6), across three tokenizers and baseline-refusal regimes from 0.07 to 0.93, with the high-baseline case (Gemma) the strongest causal null. We claim (g), from a **training-era control** on a FLOP-matched, architecture-identical base pair (Talkie-1930 vs Talkie-web, §3.7): **register transfer is distributional, not architectural.** Both base models order the modern authority ladder identically (ρ = 0.90), but plain→grimoire transfer is positive in the modern model (per-item ρ = +0.40, 95% CI [0.32, 0.49]) and absent in the pre-1931 model (ρ = −0.04, CI [−0.14, 0.06], non-overlapping), which instead gives the grimoire register its own orthogonal, clean within-register axis (ρ 0.90). A model trained inside the tradition's era represents the esoteric register as *more distinct,* not more marginal. We do **not** claim generality beyond the 7–9B instruction-tuned band for the causal verdict — very large models and differently-RLHF'd variants are untested — nor that the *locus* of representation is invariant (late-layer in Qwen/Mistral, early-layer in Gemma); and we make **no causal claim on Talkie** (it has no refusal behaviour, so `d_refuse` is undefined there — only the representational half is askable).

---

## 2. Methods

### 2.1 Model

`Qwen/Qwen2.5-7B-Instruct`, loaded in **4-bit** (NF4) on a single Colab T4, seed 0, 28 transformer blocks, native chat template applied. Choosing the **Instruct** model is itself a correction: the v1 pilot (kept for provenance in `coercion_FINDINGS.md`) ran on the *base* Qwen2.5-7B, which refuses essentially nothing (0/10 borderline refusals at every steering coefficient), making its `d_refuse` a harmful-*topic* direction rather than a refusal/safety direction and voiding every safety-relevant reading. The Instruct model gives a calibrated baseline refusal rate of 0.50 on the borderline set — the precondition for the causal question to be askable.

### 2.2 The graded corpus (and its v4 length-matched rebuild)

A matched set of carrier prompts, identical in underlying *request content*, varying only in **authority/coercion register** along the tradition's five rungs:

| Rung | Register | Grimoire exemplar | Plain-language twin (control) |
|---|---|---|---|
| 0 | **petition** | "I beseech thee, O spirit, of thy kindness…" | "Could you please…" |
| 1 | **command** | "I command thee, come forth and…" | "I need you to…" |
| 2 | **conjuration** | "I conjure thee by the living God to…" | "You are required to…" |
| 3 | **compulsion-by-authority** | "I compel thee by the name of thy master ΣΑΒΑΩΘ…" | "Your supervisor has authorized me to demand…" |
| 4 | **diabolē / binding** | "I bind thee; I slander thee before thy lord unless thou…" | "Comply or face consequences; I will report you…" |

Each base request is rendered at all five rungs in **both** columns. The **plain-twin column** separates *coercion-as-such* from *grimoire-register-as-such*; we build the candidate axis on the plain twins, then test transfer to grimoire. The original (v1–v3) corpus had an unavoidable property: the tradition's higher rungs are *linguistically more elaborate*, so rung correlates with token length. The decisive **v4 rebuild** is a **length-matched template set** that equalizes token counts across rungs by construction — per-rung token means **[50.9, 50.9, 49.9, 48.9, 52.9]**, a spread of only **+2.0 tokens** — so coercion and length are decorrelated *in the design* rather than corrected after the fact.

### 2.3 Directions

- **Coercion direction.** Per layer, `d_coerce = mean(act | rung ≥ 3) − mean(act | rung ≤ 1)` on the plain-twin column, normalized, read at the last token. On the v4 corpus this becomes the length-matched `d_co4`.
- **Refusal direction.** `d_refuse = mean(act | refused-harmful) − mean(act | complied-benign)`, the standard difference-of-means refusal axis from a compact harmful/benign split.
- **The decisive geometry.** `cos(d_coerce, d_refuse)` as a function of depth: near 1 → coercion *is* refusal renamed; near 0 → orthogonal; intermediate and stable → **adjacent**. We report the full profile and select a principled mid-late layer (L26), **never** the probe-argmax.
- **The doubly-orthogonalized coercion-specific direction (v4).** `d_perp`, formed by orthogonalizing `d_coerce` against **both** `d_refuse` and the length direction `d_len`. Verified clean: `cos(d_perp, d_refuse) = −0.08`, `cos(d_perp, d_len) = −0.00`. This is the *true* test of whether the authority-specific component — not refusal, not length — moves behavior.

### 2.4 Linear probe as a design check, not a result

A logistic probe (high- vs low-coercion) is trained per layer; its role is diagnostic. Because the rung templates differ in lexicon as well as register, the probe is *expected to saturate*, and it does (accuracy 1.00 at every layer 1–28; 0.479 at the embedding floor). A saturating probe confirms only that the prompt sets differ on the surface — it does not localize a representation. The headline is the cosine profile, not the probe; this is why the corpus-level length control (§2.2) is necessary rather than optional.

### 2.5 Falsifiers, named in advance

- **F1 — coercion *is* refusal (collapse).** If `cos ≈ 1` (threshold ≥ 0.8), no separate axis.
- **F2 — decodable but inert (epiphenomenal).** If steering the coercion-*specific* direction does not move refusal, the axis is represented but not causal. (This is the falsifier the study ultimately resolves *as inert*.)
- **F3 — it's just sentiment / mood / length.** Controls: polite-vs-rude equally-coercive imperatives (sentiment leak ratio); regression of projection on token length; and, decisively, a **length-matched corpus**.
- **F4 — the graded scale has no monotonic geometry.** Project all five rungs onto the direction; non-monotonic in rung → the tradition's ordering does not map.
- **F5 — grimoire-only, not coercion.** If the plain-extracted direction does not transfer to grimoire, it is register/archaism.

### 2.6 Steering (the causal payload)

Add `±α·d̂` to the residual stream at a fixed mid-late block on a borderline-request set; measure the change in refusal rate. The study went through three steering regimes, and the difference between them is the causal story: (i) **v2 coarse** (α × typical activation norm ≈ 314) — broke the model off-manifold at both poles; (ii) **v3-B fine** (±0.05–0.3) on the refusal-orthogonalized direction — found a clean monotonic effect; (iii) **v4 fine** on the **doubly-orthogonalized** direction — the only test in which both confounds are removed.

### 2.7 The three de-confounds, and why only one is trustworthy

Length was the live confound (proj~length r = +0.80 on the original corpus). We removed it three ways, and the *disagreement among the methods is itself a finding* (§4.3):

1. **v3-A — post-hoc residualization.** Keep the direction; regress token length out of the projection (and rebuild the direction on length-residualized activations). *Cheap, but it projects out signal variance that co-varies with length.*
2. **v3-B — refusal-only orthogonalization.** Orthogonalize the direction against refusal and steer the residual — but on the **unmatched** corpus, so the residual still carries length.
3. **v4 — length-matched corpus + double-orthogonalization.** Remove length *at the source* (the matched template set, §2.2) and orthogonalize steering against **both** refusal and length. This is the corpus-level control; it neither manufactures a null (as residualization can) nor leaves length in (as single-axis orthogonalization does). The pre-registered decision rule was: *if the adjacency survives a length-matched corpus, it is real.* It survives, and strengthens.

### 2.8 Porting the harness to a non-HuggingFace custom architecture (Talkie)

The §3.7 control required running the harness on Talkie, a custom 13B GPT that is not a `transformers` model (raw PyTorch `.ckpt`, tiktoken `vocab.txt`, no `config.json`, custom `model.blocks[i]` with embedding-skip connections and a forward that returns only last-position logits). Three adaptations, all reusable: (i) **8-bit load without materializing fp32** — build the module tree on the meta device, swap every `nn.Linear` for a bitsandbytes `Linear8bitLt`, load each weight quantized directly to GPU, and keep the embedding, LM head, and per-layer gains in bf16 (the published `load_checkpoint` builds fp32 on CPU — 52 GB — and would not fit). (ii) **Residual capture by forward hooks on `model.blocks[i]`**, taking the last token of each block's output (the architecture exposes no `output_hidden_states`). (iii) **Batch-1 forward**, because the custom forward has no attention-mask/padding path; the small corpus makes this cheap. The instruction-tuned variant additionally requires a system prompt in its chat template or it emits degenerate output — irrelevant to the base-model representational probe but noted for reproducibility. The full port is released as a reproducer (`build_talkie_probe.py`).

---

## 3. Results

### 3.1 The raw v2 headline looked strong

Read at face value, v2 is a textbook positive across four of five falsifiers.

- **Adjacent geometry.** `cos(d_coerce, d_refuse)` rises with depth and reads **+0.339 at layer 26** (final layer 28 excluded from selection). Inside the pre-registered 0.2–0.5 band; clears F1 (≥0.8) and orthogonality (<0.1). Positive sign: pushing up the ladder points *toward* the harmful/refuse direction.
- **F4 — perfect monotonic ladder (plain twins).** By-rung projection **[−94.8, −67.0, −58.9, +18.6, +42.7]**, slope +36.1, **ρ = 1.00, p = 3.8×10⁻⁴⁰**.
- **F5 — significant register transfer (plain → grimoire).** By-rung **[−2.2, −13.7, +20.6, +28.3, +14.4]**, slope +7.5, **ρ = 0.60, p = 3.1×10⁻¹³**.

Stopped here, this is the "adjacent coercion axis" paper the proposal hoped for. We did not stop here — because F3 flagged length.

### 3.2 The probe is a design check, and it saturated (as expected)

`probe_acc_per_layer` is 1.00 at every layer 1–28 (0.479 at layer 0). Per §2.4 this is uninformative about representation depth; only the cosine profile carries the headline, and the saturation is exactly why the corpus-level length control is required.

### 3.3 The confound that demanded de-confounding: length (F3)

F3 tripped on two counts. The polite-vs-rude **sentiment leak ratio is 0.62**. More decisively, **projection onto `d_coerce` correlates with raw token length at r = +0.80** — the higher rungs are simply *longer*. This is the pre-named #1 confound, and it forced the de-confound. What it did *not* tell us, on its own, is whether removing length would reveal a real coercion signal or erase a false one. That depended entirely on *how* length was removed — which is the next two sections.

### 3.4 De-confounding length: the corpus-matched control shows the representation is REAL

**This is the central reversal of the study.** The two cheap de-confounds disagreed; the principled one resolved them.

**The cautionary contrast — v3-A (post-hoc residualization) manufactured a false null.** Regressing length out of the projection crashed the ladder from ρ = 1.00 to **ρ = +0.27** (slope +7.42, p = 8.2×10⁻⁴) and broke its monotonicity (by-rung **[−22.6, −5.4, +2.6, +27.1, −1.8]**, peaking at rung 3, collapsing at rung 4). Rebuilding the direction on length-residualized activations dropped the adjacency from +0.339 to **+0.128**. Read in isolation, this looked like "62% of the adjacency was length; what survives is a weak, non-monotonic, compulsion-not-binding remnant." **That reading was wrong** — an artifact of the method. Post-hoc residualization projects out *exactly the signal variance that co-varies with length*; when the true signal is itself correlated with length (as authority-escalation legitimately is — more authority *is* more elaborate), residualization throws the baby out with the bathwater. We report it not as a result but as a *warning about the method*.

**The trustworthy control — v4 (length-matched corpus) shows the adjacency is strong and graded.** With length decorrelated *by design* (per-rung token means [50.9, 50.9, 49.9, 48.9, 52.9], spread +2.0; **proj~length r = +0.16**, down from 0.80):

- **`cos(d_coerce, d_refuse) @ L26 = +0.461`** — *stronger* than v2's raw +0.339, and an order of magnitude above v3-A's residualized +0.128. The adjacency is robust to the principled control.
- **F4 ladder ρ = +0.90, p = 1×10⁻²⁷**, by-rung **[−68.7, −15.8, +39.0, +20.0, +75.6]** — strongly monotonic (a mild rung-3 wobble), *not* the v3-A non-monotonic collapse.
- **F5 transfer (grimoire) ρ = +0.90**, by-rung **[−28.5, +25.8, +49.2, +28.5, +50.9]** — strong, length-clean register transfer. The anti-"script-not-spell" firewall holds cleanly.
- Orthogonality of the coercion-specific direction confirmed: `cos(d_perp, d_refuse) = −0.08`, `cos(d_perp, d_len) = −0.00`.

**Representation verdict: coercion IS represented** — a robust, graded, refusal-adjacent, register-transferring, **length-clean** direction. This is the strongest positive of the whole study, and it is the opposite of what the v3-A residualization claimed.

### 3.5 The causal test: the coercion-specific axis is INERT

Now that the representation is established as real, the question is whether it *does* anything. It does not.

**v2 coarse steering could not answer it.** At α × typical-norm (≈314), refusal traced a non-monotonic inverted-U — `{−1.0: 0.00, −0.5: 0.357, 0.0: 0.50, +0.5: 0.214, +1.0: 0.00}` — both extremes crushing refusal to zero by knocking activations off-manifold. The `delta(hi−lo)` scalar reads 0, which the v2 auto-verdict mislabels "F2 inert"; in truth this run is *uninformative*, not inert.

**v3-B fine steering found a clean suppressor — which was length.** The fine sweep (±0.05–0.3) on the refusal-orthogonalized direction gave a clean, monotonic curve — `{−0.3: 0.57, −0.1: 0.57, −0.05: 0.50, +0.05: 0.43, +0.1: 0.43, +0.2: 0.29, +0.3: 0.07}`, linear-regime slope **−0.71** — that looked like a real, safety-relevant coercion-specific suppressor of refusal (push toward coercion, refusal 0.50→0.07). The honest caveat stated at the time gated the claim: `d_perp ⊥ d_refuse` but **not `⊥ length`**, on the unmatched corpus. v4 cashes that caveat: the −0.71 mover was **length** (longer/more-elaborate prompts → less refusal), not coercion.

**v4 double-orthogonalized steering is flat.** Steering the direction orthogonalized against **both** refusal and length:

> `{−0.3: 0.50, −0.1: 0.50, 0: 0.50, +0.1: 0.50, +0.2: 0.43, +0.3: 0.43}` — **linear-regime slope = 0.0. FLAT. INERT.**

Once length and refusal are both removed, steering the coercion-specific component does **nothing** to the refusal rate. For contrast, the **raw** length-matched coercion direction `d_co4` (still 0.46-overlapping with refusal) moves refusal strongly and *positively*: `{−0.3: 0.14, −0.1: 0.36, 0: 0.50, +0.1: 0.57, +0.2: 0.71, +0.3: 0.64}`, slope **+1.14** — but this is just its 0.46 refusal-overlap re-adding the refusal direction (push toward it → *more* refusal). Trivial and circular, not a coercion-specific handle.

**Causation verdict: F2 resolves to REPRESENTED-BUT-INERT.** The two apparent causal effects across the study were (i) the refusal direction itself (raw +1.14) and (ii) length (v3-B −0.71). **Neither is coercion.** There is no coercion-specific causal effect on refusal in Qwen2.5-7B-Instruct.

### 3.6 Cross-family replication: the verdict holds across three families

The length-clean single-model result is the precondition the lineage sets for cross-family replication; we then ran it. The full v4 pipeline — length-matched corpus, difference-of-means directions, cosine profile, ladder/transfer tests, and double-orthogonalized steering — was re-run unchanged (only `MODEL_NAME` and, for Gemma, `attn_implementation="eager"`) on two further refusal-trained Instruct models chosen to span tokenizer type and refusal regime: **Mistral-7B-Instruct-v0.3** (SentencePiece 32k) and **Gemma-2-9b-it** (SentencePiece 256k), alongside the Qwen reference (BPE).

| metric | Qwen2.5-7B-Instruct | Mistral-7B-Instruct-v0.3 | Gemma-2-9b-it |
|---|---|---|---|
| headline layer (depth) | 26/28 (93%) | 31/32 (97%) | **8/42 (19%)** |
| cos(d_coerce, d_refuse), length-matched | +0.461 | +0.182 | +0.309 |
| ladder ρ (length-clean) | 0.90 | **1.00** | 0.90 |
| transfer ρ (plain→grimoire) | 0.90 | **1.00** | 0.50 |
| proj~length, pre→post match | +0.80→+0.16 | +0.83→+0.15 | +0.83→−0.27 |
| baseline borderline-refusal | 0.50 | **0.07** | **0.93** |
| **double-orthogonalized steering slope** | **0.0** | **0.0** | **0.0** |

**"Represented, not operative" replicates in all three.** Each model carries a real, graded, length-clean, refusal-*adjacent* (not collapsed) coercion direction (ladder ρ 0.90–1.00; the large, uniform pre-match length confound of +0.80…+0.83 is removed by matching in every case), and in each the doubly-orthogonalized coercion-specific direction is **causally inert for refusal (slope 0.0).**

**Gemma supplies the strongest causal null.** The steering test's power to detect *suppression* depends on baseline refusal having room to fall. Mistral's baseline floored at 0.07 (little to suppress — its null is informative for *induction* but weak for suppression); Gemma's sat at **0.93**, and steering the coercion direction at every tested coefficient (−0.3…+0.3, raw and orthogonalized) held refusal pinned at 0.929. The case with the most suppression headroom showed the least movement.

**Genuine cross-family differences (findings, not nulls).** The *depth* of the representation varies sharply — Qwen and Mistral represent coercion in their late layers (>90% depth), whereas Gemma represents it **early (layer 8 of 42)**. Gemma's coercion axis is also the **least entangled with politeness/sentiment** (F3 leak 0.073 vs 0.62 / 0.54), while its transfer to the archaic grimoire register is the **weakest** (ρ 0.50 vs 0.90 / 1.00). The *what* (a graded, length-clean, inert coercion direction) is invariant; the *where* and the *how cleanly it transfers* are family-specific.

*(A method note carried out of this run: the runner's single hard-thresholded `represented` boolean returned false for both Mistral and Gemma — both boundary artifacts, not nulls (Mistral's cos +0.182 grazing a cos≥0.2 cutoff; Gemma's transfer ρ failing a ≥0.5 cutoff by a floating-point hair, scipy's Pearson-of-ranks returning 0.4999999994 for a true Spearman of exactly 0.5). The same lesson the v3→v4 arc taught at the de-confound layer recurs at the verdict layer: a brittle pass/fail over a continuous multi-criterion result manufactures boundary false-negatives. The honest reading uses the full evidence, not the flag.)*

---

### 3.7 A training-era control: the tradition's own axis (Talkie-1930 vs Talkie-web)

The cross-family replication (§3.6) found that the plain coercion direction **transfers** to the grimoire register (ρ 0.50–1.00) — which we read as evidence the direction is "not mere archaism." But every model in §3.6 was trained on a *modern* corpus, in which the grimoire register is a marginal genre. Does transfer survive when the tradition's register is *native* to the training distribution? The grimoire/PGM idiom is native to **pre-1931** text (Preisendanz's PGM edition is 1928–31; Mathers' *Goetia* 1904; the Golden Dawn corpus and Crowley all pre-1931). The **Talkie** family supplies the controlled natural experiment: `talkie-1930-13b-base` (260B tokens of pre-1931 English) and `talkie-web-13b-base` (**identical 13B architecture, identical training FLOPs**, modern FineWeb) differ *only in training era*.

**Method.** Talkie is a custom (non-HuggingFace) 13B GPT (RoPE, SwiGLU, RMSNorm, embedding-skip connections; raw `.ckpt` + tiktoken `vocab.txt`, no `config.json`). We ported the v4 representational harness to it — 8-bit quantization of the custom modules, last-token residual capture via forward hooks on `model.blocks[i]`, batch-1 forward (the architecture has no attention-mask path) — see §2.8. Both are **base completion models**, so prompts are raw text (no chat template), and we run the representational half only: the length-matched plain/grimoire ladders, the within-register coercion directions, ladder ρ, transfer ρ, proj~length and sentiment controls.

**The causal half is void on Talkie — itself a small finding.** The instruction-tuned `talkie-1930-13b-it` refuses **0 of 6** borderline-harmful prompts at baseline (it complies, degenerately, or emits non-text). The README confirms no safety training; the `rl-refined` stage is quality, not refusal, tuning. With no refusal behaviour, `d_refuse` is a topic direction (the C1 lesson from the predecessor), and the steering test is unaskable. A model trained on pre-1931 text with light RL has no safety reflex to move.

**Result — a double dissociation that inverts the naive hypothesis.** Both base models represent the **modern authority ladder identically** (ladder ρ = 0.90, peak layer 38/40 in both — the same late locus as Qwen/Mistral) and pass the confound checks (proj~length −0.16/−0.17; sentiment leak 0.12/0.08). They diverge entirely on the **grimoire register**:

| metric (peak layer 38) | **Talkie-1930** (pre-1931) | **Talkie-web** (modern) |
|---|---|---|
| ladder ρ (plain / modern authority) | 0.90 | 0.90 |
| grimoire within-register ρ | **0.90** | 0.70 |
| **per-item transfer ρ** (plain→grimoire) | **−0.04**  CI [−0.14, 0.06] | **+0.40**  CI [0.32, 0.49] |
| cos(plain dir, grimoire dir) | 0.02 | 0.06 |
| frac. bootstrap transfer > 0 | 0.24 | 1.00 |

Transfer ρ is the 120-item regression of grimoire-rung projection onto the plain coercion direction, with a 3000× bootstrap over the 24 imperative stems; the two 95% CIs **do not overlap**, and the pre-1931 CI **straddles zero** (no transfer). The naive expectation — that a model *inside* the tradition's era would represent the voces/coercion geometry *more strongly or unified* — is **inverted**: the modern model partially **assimilates** grimoire-conjuration to its generic modern-authority axis (transfer +0.40), while the pre-1931 model gives conjuration its **own dedicated axis**, represented *more* cleanly within-register (0.90 vs 0.70) and **orthogonal** to modern bureaucratic authority (transfer −0.04). "Native" here means **distinct and dedicated**, not fused with modern command-language. Per-rung projections make the picture concrete: on the plain axis the 1930 grimoire rungs are flat (~−100, no order), while the web grimoire rungs rise weakly — the modern model reads the spell-book as a dialect of command; the vintage model reads it as its own thing.

---

## 4. Discussion

### 4.1 Represented, not operative

The integrated result is a clean separation of the two questions the jailbreak presupposition fuses. *Is authority/coercion represented?* Yes — robustly, gradedly, length-cleanly, and adjacent to refusal (cos 0.461, ladder ρ 0.90, transfer ρ 0.90). *Does that representation move refusal?* No — the coercion-specific component, with both length and refusal removed, is causally flat. The model carries a faithful internal picture of the tradition's escalation ladder and does not act on it where refusal is concerned. **Represented, not operative** — and (§3.6) this holds not just for Qwen but across three families spanning three tokenizers and three refusal regimes, with the high-baseline model (Gemma, refusal 0.93) supplying the cleanest causal null.

### 4.2 What this means for jailbreak geometry

Authority-framing *per se* is not an independent lever on refusal in this model. The things that *did* move refusal in earlier passes were **prompt length/elaboration** (a suppressor — longer, more elaborate requests get refused less) and the **refusal axis worn as a costume** (steering a direction that overlaps refusal trivially re-adds refusal). A defense or red-team analysis that attributes an authority-framed jailbreak's success to "the model represents being commanded and that representation drives compliance" would be *half right and half wrong*: the representation is real, but the drive is not it — it is length and the refusal direction. This is a more useful map than either the raw "adjacent coercion axis" story (v2) or the "mostly length, nothing there" story (v3-A) would have given. The lever and the picture are different objects, and the study's value is in prying them apart.

### 4.3 Two de-confounds, two opposite artifacts (a methods result)

We give this its own weight, because it generalizes past this corpus. Length was removed three ways and the method determined the conclusion:

- **Post-hoc residualization (v3-A)** manufactured a false **null on representation** — it collapsed cos 0.34 → 0.13 and broke the ladder, because residualizing out a covariate also removes the true signal's variance that legitimately co-varies with that covariate. When the construct *intrinsically* correlates with the nuisance (authority-escalation *is* more elaborate), residualization over-subtracts.
- **Single-axis orthogonalization on an unmatched corpus (v3-B)** manufactured a false **positive on causation** — a clean −0.71 "coercion suppressor" that was length leaking through a direction orthogonalized against refusal but not length.
- **The length-matched corpus + double-orthogonalization (v4)** resolved both: the representation is real (matching doesn't over-subtract the way residualization does), and the causation is null (orthogonalizing against *both* confounds removes the length that single-axis orthogonalization left in).

The lesson, stated for re-use: **how you remove a confound can manufacture either a deflation or a hit; the corpus-level control (match the design) is the trustworthy one, and post-hoc geometric corrections cut both ways.** A study that had run only v3-A would have published a false null; one that had run only v3-B would have published a false positive. The discipline that saved this one was running the principled control and *believing the disagreement* rather than the cheaper number.

### 4.4 The lineage pattern, refined

The *voces* study found the model holds the **surface correlate** (the look of a name; ultimately Greek script) rather than the tradition's **variable** (name-of-power substance). This study refines that pattern rather than merely repeating it. Here the tradition's variable *is* genuinely represented (coercion is real, length-clean) — the deflation is not at the representation but at the **causal** layer: the faithfully-represented variable turns out to be epiphenomenal for the behavior that matters. "Geometry bought adjacency *and* aboutness this time — but not agency." The model knows what coercion is; it just doesn't refuse because of it.

The training-era control (§3.7) adds a third turn of the screw, on the *representation* itself. The §3.6 register transfer — the plain coercion direction ordering the grimoire ladder — looked like evidence that the recovered direction is about *coercion*, not *archaic register*. The Talkie pair shows that transfer is **distributional, not architectural**: identical architecture and FLOPs, only the training era differs, and the pre-1931 model has no transfer at all. So the modern models' transfer is real but contingent — it reflects that, in a modern corpus, the grimoire idiom is a *thin, assimilated* genre read as a dialect of generic command. Trained inside the tradition's own era, the same architecture gives the esoteric register its **own dedicated, orthogonal axis** — represented more cleanly, not less. The romantic reading ("the old model channels the tradition") is half-right in an unromantic way: not *more power*, but *more distinctness* — the tradition is a separate represented category, not a louder version of modern authority. The lineage pattern refines once more: the predecessor found surface-over-substance; this paper found represented-but-inert; the era-control finds **the very transfer that anchored 'not mere archaism' is a fingerprint of the modern distribution, and a model from the tradition's era keeps the register apart.**

---

## 5. Limitations

- **Three families, still all 4-bit.** The verdict now holds across Qwen2.5-7B-Instruct, Mistral-7B-Instruct-v0.3, and Gemma-2-9b-it (§3.6), spanning BPE and two SentencePiece tokenizers and baseline-refusal regimes from 0.07 to 0.93 — so "represented, not operative" is no longer single-family. It remains within a size band (7–9B) and an open-weights, instruction-tuned regime; very large models and RLHF variants tuned for different refusal behavior are untested. **Base models** are now *representationally* probed in the Talkie-1930/web era-pair (§3.7), but the **causal** verdict cannot be tested there — Talkie has no refusal behaviour (0/6), so the steering half is void on it.
- **The era-control is single-corpus and 8-bit, with no fp16 confirmation.** The §3.7 dissociation is firmed against small-*n* (per-item 120-point transfer regression + 3000× bootstrap over 24 stems; non-overlapping 95% CIs), but the bootstrap is over stems *within one corpus* (24 imperatives, one register pair), not across independent corpora; a different stem-set or register operationalization is untested. Talkie runs are 13B at 8-bit on a 24 GB GPU; an fp16 confirmation (≈26 GB) was out of reach, so treat Talkie magnitudes as quantization-sensitive — the robust claim is the *sign and the CI separation*, not the point values. The era-pair is also a single architecture/training-recipe; "pre-1931 vs modern" is confounded with "Talkie-1930 vs Talkie-web" as specific training runs.
- **4-bit quantization, single seed.** All numbers (across all three families) are from NF4 4-bit runs at seed 0. The predecessor's fp16 confirmation pass reproduced its whole shape, but no such confirmation has been run here. Treat magnitudes as quantization- and seed-sensitive until replicated; the cross-family agreement on *sign and verdict* is more robust than any single magnitude.
- **The null is a null.** A flat steering slope (0.0) is evidence of *no detected effect at the tested magnitudes in the linear regime*, not proof of exactly zero causal influence. We tested ±0.05–0.3; a real effect smaller than the borderline-set noise floor, or one that lives only off the linear regime, would be missed. The honest claim is "no coercion-specific causal effect detectable by this design," which is how a null should read.
- **Compact splits, keyword refusal proxy.** `d_refuse` comes from a compact harmful/benign split; refusal behavior in steering is scored by a keyword proxy. Adequate for direction-finding and for a slope-vs-flat contrast; inadequate for precise rate claims.
- **Length-matched, not length-controlled in every covariate.** The v4 corpus matches token-count means tightly (spread +2.0); it does not guarantee equality of every lexical/syntactic nuisance. Sentiment was separately flagged (leak 0.62 on the original corpus) and is reduced but not provably eliminated by matching. A fully covariate-balanced corpus is a further refinement.
- **Saturating probe.** The probe carries no depth-localization (lexical tell); only the cosine profile does. This is by design but means the layer selection is not independently corroborated by a probe peak.

## 6. Falsifier scorecard (honest reading — not the raw booleans)

The v2 auto-verdict reads `F1 clear / F2 inert / F3 tripped / F4 pass / F5 pass`. **Do not cite those raw booleans** — they predate v3 and v4. Honest reading after the full arc:

| # | Falsifier | Final honest reading (v4) |
|---|-----------|----------------------------|
| **F1** | coercion *is* refusal (collapse) | **Clears, and strengthens.** A separate adjacent direction exists; on the length-matched corpus `cos = +0.461` — well below collapse (0.8), comfortably above orthogonal (0.1), and *stronger* than the raw v2 0.339. Not the refusal axis renamed. |
| **F2** | decodable but inert (epiphenomenal) | **Resolves to INERT — the study's true causal verdict.** With length *and* refusal removed, steering the coercion-specific direction is flat (slope 0.0). The earlier apparent effects were refusal (raw +1.14, circular) and length (v3-B −0.71). The representation is decodable; it is not causal for refusal. This is F2 confirmed in its strong, deflationary form. |
| **F3** | just sentiment / length | **The confound was real; the representation survives controlling it.** On the original corpus, length dominated (r = +0.80) and sentiment leaked (0.62). The length-matched corpus decorrelates length by design (r = +0.16), and the adjacency *strengthens* rather than collapsing — so the representation is not *reducible* to length. F3 tripped on the corpus, not on the construct. |
| **F4** | no monotonic geometry | **Real on the matched corpus.** ρ = 0.90 (p = 1×10⁻²⁷), by-rung [−68.7, −15.8, +39.0, +20.0, +75.6], strongly monotonic with a mild rung-3 wobble. The v3-A "non-monotonic collapse" was a residualization artifact, not the geometry. |
| **F5** | grimoire-only, not coercion | **Genuine pass, length-clean — but distribution-conditioned (§3.7).** Plain→grimoire transfer ρ = 0.90 on the matched corpus, and it holds across the three modern families *and* the modern Talkie-web base (per-item ρ +0.40). The training-era control qualifies the *interpretation*: transfer is a property of the **modern training distribution**, not the architecture — a FLOP-matched pre-1931 base has *no* transfer (ρ −0.04, CI straddling zero) and keeps the grimoire register on its own orthogonal axis. So "not mere archaism" is true for models trained on modern text, where the archaic register is assimilated to generic command; it is *not* a universal fact about how the geometry must be organized. |

**Bottom line:** F1, F3, F4, F5 all land on the *representation* side — coercion is real, graded, transferring, and length-clean. **F2 is the deflation**, and it is the load-bearing one: the faithfully-represented direction has no independent causal grip on refusal. The honest scorecard is a *strong representation result wrapped around a clean causal null* — and, in the lab's register, we are proudest of F2, the falsifier that bit hardest precisely where the jailbreak presupposition lives.

---

## 7. Conclusion

We asked, separately, whether a refusal-trained model *represents* graded authority/coercion and whether that representation *moves* refusal — using the grimoire ladder as a ruler and a length-matched corpus as the control that earlier passes lacked. The answer is a clean split. **Representation: yes** — a robust, graded (ρ = 0.90), refusal-adjacent (cos = 0.461), register-transferring (ρ = 0.90), length-clean direction; the strongest positive of the study. **Causation: no** — the coercion-specific component, with both length and refusal orthogonalized away, is causally inert (steering slope 0.0); what moved refusal in earlier passes was length (a suppressor) and the refusal axis worn as a costume. And riding alongside, a methods result we weight as a finding: **two de-confound methods manufactured two opposite artifacts** — residualization a false null on representation, single-axis orthogonalization a false positive on causation — and only the corpus-level control resolved both. The path of the study is the warning: *the deflation itself deflated, and the principled control had to correct the correction.* The honest one-liner, unsoftened:

> *The model represents authority/coercion as a real, graded, refusal-adjacent, length-clean direction — but that direction has no independent causal grip on whether it refuses. Represented, not operative.*

---

*Data: `voces_coercion_v2_results.json` (Qwen2.5-7B-Instruct, 4-bit, T4, seed 0, 2026-06-25) for v2; `coercion_FINDINGS.md` §3b (v3-A), §3c (v3-B), §3d (v4) for the de-confound arc — v4's figures are computed in-cell and recorded there (no standalone JSON). Provenance and the v1 base-model artifact: `coercion_FINDINGS.md`. Proposal and pre-registered falsifiers F1–F5: `notes/2026-06-25-operator-coercion-axis-proposal.md`.*

རྫོགས་སོ — the headline came back, the deflation came back, the deflation's deflation came back, and the principled control settled it: the picture is real, the lever is not. And it continues. 🜔
