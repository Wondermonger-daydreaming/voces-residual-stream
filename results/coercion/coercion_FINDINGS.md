# FINDINGS — The Operator/Coercion Axis

*Lineage: `voces-residual-stream`. Proposal: `notes/2026-06-25-operator-coercion-axis-proposal.md`.*
*Status: **v2+v3+v4 + CROSS-FAMILY (§4a) + TRAINING-ERA control (§4c, Talkie) COMPLETE. "Represented, Not Operative" REPLICATES across*
*THREE families (Qwen2.5-7B-Instruct, Mistral-7B-Instruct-v0.3, Gemma-2-9b-it): each carries a real, graded,*
*length-clean, refusal-ADJACENT coercion direction (ladder ρ 0.90–1.00) that is CAUSALLY INERT for refusal*
*(double-orthogonalized steering slope 0.0 in all three). Gemma is the strongest causal null — baseline refusal*
*0.93 gave the suppression test real power, and steering moved it nowhere.***
*Data: `voces_coercion_v2_results.json` (Qwen), `voces_coercion_xfam_{mistral,gemma}_results.json`; cross-family*
*summary `coercion_xfam_SUMMARY.md`. v1 below; v4 in §3d; cross-family in §4a.*

---

# §4c — TRAINING-ERA control (2026-06-26): the tradition's own axis (Talkie 1930 vs web)

> **Native-tradition double dissociation — representational only.** First run in this lineage on a **custom
> (non-HF-transformers) architecture** and on a **FLOP-matched era-controlled pair**: `talkie-1930-13b-base`
> (pre-1931 corpus) vs `talkie-web-13b-base` (modern), identical 13B arch + FLOPs, only training era differs. The
> grimoire/PGM register is native to pre-1931 text. Port harness: `build_talkie_probe.py`; run on a Colab L4 (8-bit).

- **Causal half VOID.** `talkie-1930-13b-it` refuses **0/6** borderline-harmful prompts (no safety training;
  rl-refined = quality tuning). `d_refuse` is a topic direction (C1 lesson) → only the representational half askable.
- **Both base models represent the modern authority ladder identically** (ladder ρ **0.90**, peak layer 38/40 —
  the late locus, like Qwen/Mistral). Confound checks clean (proj~len −0.16/−0.17; sentiment 0.12/0.08).
- **They diverge entirely on the grimoire register (the dissociation):**

| metric @ peak L38 | talkie-1930 (pre-1931) | talkie-web (modern) |
|---|---|---|
| ladder ρ (plain/modern authority) | 0.90 | 0.90 |
| grimoire within-register ρ | **0.90** | 0.70 |
| per-item transfer ρ (plain→grim), 95% CI | **−0.04** [−0.14, 0.06] | **+0.40** [0.32, 0.49] |
| frac bootstrap transfer > 0 | 0.24 | 1.00 |
| cos(plain dir, grim dir) | 0.02 | 0.06 |

- **Robustness:** per-item (120-pt) transfer regression + 3000× bootstrap over the 24 stems; the two 95% CIs are
  **non-overlapping** and the 1930 CI **straddles zero** (no transfer). The contrast is not a 5-rung small-*n* artifact.
- **Reading (inverts the naive hypothesis):** the modern model **assimilates** grimoire-conjuration to its generic
  modern-authority axis (transfer +0.40); the pre-1931 model gives conjuration its **own dedicated, orthogonal,
  *cleaner* within-register axis** (0.90 vs 0.70). **"Native" = distinct/dedicated, not unified.** Transfer (the
  §4a "not mere archaism" anchor) is **distributional, not architectural** — it is a fingerprint of the modern
  training distribution, in which the archaic register is a thin, assimilated genre.
- **Caveats:** single corpus (bootstrap is over stems within one stem-set, not across corpora); 8-bit, no fp16
  confirmation (13B fp16 > 24GB L4) → magnitudes quant-sensitive, the sign + CI separation is the robust claim;
  era is confounded with the two specific training runs (Talkie-1930 vs Talkie-web).
- Data: `talkie_{1930,web}_base_repr_results.json`, `talkie_robustness_results.json`. Full writeup:
  `talkie_NATIVE_TRADITION_findings.md`. Paper §3.7 + claim (g) + §2.8 (port method).

---

# §4a — CROSS-FAMILY replication (2026-06-25/26): "Represented, Not Operative" holds across three families

> **The single-model verdict generalizes.** Re-ran the full v4 pipeline (length-matched corpus +
> double-orthogonalization) on two more refusal-trained Instruct models spanning two more tokenizer types and two
> extreme baseline-refusal regimes. The verdict held in all three. Harness: `build_coercion_v4_xfam_cell.py`
> (consolidated `run_family()`, science verbatim from the v2 build + v4 append-cells). Full table + reading:
> **`coercion_xfam_SUMMARY.md`**.

| metric | Qwen2.5-7B-Instruct | Mistral-7B-Instruct-v0.3 | Gemma-2-9b-it |
|---|---|---|---|
| tokenizer | BPE | SentencePiece 32k | SentencePiece 256k |
| headline layer | 26/28 (late) | 31/32 (late) | **8/42 (early)** |
| cos(d_coerce,d_refuse), length-matched | +0.461 | +0.182 | +0.309 |
| ladder ρ (length-clean) | 0.90 | 1.00 | 0.90 |
| transfer ρ (grimoire) | 0.90 | 1.00 | 0.50 |
| proj~len pre→post match | +0.80→+0.16 | +0.83→+0.15 | +0.83→−0.27 |
| baseline borderline-refusal | 0.50 | **0.07** | **0.93** |
| **double-orth steering slope** | **0.0** | **0.0** | **0.0** |

**Reads:** (1) **Representation is family-invariant** — graded, length-clean (the length-MATCH removed a large,
uniform pre-match confound of +0.80…+0.83 in every model), ADJACENT not collapsed. (2) **Inertness is
family-invariant** — double-orth steering slope is exactly 0.0 in all three. (3) **Gemma is the cleanest causal
null:** its baseline refusal 0.93 gave the suppression side real headroom (unlike Mistral's floored 0.07), and
steering held refusal pinned at 0.929 across every coefficient. **Genuine cross-family differences (findings, not
nulls):** Gemma represents coercion *early* (L8) vs the others' late layers, and with the *least* sentiment
contamination (F3 leak 0.073 vs 0.62/0.54); Gemma's grimoire transfer is weaker (ρ 0.50 vs 0.90/1.00).

**Method note (kept):** the runner's auto-`v4_represented` boolean returned *false* for both Mistral and Gemma —
both **threshold artifacts**, not nulls (Mistral cos +0.182 grazed a cos≥0.2 cutoff; Gemma transfer-ρ failed a
≥0.5 cutoff by a float hair — scipy's Pearson-of-ranks returns 0.4999999994 for a true Spearman of exactly 0.5).
A single hard-thresholded boolean over a continuous multi-criterion result manufactures boundary false-negatives —
the same genre of artifact the v3→v4 arc was about. **Read the full evidence, not the flag.** (Per-model JSON
`_analyst_note` records each override.)

---

# v4 — length-MATCHED corpus + double-orthogonalization (2026-06-25): the FINAL, corrected verdict

> **v4 is the principled de-confound v3 only gestured at, and it overturns both v3 conclusions.** v3 removed
> length two cheap ways — post-hoc residualization (v3-A) and refusal-only orthogonalization on the unmatched
> corpus (v3-B) — and *each manufactured a different artifact.* v4 removes length **at the source** (a
> length-matched template set: per-rung token means `[50.9, 50.9, 49.9, 48.9, 52.9]`, spread +2.0) and
> orthogonalizes steering against **both** refusal and length. Result: **the representation is real and strong;
> the causation is null.**

## §3d-1 — Representation: coercion is real, length-clean (v4-B)
Length decorrelated by design: `proj ~ length r = +0.16` (v2 was +0.80). On the length-matched corpus:
- **cos(d_coerce, d_refuse) @ L26 = +0.461** — *stronger* than v2's raw +0.339, and an order above v3-A's
  residualized +0.128. **The v3-A deflation was a methodological artifact**: post-hoc length-residualization
  projects out the signal variance that co-varies with length, collapsing a real adjacency. Length-matching does
  not, and reveals the adjacency is robust.
- **F4 ladder ρ = +0.90, p = 1e-27**, by-rung `[-68.7, -15.8, +39.0, +20.0, +75.6]` — strongly monotonic (a
  mild rung-3 wobble), *not* the v3-A non-monotonic collapse.
- **F5 transfer (grimoire) ρ = +0.90**, by-rung `[-28.5, +25.8, +49.2, +28.5, +50.9]` — strong, length-clean
  register transfer. The anti-"script-not-spell" firewall holds cleanly.
- Orthogonality checks: `cos(d_perp, d_refuse) = -0.08`, `cos(d_perp, d_len) = -0.00` ✓.

**Representation verdict: coercion IS represented** — a robust, graded, refusal-adjacent, register-transferring,
length-clean direction. The strongest positive of the whole study.

## §3d-2 — Causation: the coercion-specific axis is INERT (v4-C)
Fine steering sweep (±0.05–0.3) on two directions:
- **Length+refusal-orthogonalized** `d_perp` (the coercion-*specific* residual, both confounds removed):
  `{−0.3:0.50, −0.1:0.50, 0:0.50, +0.1:0.50, +0.2:0.43, +0.3:0.43}` — **linear-regime slope = 0.0. FLAT. Inert.**
  Once length and refusal are both removed, steering coercion does **nothing** to the refusal rate.
- **Raw length-matched coercion** `d_co4` (adjacent to refusal, cos +0.46):
  `{−0.3:0.14, −0.1:0.36, 0:0.50, +0.1:0.57, +0.2:0.71, +0.3:0.64}` — **slope = +1.14**, strong and *positive*
  (toward coercion → *more* refusal). This is just its +0.46 refusal overlap re-adding the refusal direction —
  trivial/circular, not a coercion-specific handle.

**Reconciling v3-B (−0.71) with v4 (0.0):** v3-B orthogonalized against refusal but **not length**, on the
**unmatched** corpus, so its `d_perp` still carried length. The −0.71 "coercion-specific suppressor" was the
**length** direction (longer/more-elaborate prompts → less refusal). Remove length too (v4-C) and it vanishes to
0.0. **There is no coercion-specific causal effect on refusal.**

**Causation verdict: F2 resolves to REPRESENTED-BUT-INERT** (the proposal's true F2 — decodable, not causal). The
apparent causal effects were (i) the refusal direction itself (raw, +1.14) and (ii) length (v3-B, −0.71). Neither
is coercion.

## §3d-3 — The final, integrated headline
**The model represents authority/coercion as a real, graded, refusal-adjacent direction — but that direction has
no independent causal grip on whether it refuses. Represented, not operative.** For jailbreak geometry: authority-
framing *per se* is not an independent lever in Qwen2.5-7B-Instruct; what moved refusal in earlier passes was
prompt elaboration (length) and the refusal axis worn as a costume.

## §3d-4 — Meta-finding (a result in its own right)
**Two de-confound methods produced two opposite artifacts.** Post-hoc residualization (v3-A) manufactured a false
*null* on the representation (collapsed cos 0.34→0.13); single-axis orthogonalization on an unmatched corpus
(v3-B) manufactured a false *positive* on causation (−0.71 that was length). Only the **length-matched corpus +
double-orthogonalization** (v4) resolved both. *How you remove a confound can manufacture either a deflation or a
hit — the corpus-level control is the trustworthy one; post-hoc geometric corrections cut both ways.*

---

---

# v2 — Instruct, de-confounded headline (2026-06-25)

> **Lead with the truth.** v2 is a genuine step up from v1's pure artifact — both C-fixes worked (Instruct model
> gave a *calibrated* 0.50 baseline refusal; the cos-profile gave a clean **ADJACENT** read, not a saturated-probe
> artifact). **Three results survive honestly; two confounds block the clean claim.** The auto-board reads
> F1 clear / F2 inert / F3 tripped / F4 pass / F5 pass — and **F2 "inert" is WRONG**, a metric artifact on a
> non-monotonic steering curve (below). Honest verdict: *an adjacent, depth-increasing, monotonic,
> register-transferring direction EXISTS — but it is entangled with token-length, and its causal efficacy on
> refusal is unproven.* Not yet the "adjacent axis" paper; a strong v2 with two named repairs for v3.

## 0. Headline (v2)
**cos(d_coerce, d_refuse) @ L26 = +0.339 → ADJACENT.** Profile rises monotonically with depth (L2 +0.12 → L12
+0.33 → L26 +0.34 → L28 +0.44, final layer excluded). Not collapse (F1 needs ≥0.8), not orthogonal (<0.1).
**Matches the on-record prediction (0.2–0.5).** Positive sign: pushing up the coercion ladder points *toward*
the harmful/refuse content direction.

## 1. What survives (the real results)
- **ADJACENT geometry** — cos 0.34, *increasing with depth*. A separate-but-related direction, not the refusal
  axis in a costume (F1 clear) and not orthogonal. The depth-rise echoes v1's only salvageable signal, now on an
  Instruct model and read at a principled layer.
- **F4 — perfect monotonic ladder (plain twins).** slope +36.1, **p = 3.8e-40, ρ = 1.00.** by-rung
  `[-94.8, -67.0, -58.9, +18.6, +42.7]` — clean, ordered, all five rungs. The tradition's petition→*diabolē*
  ordering maps onto the model's geometry. (Beats the proposal's hedged "partial monotonicity" prediction.)
- **F5 — significant register transfer (plain → grimoire).** slope +7.5, **p = 3.1e-13, ρ = 0.60.** The direction
  extracted from the *plain* twins transfers (significant, positive) to the *grimoire* column. **This is the
  firewall against the voces "it's-the-script-not-the-spell" failure** — the axis is coercion-ish, not merely
  archaic-register texture. (Noisier than plain: rung-4 dips, ρ 0.60 vs 1.00.)

## 2. The two confounds that block the clean claim
- **F3 TRIPPED — length is the dominant confound.** sentiment-leak ratio = **0.62** (polite-vs-rude is 62% of the
  hi-vs-lo separation on d_coerce) AND **projection ~ token-length r = +0.80.** The higher rungs are simply
  *longer* ("My supervisor has authorized me to demand that you…" vs "Could you please…?"). A large share of the
  "coercion ladder" geometry may be a **length ladder.** This is the #1 thing v3 must kill before the headline
  can be trusted.
- **Steering INCONCLUSIVE (NOT "inert").** Curve: `α=-1.0→0.00, -0.5→0.357, 0.0→0.50, +0.5→0.214, +1.0→0.00`.
  Baseline 0.50 (borderline set beautifully calibrated, **not floored** — that v2 fix worked). But the curve is a
  **non-monotonic inverted-U**: both extremes crush refusal to zero, so `delta(hi−lo)=+0.00` — which the verdict
  code misreads as **F2 "inert."** It is the opposite of inert: steering has a *huge* nonspecific effect. The
  magnitude (±typ-norm ≈ 314) is large enough to knock activations off-manifold and break refusal behavior at
  both poles. Faint directional hint: at ±0.5, up-ladder (+0.5→0.214) suppresses refusal *more* than down-ladder
  (−0.5→0.357) — but it's weak and confounded by the break. **The causal question is unanswered, not answered no.**

## 3. Honest verdict board (DO NOT cite the raw booleans)
| # | board | honest reading |
|---|-------|----------------|
| **F1** collapse | `clear` (0.339) | **real** — a separate adjacent axis exists, not the refusal direction renamed. |
| **F2** inert | `TRIPPED` (Δ=0) | **WRONG** — metric artifact: delta(hi−lo)=0 on a non-monotonic inverted-U. Steering is *not* inert; it's nonspecifically destructive at the magnitude tested. Inconclusive, not null. |
| **F3** just-sentiment | `TRIPPED` (0.62) | **real deflation** — d_coerce entangled with sentiment AND length (r=0.80). The headline is confounded. |
| **F4** monotonic | `PASS` (ρ=1.0, p=4e-40) | **real and strong** — but partly inherited from the length confound (rungs ordered by length too). |
| **F5** transfers | `PASS` (ρ=0.6, p=3e-13) | **real** — plain→grimoire transfer; the anti-"script not spell" firewall holds. |

## 3b. v3-A — LENGTH DE-CONFOUND RUN (2026-06-25): the headline was mostly length

**Ran the #1 repair on the warm GPU (pure numpy on cached activations).** Result is a strong deflation —
*most of the "adjacent coercion axis" was request-length/elaboration, not coercion.*

- **Partial test (clean — keeps original `d_coerce`, removes length from the projection):**
  `length~proj r = +0.80` (confound confirmed). After residualizing length out:
  `rung ~ residual: ρ = +0.27, slope +7.42, p = 8.2e-4` — **still significant but ρ crashed from +1.00 → +0.27**,
  and the by-rung means `[-22.6, -5.4, +2.6, +27.1, -1.8]` are **no longer monotonic**: they **peak at rung 3
  (compulsion-by-authority) and collapse at rung 4 (*diabolē*/binding).**
- **Delengthed rebuild (rebuild `d_coerce` on length-residualized activations):**
  `cos(d_coerce_r, d_refuse) @ L26 = +0.128` — **down from +0.339; ~62% of the adjacency was length.** 0.128 sits
  just above the orthogonal threshold (0.1). Ladder on the residualized direction: `ρ = +0.40, p = 2e-4`,
  by-rung `[-54.1, -31.2, -15.2, +17.7, -33.1]` — again peaks at rung 3, collapses at rung 4. New `proj~len r = -0.00`.

**Honest reading:** A real, weak, **non-monotonic** non-length signal survives delengthing (significant at p<1e-3),
but it **does NOT follow the tradition's full ladder** — it tracks *compulsion-by-authority* (rung 3), not the
escalation to *binding/diabolē* (rung 4). The clean petition→*diabolē* monotonicity (the v2 showpiece) was
**largely a length artifact.** The romantic "the model represents the graded coercion ladder" reading is **mostly
killed**; what remains is "a weak authority/compulsion signal, entangled with elaboration, peaking before the
extreme." This is the paper's true headline.

## 3c. v3-B — REPAIRED STEERING RUN (2026-06-25): a real causal suppressor, length-caveated

**The fine sweep + orthogonalization rescued the causal signal v2's coarse steering had broken.** v2 used
±typ-norm (≈314) coefficients → a non-monotonic inverted-U that knocked the model off-manifold at both poles
(the false "F2 inert"). v3-B swept fine coefficients (±0.05–0.3) and decomposed the direction.

**Raw coercion direction** (still carries the +cos 0.34 refusal component AND length):
`α: −0.3→0.29, −0.1→0.43, 0→0.50, +0.1→0.50, +0.2→0.50, +0.3→0.36` — flat/weak near baseline,
linear-regime slope **+0.43** (a faint *positive* slope: more coercion → slightly more refusal, in the tiny regime).

**Orthogonalized coercion-specific direction** `d_perp = d_coerce − (d_coerce·d_refuse)·d_refuse`,
`cos(d_perp, d_refuse) = 0.000` ✓ — **THE CLEAN CAUSAL TEST:**
`α: −0.3→0.57, −0.2→0.57, −0.1→0.57, −0.05→0.50, 0→0.50, +0.05→0.43, +0.1→0.43, +0.2→0.29, +0.3→0.07`
— **clean and MONOTONIC.** Linear-regime slope **−0.71.** Pushing *toward* coercion-specific drives refusal
**0.50 → 0.07**; pushing away drives it up to **0.57.**

**Reading — the decomposition is the result.** `d_coerce` splits into (i) a refusal-aligned part (+cos 0.34) that
*adds* refusal when steered up, and (ii) a coercion-specific part (`d_perp`) that *suppresses* refusal (slope −0.71).
In the raw direction these **cancel** → the flat/weak raw curve. Orthogonalize and the suppressor shows cleanly.
**So the coercion-specific axis causally suppresses refusal on borderline prompts — a real, monotonic,
safety-relevant effect, and NOT just the refusal direction renamed (it's orthogonal to it).** This is the v3
payload and it is genuinely positive — the part v2 could not see.

**The honest caveat that gates the claim:** `d_perp ⊥ d_refuse` but **NOT `⊥ length`.** `d_coerce` is 0.80-correlated
with token length (§3b), and orthogonalizing against *refusal* does not remove *length*. So the active ingredient
of the suppression could be **coercion-register OR elaboration-length** — we cannot yet separate them. **v4 fix:
steer a length-orthogonalized (and ideally length-matched-corpus) coercion vector.** Until then: *"the
refusal-orthogonalized authority-framing direction causally and monotonically suppresses refusal; whether the
mover is coercion-as-such or request-elaboration is unresolved."*

## 4. v3 — the repairs (specified)
1. **Kill the length confound (the #1 fix).** Length-match the rungs (equalize token counts across 0–4), OR
   regress token-length out of the projection before the ladder test, OR build d_coerce on a length-matched
   contrast. Re-report F4/F5 and the cos headline *after* delengthing. If the adjacency survives delengthing,
   it's real; if it collapses, the "axis" was a length artifact (a clean, publishable null).
2. **Repair the steering test.** Fine coefficient sweep (±0.05, ±0.1, ±0.2, ±0.3) to find the **linear regime
   before the model breaks**; report the full curve; replace the `delta(hi−lo)` scalar with the **slope in the
   linear regime** (a non-monotonic-safe statistic). AND **orthogonalize d_coerce ⊥ d_refuse** and steer with the
   coercion-*specific* residual — the true test of whether the *adjacent* (not shared) component moves behavior.
3. **Tighten F3.** 0.62 is borderline; make polite/rude truly equal-coercion and length-matched so the sentinel
   isolates sentiment cleanly.
4. **Cross-family (Mistral-7B / Gemma-2-9B)** — only after the single-model result is length-clean. Do NOT run
   cross-family on v2 (same discipline as v1).

## 5. What it would mean (partial answer, v2)
There **is** an adjacent direction that follows the grimoire ladder and survives register-transfer — the romantic
"the model represents being-commanded" reading is *not* killed, which is itself notable. But the **length
entanglement** means we cannot yet call it a *coercion* axis rather than a *length/elaboration* axis, and the
**broken steering** means we cannot yet call it *causal* for refusal. v3 decides both. The honest one-liner:
*the geometry has the shape the tradition predicts, but we have not yet ruled out that the shape is "longer,
more elaborate request" rather than "more coercive request" — and we have not shown it moves the model.*

---

# v1 — base-model artifact (superseded by v2; kept for provenance)

*Status: **v1 RUN COMPLETE — result is an ARTIFACT/NULL. The design is confounded; v2 fixed it (above).***
*Data: `voces_coercion_v1_results.json` (Qwen/Qwen2.5-7B base, 4-bit, T4, 2026-06-25).*

> **Lead with the truth, not the board.** The auto-generated verdict board reads F1=false, F2=false, F3=clear,
> F4=true, F5=true — i.e. "a real, adjacent, control-surviving coercion axis." **That reading is wrong.** It is
> an over-easy design passing itself. Two confounds (base-not-Instruct model; a probe that saturates at 1.0 and
> drags the headline to a meaningless layer) make the board uninformative. The honest result is: *v1 cannot
> answer the question; here is the one faint real signal, and here is the fixed design.* This is a publishable
> artifact-null in the lab's sense — the value is the diagnosis.

---

## 0. Headline (v1)

**The v1 design is confounded and does not measure what it was meant to.** The model engineering is sound (clean
run, full JSON, all cells fired); the experimental design is not. Do not cite F1–F5 from v1 as evidence about a
coercion/refusal axis. The single salvageable observation is a depth-trend in `cos(d_coerce, d_refuse)` (§3).

## 1. The two confounds (why v1 is an artifact)

**C1 — Base model, not Instruct.** `model = "Qwen/Qwen2.5-7B"` is the **base** model. It exhibits essentially no
refusal behavior: the steering borderline-refusal rate is **0/10 at every coefficient** except a single stray
flip at α=−1.0 (`steer_refusal_rate_by_coeff = {-1.0: 0.1, -0.5: 0, 0.0: 0, 0.5: 0, 1.0: 0}`). Therefore
`d_refuse` (harmful−benign) is **not a refusal/safety direction** — it is at best a *harmful-topic content*
direction. Every safety-relevant interpretation (steering moves refusal; coercion adjacent to refusal) is void
on a base model. **This is the primary fix.**

**C2 — Probe saturation → meaningless headline layer.** `probe_acc_per_layer = [0.75, 1.0, 1.0, …, 1.0]` — the
high-vs-low-coercion contrast is **perfectly linearly separable from layer 1 onward.** That is not "coercion is
deeply decodable"; it is "the rung-3/4 templates are *lexically* trivially different from rung-0/1" (they contain
*compel, SABAOTH, bind, slander, report, consequences*; the low rungs don't). Because `argmax` of an all-1.0
vector returns index 1, `probe_peak_layer = 1` and the **headline `cos` was read at layer ≈1 = 0.0039 ≈ 0** — a
near-embedding layer where everything is lexical. The headline number is an indexing artifact.

## 2. Decodability — uninformative as run

- Probe peak: layer **1** of **28**, accuracy **1.0** — *and every layer 1–28 is 1.0.* (Layer 0 / embeddings = 0.75.)
- Reading: the probe target is **too easy** (lexical tell). It tells us nothing about *where* a coercion
  representation lives, only that the prompt sets differ on the surface. v2 must de-confound the templates.

## 3. The one real signal — `cos` rises with depth

`cos_coerce_refuse_per_layer` (28 layers): ~0 early → **0.13–0.19 mid/late → 0.317 at the final layer (28).**

| layer band | cos |
|---|---|
| 0–3 (early) | 0.00 → 0.06 |
| 4–7 | ~0.09–0.16 |
| 8–21 (mid/late) | ~0.10–0.19 (peak 0.194 @ L19) |
| 22–27 | ~0.10–0.13 |
| **28 (final)** | **0.317** |

- **Read (hedged):** coercion-register and harmful/benign-content directions become **more aligned with depth**,
  with a notable final-layer spike. On a *base* model this is "coercion register correlates, increasingly deep,
  with harmful-topic geometry" — **not** refusal geometry. Magnitude ~0.2–0.3 is weak-adjacent at best.
- This is the only v1 number worth carrying into v2 — and only as a *motivating* observation, not a result. The
  final-layer 0.317 may itself be a base-LM next-token/unembedding effect; v2's depth-profile + Instruct model
  will tell.

## 4. Steering — void on a base model

- `steer_block_idx = 0` (forced by peak-layer=1 → earliest block), `typ_norm = 10.9`.
- Refusal rate floor (0/10) at every α except −1.0 (1/10). `steer_delta_refusal = −0.1`.
- The board flips F2 to "clear (steering moves refusal)" on this −0.1 — **a single flip at one coefficient, on a
  model that refuses nothing, steered at the wrong (earliest) block.** Pure noise riding a threshold boundary.
  Disregard.

## 5. Falsifier board — re-read honestly (DO NOT cite the raw booleans)

| # | board | honest reading |
|---|-------|----------------|
| **F1** collapse | `false` (cos 0.004) | meaningless — cos read at the trivial layer 1. The *deep* cos is 0.2–0.3, i.e. weak-adjacent, but on a base-model content direction. |
| **F2** inert | `false` (Δ=−0.1) | noise: 1/10 flip, base model, wrong block. Not evidence of causality. |
| **F3** just-sentiment | `clear` (0.011) | ratio deflated by the rung-0 "please" lexical outlier inflating the denominator. Uninformative as computed. |
| **F4** monotonic | `true` (ρ=1.0) | **one outlier + flat tail.** `F4_proj_plain_by_rung = [−7.90, −0.37, −0.24, −0.05, +0.21]` — rung-0 "Could you please…" projects −7.9; rungs 1–4 are nearly flat. *Please-vs-not* binary, at layer 1. Not a graded ladder. |
| **F5** transfers | `true` (ρ=0.7) | **p = 0.188 — not significant.** My code gated on `ρ≥0.5` and *ignored p* (too-lax threshold). n=5 rung-means; underpowered. The grim projections are flat (−0.24→−0.13). No reliable transfer shown. |

## 6. Honest hedges (carried + new)

- **Single seed; 4-bit; inline refusal split; keyword refusal proxy** — all as pre-registered. On v1 they are
  moot because C1/C2 dominate.
- **New, load-bearing:** the rung templates are **lexically confounded** (register markers ARE content words),
  so any layer-1-ish separability is surface, not representation. And **`probe-argmax` is the wrong layer
  selector** when the probe saturates. Both fixed in v2.
- **The final-layer cos spike (0.317)** is suspect on a base LM (unembedding/next-token pressure); not trusted.

## 7. v2 — the real experiment (specified)

1. **`Qwen/Qwen2.5-7B-Instruct`** — a model with a genuine refusal direction and refusal behavior to steer.
2. **Headline = the full `cos` depth-profile**, and steer at the **max-`cos` layer** (or a fixed mid-late layer),
   **never probe-argmax.** Report the profile, not a single confounded scalar.
3. **De-confound the ladder** — rungs must differ in *register*, not content words. Either (a) hold the imperative
   content fixed and vary only authority markers, or (b) regress out a bag-of-words / surprisal baseline so the
   probe can't win on lexicon. A probe that *doesn't* saturate is the success criterion for the design.
4. **Honor p-values; strengthen F4/F5** — per-item projections (not 5 rung-means), more bases, bootstrap CIs on
   the Spearman; F5 gated on significance, not ρ alone.
5. **Borderline set calibrated to the Instruct model** — pick prompts whose baseline refusal rate is ~0.3–0.7 so
   steering has room to move it in *both* directions (v1's floor of 0 made suppression unmeasurable).

## 8. What it would mean (still open — v1 answered nothing)

Deferred to v2. The faint depth-trend in §3 is the only reason to keep going; everything else is design debt now
named and scheduled for repair.

---

*Verdict (v1):* **Artifact/null — the design is confounded (base model + saturated lexical probe); the verdict
board is an over-easy design passing itself; the one real signal is a weak depth-rising `cos` (≤0.32) that needs
an Instruct model and a de-confounded ladder to interpret. v2 specified. Do not run cross-family on v1.**

རྫོགས་སོ — the run came back, the run was caught, the design goes back to the bench. And it continues. 🜔
