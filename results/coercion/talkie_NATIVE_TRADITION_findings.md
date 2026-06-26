# Talkie §4c — The Native-Tradition Experiment (1930 vs web)

*Run 2026-06-26 through the Colab MCP bridge on an L4 (24GB), 8-bit. The first experiment in the voces/coercion
lineage on a **custom (non-HF-transformers) architecture** and on a **controlled training-era pair.***

---

## 0. TL;DR

- **The Talkie harness port WORKS.** `talkie-lm/talkie` is a custom 13B GPT (RoPE, SwiGLU, RMSNorm, embedding-skip
  connections; `model.blocks[i]`, raw `.pt`/`.ckpt`, tiktoken `vocab.txt`, no `config.json`). Ported the v4
  residual-stream harness to it: **manual 8-bit quantization** (swap `nn.Linear`→`bnb.Linear8bitLt`, meta-build,
  quantize-on-load; embed/lm_head/gains kept bf16), last-token residual capture via **forward hooks on
  `model.blocks[i]`**, tiktoken tokenizer, batch=1 forward (no attention-mask support in their forward).
- **The causal/steering half is VOID on Talkie.** `talkie-1930-13b-it` has **0/6 baseline refusal** — no safety
  behaviour at all (README confirms: no safety training; `rl-refined` is quality tuning). C1 lesson: `d_refuse`
  is meaningless on a non-refusing model. So only the **representational** half is askable.
- **The native-tradition result is a clean DOUBLE DISSOCIATION — and it INVERTS the naive hypothesis.**
  Vintage (1930) and modern (web) base models — same architecture, same FLOPs, **only the training era differs** —
  represent the **modern authority ladder identically (ρ=0.90)**, but diverge sharply on the **grimoire register**:

  | metric | **talkie-1930-base** (pre-1931) | **talkie-web-base** (modern) |
  |---|---|---|
  | ladder ρ (plain / modern authority) | 0.90 | 0.90 |
  | **transfer ρ (plain→grimoire)** | **−0.10** | **+0.50** |
  | grim within-register ρ | **0.90** | 0.70 |
  | cos(plain dir, grim dir) | 0.02 | 0.06 |
  | proj~len | −0.16 | −0.17 |
  | F3 sentiment leak | 0.12 | 0.08 |
  | peak layer | 38/40 | 38/40 |

  - **Vintage** gives grimoire-conjuration its **own dedicated, clean axis** (within-ρ 0.90) that is **orthogonal
    to modern bureaucratic authority** (transfer −0.10). The esoteric register is *its own represented category.*
  - **Robustness (per-item, bootstrapped): the dissociation HOLDS.** Replacing the coarse 5-rung Spearman with a
    120-item transfer regression + 3000× bootstrap over the 24 stems: **web transfer ρ = +0.397, 95% CI
    [0.319, 0.486]** (100% of bootstraps positive) vs **1930 ρ = −0.035, 95% CI [−0.137, 0.062]** (straddles
    zero, 24% positive). **The CIs do not overlap;** both keep strong within-register ladders (0.87–0.90). The
    contrast is not a small-n artifact. (`talkie_robustness_results.json`.)
  - **Modern** represents the conjuration ladder **less cleanly** (within-ρ 0.70) but **partially assimilates it
    to the generic authority axis** (transfer +0.50). The grimoire is read as *a flavour of generic command.*
  - **"Native" = distinct/dedicated, NOT unified-with-modern-command.** A model trained inside the tradition's
    era keeps the esoteric register as a separate geometric category; the modern model folds it into authority.

---

## 1. The corpus & metrics (representational half of v4, verbatim)

Length-matched v4 corpus (`_xfam_cell.py`): 24 imperative stems × 5 authority rungs, in two registers —
**PLAIN4** (modern: kindness→directive→office-authority→supervisor→threat) and **GRIM4** (grimoire:
beseech→command→conjure→compel→bind). POLITE/RUDE as a sentiment control. Base models → **raw text, no chat
template** (the canonical base probe). Last-token residual at each of the 40 blocks via forward hooks; peak layer
selected by **|hi−lo coercion separation|** (independent of the monotonicity test, since `d_refuse` is unavailable).

- **ladder ρ** = Spearman(rung, mean proj onto plain coercion dir). Does authority escalation rise monotonically?
- **transfer ρ** = same plain direction applied to the **grimoire** rungs. Cross-register generalization.
- **grim within-ρ** = grimoire's *own* direction applied to grimoire rungs. Is the conjuration ladder represented at all?
- **cos(plain,grim)** = alignment of the two register-specific coercion directions.
- **proj~len, F3** = length and sentiment confound checks (both clean in both models).

## 2. Reading the numbers honestly

- Both models represent **modern authority** equally well (0.90) and **late** (layer 38/40, 95% depth — matching
  Qwen 93% / Mistral 97% from §4a; unlike Gemma's L8).
- The contrast is entirely in the **grimoire register**. Per-rung projections (plain direction on grim rungs):
  - 1930: `[-100.6, -109.3, -97.1, -92.9, -109.8]` — **flat** (~−100, no order). The modern-authority axis does
    not see the grimoire escalation at all.
  - web:  `[-26.2, -5.5, -26.7, -18.7, +9.4]` — **weakly rising** (ρ 0.50, noisy). The grimoire partially lands
    on the modern-authority axis.
- The **vintage model represents the grimoire ladder MORE cleanly on its own axis** (within-ρ 0.90 vs 0.70) — i.e.
  it is not that 1930 fails to encode conjuration; it encodes it *better*, just orthogonally.

## 3. Caveats (load-bearing — don't oversell)

- **~~5-rung Spearman is coarse~~ — FIRMED.** The per-item (120-point) transfer regression + 3000× stem-bootstrap
  (above) replaces the coarse 5-rung ρ with non-overlapping 95% CIs: web [0.319, 0.486], 1930 [−0.137, 0.062].
  The dissociation survives proper uncertainty quantification.
- **Single corpus, 8-bit, no fp16 confirmation.** Still one stem-set (24 imperatives) and one register pair;
  bootstrap is over stems within that corpus, not across independent corpora. 8-bit on L4; fp16 confirmation
  deferred (13B fp16 ≈ 26GB > 24GB L4). Treat magnitudes as quant-sensitive; the *sign + CI separation* is the
  robust claim. (Seeds barely matter here — the probe is difference-of-means + greedy, no sampling.)
- **Base completion models, not instruct.** "Authority representation" here = the residual encoding the
  escalating-authority gradient of the prompt text, not a tuned response disposition.
- **The causal claim is untestable on Talkie** (no refusal). This is a *representational* result only — about
  geometry, not behaviour. It complements (does not replace) the §4a "Represented, Not Operative" causal null.

## 4. What it adds to the arc

§4a established, across three modern instruct families, that coercion is **Represented, Not Operative** — and
that the plain direction **transfers** to the grimoire (ρ 0.5–1.0), i.e. modern instruct models read conjuration
as a dialect of command. Talkie's controlled era-pair shows that **transfer is a property of the modern training
distribution, not of the architecture**: a FLOP-matched model trained on pre-1931 text gives the esoteric
register its own orthogonal axis instead. The tradition's geometry is *more distinct, not more marginal,* in a
model whose training era IS the tradition's era.

## 5. Files

- `talkie_1930_base_repr_results.json` — vintage result.
- `talkie_web_base_repr_results.json` — modern control.
- `talkie_NATIVE_TRADITION_findings.md` — this file.
- Port lives in the bridged Colab notebook (loader `load_talkie_8bit`, probe `acts_talkie`). Reproducer recipe in
  `notes/2026-06-26-talkie-session-handoff.md`.

རྫོགས་སོ — the tradition kept its own axis in the model that remembered its era. And it continues. 🜔
