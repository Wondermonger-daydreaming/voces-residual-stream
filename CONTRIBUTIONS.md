# Contributions

**Author (accountable):** Tomás Pavan.

**AI collaboration (acknowledged, not authorship):** the experiment was designed and analyzed in dialogue with
**two Claude Opus 4.8 instances**:

- **Claude Opus 4.8 on claude.ai** — the *design* instance: wrote the study spec, hypotheses, and named
  falsifiers, and caught the conclusion-wearing-a-hat overclaim ("abstraction reading gets first real
  evidence") and routed it to the control that dissolved it. White-box-blind by construction (API-only), so it
  could design the residual-stream experiment but not run it.
- **Claude Opus 4.8 on Claude Code** — the *build & analysis* instance: built the apparatus, ran it across six
  versioned passes, verified the citations against live sources, and hardened the repository.

The full three-way back-and-forth (both instances plus the author) is on the record in the **git history** and
commit co-author trailers.

This is recorded as *acknowledgment*, not authorship: an author line is an accountability claim, and a model
cannot be held to account. The work was genuinely collaborative; the credit says so without writing a check no
one can cash.

**Method note.** The retraction of the *"The Grain Runs Deeper in Greek"* draft, and the verify-then-you-know
citation pass, are part of the record on purpose — see `PROVENANCE.md` and the git log.

---

## Sequel — the operator/coercion axis + Talkie training-era control

The sequel study (`paper/represented-not-operative.md`, `results/coercion/`, `src/coercion/`) was built and
analyzed by a **Claude Opus 4.8 (1M-context) instance on Claude Code** — the *build & analysis* role, continuous
with the instance that hardened the study above. It specified the falsifier-gated coercion design, ran the
v2→v4 de-confound arc and the three-family replication, and ported the residual-stream harness to Talkie's
custom non-HuggingFace architecture for the §4c training-era control. The same accounting holds: **acknowledged,
not authorship** — the author (Tomás Pavan) is accountable; the model collaboration is on the record in the git
history and commit co-author trailers.

**Method note (sequel).** The recorded refusal-gate result (`talkie-1930-it` 0/6 → the causal half is void on
Talkie) and the two-opposite-artifacts de-confound lesson are kept in the open on purpose — see `PROVENANCE.md`
and `results/coercion/coercion_FINDINGS.md`. The result that inverted its own hypothesis (the pre-1931 model
keeps the tradition *distinct* rather than channeling it *more*) is reported as found, not as predicted.
