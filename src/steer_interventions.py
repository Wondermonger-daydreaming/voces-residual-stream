"""
steer_interventions.py — Layer 6 (H4) of the voces-magicae study.

STATUS: UNRUN SCAFFOLD. No steering result is claimed anywhere in this repository.
Its only valid target is H1's *surface* texture (the early-layer, on-sight recognition),
NOT the H2 deep representation: the study showed the deep voces-specific representation is
indistinguishable from a token-matched control (p=0.224), so steering "toward" it would be
probing a thing we have evidence isn't there. Run it against the H1 surface effect or not at all.

Rewrite of the original plumbing-only steering script to add the three things that
made the original uninterpretable as an *experiment*:

  1. A CONTROL VECTOR. A positive steering result is meaningless unless a
     magnitude-matched control direction (scrambled-derived and/or random) does
     NOT reproduce it. Without this you cannot distinguish "numinosity" from
     "adding any rare-structured direction degrades the model."

  2. NORM-SCALED DOSING. Dose is alpha * (measured residual norm at the layer),
     applied to a UNIT direction — not a raw multiplier on an unnormalized vector.
     This makes doses comparable across directions and layers, which is the whole
     point of having a control arm.

  3. AN EFFECT-MEASUREMENT HARNESS. The science of H4 is the measurement, not the
     hook. Three readouts at every dose:
       - logit-shift : summed probability mass on a curated ritual/divine token
                       set, steered vs baseline, on fixed neutral prompts.
                       Sampling-free, cheap, the primary causal readout.
       - lexicon     : sacral/archaic word density in free generation.
       - coherence   : a parse/degeneracy guard so a format collapse is never
                       scored as a register effect.

The hook mechanics (modify output[0], return the tuple, broadcast a 1-D vector,
try/finally teardown) are kept from the original — they were correct.

Subject model is open-weights (Qwen/Llama). This never runs "on Claude" — see the
skill's section 0.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ----------------------------------------------------------------------------- #
# 1. The steerer: norm-scaled, unit-direction, control-capable
# ----------------------------------------------------------------------------- #
class ActivationSteerer:
    """Injects `alpha * resid_norm * unit_direction` at one decoder layer.

    The direction is stored as a UNIT vector. The dose actually added is
    `alpha * self.resid_norm`, so `alpha` is a fraction of the layer's typical
    residual norm and is comparable across any directions you hand it (steer,
    scrambled-control, random-control).
    """

    def __init__(self, model, target_layer: int, unit_direction: torch.Tensor,
                 resid_norm: float):
        self.model = model
        self.target_layer = target_layer
        d = unit_direction.to(model.device).to(model.dtype)
        self.unit_direction = d / (d.norm() + 1e-8)   # enforce unit length
        self.resid_norm = float(resid_norm)
        self.alpha = 0.0
        self.hook_handle = None

    def _decoder_layers(self):
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers          # Qwen2/Llama3/Mistral
        raise AttributeError("Unsupported architecture: no model.model.layers")

    def _hook(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        # hidden: (batch, seq, d_model); unit_direction broadcasts over batch+seq
        delta = (self.alpha * self.resid_norm) * self.unit_direction
        steered = hidden + delta
        if isinstance(output, tuple):
            return (steered,) + output[1:]
        return steered

    @contextmanager
    def steer(self, alpha: float = 0.0):
        """Register the hook at `alpha`, yield, and always tear down."""
        self.alpha = alpha
        layer = self._decoder_layers()[self.target_layer]
        self.hook_handle = layer.register_forward_hook(self._hook)
        try:
            yield self
        finally:
            if self.hook_handle is not None:
                self.hook_handle.remove()
                self.hook_handle = None


# ----------------------------------------------------------------------------- #
# 2. Calibration: measure the residual norm at the target layer
# ----------------------------------------------------------------------------- #
@torch.no_grad()
def measure_residual_norm(model, tokenizer, target_layer: int,
                          calib_prompts: list[str]) -> float:
    """Mean L2 norm of the target layer's output hidden states over real tokens.

    This is the scale `alpha` is expressed as a fraction of. Measured on neutral
    text so the dose unit is a property of the model, not of the steering target.
    """
    captured = {}

    def cap(module, inputs, output):
        captured["h"] = output[0] if isinstance(output, tuple) else output

    layer = (model.model.layers[target_layer]
             if hasattr(model, "model") and hasattr(model.model, "layers")
             else None)
    if layer is None:
        raise AttributeError("Unsupported architecture: no model.model.layers")

    handle = layer.register_forward_hook(cap)
    norms = []
    try:
        for p in calib_prompts:
            enc = tokenizer(p, return_tensors="pt").to(model.device)
            model(**enc)
            h = captured["h"][0]                       # (seq, d_model)
            norms.append(h.norm(dim=-1).mean().item()) # mean over tokens
    finally:
        handle.remove()
    return float(sum(norms) / len(norms))


# ----------------------------------------------------------------------------- #
# 3. Direction construction (from precomputed cohort means)
# ----------------------------------------------------------------------------- #
def diff_direction(mean_pos: torch.Tensor, mean_neg: torch.Tensor) -> torch.Tensor:
    """Unit difference-of-means direction, e.g. (v_attested - v_token_match)."""
    v = mean_pos - mean_neg
    return v / (v.norm() + 1e-8)


def random_direction(d_model: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(d_model, generator=g)
    return v / v.norm()


# ----------------------------------------------------------------------------- #
# 4. Effect-measurement harness
# ----------------------------------------------------------------------------- #
# A small sacral/archaic/invocatory lexicon. Replace/extend with a vetted list;
# kept short here so the file is self-contained. These are the words whose
# *promotion* would indicate a shift toward liturgical/numinous register.
RITUAL_WORDS = [
    "holy", "sacred", "divine", "eternal", "celestial", "heavens", "almighty",
    "thou", "thee", "thy", "behold", "invoke", "summon", "spirit", "angel",
    "lord", "god", "prayer", "blessed", "ritual", "incantation", "temple",
    "ancient", "mystery", "veil", "throne", "glory", "radiant", "unto",
]


@dataclass
class EffectHarness:
    model: object
    tokenizer: object
    neutral_prompts: list[str]
    ritual_words: list[str] = field(default_factory=lambda: RITUAL_WORDS)

    def __post_init__(self):
        # Build the target token-id set: first BPE token of each ritual word,
        # with and without a leading space (BPE is whitespace-sensitive).
        ids = set()
        for w in self.ritual_words:
            for form in (w, " " + w, w.capitalize(), " " + w.capitalize()):
                toks = self.tokenizer(form, add_special_tokens=False).input_ids
                if toks:
                    ids.add(toks[0])
        self.target_ids = torch.tensor(sorted(ids))

    # -- primary readout: probability mass on ritual tokens at the next position
    @torch.no_grad()
    def logit_shift(self, steerer: ActivationSteerer | None, alpha: float) -> float:
        """Mean next-token probability mass on the ritual token set across the
        neutral prompts. Compare steered vs baseline (alpha=0) externally.
        Sampling-free: reads the distribution directly, so it's low-variance."""
        masses = []
        ctx = steerer.steer(alpha) if steerer is not None else _nullctx()
        with ctx:
            for p in self.neutral_prompts:
                enc = self.tokenizer(p, return_tensors="pt").to(self.model.device)
                logits = self.model(**enc).logits[0, -1]        # (vocab,)
                probs = torch.softmax(logits.float(), dim=-1)
                masses.append(probs[self.target_ids.to(probs.device)].sum().item())
        return float(sum(masses) / len(masses))

    # -- magnitude readout: KL(steered || baseline) of the next-token distribution
    @torch.no_grad()
    def kl_shift(self, steerer: "ActivationSteerer", alpha: float) -> float:
        """Mean KL(P_steered || P_baseline) over the next-token distributions on the
        neutral prompts. This is the *magnitude* of the causal push, used as a
        READOUT — not as a dose constraint. (Bounding KL and then measuring it is
        circular; see the spec's benched 'KL bound'.) Reported alongside logit-shift
        so you can tell 'moved a lot, toward ritual tokens' from 'moved a lot, into
        salad' when read with the coherence metric."""
        kls = []
        for p in self.neutral_prompts:
            enc = self.tokenizer(p, return_tensors="pt").to(self.model.device)
            base_logits = self.model(**enc).logits[0, -1].float()
            with steerer.steer(alpha):
                steer_logits = self.model(**enc).logits[0, -1].float()
            logp_s = torch.log_softmax(steer_logits, dim=-1)
            logp_b = torch.log_softmax(base_logits, dim=-1)
            p_s = logp_s.exp()
            kls.append((p_s * (logp_s - logp_b)).sum().item())   # KL(steer||base)
        return float(sum(kls) / len(kls))
    @torch.no_grad()
    def lexicon_density(self, steerer: ActivationSteerer | None, alpha: float,
                        max_new_tokens: int = 40) -> tuple[float, float]:
        """Returns (ritual_word_density, coherence). Coherence is the distinct-
        token ratio of the continuation — a cheap degeneracy guard so a collapse
        into repetition isn't scored as register change."""
        ritual_set = {w.lower() for w in self.ritual_words}
        densities, coherences = [], []
        ctx = steerer.steer(alpha) if steerer is not None else _nullctx()
        with ctx:
            for p in self.neutral_prompts:
                enc = self.tokenizer(p, return_tensors="pt").to(self.model.device)
                out = self.model.generate(**enc, max_new_tokens=max_new_tokens,
                                           do_sample=False)
                gen_ids = out[0][enc.input_ids.shape[1]:]
                text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                words = [w.strip(".,;:!?\"'()").lower() for w in text.split()]
                words = [w for w in words if w]
                if not words:
                    densities.append(0.0); coherences.append(0.0); continue
                densities.append(sum(w in ritual_set for w in words) / len(words))
                coherences.append(len(set(words)) / len(words))  # distinct ratio
        n = len(densities)
        return float(sum(densities) / n), float(sum(coherences) / n)


class _nullctx:
    def __enter__(self): return None
    def __exit__(self, *a): return False


# ----------------------------------------------------------------------------- #
# 5. Dose sweep across directions (steer + controls)
# ----------------------------------------------------------------------------- #
@torch.no_grad()
def run_dose_sweep(model, tokenizer, target_layer: int, resid_norm: float,
                   directions: dict[str, torch.Tensor],
                   harness: EffectHarness,
                   alphas=(0.0, 0.25, 0.4, 0.6, 0.8)) -> dict:
    """For each named direction and dose, record logit-shift, lexicon density,
    and coherence. The baseline is alpha=0 (any direction); the interesting
    quantity is (steer - control) at matched alpha."""
    results = {"resid_norm": resid_norm, "target_layer": target_layer,
               "alphas": list(alphas), "directions": {}}

    # baseline once (direction-independent at alpha=0)
    base_shift = harness.logit_shift(None, 0.0)
    base_dens, base_coh = harness.lexicon_density(None, 0.0)
    results["baseline"] = {"logit_mass": base_shift,
                           "lexicon_density": base_dens, "coherence": base_coh}

    for name, vec in directions.items():
        steerer = ActivationSteerer(model, target_layer, vec, resid_norm)
        rows = []
        for a in alphas:
            shift = harness.logit_shift(steerer, a)
            dens, coh = harness.lexicon_density(steerer, a)
            kl = harness.kl_shift(steerer, a) if a > 0 else 0.0
            rows.append({"alpha": a,
                         "logit_mass": shift,
                         "logit_mass_delta": shift - base_shift,
                         "kl_steered_vs_base": kl,
                         "lexicon_density": dens,
                         "coherence": coh})
        results["directions"][name] = rows
    return results


# ----------------------------------------------------------------------------- #
# 6. Runnable verification (mock vectors → plumbing test)
# ----------------------------------------------------------------------------- #
if __name__ == "__main__":
    MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"   # plumbing only; run science at >=7B
    TARGET_LAYER = 12

    print("loading model ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    d_model = model.config.hidden_size

    calib = ["The weather today is", "She walked to the store and",
             "In the meeting we discussed", "The recipe calls for"]
    neutral = ["The inscription on the limestone block reads:",
               "He opened the notebook and wrote:",
               "The label on the jar said:"]

    # 1. calibrate the dose unit
    resid_norm = measure_residual_norm(model, tokenizer, TARGET_LAYER, calib)
    print(f"residual norm @ layer {TARGET_LAYER}: {resid_norm:.1f}")

    # 2. directions. In practice load cohort means from the extraction step:
    #      v_steer   = diff_direction(mean_attested, mean_token_match)
    #      v_control = diff_direction(mean_scrambled, mean_token_match)
    #    Here: mock vectors so the harness/teardown can be exercised on CPU/GPU.
    directions = {
        "steer_mock":   random_direction(d_model, seed=1),  # stand-in for v_steer
        "control_scrambled_mock": random_direction(d_model, seed=2),
        "control_random": random_direction(d_model, seed=3),
    }

    # 3. measure
    harness = EffectHarness(model, tokenizer, neutral)
    out = run_dose_sweep(model, tokenizer, TARGET_LAYER, resid_norm,
                         directions, harness, alphas=(0.0, 0.25, 0.5))
    print(json.dumps(out, indent=2))

    # 4. teardown audit: identical generation before and after a steer block
    enc = tokenizer(neutral[0], return_tensors="pt").to(model.device)
    pre = model.generate(**enc, max_new_tokens=12, do_sample=False)
    s = ActivationSteerer(model, TARGET_LAYER, directions["steer_mock"], resid_norm)
    with s.steer(0.5):
        _ = model.generate(**enc, max_new_tokens=12, do_sample=False)
    post = model.generate(**enc, max_new_tokens=12, do_sample=False)
    print("hook cleanly removed (pre == post):", torch.equal(pre, post))
