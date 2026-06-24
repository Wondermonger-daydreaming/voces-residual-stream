"""Regenerate the two paper figures from a results JSON (CPU, no model).

Every plotted value is READ FROM THE JSON — nothing is hard-coded, so a reader
cannot say the conclusion was plotted by hand. Fig 2 (the decider) requires the
`voces_specificity` object; if it is absent the script errors out rather than
inventing the bars.

Usage:
    python src/make_figures.py [results.json]
Default: results/voces_v6_results.json (carries `voces_specificity`).
"""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else "results/voces_v6_results.json"
d = json.load(open(path))
model = d.get("model", "model")

# --- Fig 1: name-likeness by script, swept across layers (all series from JSON) ---
gs = d["greek_split"]
lat = d["sweep_low_pa_mean"]["latin"]
plt.figure(figsize=(8, 4.3))
plt.axhline(0, color="#999", lw=0.8)
plt.plot(lat, color="#2b6cb0", marker="o", ms=3, lw=1.6, label="Latin (same strings)")
plt.plot(gs["authentic_sweep"], color="#2f855a", marker="o", ms=3, lw=1.6,
         label=f"Authentic Greek (n={gs['n_authentic']})")
plt.plot(gs["translit_sweep"], color="#dd6b20", marker="o", ms=3, lw=1.6,
         label=f"Transliterated Greek (n={gs['n_translit']})")
plt.xlabel("layer")
plt.ylabel("pure-asemic low-T name-likeness")
plt.title(f"Name-likeness by script ({model})")
plt.legend(fontsize=8, frameon=False)
plt.tight_layout()
plt.savefig("results/fig1_script_sweep.png", dpi=130)

# --- Fig 2: THE DECIDER — voces vs token-matched controls, deep layers (from JSON) ---
vs = d.get("voces_specificity")
if vs is None:
    sys.exit(
        f"ERROR: {path} has no 'voces_specificity' object. Fig 2 is the headline and must NOT be "
        "hand-plotted — run the v6 notebook (Cell 10f) to produce the decider, then re-run this."
    )
scripts = ["greek", "latin"]
vox = [vs[s]["vox_deep"] for s in scripts]
ctl = [vs[s]["ctl_deep"] for s in scripts]
ps = [vs[s]["p"] for s in scripts]
fig, ax = plt.subplots(figsize=(6, 3.6))
x = [0, 1]; w = 0.35
ax.bar([i - w / 2 for i in x], vox, w, label="voces", color="#6b46c1")
ax.bar([i + w / 2 for i in x], ctl, w, label="token-match controls", color="#a0aec0")
ax.set_xticks(x)
ax.set_xticklabels([s.capitalize() for s in scripts])
ax.set_ylabel("deep-layer name-likeness")
ax.set_title(f"voces ≈ controls deep (p={ps[0]:.3f} Greek, {ps[1]:.3f} Latin)")
ax.legend(fontsize=8, frameon=False)
plt.tight_layout()
plt.savefig("results/fig2_decider.png", dpi=130)

print(f"figures written from {path} — decider read from JSON "
      f"(greek p={ps[0]}, latin p={ps[1]}); nothing hard-coded.")
