"""Regenerate the two paper figures from results/voces_v5_results.json (CPU, no model)."""
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
d = json.load(open("results/voces_v5_results.json"))
gs = d["greek_split"]; lat = d["sweep_low_pa_mean"]["latin"]
plt.figure(figsize=(8,4.3)); plt.axhline(0,color="#999",lw=0.8)
plt.plot(range(len(lat)),lat,color="#2b6cb0",marker="o",ms=3,lw=1.6,label="Latin (same strings)")
plt.plot(range(len(gs["authentic_sweep"])),gs["authentic_sweep"],color="#2f855a",marker="o",ms=3,lw=1.6,label="Authentic Greek (n=13)")
plt.plot(range(len(gs["translit_sweep"])),gs["translit_sweep"],color="#dd6b20",marker="o",ms=3,lw=1.6,label="Transliterated Greek (n=39)")
plt.xlabel("layer"); plt.ylabel("pure-asemic low-T name-likeness")
plt.title("Name-likeness by script (Qwen2.5-3B)"); plt.legend(fontsize=8,frameon=False)
plt.tight_layout(); plt.savefig("results/fig1_script_sweep.png",dpi=130)
fig,ax=plt.subplots(figsize=(6,3.6)); x=[0,1]; w=0.35
ax.bar([i-w/2 for i in x],[0.027,0.004],w,label="voces",color="#6b46c1")
ax.bar([i+w/2 for i in x],[0.020,0.010],w,label="token-match controls",color="#a0aec0")
ax.set_xticks(x); ax.set_xticklabels(["Greek","Latin"]); ax.set_ylabel("deep-layer name-likeness")
ax.set_title("voces \u2248 controls deep (p=0.224 Greek, 0.886 Latin)"); ax.legend(fontsize=8,frameon=False)
plt.tight_layout(); plt.savefig("results/fig2_decider.png",dpi=130)
print("figures written to results/")
