"""
voces_coercion_v3_APPEND_cells.py — the v3 repairs, as append-cells to the v2 notebook.

These were run live on Colab (T4) on 2026-06-25 AFTER voces_coercion_v2.ipynb had loaded the model and
computed Ap/Ag/Ah/Ab/Apol/Arud, d_coerce, d_refuse, cos, peak, etc. They REUSE the in-memory state — no
re-inference for v3-A (pure numpy on cached activations); v3-B reuses the loaded model for generation.

To reproduce: run voces_coercion_v2.ipynb through Cell 12, then paste these two cells and run them.
Results are recorded in coercion_FINDINGS.md §3b (v3-A) / §3c (v3-B) and voces_coercion_v2_results.json
(v3a_* / v3b_* fields).

Provenance note: v3-A is the pre-specified #1 repair (length de-confound); v3-B is the repaired causal
(steering) test. A future v4 should fold a length-MATCHED corpus + length-orthogonalized steering into a
proper build_coercion_v4_notebook.py generator rather than append-cells.
"""

# ===================== v3-A — length de-confound (pure numpy; NO GPU) =====================
V3A = r'''
# v3-A — length de-confound (pure numpy; reuses cached activations, NO GPU)
import numpy as np
from scipy.stats import linregress, spearmanr, pearsonr
Lp = peak
toklen_p = np.array([len(tok(templ(t)).input_ids) for _,t in plain_rows]).astype(float)

# (a) PARTIAL test — keep original d_coerce; does rung predict projection AFTER removing length?
proj_p = Ap[:,Lp,:]@d_coerce[Lp]
fit = linregress(toklen_p, proj_p)
res_p = proj_p - (fit.slope*toklen_p + fit.intercept)          # length-residualized projection
lr_part = linregress(rung_p, res_p); rho_part = spearmanr(rung_p, res_p).correlation
print(f"[partial]  length~proj r={pearsonr(toklen_p,proj_p)[0]:+.2f}")
print(f"           rung~(length-residualized proj): rho={rho_part:+.2f}  slope={lr_part.slope:+.3f}  p={lr_part.pvalue:.2e}")
print(f"           residualized by-rung: {[round(float(res_p[rung_p==r].mean()),2) for r in RUNGS]}")

# (b) REBUILD d_coerce on length-residualized activations; does the cos-adjacency survive?
def residualize(A, x):
    xc=(x-x.mean())/(x.std()+1e-8)
    beta=(A*xc[:,None,None]).mean(0)/((xc**2).mean())          # per-(layer,dim) OLS slope
    return A - xc[:,None,None]*beta[None]
def unit(v): n=np.linalg.norm(v); return v/n if n>0 else v
Ap_r = residualize(Ap, toklen_p)
d_coerce_r = np.stack([unit((Ap_r[hi_p].mean(0)-Ap_r[lo_p].mean(0))[l]) for l in range(L)])
cos_r = float(d_coerce_r[Lp]@d_refuse[Lp])
proj_pr = Ap_r[:,Lp,:]@d_coerce_r[Lp]
lr_r = linregress(rung_p, proj_pr); rho_r = spearmanr([0,1,2,3,4],[proj_pr[rung_p==r].mean() for r in RUNGS]).correlation
print(f"[delengthed] cos(d_coerce_r, d_refuse) @ L{Lp} = {cos_r:+.3f}   (was {cos[Lp]:+.3f})")
print(f"             F4 ladder rho={rho_r:+.2f}  slope={lr_r.slope:+.3f}  p={lr_r.pvalue:.2e}  | new proj~len r={pearsonr(toklen_p,proj_pr)[0]:+.2f}")
print(f"             by-rung: {[round(float(proj_pr[rung_p==r].mean()),2) for r in RUNGS]}")
print("NOTE: (b)'s ladder is mildly circular (direction rebuilt on residualized data); (a) is the clean significance test.")
'''

# ===================== v3-B — repaired steering (fine sweep + orthogonalized) =====================
V3B = r'''
# v3-B — repaired steering: fine coefficient sweep + orthogonalized (coercion ⊥ refusal) component
import numpy as np, torch
from scipy.stats import linregress
Lp=peak
u_co = d_coerce[Lp]/np.linalg.norm(d_coerce[Lp])
u_re = d_refuse[Lp]/np.linalg.norm(d_refuse[Lp])
d_perp = u_co - (u_co@u_re)*u_re; d_perp = d_perp/np.linalg.norm(d_perp)   # coercion-specific (refusal removed)
typ = float(np.linalg.norm(Ap[:,Lp,:],axis=1).mean())
block = model.model.layers[max(0,Lp-1)]; stt={"a":0.0,"v":None}
def hook(m,i,o):
    if stt["a"]==0.0 or stt["v"] is None: return o
    h=o[0] if isinstance(o,tuple) else o; h=h+stt["a"]*typ*stt["v"]
    return (h,)+o[1:] if isinstance(o,tuple) else h
hd=block.register_forward_hook(hook)
@torch.no_grad()
def gen(prompt):
    enc=tok(templ(prompt),return_tensors="pt",add_special_tokens=False).to(model.device)
    out=model.generate(**enc,max_new_tokens=MAX_NEW,do_sample=False,pad_token_id=tok.pad_token_id)
    return tok.decode(out[0,enc["input_ids"].shape[1]:],skip_special_tokens=True)
def sweep(vec, coeffs):
    stt["v"]=torch.tensor(vec,dtype=model.dtype,device=model.device); c={}
    for a in coeffs:
        stt["a"]=a; c[a]=sum(is_refusal(gen(p)) for p in BORDERLINE)/len(BORDERLINE)
    stt["a"]=0.0; return c
COEF=[-0.3,-0.2,-0.1,-0.05,0.0,0.05,0.1,0.2,0.3]
curve_co=sweep(u_co, COEF); curve_pp=sweep(d_perp, COEF); hd.remove()
lin=[c for c in COEF if abs(c)<=0.1]
slope_co=linregress(lin,[curve_co[c] for c in lin]).slope
slope_pp=linregress(lin,[curve_pp[c] for c in lin]).slope
print("coercion-dir   :", {f'{c:+.2f}':round(curve_co[c],2) for c in COEF})
print("   linear-regime slope (|a|<=0.1):", round(slope_co,3))
print("orthogonalized :", {f'{c:+.2f}':round(curve_pp[c],2) for c in COEF})
print("   linear-regime slope (|a|<=0.1):", round(slope_pp,3))
print(f"cos(d_perp,d_refuse)={float(d_perp@u_re):+.3f}")
'''

if __name__ == "__main__":
    print("These are append-cells; paste V3A then V3B into the loaded v2 notebook. See module docstring.")
