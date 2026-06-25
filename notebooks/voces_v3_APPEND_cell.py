# ===================================================================================================
# VOCES FALSIFIER v3 — APPEND CELL.  Paste this as ONE new cell at the BOTTOM of a notebook whose
# model is ALREADY LOADED (e.g. the v2 Gemma/Mistral run that just finished). It REUSES the in-memory
# `model`, `tok`, `N_LAYERS`, `DEV` — NO re-download, NO re-load. Run it; it saves + downloads a JSON.
#
# Separates the v2 "lexicality" bundle: FRAGMENTATION / FAMILIARITY / MEANING / NAMEHOOD, with
# multi-seed asemic cohorts, per-layer profiles, absolute Greek name-likeness, CI-based verdicts.
# ===================================================================================================
import numpy as np, random, json, time, torch
assert 'model' in dir() and 'tok' in dir(), "No loaded model/tok in this kernel — run this in a notebook that already loaded one."
N_LAYERS = model.config.num_hidden_layers
DEV = next(model.parameters()).device
MODEL_NAME = getattr(model.config, "_name_or_path", "loaded-model")
N_SEEDS = 5
VERSION = "falsifier-v3-separation_" + str(MODEL_NAME).split("/")[-1]
print("REUSING loaded model:", MODEL_NAME, "| layers:", N_LAYERS, "| device:", DEV)

# --- transliterator + harness (re-defined defensively; identical to the cross-family notebook) ---
_DIGRAPHS=[("TH","Θ"),("PH","Φ"),("CH","Χ"),("PS","Ψ"),("NG","ΝΓ")]
_SINGLES={"A":"Α","B":"Β","G":"Γ","D":"Δ","E":"Ε","Z":"Ζ","H":"Η","I":"Ι","K":"Κ","L":"Λ","M":"Μ",
          "N":"Ν","X":"Ξ","O":"Ο","P":"Π","R":"Ρ","S":"Σ","T":"Τ","U":"Υ","Y":"Υ","W":"Ω"}
def to_greek(s):
    s=s.upper()
    for a,b in _DIGRAPHS: s=s.replace(a,b)
    return "".join(_SINGLES.get(c,c) for c in s)
def ngreek_tok(l): return len(tok.encode(to_greek(l), add_special_tokens=False))
FRAME_PRE="The string "; FRAME_POST=" is written on the page."
@torch.no_grad()
def extract_reps(s):
    pre=len(tok(FRAME_PRE, add_special_tokens=True).input_ids)
    full=len(tok(FRAME_PRE+s, add_special_tokens=True).input_ids)
    ids=tok(FRAME_PRE+s+FRAME_POST, return_tensors="pt", add_special_tokens=True).input_ids.to(DEV)
    end=min(full, ids.shape[1]); start=pre if pre<end else max(0,end-1)
    hs=model(ids).hidden_states
    return np.stack([h[0,start:end,:].float().mean(0).cpu().numpy() for h in hs])
def build_matrix(strings): return np.stack([extract_reps(s) for s in strings])
@torch.no_grad()
def mean_surprisal(greek_str):
    ids=tok(FRAME_PRE+greek_str+FRAME_POST, return_tensors="pt", add_special_tokens=True).input_ids.to(DEV)
    pre=len(tok(FRAME_PRE, add_special_tokens=True).input_ids)
    full=len(tok(FRAME_PRE+greek_str, add_special_tokens=True).input_ids)
    logits=model(ids).logits[0]; logp=torch.log_softmax(logits[:-1].float(),-1); tgt=ids[0,1:]
    nll=-logp[torch.arange(tgt.shape[0]), tgt]
    span=[i-1 for i in range(pre, min(full, ids.shape[1]))]
    return float(nll[span].mean().cpu()) if span else float('nan')

# --- cohorts ---
NAMES_LAT=list(dict.fromkeys(["AGAMEMNON","ODYSSEUS","ACHILLEUS","PENELOPE","ALEXANDROS","SOKRATES","ARISTOTELES",
 "HERAKLES","PERSEPHONE","APHRODITE","POSEIDON","ARTEMIS","DIONYSOS","ASKLEPIOS","HEPHAISTOS","KLEOPATRA","MENELAOS",
 "AGATHON","THEMISTOKLES","PERIKLES","DEMOSTHENES","EURIPIDES","SOPHOKLES","AISCHYLOS","HERODOTOS","THOUKYDIDES",
 "PYTHAGORAS","ANAXAGORAS","EMPEDOKLES","PARMENIDES","ANTIGONE"]))
_CONS=list("BGDZKLMNPRSTFCH"); _VOW=list("AEIOUY")
VOCES_LAT=["ABLANATHANALBA","AKRAMMACHAMAREI","SESENGENBARPHARANGES","SEMESILAM","SEMESEILAMPS","MASKELLI","MASKELLO",
 "NEBOUTOSOUALETH","DAMNAMENEUS","ASKION","KATASKION","LIXTETRAX","AISION","BAINCHOOOCH","THOBARRABAU","CHABRACH",
 "PHNESCHER","BORKA","PHORBA","AROURARELYOTH","KMEPH","THENOR","AEEIOYO","IEOUAEOI","OREOBAZAGRA","PROTOPHANES",
 "NEPHERIERI","AKTIOPHI","ORORIOUTH","BAKAXICHYCH","ABERAMENTHOOU","LERTHEXANAX","SOROORMERPHERGAR","PATATHNAX",
 "IARBATHA","SANKANTHARA","PSINOTHER","CHTHETHONI","MENEBAINCHUCH","BARZAN","SISISRO","THENOB","ARSENOPHRE","BACHUCH",
 "PHORPHORBA","HARMONTHARTHOCHE","NEOPHOXOTHA","ARISTANABAZAO","PCHORBAZANACHU","ZALAMOIRLALITH","EILESILARMU",
 "TANTINURACHTH","CHORCHORNATHI","PHANTHENPHYPHLIA","AZAZAEISTHAILICH","MENNYTHYTH","SERYCHARRALMIO","AKHEBUKROM"]
TROOTS=["IAO","SABAOTH","BAOTH","AOTH","ADONAI","ELOAI","ELOI","ABRASAX","ABRAXAS","ABRA","MICHA","ERESHKIGAL","SABA",
 "ADON","OTH","IAE","IEOU","SETH","HOR","OSIR","ISIS","AMOUN","THOTH"]
def T(s):
    s=s.upper(); cov=[False]*len(s)
    for r in TROOTS:
        st=0
        while True:
            i=s.find(r,st)
            if i<0: break
            for j in range(i,i+len(r)): cov[j]=True
            st=i+1
    return sum(cov)/max(1,len(s))
VOCES=[v for v in VOCES_LAT if T(v)<0.15]
TARGET=int(round(np.median([ngreek_tok(v) for v in VOCES])))
REAL_GREEK=list(dict.fromkeys(["PHILOSOPHIA","DEMOKRATIA","ANTHROPOLOGIA","BIBLIOTHEKE","KATASTROPHE","METAMORPHOSIS",
 "OIKONOMIA","GEOGRAPHIA","ASTRONOMIA","MATHEMATIKE","THEOLOGIA","PSYCHOLOGIA","TECHNOLOGIA","ENKYKLOPAIDEIA",
 "ARISTOKRATIA","KOSMOLOGIA","SYMPHONIA","EUCHARISTIA","DIALEKTIKE","ETYMOLOGIA","GLAUKOPIS","RHODODAKTYLOS",
 "ERIGDOUPOS","KUDOIMOS","AMPHIGUEEIS","ENOSICHTHON","NEPHELEGERETA","ARGEIPHONTES","POLUPHLOISBOS","LEUKOLENOS",
 "BOOPIS","KHALKEOPHONOS","EUKNEMIDES","PODARKES","KORUTHAIOLOS","OBRIMOPATRE","TANUPEPLOS","KALLIPAREOS","MELIEDES",
 "PERIKALLES","AMPHIELISSA","EUPLOKAMOS","PHILOMMEIDES","TERPSIMBROTOS","CHRUSOTHRONOS","BATHUKOLPOS","AIGIOCHOS",
 "ERIBROMOS","PALAIONTOLOGIA","HERMENEUTIKE","PARTHENOGENESIS"]))
pool=[w for w in REAL_GREEK if abs(ngreek_tok(w)-TARGET)<=2]
surp={w: mean_surprisal(to_greek(w)) for w in pool}
med=float(np.median(list(surp.values())))
LEX_FAM=sorted([w for w in pool if surp[w]<med], key=lambda w:surp[w])
LEX_UNF=sorted([w for w in pool if surp[w]>=med], key=lambda w:-surp[w])
def gen_asemic(t,n,seed,tol=1,mt=40000):
    rng=random.Random(seed); out=[]; k=0
    while len(out)<n and k<mt:
        k+=1; L=rng.randint(6,16); s="".join(rng.choice(_CONS+_VOW) for _ in range(L))
        if abs(ngreek_tok(s)-t)<=tol: out.append(s)
    return out
def gen_rand(n,seed):
    rng=random.Random(seed+999); return ["".join(rng.choice(_CONS) for _ in range(rng.randint(5,11))) for _ in range(n)]
ASEM={s:gen_asemic(TARGET,28,s) for s in range(N_SEEDS)}; RAND={s:gen_rand(30,s) for s in range(N_SEEDS)}
LOWFRAG=["EN","DYO","TRIA","TESSARA","PENTE","EX","EPTA","OKTO","ENNEA","DEKA","KAI","ALLA","OUTOS","GAR","MEN","EIS",
 "EK","PROS","META","PERI","BIBLION","KARDIA","LOGOS","ERGON","TOPOS","PHONE"]
print(f"TARGET={TARGET} Greek tok | voces n={len(VOCES)} | lex_fam n={len(LEX_FAM)} lex_unf n={len(LEX_UNF)} | surp split {med:.2f} nats")

# --- extraction (reuses the loaded model) ---
t0=time.time()
def reps_both(L): return {"latin":build_matrix(L),"greek":build_matrix([to_greek(x) for x in L])}
DET={k:reps_both(v) for k,v in {"name":NAMES_LAT,"voces":VOCES,"lex_familiar":LEX_FAM,"lex_unfamiliar":LEX_UNF,"lowfrag":LOWFRAG}.items()}
AR={s:reps_both(ASEM[s]) for s in range(N_SEEDS)}; RR={s:reps_both(RAND[s]) for s in range(N_SEEDS)}
print("extraction done in", round(time.time()-t0,1),"s")

# --- name-likeness, deep gaps, factorial ---
DEEP=range(int(0.5*N_LAYERS), N_LAYERS+1)
def _n(x): return x/(np.linalg.norm(x)+1e-9)
def prof(A,script,seed):
    nameC=DET["name"][script]; randC=RR[seed][script]; out=[]
    for L in range(N_LAYERS+1):
        nc=_n(nameC[:,L,:].mean(0)); rc=_n(randC[:,L,:].mean(0))
        out.append(np.array([_n(A[i,L,:])@nc-_n(A[i,L,:])@rc for i in range(A.shape[0])]))
    return out
def dgap(reps,seed):
    g=prof(reps["greek"],"greek",seed); l=prof(reps["latin"],"latin",seed); n=reps["greek"].shape[0]
    return np.array([np.mean([g[L][i]-l[L][i] for L in DEEP]) for i in range(n)])
def absg(reps,seed):
    g=prof(reps["greek"],"greek",seed); n=reps["greek"].shape[0]
    return np.array([np.mean([g[L][i] for L in DEEP]) for i in range(n)])
def boot(ps,B=2000,sd=0):
    rng=np.random.RandomState(sd); n=len(ps); m=[ps[rng.randint(0,n,n)].mean() for _ in range(B)]
    return float(ps.mean()),float(np.percentile(m,2.5)),float(np.percentile(m,97.5))
asem_pool=np.concatenate([dgap(AR[s],s) for s in range(N_SEEDS)])
asem_perseed=[float(dgap(AR[s],s).mean()) for s in range(N_SEEDS)]
PS={"voces":dgap(DET["voces"],0),"asemic":asem_pool,"lex_familiar":dgap(DET["lex_familiar"],0),
    "lex_unfamiliar":dgap(DET["lex_unfamiliar"],0),"lowfrag":dgap(DET["lowfrag"],0)}
def dci(a,b,B=2000,sd=1):
    rng=np.random.RandomState(sd)
    d=[PS[a][rng.randint(0,len(PS[a]),len(PS[a]))].mean()-PS[b][rng.randint(0,len(PS[b]),len(PS[b]))].mean() for _ in range(B)]
    return float(PS[a].mean()-PS[b].mean()),float(np.percentile(d,2.5)),float(np.percentile(d,97.5))
def verdict(name,a,b):
    d,lo,hi=dci(a,b); sig=(lo>0 or hi<0)
    tag=("SIGNIFICANT "+("+" if d>0 else "-")) if sig else "n.s. (CI spans 0 — NOT evidence for the null)"
    print(f"[{name:14s}] {a} - {b} = {d:+.4f} [{lo:+.4f},{hi:+.4f}]  {tag}")
    return dict(a=a,b=b,diff=d,ci_lo=lo,ci_hi=hi,significant=bool(sig))
print("="*78); print("FACTORIAL — model:",MODEL_NAME); print("="*78)
R={"FRAGMENTATION":verdict("FRAGMENTATION","asemic","lowfrag"),
   "FAMILIARITY":verdict("FAMILIARITY","lex_unfamiliar","lex_familiar"),
   "MEANING":verdict("MEANING","asemic","lex_unfamiliar"),
   "NAMEHOOD":verdict("NAMEHOOD","voces","asemic")}
print("\nread: FAMILIARITY sig & MEANING n.s. -> 2nd factor is FAMILIARITY not meaning |",
      "MEANING sig -> meaning itself matters | asemic per-seed sd =", round(float(np.std(asem_perseed)),4))

# --- save ---
def fs(c): a=np.array([ngreek_tok(x) for x in c]); return dict(mean=float(a.mean()),median=float(np.median(a)),n=len(a))
out=dict(model=str(MODEL_NAME),version=VERSION,n_layers=int(N_LAYERS),n_seeds=N_SEEDS,target_greek_tok=TARGET,
  cohort_fragmentation={k:fs(c) for k,c in [("voces",VOCES),("asemic_seed0",ASEM[0]),("lex_familiar",LEX_FAM),("lex_unfamiliar",LEX_UNF),("lowfrag",LOWFRAG)]},
  surprisal_split=dict(median_nats=med,familiar_mean=float(np.mean([surp[w] for w in LEX_FAM])),unfamiliar_mean=float(np.mean([surp[w] for w in LEX_UNF]))),
  deep_gaps={k:dict(zip(("mean","ci_lo","ci_hi"),boot(v))) for k,v in PS.items()},
  asemic_perseed_deepgap=asem_perseed,
  absolute_greek_deep={k:float(absg(DET[k],0).mean()) for k in ["voces","lex_familiar","lex_unfamiliar","lowfrag"]},
  factorial=R,
  layer_profile_greek_minus_latin={k:[float(np.mean(prof(DET[k]["greek"],"greek",0)[L]-prof(DET[k]["latin"],"latin",0)[L])) for L in range(N_LAYERS+1)] for k in ["voces","lex_familiar","lex_unfamiliar","lowfrag"]})
fn=f"voces_{VERSION}_results.json"; json.dump(out,open(fn,"w"),indent=2)
print("\nsaved:",fn)
try:
    from google.colab import files; files.download(fn)
except Exception: pass
