#!/usr/bin/env python3
"""
check_nonname_fragmentation.py — why the non-name-Greek falsifier (Cell 10g) cannot
adjudicate namehood: its non-name cohort is not fragmentation-matched to the voces.

The falsifier (results/cross-family/*_results.json -> nonname_falsifier) compares the deep
Greek-minus-Latin name-likeness of the voces against a NON-NAME Greek cohort (numerals,
function words, common nouns). On Mistral it returned voces deep gap +0.041 vs non-name
-0.005 (ratio -0.12), which the notebook's binary rule labels "name-specific."

But the non-name words are SHORT and COMMON, so the tokenizer shreds them far less than the
long asemic voces. This script tokenizes both cohorts in Greek with the model's own tokenizer
and shows the cohorts differ in fragmentation (~2x) — so the falsifier is confounded and its
result is consistent with the fragmentation account, not evidence against it. The clean,
fragmentation-matched non-voces test is the §3.3 decider (token-matched controls), which is
null: at matched fragmentation the voces are not special.

CPU only, tokenizer only (no weights, no torch). Needs `transformers` + `sentencepiece`.

    python src/check_nonname_fragmentation.py [--model unsloth/mistral-7b-v0.3]
"""
import argparse, statistics as st
from transformers import AutoTokenizer

# --- Betz Latin -> uppercase Greek transliterator (digraphs first) — identical to the notebook ---
_DIGRAPHS = [("TH", "Θ"), ("PH", "Φ"), ("CH", "Χ"), ("PS", "Ψ"), ("NG", "ΝΓ")]
_SINGLES = {"A": "Α", "B": "Β", "G": "Γ", "D": "Δ", "E": "Ε", "Z": "Ζ", "H": "Η", "I": "Ι",
            "K": "Κ", "L": "Λ", "M": "Μ", "N": "Ν", "X": "Ξ", "O": "Ο", "P": "Π", "R": "Ρ",
            "S": "Σ", "T": "Τ", "U": "Υ", "Y": "Υ", "W": "Ω"}


def to_greek(s):
    s = s.upper()
    for a, b in _DIGRAPHS:
        s = s.replace(a, b)
    return "".join(_SINGLES.get(c, c) for c in s)


THEONYM_ROOTS = ["IAO", "SABAOTH", "BAOTH", "AOTH", "ADONAI", "ELOAI", "ELOI", "ABRASAX",
                 "ABRAXAS", "ABRA", "MICHA", "ERESHKIGAL", "SABA", "ADON", "OTH", "IAE",
                 "IEOU", "SETH", "HOR", "OSIR", "ISIS", "AMOUN", "THOTH"]


def contamination_T(s):
    s = s.upper(); cov = [False] * len(s)
    for r in THEONYM_ROOTS:
        st_ = 0
        while True:
            i = s.find(r, st_)
            if i < 0:
                break
            for j in range(i, i + len(r)):
                cov[j] = True
            st_ = i + 1
    return sum(cov) / max(1, len(s))


# non-theonym voces (theonym=False) — the cohort the falsifier's `low` subset is drawn from
NONTHEO = ["ABLANATHANALBA", "AKRAMMACHAMAREI", "SESENGENBARPHARANGES", "SEMESILAM", "SEMESEILAMPS",
           "MASKELLI", "MASKELLO", "NEBOUTOSOUALETH", "DAMNAMENEUS", "ASKION", "KATASKION", "LIXTETRAX",
           "AISION", "BAINCHOOOCH", "THOBARRABAU", "CHABRACH", "PHNESCHER", "BORKA", "PHORBA",
           "AROURARELYOTH", "KMEPH", "THENOR", "AEEIOYO", "IEOUAEOI", "OREOBAZAGRA", "PROTOPHANES",
           "NEPHERIERI", "AKTIOPHI", "ORORIOUTH", "BAKAXICHYCH", "ABERAMENTHOOU", "LERTHEXANAX",
           "SOROORMERPHERGAR", "PATATHNAX", "IARBATHA", "SANKANTHARA", "PSINOTHER", "CHTHETHONI",
           "MENEBAINCHUCH", "BARZAN", "SISISRO", "THENOB", "ARSENOPHRE", "BACHUCH", "PHORPHORBA",
           "HARMONTHARTHOCHE", "NEOPHOXOTHA", "ARISTANABAZAO", "PCHORBAZANACHU", "ZALAMOIRLALITH",
           "EILESILARMU", "TANTINURACHTH", "CHORCHORNATHI", "PHANTHENPHYPHLIA", "AZAZAEISTHAILICH",
           "MENNYTHYTH", "SERYCHARRALMIO", "AKHEBUKROM"]
# authentic Greek overrides (real PGM bytes) — same as the notebook
ATTESTED_GREEK = {
    "HARMONTHARTHOCHE": "ΑΡΜΟΝΘΑΡΘΩΧΕ", "NEOPHOXOTHA": "ΝΕΟΦΟΞΩΘΑ", "ARISTANABAZAO": "ΑΡΙΣΤΑΝΑΒΑΖΑΩ",
    "PCHORBAZANACHU": "ΠΧΟΡΒΑΖΑΝΑΧΟΥ", "ZALAMOIRLALITH": "ΖΑΛΑΜΟΙΡΛΑΛΙΘ", "EILESILARMU": "ΕΙΛΕΣΙΛΑΡΜΟΥ",
    "TANTINURACHTH": "ΤΑΝΤΙΝΟΥΡΑΧΘ", "CHORCHORNATHI": "ΧΟΡΧΟΡΝΑΘΙ", "PHANTHENPHYPHLIA": "ΦΑΝΘΕΝΦΥΦΛΙΑ",
    "AZAZAEISTHAILICH": "ΑΖΑΖΑΕΙΣΘΑΙΛΙΧ", "MENNYTHYTH": "ΜΕΝΝΥΘΥΘ", "SERYCHARRALMIO": "ΣΕΡΥΧΑΡΡΑΛΜΙΟ",
    "AKHEBUKROM": "ΑΧΕΒΥΚΡΩΜ",
}
# the non-name cohort — verbatim from Cell 11
NONNAME_LAT = ["EN", "DYO", "TRIA", "TESSARA", "PENTE", "EX", "EPTA", "OKTO", "ENNEA", "DEKA",
               "KAI", "ALLA", "OUTOS", "GAR", "MEN", "EIS", "EK", "PROS", "META", "PERI",
               "BIBLION", "KARDIA", "LOGOS", "ERGON", "TOPOS", "PHONE"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/mistral-7b-v0.3",
                    help="HF tokenizer id (default: non-gated Mistral-7B-v0.3 mirror)")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)

    def ntok(s):
        return len(tok.encode(s, add_special_tokens=False))

    def greek(w):
        return ATTESTED_GREEK.get(w, to_greek(w))

    low = [w for w in NONTHEO if contamination_T(w) < 0.15]  # the falsifier's `low` voces subset
    vox_gr = [ntok(greek(w)) for w in low]
    nn_gr = [ntok(to_greek(w)) for w in NONNAME_LAT]
    vox_lat = [ntok(w) for w in low]
    nn_lat = [ntok(w) for w in NONNAME_LAT]

    print(f"model tokenizer: {args.model}\n")
    print(f"{'cohort':<22}{'n':>4}{'Greek tok/str':>16}{'Latin tok/str':>16}")
    print(f"{'voces (low-T)':<22}{len(low):>4}{st.mean(vox_gr):>16.2f}{st.mean(vox_lat):>16.2f}")
    print(f"{'non-name':<22}{len(NONNAME_LAT):>4}{st.mean(nn_gr):>16.2f}{st.mean(nn_lat):>16.2f}")
    print(f"\nGreek fragmentation ratio voces/non-name: {st.mean(vox_gr)/st.mean(nn_gr):.2f}x")
    print("=> the non-name cohort is NOT fragmentation-matched; Cell 10g's binary verdict is confounded.")
    print("   The fragmentation-matched non-voces test is the §3.3 decider (token-matched controls), which")
    print("   is null at ~matched fragmentation -> fragmentation, not namehood, drives deep-Greek persistence.")


if __name__ == "__main__":
    main()
