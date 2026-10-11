import numpy as np, pandas as pd, glob
from scipy.stats import beta
pd.set_option("display.width", 220); W = "/path/to/sharc_work"
def cp(k, n): return (beta.ppf(.025, k, n - k + 1) if k else 0.0, beta.ppf(.975, k + 1, n - k) if k < n else 1.0)
def grp(s, e): return "Real" if e == "E2" else ("Concept" if s.startswith("concept") else ("Appearance" if s.startswith("deg:") or s == "quality" else "Prevalence"))
rows = []
for tag in ["convnext5", "resnet50", "dinov2"]:
    R = pd.read_parquet(f"{W}/results/e_v9_{tag}_raw.parquet", columns=["exp", "site", "alpha", "method", "miss", "flag", "n_lab_pos", "n_test_pos"])
    R["g"] = [grp(s, e) for s, e in zip(R.site, R.exp)]; R = R[(R.alpha == 0.05) & (R.g != "Concept")]
    for g, d in R.groupby("g"):
        fl = d[(d.method == "TGT_CP") & d.flag]; un = d[(d.method == "CRC") & ~d.flag]   # flagged: exact-binomial branch; unflagged: source CRC branch
        for name, x in [("flagged -> exact binomial (90% CP)", fl), ("unflagged -> source CRC (expected risk only)", un)]:
            if len(x) == 0: continue
            k, n = int((x.miss > .05).sum()), len(x); lo, hi = cp(k, n)
            rows.append(dict(backbone=tag, group=g, branch=name, runs=n, exceed=k, rate=round(k / n, 4), ci95=f"[{lo:.3f}, {hi:.3f}]",
                             median_test_positives=int(x.n_test_pos.median()) if x.n_test_pos.notna().any() else None))
T = pd.DataFrame(rows); print("== Exceedance P(observed test FNR > 0.05) by gate branch (v9, alpha 0.05)"); print(T.to_string(index=False))
T.to_csv(f"{W}/results/paper_final_v9/TA9_exceedance_by_branch.csv", index=False)
S = pd.read_csv(f"{W}/results/cp_exceedance_simulation.csv"); S["mc_se"] = np.sqrt(S.exceed * (1 - S.exceed) / 20000).round(4)
print("\n== simulation (20,000 draws each; MC standard error)"); print(S.to_string(index=False))
L = pd.concat([pd.read_csv(f) for f in glob.glob(f"{W}/results/lineage_v9_*.csv")])
print("\n== lineage: runs", len(L), "| per-site median pool sizes (units)"); print(L.drop(columns=["seed", "rep"]).groupby("site").median().round(0).astype(int).to_string())
U = pd.read_parquet(f"{W}/data/proc/metadata_rows.parquet")
print("\n== unit check: images per unit by dataset (paired datasets must group eyes)"); print(U.groupby(["ds"]).apply(lambda d: d.groupby("patient").size().value_counts().sort_index().to_dict(), include_groups=False).to_string())
