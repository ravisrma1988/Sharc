import numpy as np, pandas as pd
from scipy import stats
# P0 independent re-verification of authoritative v9 tables from the frozen raw Parquets (fresh code, not the analysis scripts).
W = "/path/to/sharc_work/results"; P = W + "/paper_final_v9/"; BB = {"convnext5": "ConvNeXt-T", "resnet50": "ResNet-50", "dinov2": "DINOv2-S (frozen)"}
def grp(site, exp):
    if exp == "E2": return "Real sites"
    if site.startswith("concept"): return "Concept shift (image-invisible)"
    return "Appearance shift" if site.startswith("deg:") or site == "quality" else "Prevalence shift"
T2 = pd.read_csv(P + "TA2_main_summary_seedCI.csv"); T9 = pd.read_csv(P + "TA9_exceedance_by_branch.csv"); out, bad = [], 0
for tag, name in BB.items():
    R = pd.read_parquet(f"{W}/e_v9_{tag}_raw.parquet"); R["g"] = [grp(s, e) for s, e in zip(R.site, R.exp)]
    k = ["exp", "seed", "rep", "site", "alpha"]
    # 1) the gated policy rows (G4_CRC) must equal the analysis-constructed FINAL_CRC (flag ? TGT_CRC : CRC), run by run
    g4 = R[R.method == "G4_CRC"].set_index(k).sort_index(); c = R[R.method == "CRC"].set_index(k).reindex(g4.index); t = R[R.method == "TGT_CRC"].set_index(k).reindex(g4.index)
    rebuilt = np.where(g4.flag.astype(bool), t.miss, c.miss); d1 = float(np.abs(rebuilt - g4.miss).max()); bad += d1 > 1e-12
    # 2) seed-level mean and t-interval, worst scenario, labels, for the final method at every alpha and group
    for (a, g), d in R[R.method == "G4_CRC"].groupby(["alpha", "g"]):
        sm = d.groupby("seed").miss.mean(); n = len(sm); m = sm.mean(); h = stats.t.ppf(.975, n - 1) * sm.std(ddof=1) / np.sqrt(n)
        worst = d.groupby("site").miss.mean().max(); row = T2[(T2.backbone == name) & (T2.alpha == a) & (T2.group == g) & (T2.method == "FINAL_CRC")]
        diff = max(abs(m - row.miss.iloc[0]), abs(m - h - row.miss_lo.iloc[0]), abs(worst - row.worst_scenario_miss.iloc[0]), abs(d.n_lab.mean() - row.n_lab.iloc[0]))
        bad += diff > 1e-4; out.append(dict(backbone=name, alpha=a, group=g, seeds=n, miss=round(m, 5), ci_lo=round(m - h, 5), ci_hi=round(m + h, 5), worst=round(worst, 5), max_abs_diff_vs_TA2=diff))
    # 3) exceedance by branch (alpha 0.05, non-concept) versus TA9
    for g in ["Appearance shift", "Real sites", "Prevalence shift"]:
        d = R[(R.alpha == 0.05) & (R.g == g)]; fl = d[(d.method == "TGT_CP") & d.flag.astype(bool)]
        if len(fl):
            r9 = T9[(T9.backbone == tag) & (T9.group == g.split()[0]) & T9.branch.str.startswith("flagged")]
            dd = abs((fl.miss > .05).mean() - r9.rate.iloc[0]) if len(r9) else np.nan; bad += (dd > 1e-4) if dd == dd else 0
            out.append(dict(backbone=name, alpha=.05, group=g + " [TA9 flagged CP]", seeds=None, miss=round((fl.miss > .05).mean(), 4), max_abs_diff_vs_TA2=dd))
    print(name, "| G4_CRC rebuilt from flag/CRC/TGT_CRC: max |diff| =", d1)
O = pd.DataFrame(out); O.to_csv(P + "VERIFY_independent_recompute.csv", index=False)
print(O[O.alpha == 0.05].to_string(index=False)); print("rows checked:", len(O), "| mismatches > 1e-4:", int(bad)); print("VERIFY", "PASS" if bad == 0 else "FAIL")
