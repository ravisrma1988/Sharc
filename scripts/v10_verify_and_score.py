import numpy as np, pandas as pd, glob
from scipy import stats
pd.set_option("display.width", 220); W = "/path/to/sharc_work/results"; P = W + "/paper_final_v10/"
BB = {"convnext5": "ConvNeXt-T", "resnet50": "ResNet-50", "dinov2": "DINOv2-S (frozen)"}
def setting(app, p):
    if app.startswith("concept"): return "E concept (boundary)"
    if app == "none": return "A no shift" if p == 1 else "B prevalence only"
    return "C appearance" if p == 1 else "D mixed"
# ---------- (A) independent re-verification of v10 TD2 from raw Parquets (fresh code)
T = pd.read_csv(P + "TD2_by_setting.csv"); bad = 0; n = 0; inc = []
for tag, b in BB.items():
    R = pd.read_parquet(f"{W}/e_v10_{tag}_raw.parquet"); R = R[~R.method.str.endswith("_3w") & (R.alpha == 0.05)]
    R["setting"] = [setting(a, p) for a, p in zip(R.appearance, R.prev_mult)]
    src = R[R.method == "SOURCE"].groupby("site").miss.mean(); harm = set(src[src > 0.05].index)
    assert len([c for c in harm if not c.startswith("concept")]) == 36 and R.site.nunique() == 41, "denominator mismatch"
    for (st, m), d in R[R.method.isin(["SOURCE", "GATED_CRC", "ALWAYS_TARGET_CRC", "FIXED_BUDGET_CRC", "GATED_SCORE_ONLY_CRC", "GATED_FEATURE_ONLY_CRC"])].groupby(["setting", "method"]):
        sm = d.groupby("seed").miss.mean(); row = T[(T.backbone == b) & (T.alpha == 0.05) & (T.setting == st) & (T.method == m)].iloc[0]
        diffs = [abs(sm.mean() - row.miss), abs((d.n_lab == 0).mean() - row.zero_label), abs(d.n_lab.mean() - row.labels_mean), abs(d.groupby("site").miss.mean().max() - row.worst_cell_miss)]
        fl = {"GATED_CRC": "flag", "GATED_SCORE_ONLY_CRC": "flag_score", "GATED_FEATURE_ONLY_CRC": "flag_feat"}.get(m)
        if fl and st in ("C appearance", "D mixed"): diffs.append(abs((~d[d.site.isin(harm)][fl].astype(bool)).mean() - row.harmful_not_flagged))
        n += len(diffs); bad += sum(x > 1e-4 for x in diffs)
    # ---------- (B) incremental value of the score test: runs flagged ONLY by the score test (p_s < .025, p_f >= .025), by setting
    g = R[R.method == "GATED_CRC"].assign(only_score=lambda d: (d.gof_p < .025) & (d.feat_p >= .025), only_feat=lambda d: (d.gof_p >= .025) & (d.feat_p < .025), both=lambda d: (d.gof_p < .025) & (d.feat_p < .025))
    s = g.groupby("setting")[["flag", "only_score", "only_feat", "both"]].mean(); s["fa_feature_only_at_0.05"] = g.groupby("setting").flag_feat.mean(); s.insert(0, "backbone", b); inc.append(s.reset_index())
print("v10 TD2 independent check:", n, "quantities,", bad, "mismatches >1e-4; harmful non-concept cells = 36 of 41 for every backbone ->", "PASS" if bad == 0 else "FAIL")
I = pd.concat(inc); print("\n== who triggers the combined gate (fraction of runs), v10"); print(I.round(4).to_string(index=False))
# v9 (main grid + real sites): does the score test ever trigger alone?
v9 = []
for tag, b in BB.items():
    R = pd.read_parquet(f"{W}/e_v9_{tag}_raw.parquet", columns=["exp", "site", "alpha", "method", "gof_p", "feat_p", "flag"]); R = R[(R.method == "CRC") & (R.alpha == 0.05)]
    R["grp"] = np.where(R.exp == "E2", "real sites", np.where(R.site.str.startswith("concept"), "concept", np.where(R.site.str.startswith("deg:") | (R.site == "quality"), "appearance", "prevalence/none")))
    v9.append(R.groupby("grp").apply(lambda d: pd.Series(dict(flag=d.flag.mean(), only_score=((d.gof_p < .025) & (d.feat_p >= .025)).mean(), only_feat=((d.gof_p >= .025) & (d.feat_p < .025)).mean())), include_groups=False).assign(backbone=b).reset_index())
print("\n== v9: who triggers the combined gate"); print(pd.concat(v9).round(4).to_string(index=False))
B = pd.read_csv(W + "/paper_final_v9/efficiency_benchmark.csv"); print("\n== runtime (s) per site, one CPU worker"); print(B[["site", "score_test_B200", "feature_test_B200", "full_pipeline_all_methods_3_alphas"]].to_string(index=False))
I.to_csv(P + "TD4_gate_component_contribution.csv", index=False)
