import pandas as pd, numpy as np
W = "/path/to/sharc_work/results"; OUT = W + "/paper_final_v10"; S = pd.read_csv(OUT + "/TD2_by_setting.csv")
FL = {"GATED_CRC": "flag", "GATED_CP": "flag", "GATED_SCORE_ONLY_CRC": "flag_score", "GATED_SCORE_ONLY_CP": "flag_score", "GATED_FEATURE_ONLY_CRC": "flag_feat", "GATED_FEATURE_ONLY_CP": "flag_feat"}
BB = {"convnext5": "ConvNeXt-T", "resnet50": "ResNet-50", "dinov2": "DINOv2-S (frozen)"}; fix, ref = [], []
def setting(app, p):
    if app.startswith("concept"): return "E concept (boundary)"
    if app == "none": return "A no shift" if p == 1 else "B prevalence only"
    return "C appearance" if p == 1 else "D mixed"
for tag, b in BB.items():
    R = pd.read_parquet(f"{W}/e_v10_{tag}_raw.parquet", columns=["seed", "rep", "site", "alpha", "method", "miss", "refer", "n_lab", "flag", "flag_score", "flag_feat", "prev_mult", "appearance", "defer", "auto"])
    R["setting"] = [setting(a, p) for a, p in zip(R.appearance, R.prev_mult)]
    src = R[R.method == "SOURCE"].groupby(["alpha", "site"]).miss.mean(); harmful = {k for k, v in src.items() if v > k[0]}
    R["harmful"] = [(a, s) in harmful for a, s in zip(R.alpha, R.site)]
    for (a, st, m), d in R[R.method.isin(FL)].groupby(["alpha", "setting", "method"]):
        h = d[d.harmful]; fix.append(dict(backbone=b, alpha=a, setting=st, method=m, harmful_not_flagged=(~h[FL[m]].astype(bool)).mean() if len(h) else np.nan))
    w = R[R.alpha == 0.05].groupby(["setting", "method"]).agg(refer=("refer", "mean"), workload_3w=("auto", lambda v: np.nan)).reset_index()
    t3 = R[(R.alpha == 0.05) & R.method.str.endswith("_3w")].assign(work=lambda d: d.auto + d.defer).groupby(["setting", "method"]).work.mean().reset_index()
    ref.append(w.assign(backbone=b)); ref.append(t3.assign(backbone=b))
F = pd.DataFrame(fix); S = S.drop(columns=["harmful_not_flagged"]).merge(F, on=["backbone", "alpha", "setting", "method"], how="left"); S.round(5).to_csv(OUT + "/TD2_by_setting.csv", index=False)
print(F[F.alpha == 0.05].pivot_table(index=["backbone", "method"], columns="setting", values="harmful_not_flagged").round(3).to_string())
X = pd.concat(ref); x = X[X.method.isin(["SOURCE", "GATED_CRC", "ALWAYS_TARGET_CRC", "FIXED_BUDGET_CRC", "FIXED_BUDGET_CP"])]
print("\nreferral rate (two-way), alpha 0.05"); print(x.pivot_table(index=["backbone", "method"], columns="setting", values="refer").round(3).to_string())
y = X[X.method.str.endswith("_3w")]; print("\nthree-way workload (auto + defer), alpha 0.05"); print(y.pivot_table(index=["backbone", "method"], columns="setting", values="work").round(3).to_string())
