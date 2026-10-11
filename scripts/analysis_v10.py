import os, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
# Preregistered analysis (PREREG_DEPLOYMENT_GRID.md, SHA-256 0518621e...): all 41 cells reported; seed-level intervals; harmful = SOURCE cell-mean FNR > alpha.
W = "/path/to/sharc_work/results"; OUT = W + "/paper_final_v10"; os.makedirs(OUT, exist_ok=True)
BB = {"convnext5": "ConvNeXt-T", "resnet50": "ResNet-50", "dinov2": "DINOv2-S (frozen)"}
POL = ["SOURCE", "GATED_CRC", "GATED_CP", "ALWAYS_TARGET_CRC", "ALWAYS_TARGET_CP", "FIXED_BUDGET_CRC", "FIXED_BUDGET_CP", "GATED_SCORE_ONLY_CRC", "GATED_FEATURE_ONLY_CRC", "ORACLE_DECISION"]
def setting(r):
    if r.appearance.startswith("concept"): return "E concept (boundary)"
    if r.appearance == "none": return "A no shift" if r.prev_mult == 1 else "B prevalence only"
    return "C appearance" if r.prev_mult == 1 else "D mixed"
def tci(x):
    x = np.asarray(x, float); n = len(x); m = x.mean(); h = stats.t.ppf(.975, n - 1) * x.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan; return m, m - h, m + h
cell_rows, sum_rows, pair_rows = [], [], []
for tag, bbn in BB.items():
    R = pd.read_parquet(f"{W}/e_v10_{tag}_raw.parquet"); R = R[~R.method.str.endswith("_3w")].copy()
    k = ["seed", "rep", "site", "alpha"]
    src = R[R.method == "SOURCE"].set_index(k); tgt = R[R.method == "ALWAYS_TARGET_CRC"].set_index(k).reindex(src.index)
    o = src.copy(); use = (src.miss > src.alpha.values) if False else (src.miss.values > src.index.get_level_values("alpha"))
    o["miss"] = np.where(use, tgt.miss.values, src.miss.values); o["n_lab"] = np.where(use, tgt.n_lab.values, 0); o["refer"] = np.where(use, tgt.refer.values, src.refer.values)
    o["method"] = "ORACLE_DECISION"; R = pd.concat([R, o.reset_index()], ignore_index=True)       # analysis-only oracle decision
    R["setting"] = R.apply(setting, axis=1); R["viol"] = R.miss > R.alpha
    harm = R[R.method == "SOURCE"].groupby(["alpha", "site"]).miss.mean().gt(R[R.method == "SOURCE"].groupby(["alpha", "site"]).alpha.first()).rename("harmful")
    R = R.join(harm, on=["alpha", "site"])
    for (a, site, m), d in R[R.method.isin(POL)].groupby(["alpha", "site", "method"]):
        sm = d.groupby("seed").miss.mean(); mm, lo, hi = tci(sm)
        cell_rows.append(dict(backbone=bbn, alpha=a, site=site, setting=d.setting.iloc[0], prev_mult=d.prev_mult.iloc[0], appearance=d.appearance.iloc[0], level=d.level.iloc[0],
            method=m, harmful=bool(d.harmful.iloc[0]), miss=mm, miss_lo=lo, miss_hi=hi, exceed=d.viol.mean(), flag_rate=d.flag.mean(), zero_label=(d.n_lab == 0).mean(),
            labels_mean=d.n_lab.mean(), labels_median=d.n_lab.median(), labels_p90=d.n_lab.quantile(.9), refer=d.refer.mean(), refer_all=d.refer_all.mean() if "refer_all" in d else np.nan))
    for (a, st, m), d in R[R.method.isin(POL)].groupby(["alpha", "setting", "method"]):
        sm = d.groupby("seed").miss.mean(); mm, lo, hi = tci(sm); h = d[d.harmful]
        sum_rows.append(dict(backbone=bbn, alpha=a, setting=st, method=m, cells=d.site.nunique(), harmful_cells=h.site.nunique(), miss=mm, miss_lo=lo, miss_hi=hi,
            worst_cell_miss=d.groupby("site").miss.mean().max(), zero_label=(d.n_lab == 0).mean(), labels_mean=d.n_lab.mean(), labels_median=d.n_lab.median(), labels_p90=d.n_lab.quantile(.9),
            harmful_not_flagged=(~h.flag.astype(bool)).mean() if len(h) and m.startswith("GATED") else np.nan, refer=d.refer.mean(), refer_all=d.refer_all.mean()))
    G = R[(R.alpha == 0.05)].set_index(["seed", "rep", "site", "method"])
    for st in sorted(R.setting.unique()):
        Gs = G[G.setting == st]; g = Gs.xs("GATED_CRC", level="method")
        for b in ["SOURCE", "ALWAYS_TARGET_CRC", "FIXED_BUDGET_CRC", "GATED_SCORE_ONLY_CRC", "GATED_FEATURE_ONLY_CRC", "ORACLE_DECISION"]:
            c = Gs.xs(b, level="method"); j = g.index.intersection(c.index); s = j.get_level_values("seed")
            dm = pd.Series(g.loc[j, "miss"].values - c.loc[j, "miss"].values, index=s).groupby(level=0).mean(); dl = pd.Series(g.loc[j, "n_lab"].values - c.loc[j, "n_lab"].values, index=s).groupby(level=0).mean()
            r = dict(backbone=bbn, setting=st, comparator=b); r["d_miss"], r["d_miss_lo"], r["d_miss_hi"] = tci(dm); r["d_labels"], r["d_labels_lo"], r["d_labels_hi"] = tci(dl); pair_rows.append(r)
    print(bbn, "analysed", flush=True)
C = pd.DataFrame(cell_rows); S = pd.DataFrame(sum_rows); D = pd.DataFrame(pair_rows)
C.round(5).to_csv(OUT + "/TD1_all_41_cells.csv", index=False); S.round(5).to_csv(OUT + "/TD2_by_setting.csv", index=False); D.round(5).to_csv(OUT + "/TD3_paired_vs_comparators.csv", index=False)
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300, "font.family": "DejaVu Sans"})
def save(fig, n): fig.savefig(f"{OUT}/{n}.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/{n}.png", bbox_inches="tight"); plt.close(fig)
c = C[(C.backbone == "ConvNeXt-T") & (C.alpha == 0.05) & ~C.appearance.str.startswith("concept")]
cols = ["none"] + [f"{k}{l}" for k in ["blur", "contrast", "dark", "colorcast"] for l in (1, 2, 3)]; rows = [0.4, 1.0, 3.0]
def grid(m, val):
    x = c[c.method == m].assign(col=lambda d: np.where(d.appearance == "none", "none", d.appearance + d.level.astype(str)))
    return x.pivot_table(index="prev_mult", columns="col", values=val).reindex(index=rows, columns=cols)
fig, axs = plt.subplots(2, 1, figsize=(7.4, 3.6))
for ax, (m, val, ttl, cm) in zip(axs, [("GATED_CRC", "flag_rate", "Shift flagged (labels bought), %", "Greys"), ("SOURCE", "miss", "Source-only miss rate, % (target 5%)", "Greys")]):
    G = grid(m, val) * 100; im = ax.imshow(G.values, cmap=cm, aspect="auto", vmin=0, vmax=100 if val == "flag_rate" else max(15, np.nanmax(G.values)))
    ax.set_yticks(range(3)); ax.set_yticklabels(["0.4x", "1x", "3x"]); ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=7); ax.set_title(ttl, fontsize=8.5)
    for i in range(3):
        for j in range(len(cols)): ax.text(j, i, f"{G.values[i, j]:.0f}" if val == "flag_rate" else f"{G.values[i, j]:.1f}", ha="center", va="center", fontsize=6.5, color="red" if (val == "miss" and G.values[i, j] > 5) else "black")
    ax.set_ylabel("Prevalence")
fig.tight_layout(); save(fig, "FD1_gate_decision_heatmap")
s = S[(S.alpha == 0.05)]; order = ["A no shift", "B prevalence only", "C appearance", "D mixed", "E concept (boundary)"]
fig, ax = plt.subplots(figsize=(6.2, 2.6)); x = np.arange(len(order))
for i, (m, col) in enumerate([("GATED_CRC", "blue"), ("GATED_SCORE_ONLY_CRC", "0.6"), ("GATED_FEATURE_ONLY_CRC", "0.35"), ("ALWAYS_TARGET_CRC", "black")]):
    v = [100 * s[(s.backbone == "ConvNeXt-T") & (s.setting == o) & (s.method == m)].zero_label.mean() for o in order]
    ax.bar(x + (i - 1.5) * 0.2, v, 0.2, color=col, label={"GATED_CRC": "Gated (proposed)", "GATED_SCORE_ONLY_CRC": "Score-only gate", "GATED_FEATURE_ONLY_CRC": "Feature-only gate", "ALWAYS_TARGET_CRC": "Always label"}[m])
ax.set_xticks(x); ax.set_xticklabels(order, fontsize=7.5); ax.set_ylabel("Deployments with zero labels (%)"); ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1, 1)); save(fig, "FD2_zero_label_rate")
fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.6), sharey=True)
for ax, b in zip(axs, BB.values()):
    for m, col, mk in [("SOURCE", "black", "o"), ("GATED_CRC", "blue", "s"), ("ALWAYS_TARGET_CRC", "olive", "^"), ("FIXED_BUDGET_CRC", "orange", "D")]:
        d = C[(C.backbone == b) & (C.alpha == 0.05) & (C.method == m) & ~C.appearance.str.startswith("concept")]
        ax.scatter(d.labels_mean, 100 * d.miss, s=10, color=col, marker=mk, label={"SOURCE": "Source only", "GATED_CRC": "Gated (proposed)", "ALWAYS_TARGET_CRC": "Always label until K", "FIXED_BUDGET_CRC": "Fixed 100 labels"}[m])
    ax.axhline(5, color="red", ls="--", lw=0.8); ax.set_title(b, fontsize=8); ax.set_xlabel("Target units labelled (mean)")
axs[0].set_ylabel("Miss rate per cell (%)"); axs[-1].legend(fontsize=6.5, frameon=False, loc="upper left", bbox_to_anchor=(1, 1)); save(fig, "FD3_safety_vs_labels")
print("FIGURES DONE"); pd.set_option("display.width", 250)
print(S[(S.alpha == 0.05) & S.method.isin(["SOURCE", "GATED_CRC", "GATED_CP", "ALWAYS_TARGET_CRC", "FIXED_BUDGET_CRC", "GATED_SCORE_ONLY_CRC", "ORACLE_DECISION"])][
    ["backbone", "setting", "method", "harmful_cells", "miss", "miss_lo", "miss_hi", "worst_cell_miss", "zero_label", "labels_mean", "labels_p90", "harmful_not_flagged", "refer_all"]].round(4).to_string(index=False))
print(D[D.comparator.isin(["SOURCE", "ALWAYS_TARGET_CRC", "FIXED_BUDGET_CRC"]) & (D.backbone == "ConvNeXt-T")].round(4).to_string(index=False)); print("V10 ANALYSIS DONE")
