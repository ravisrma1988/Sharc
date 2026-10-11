import os
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
W = "/path/to/sharc_work"; OUT = W + "/results/" + os.environ.get("OUTDIR", "paper_final")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300, "font.family": "DejaVu Sans"})
LBL = dict(FIXED="Fixed threshold", CRC="Source CRC", NAIVE="Weighted CRC (density ratio)", B5_ordinal_LAC="Ordinal conformal sets",
           B6_entropy_d20="Entropy abstention (20%)", B8_PAC_labelshift="PAC sets, label shift", BBSE_l1="Unlabelled weighting only",
           TGT_CRC="Always label K (CRC)", FINAL_CRC="Proposed (gate + CRC)", FINAL_CP="Proposed (gate + exact bound)", ORACLE="Oracle")
ORDER = ["FIXED", "CRC", "NAIVE", "B5_ordinal_LAC", "B6_entropy_d20", "B8_PAC_labelshift", "BBSE_l1", "TGT_CRC", "FINAL_CRC", "FINAL_CP"]
BB = ["ConvNeXt-T", "ResNet-50", "DINOv2-S (frozen)"]
def save(fig, n): fig.savefig(f"{OUT}/{n}.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/{n}.png", bbox_inches="tight"); plt.close(fig)
S = pd.read_csv(OUT + "/TA2_main_summary_seedCI.csv"); P = pd.read_csv(OUT + "/TA4_per_scenario_all_methods.csv")
# F1 worst-scenario miss
fig, axs = plt.subplots(1, 3, figsize=(7.4, 3.4), sharey=True); x = np.arange(len(ORDER))
for ax, g in zip(axs, ["Appearance shift", "Real sites", "Concept shift (image-invisible)"]):
    for i, b in enumerate(BB):
        d = S[(S.alpha == 0.05) & (S.group == g) & (S.backbone == b)].set_index("method").reindex(ORDER)
        ax.bar(x + (i - 1) * 0.27, 100 * d.worst_scenario_miss, 0.27, color=["0.15", "0.5", "0.8"][i], label=b)
    ax.axhline(5, color="red", ls="--", lw=1, label="Target (5%)"); ax.set_yscale("log"); ax.set_title(g, fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels([LBL[m] for m in ORDER], rotation=65, ha="right", fontsize=6.5)
axs[0].set_ylabel("Worst-scenario miss rate (%)"); h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=4, fontsize=7, frameon=False, bbox_to_anchor=(0.5, 1.03)); fig.tight_layout(rect=(0, 0, 1, 0.93)); save(fig, "F1_worst_case_miss")
# F2 severity
p = P[(P.backbone == "ConvNeXt-T") & (P.alpha == 0.05)]; COL = dict(CRC="black", NAIVE="orange", BBSE_l1="green", FINAL_CRC="blue", FINAL_CP="red")
fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.5), sharey=True)
for ax, (k, t) in zip(axs, [("blur", "Blur"), ("contrast", "Contrast loss"), ("dark", "Darkening"), ("colorcast", "Colour cast")]):
    for m, c in COL.items():
        v = [p[(p.site == "none") & (p.method == m)].miss.mean()] + [p[(p.site == f"deg:{k}{l}") & (p.method == m)].miss.mean() for l in (1, 2, 3)]
        ax.plot([0, 1, 2, 3], 100 * np.array(v), marker="o", ms=3, color=c, label=LBL[m])
    ax.axhline(5, color="red", ls="--", lw=0.8); ax.set_title(t); ax.set_xticks([0, 1, 2, 3]); ax.set_xlabel("Severity level"); ax.set_ylim(0, 80)
axs[0].set_ylabel("Miss rate (%)"); axs[-1].legend(fontsize=6, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1)); fig.tight_layout(); save(fig, "F2_severity")
# F3 real-site workload with labels needed
p3 = P[(P.backbone == "ConvNeXt-T") & (P.alpha == 0.05) & (P.exp == "E2")]; sites = ["messidor2", "deepdrid", "ddr", "aptos", "idrid"]
fig, ax = plt.subplots(figsize=(5.2, 2.7)); x = np.arange(5)
for i, (m, c) in enumerate([("CRC", "black"), ("FINAL_CRC", "blue"), ("ORACLE", "olive")]):
    v = [100 * p3[(p3.site == s) & (p3.method == m)].workload.mean() for s in sites]; ax.bar(x + (i - 1) * 0.27, v, 0.27, color=c, label=LBL[m])
    if m == "FINAL_CRC":
        for xi, s, vi in zip(x, sites, v): ax.text(xi, vi + 1, f"{p3[(p3.site == s) & (p3.method == m)].n_lab.mean():.0f}", ha="center", fontsize=6.5, color="blue")
ax.set_xticks(x); ax.set_xticklabels(["Messidor-2", "DeepDRiD", "DDR", "APTOS", "IDRiD"]); ax.set_ylabel("Referred + deferred (%)")
ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1, 1)); ax.set_title("Numbers above blue bars: target patients labelled", fontsize=7.5); save(fig, "F3_real_site_workload")
# F4 diagnostic power vs unlabelled patients (ablation run, ConvNeXt)
a = pd.read_parquet(W + "/results/e_v5abl_convnext5_nad_raw.parquet"); a = a[a.method == "G4_CRC_K45"].groupby(["site", "nad"]).flag.mean().reset_index()
fig, ax = plt.subplots(figsize=(4.4, 2.7))
for s, c, l in [("none", "black", "No shift"), ("prev_high", "gray", "Prevalence x3"), ("prev_low", "0.7", "Prevalence x0.4"), ("deg:blur1", "blue", "Blur (mild)"),
                ("deg:contrast1", "red", "Contrast (mild)"), ("deg:dark1", "purple", "Darkening (mild)"), ("quality", "orange", "Poor image quality"), ("ddr", "green", "DDR (real)")]:
    d = a[a.site == s].sort_values("nad"); ax.plot(d.nad, 100 * d.flag, marker="o", ms=3, color=c, label=l)
ax.axhline(5, color="red", ls=":", lw=0.8); ax.set_xscale("log"); ax.set_xlabel("Unlabelled target patients"); ax.set_ylabel("Shift flagged (%)")
ax.legend(fontsize=6.5, frameon=False, loc="upper left", bbox_to_anchor=(1, 1)); save(fig, "F4_diagnostic_power")
# F5 calibration scale (within EyePACS test pool), within-model threshold SD and mean miss of source CRC
S5 = pd.read_parquet(W + "/results/e_v6scale_convnext5_raw.parquet"); S5 = S5[(S5.method == "CRC") & (S5.site == "none")]
w = S5.groupby(["seed", "alpha", "ncal"]).agg(sd=("thr", "std"), miss=("miss", "mean")).reset_index().groupby(["alpha", "ncal"]).mean(numeric_only=True).reset_index()
w.round(5).to_csv(OUT + "/TA8_calibration_scale.csv", index=False)
fig, axs = plt.subplots(1, 2, figsize=(6.2, 2.6))
for al, c in [(0.02, "red"), (0.05, "blue"), (0.10, "black")]:
    d = w[w.alpha == al]; axs[0].plot(d.ncal, d.sd, marker="o", ms=3, color=c, label=f"alpha = {al:g}"); axs[1].plot(d.ncal, 100 * d.miss, marker="o", ms=3, color=c); axs[1].axhline(100 * al, color=c, ls=":", lw=0.8)
for ax in axs: ax.set_xscale("log"); ax.set_xlabel("Source calibration patients")
axs[0].set_ylabel("Threshold SD (within model)"); axs[1].set_ylabel("Mean miss rate (%)"); axs[0].legend(fontsize=7, frameon=False); fig.tight_layout(); save(fig, "F5_calibration_scale")
# F6 three-way pathway at real sites (ConvNeXt, gate + CRC, gamma 0.10)
t = pd.read_csv(OUT + "/TA6_three_way_pathway.csv"); t = t[(t.backbone == "ConvNeXt-T") & (t.method == "G4_CRC_3w_g10")].set_index("site").reindex(sites)
fig, ax = plt.subplots(figsize=(5.0, 2.7)); bot = np.zeros(5)
for col, c, l in [("auto", "blue", "Refer to specialist"), ("defer", "orange", "Defer to grader"), ("clear", "0.8", "Clear (rescreen)")]:
    ax.bar(np.arange(5), 100 * t[col].values, 0.6, bottom=bot, color=c, label=l); bot += 100 * t[col].values
ax.set_xticks(np.arange(5)); ax.set_xticklabels(["Messidor-2", "DeepDRiD", "DDR", "APTOS", "IDRiD"]); ax.set_ylabel("Patients (%)")
ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1, 1)); save(fig, "F6_three_way")
# F7 label budget against prevalence: expected K / prevalence and observed means
b = pd.read_csv(OUT + "/TA1b_label_budget_real_sites.csv"); c7 = pd.read_csv(OUT + "/TA1c_label_budget_prevalence.csv")
fig, ax = plt.subplots(figsize=(4.4, 2.8)); pr = np.linspace(0.05, 0.7, 100)
for al, K, c in [(0.02, 114, "red"), (0.05, 45, "blue"), (0.10, 22, "black")]:
    ax.plot(pr, K / pr, color=c, lw=1, label=f"K/prevalence, alpha = {al:g}")
    d = pd.concat([b[b.alpha == al][["prevalence", "labelled_mean"]], c7[c7.alpha == al][["prevalence", "labelled_mean"]]]); ax.scatter(d.prevalence, d.labelled_mean, color=c, s=12)
ax.set_yscale("log"); ax.set_xlabel("Referable prevalence at the site"); ax.set_ylabel("Patients to label"); ax.legend(fontsize=6.5, frameon=False); save(fig, "F7_label_budget")
# F8 concept shift: invisible to the gate
c8 = P[(P.alpha == 0.05) & P.site.isin(["concept_g1", "concept_half"])]
fig, ax = plt.subplots(figsize=(4.6, 2.6)); ms = ["CRC", "BBSE_l1", "FINAL_CRC", "TGT_CRC", "ORACLE"]; x = np.arange(len(ms))
for i, (s, c) in enumerate([("concept_half", "0.6"), ("concept_g1", "0.2")]):
    d = c8[(c8.site == s) & (c8.backbone == "ConvNeXt-T")].set_index("method").reindex(ms); ax.bar(x + (i - 0.5) * 0.35, 100 * d.miss, 0.35, color=c, label={"concept_half": "Half of mild DR now referable", "concept_g1": "All mild DR now referable"}[s])
ax.axhline(5, color="red", ls="--", lw=0.8); ax.set_xticks(x); ax.set_xticklabels([LBL[m] for m in ms], rotation=30, ha="right", fontsize=7); ax.set_ylabel("Miss rate (%)")
ax.legend(fontsize=7, frameon=False); save(fig, "F8_concept_shift"); print("FIGURES DONE")
