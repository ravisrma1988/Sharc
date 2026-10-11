import os, numpy as np, pandas as pd, json
W = "/path/to/sharc_work"; OUT = W + "/results/" + os.environ.get("OUTDIR", "paper_final"); os.makedirs(OUT, exist_ok=True)
BBS = {"convnext5": "ConvNeXt-T", "resnet50": "ResNet-50", "dinov2": "DINOv2-S (frozen)"}
SITES = ["messidor2", "deepdrid", "ddr", "aptos", "idrid"]
def group(site, exp):
    if exp == "E2": return "Real sites"
    if str(site).startswith("concept"): return "Concept shift (image-invisible)"
    return "Appearance shift" if str(site).startswith("deg:") or site == "quality" else "Prevalence shift"
KEY = ["exp", "seed", "rep", "site", "alpha"]
def load(tag):
    R = pd.read_parquet(f"{W}/results/e_{os.environ.get('VER', 'v8')}_{tag}_raw.parquet"); R["group"] = [group(s, e) for s, e in zip(R.site, R.exp)]
    R["workload"] = R.refer + R.defer.fillna(0); R["viol"] = R.miss > R.alpha
    # FINAL method (theory Section 7): unflagged -> unweighted source CRC; flagged -> target CRC / exact binomial on K positives
    base = R[R.method == "CRC"].set_index(KEY); out = []
    for tgt, name in [("TGT_CRC", "FINAL_CRC"), ("TGT_CP", "FINAL_CP")]:
        t = R[R.method == tgt].set_index(KEY).reindex(base.index); f = base.flag.astype(bool).values
        F = base.copy(); F["method"] = name
        for c in ["thr", "miss", "refer", "false_refer", "workload", "viol"]: F[c] = np.where(f, t[c].values, base[c].values)
        F["n_lab"] = np.where(f, t.n_lab.values, 0); F["n_lab_pos"] = np.where(f, t.n_lab_pos.values, 0); out.append(F.reset_index())
    return pd.concat([R] + out, ignore_index=True)
def seed_ci(df, col):
    """Mean over repetitions within each seed, then mean and 95% t-interval across seeds (training seed = independent unit)."""
    from scipy.stats import t as tdist
    m = df.groupby("seed")[col].mean(); k = len(m); se = m.std(ddof=1) / np.sqrt(k) if k > 1 else np.nan; h = tdist.ppf(0.975, k - 1) * se if k > 1 else np.nan
    return m.mean(), m.mean() - h, m.mean() + h
METHODS = ["FIXED", "CRC", "NAIVE", "B5_ordinal_LAC", "B6_entropy_d20", "B8_PAC_labelshift", "BBSE_l1", "TGT_CRC", "TGT_CP", "G4_CRC", "G4_CP", "FINAL_CRC", "FINAL_CP", "ORACLE"]
if __name__ == "__main__":
    T2, T3, T4, T5, T6 = [], [], [], [], []
    for tag, bbn in BBS.items():
        R = load(tag); R = R[R.method.isin(METHODS) | R.method.str.contains("_3w_") | R.method.str.startswith(("Gscore", "Gfeat", "B6"))]
        # --- per-scenario means (all methods) and group summaries with seed-level 95% CIs
        sc = R.groupby(["alpha", "group", "exp", "site", "method"]).agg(miss=("miss", "mean"), exceed=("viol", "mean"), refer=("refer", "mean"),
                workload=("workload", "mean"), n_lab=("n_lab", "mean"), flag=("flag", "mean")).reset_index(); sc.insert(0, "backbone", bbn); T2.append(sc)
        for (a, g, m), d in R[R.method.isin(METHODS)].groupby(["alpha", "group", "method"]):
            mm, lo, hi = seed_ci(d, "miss"); s = sc[(sc.alpha == a) & (sc.group == g) & (sc.method == m)]; wl, wlo, whi = seed_ci(d, "workload")
            T3.append(dict(backbone=bbn, alpha=a, group=g, method=m, miss=mm, miss_lo=lo, miss_hi=hi, worst_scenario_miss=s.miss.max(),
                           worst_scenario=s.loc[s.miss.idxmax(), "site"], exceed=d.viol.mean(), workload=wl, workload_lo=wlo, workload_hi=whi, n_lab=d.n_lab.mean()))
        # --- paired differences FINAL_CRC minus baseline, per seed (training seed = independent unit), alpha 0.05
        A = R[(R.alpha == 0.05)].set_index(["exp", "seed", "rep", "site", "method"])
        for g in R.group.unique():
            Ag = A[A.group == g]; P = Ag.xs("FINAL_CRC", level="method")
            for b in ["FIXED", "CRC", "NAIVE", "B5_ordinal_LAC", "B6_entropy_d20", "B8_PAC_labelshift", "BBSE_l1", "TGT_CRC"]:
                Bb = Ag.xs(b, level="method"); j = P.index.intersection(Bb.index)
                d = pd.DataFrame(dict(seed=j.get_level_values("seed"), dm=P.loc[j, "miss"].values - Bb.loc[j, "miss"].values,
                                      dw=P.loc[j, "workload"].values - Bb.loc[j, "workload"].values, dl=P.loc[j, "n_lab"].values - Bb.loc[j, "n_lab"].fillna(0).values))
                r = dict(backbone=bbn, group=g, baseline=b, pairs=len(d))
                for c in ["dm", "dw", "dl"]: r[c], r[c + "_lo"], r[c + "_hi"] = seed_ci(d, c)
                T4.append(r)
        # --- three-way pathway at real sites with grader sensitivity
        tw = R[(R.exp == "E2") & R.method.str.contains("_3w_") & (R.alpha == 0.05)].copy()
        t = tw.groupby(["site", "method"]).agg(miss=("miss", "mean"), pos_defer=("pos_defer", "mean"), pos_auto=("pos_auto", "mean"), auto=("auto", "mean"),
                defer=("defer", "mean"), clear=("clear", "mean"), fpr_auto=("fpr_auto", "mean"), n_lab=("n_lab", "mean")).reset_index()
        for sg in (0.95, 0.90): t[f"miss_grader_sens_{sg}"] = t.miss + (1 - sg) * t.pos_defer
        t.insert(0, "backbone", bbn); T5.append(t)
        # --- gate-component ablation and label budget
        ga = R[R.method.str.match(r"^(G4|Gscore|Gfeat|FINAL|TGT)_(CRC|CP)$")].groupby(["alpha", "group", "method"]).agg(
            miss=("miss", "mean"), exceed=("viol", "mean"), n_lab=("n_lab", "mean"), flag=("flag", "mean")).reset_index()
        ga["worst_scenario_miss"] = [sc[(sc.alpha == a) & (sc.group == g) & (sc.method == m)].miss.max() for a, g, m in zip(ga.alpha, ga.group, ga.method)]
        ga.insert(0, "backbone", bbn); T6.append(ga)
        print(bbn, "done", flush=True)
    pd.concat(T2).round(5).to_csv(OUT + "/TA4_per_scenario_all_methods.csv", index=False); pd.DataFrame(T3).round(5).to_csv(OUT + "/TA2_main_summary_seedCI.csv", index=False)
    pd.DataFrame(T4).round(5).to_csv(OUT + "/TA3_paired_differences_seedCI.csv", index=False); pd.concat(T5).round(5).to_csv(OUT + "/TA6_three_way_pathway.csv", index=False)
    pd.concat(T6).round(5).to_csv(OUT + "/TA5_gate_ablation_label_budget.csv", index=False); print("TABLES DONE", flush=True)
