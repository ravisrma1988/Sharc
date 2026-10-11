import pandas as pd
R = "/path/to/sharc_work/results/"; V9 = R + "paper_final_v9/"; V10 = R + "paper_final_v10/"; out = []
BBS = {"ConvNeXt-T": "CNX", "ResNet-50": "R50", "DINOv2-S (frozen)": "DINO"}
def block(n, df): out.append(f"### {n}\n" + df.to_csv(index=False))
t2 = pd.read_csv(V9 + "TA2_main_summary_seedCI.csv"); t2["backbone"] = t2.backbone.map(BBS)
A = ["FIXED", "CRC", "NAIVE", "BBSE_l1", "B5_ordinal_LAC", "B6_entropy_d20", "B8_PAC_labelshift", "TGT_CRC", "FINAL_CRC", "FINAL_CP", "ORACLE"]
g = t2[((t2.alpha == 0.05) & t2.method.isin(A)) | (t2.alpha.isin([0.02, 0.10]) & t2.method.isin(["CRC", "FINAL_CRC", "FINAL_CP", "TGT_CRC"]))]
block("G", g[["backbone", "alpha", "group", "method", "miss", "miss_lo", "miss_hi", "worst_scenario_miss", "exceed", "workload", "n_lab"]].round(3))
t4 = pd.read_csv(V9 + "TA4_per_scenario_all_methods.csv"); t4["backbone"] = t4.backbone.map(BBS)
s = t4[(t4.alpha == 0.05) & (t4.exp == "E2") & t4.method.isin(["FIXED", "CRC", "NAIVE", "BBSE_l1", "B8_PAC_labelshift", "TGT_CRC", "FINAL_CRC", "FINAL_CP", "ORACLE"])]
block("S", s[["backbone", "site", "method", "miss", "workload", "n_lab"]].round(3))
c = t4[(t4.alpha == 0.05) & (t4.backbone == "CNX") & (t4.exp == "E1") & t4.method.isin(["CRC", "NAIVE", "BBSE_l1", "FINAL_CRC", "FINAL_CP"])]
block("E1", c[["site", "method", "miss", "flag"]].round(3))
t6 = pd.read_csv(V9 + "TA6_three_way_pathway.csv"); t6 = t6[t6.method == "G4_CRC_3w_g10"]; t6["backbone"] = t6.backbone.map(BBS)
block("TW", t6[["backbone", "site", "miss", "miss_grader_sens_0.9", "auto", "defer", "clear", "fpr_auto"]].round(3))
lb = pd.read_csv(V9 + "TA1b_label_budget_real_sites.csv"); block("LB", lb[["site", "alpha", "labelled_mean", "referable_found", "pool_exhausted_rate", "prevalence", "K"]].round(3))
lp = pd.read_csv(V9 + "TA1c_label_budget_prevalence.csv"); block("LP", lp.round(3))
block("EX", pd.read_csv(V9 + "TA9_exceedance_by_branch.csv")[["backbone", "group", "branch", "runs", "rate", "ci95", "median_test_positives"]])
d1 = pd.read_csv(V10 + "TD1_all_41_cells.csv"); d1 = d1[d1.alpha == 0.05].copy(); d1["backbone"] = d1.backbone.map(BBS)
keep = (d1.method.isin(["SOURCE", "GATED_CRC", "ALWAYS_TARGET_CRC", "FIXED_BUDGET_CRC"])) | ((d1.backbone == "CNX") & d1.method.isin(["GATED_SCORE_ONLY_CRC", "GATED_FEATURE_ONLY_CRC"]))
block("C10", d1[keep][["backbone", "site", "prev_mult", "appearance", "level", "method", "miss", "flag_rate", "labels_mean"]].round(3))
d2 = pd.read_csv(V10 + "TD2_by_setting.csv"); d2 = d2[d2.alpha == 0.05].copy(); d2["backbone"] = d2.backbone.map(BBS)
block("S10", d2[["backbone", "setting", "method", "harmful_cells", "miss", "worst_cell_miss", "zero_label", "labels_mean", "labels_p90", "harmful_not_flagged", "refer", "refer_all"]].round(3))
d4 = pd.read_csv(V10 + "TD4_gate_component_contribution.csv"); d4["backbone"] = d4.backbone.map(BBS); block("GC", d4.round(3))
e = pd.read_csv(V9 + "efficiency_benchmark.csv"); block("EF", e[["site", "n_adapt", "score_test_B200", "feature_test_B200", "full_pipeline_all_methods_3_alphas"]])
txt = "\n".join(out); open(R + "paper_export_slim.txt", "w").write(txt); print("chars", len(txt))
