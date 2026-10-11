import os
import numpy as np, pandas as pd
W = "/path/to/sharc_work"; OUT = W + "/results/" + os.environ.get("OUTDIR", "paper_final")
M = pd.read_parquet(W + "/data/proc/metadata_rows.parquet"); rows = []
for ds, d in M.groupby("ds"):
    pu = d.groupby("patient").agg(n_img=("img_id", "size"), y=("referable", "max"))
    r = dict(dataset=ds, images=len(d), units=len(pu), unit_type="patient" if ds in ("eyepacs", "messidor2", "deepdrid") else "image",
             referable_units=int(pu.y.sum()), referable_rate=round(pu.y.mean(), 4), images_per_unit=dict(pu.n_img.value_counts().sort_index()))
    if ds == "eyepacs": r.update({f"role_{k}": int(v) for k, v in d.groupby("role").patient.nunique().items()})
    rows.append(r)
T = pd.read_parquet(W + "/data/proc/metadata_eyepacs_test.parquet"); pu = T.groupby("patient").agg(n=("img_id", "size"), y=("referable", "max"))
rows.append(dict(dataset="eyepacs_test_pool (scale experiment only)", images=len(T), units=len(pu), unit_type="patient", referable_units=int(pu.y.sum()),
                 referable_rate=round(pu.y.mean(), 4), images_per_unit=dict(pu.n.value_counts().sort_index())))
F = pd.DataFrame(rows); F.to_csv(OUT + "/TA1a_sample_flow_datasets.csv", index=False); print(F.to_string(index=False))
# messidor-2 graded-eye completeness
m = M[M.ds == "messidor2"].groupby("patient").size(); print("Messidor-2 exams with 2 graded eyes:", int((m == 2).sum()), "| with 1 graded eye:", int((m == 1).sum()))
# per-site split sizes and label budget from the final v8 run (ConvNeXt), alpha-specific
R = pd.read_parquet(W + "/results/e_" + os.environ.get("VER", "v8") + "_convnext5_raw.parquet", columns=["exp", "site", "alpha", "method", "n_gate", "n_label_pool", "n_lab", "n_lab_pos", "pool_exhausted", "piT_true"])
t = R[(R.exp == "E2") & (R.method == "TGT_CP")].groupby(["site", "alpha"]).agg(gate_units=("n_gate", "mean"), label_pool=("n_label_pool", "mean"),
      labelled_mean=("n_lab", "mean"), labelled_p90=("n_lab", lambda v: v.quantile(.9)), referable_found=("n_lab_pos", "mean"),
      pool_exhausted_rate=("pool_exhausted", "mean"), prevalence=("piT_true", "mean")).reset_index()
t["K"] = t.alpha.map({0.02: 114, 0.05: 45, 0.1: 22}); t["K_over_prevalence"] = t.K / t.prevalence
t.round(3).to_csv(OUT + "/TA1b_label_budget_real_sites.csv", index=False); print(t.round(2).to_string(index=False))
e = R[(R.exp == "E1") & (R.method == "TGT_CP") & R.site.isin(["none", "prev_low", "prev_high"])].groupby(["site", "alpha"]).agg(
      labelled_mean=("n_lab", "mean"), prevalence=("piT_true", "mean"), pool_exhausted_rate=("pool_exhausted", "mean")).reset_index()
e.round(3).to_csv(OUT + "/TA1c_label_budget_prevalence.csv", index=False); print(e.round(2).to_string(index=False))
# deduplicated sensitivity versus main (ConvNeXt, real sites, alpha 0.05)
try:
    A = pd.read_parquet(W + "/results/e_" + os.environ.get("VER", "v8") + "_convnext5_raw.parquet"); B = pd.read_parquet(W + "/results/e_" + os.environ.get("VER", "v8") + "_convnext5_dedup_raw.parquet")
    pick = ["CRC", "TGT_CRC", "TGT_CP", "G4_CRC", "G4_CP", "ORACLE"]
    c = pd.concat([x[(x.exp == "E2") & (x.alpha == 0.05) & x.method.isin(pick)].groupby(["site", "method"]).agg(miss=("miss", "mean"), refer=("refer", "mean")).assign(version=v)
                   for x, v in [(A, "all images"), (B, "deduplicated")]]).reset_index()
    c = c.pivot_table(index=["site", "method"], columns="version", values=["miss", "refer"]); c.round(4).to_csv(OUT + "/TA7_dedup_sensitivity.csv"); print(c.round(3).to_string())
except FileNotFoundError: print("dedup run not finished yet")
