import os, sys, functools
os.environ.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
import numpy as np, pandas as pd, json, time
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e_v2 as V, e_v4 as V4, e_v5_main as M5, e_v6 as V6      # importing e_v6 also adds B8 to M5.run
W = V.W; SMOKE = V.SMOKE; BB = V.BB
NCAL = [250, 500, 1000, 2500, 5000, 7229]; SCEN = ["none", "prev_high", "prev_low"]
KEEP = {"CRC", "NAIVE", "BBSE_l1", "ORACLE", "B5_ordinal_LAC", "B8_PAC_labelshift"} | {f"G4_{t}_K{k}" for t in ("CRC", "CP") for k in (22, 45, 114)}
@functools.lru_cache(None)
def bank(seed):
    """Source calibration bank: EyeQ-labelled EyePACS test patients (disjoint from all training / validation / calibration patients)."""
    d = pd.read_parquet(f"{W}/preds/eyepacs_test/{BB}_s{seed}.parquet"); F = np.load(f"{W}/preds/eyepacs_test/{BB}_s{seed}_feat.npy")
    B = V.to_units(d, F); d = d.assign(so=d[["p_g2", "p_g3", "p_g4"]].max(1), nc=1 - d[[f"p_g{k}" for k in range(5)]].values[np.arange(len(d)), d.grade.values])
    O = d.groupby("patient").agg(so=("so", "max"), nc=("nc", list)); B["so"] = O.so.reindex(B.patient).values; B["nc"] = O.nc.reindex(B.patient).values
    return B.reset_index(drop=True)
def task(args):
    seed, rep, sc = args; U, deg, lr, calib, pca = M5.load5(seed); val = U[U.role == "val"]; Bk = bank(seed)
    rng = np.random.default_rng(110_000 * seed + 100 * rep + SCEN.index(sc)); perm = rng.permutation(len(Bk)); n = len(perm)
    Pa = Bk.iloc[perm[:n // 5]].reset_index(drop=True); Pt = Bk.iloc[perm[n // 5:2 * n // 5]].reset_index(drop=True); pool_src = Bk.iloc[perm[2 * n // 5:]]   # all within the test pool
    wa = V4.scen_w(Pa, sc); nz = int((wa > 0).sum()); order = rng.choice(len(Pa), nz, replace=False, p=wa / wa.sum()); Pseq = Pa.iloc[order].reset_index(drop=True)
    Tt = Pt.copy(); Tt["w"] = V4.scen_w(Pt, sc); Tt = Tt[Tt.w > 0].reset_index(drop=True); out = []
    for nc in NCAL:
        C = pool_src.iloc[rng.permutation(len(pool_src))[:min(nc, len(pool_src))]]
        out += [dict(**r, ncal=len(C)) for r in M5.run(C, Pseq.iloc[:600], Pseq, Tt, lr, val, pca, rng, dict(exp="scale", seed=seed, rep=rep, site=sc)) if r["method"] in KEEP]
    return out
if __name__ == "__main__":
    t0 = time.time(); R = int(os.environ.get("REPS_ABL", "1" if SMOKE else "30")); T = [(s, r, sc) for s in V.SEEDS for r in range(R) for sc in SCEN]
    with Pool(1 if SMOKE else 36) as pool: D = pd.DataFrame([x for o in pool.imap_unordered(task, T) for x in o])
    print(json.dumps(dict(stage="scale", rows=len(D), mins=round((time.time() - t0) / 60, 1))), flush=True)
    tag = ("smoke6scale" if SMOKE else "v6scale") + os.environ.get("RUN_TAG", ""); D["viol"] = D.miss > D.alpha; D.to_parquet(f"{W}/results/e_{tag}_raw.parquet")
    print(D[D.method == "CRC"].groupby(["alpha", "ncal"]).agg(miss=("miss", "mean"), viol=("viol", "mean"), thr_sd=("thr", "std")).round(3).unstack(0).to_string())
    print(D.groupby(["site", "ncal"]).flag.mean().unstack().round(2).to_string()); print(f"{tag.upper()} DONE", flush=True)
