import os, sys, functools
os.environ.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
import numpy as np, pandas as pd, json, time
from multiprocessing import Pool
from scipy.stats import beta, binom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e_v2 as V, e_v4 as V4, e_v5_main as M5
W = V.W; SMOKE = V.SMOKE; DELTA = 0.1
def cp(k, n, conf):
    lo = beta.ppf((1 - conf) / 2, k, n - k + 1) if k > 0 else 0.0; hi = beta.ppf(1 - (1 - conf) / 2, k + 1, n - k) if k < n else 1.0; return lo, hi
def b8(C, Ta, Tt, lr, a, rng):
    """B8: PAC prediction sets under label shift (after Si et al., ICLR 2024), binary referable / non-referable classes.
    Prevalence interval from BBSE with Clopper-Pearson intervals (union bound), conservative class weights, rejection sampling,
    then a binomial PAC threshold on nonconformity 1 - p(true class). Refer if the prediction set contains 'referable'."""
    yC = C.y.values; qC, qT = V.cal(lr, C.s.values), V.cal(lr, Ta.s.values); conf = 1 - DELTA / 6
    a_lo, a_hi = cp(int(((qC > .5) & (yC == 1)).sum()), int((yC == 1).sum()), conf); b_lo, b_hi = cp(int(((qC > .5) & (yC == 0)).sum()), int((yC == 0).sum()), conf)
    r_lo, r_hi = cp(int((qT > .5).sum()), len(qT), conf)
    pi_hi = float(np.clip((r_hi - b_lo) / max(a_lo - b_hi, 1e-3), 0, 1)); pi_lo = float(np.clip((r_lo - b_hi) / max(a_hi - b_lo, 1e-3), 0, 1))
    piS = yC.mean(); w = np.where(yC == 1, pi_hi / piS, (1 - pi_lo) / (1 - piS)); acc = rng.random(len(C)) < w / w.max()
    nc = np.where(yC == 1, 1 - C.s.values, C.s.values)[acc]; m = len(nc)
    ks = np.where(binom.cdf(np.arange(m + 1), m, a) <= DELTA / 2)[0]
    if len(ks) == 0 or m == 0: thr = 0.0
    else: k = ks.max(); q = np.sort(nc)[::-1][k] if k < m else 0.0; thr = 1 - q
    return dict(V4.ev("B8_PAC_labelshift", a, thr, Tt), pi_lo=pi_lo, pi_hi=pi_hi, n_accept=m)
_run5 = M5.run
def run6(C, Ta, Lab, Tt, lr, val, pca, rng, base):
    rows = _run5(C, Ta, Lab, Tt, lr, val, pca, rng, base); info = {k: rows[0][k] for k in ("gof_p", "feat_auc", "feat_p", "flag", "piT_bbse", "piT_true")}
    return rows + [dict(**info, **base, **b8(C, Ta, Tt, lr, a, rng)) for a in V.ALPHAS]
M5.run = run6                                    # e1_task / e2_task in e_v5_main now also emit B8 rows on identical splits
if __name__ == "__main__":
    t0 = time.time(); sites = ["messidor2", "deepdrid", "ddr", "aptos", "idrid"]
    T1 = [(s, r) for s in V.SEEDS for r in range(V.REPS)]; T2 = [(s, si, r) for s in V.SEEDS for si in sites for r in range(V.REPS)]
    with Pool(1 if SMOKE else 36) as pool:
        R1 = [x for o in pool.imap_unordered(M5.e1_task, T1) for x in o]; print(json.dumps(dict(stage="E1", rows=len(R1), mins=round((time.time() - t0) / 60, 1))), flush=True)
        R2 = [x for o in pool.imap_unordered(M5.e2_task, T2) for x in o]; print(json.dumps(dict(stage="E2", rows=len(R2), mins=round((time.time() - t0) / 60, 1))), flush=True)
    tag = ("smoke6" if SMOKE else "v6") + os.environ.get("RUN_TAG", ""); R = pd.DataFrame(R1 + R2); R.to_parquet(f"{W}/results/e_{tag}_raw.parquet"); R["viol"] = R.miss > R.alpha
    agg = dict(miss=("miss", "mean"), violation=("viol", "mean"), refer=("refer", "mean"), false_refer=("false_refer", "mean"), auto=("auto", "mean"),
               defer=("defer", "mean"), spec_false=("spec_false", "mean"), flag=("flag", "mean"), n_lab=("n_lab", "mean"), n_lab_pos=("n_lab_pos", "mean"))
    S = R.groupby(["exp", "site", "alpha", "method"]).agg(**agg).reset_index().round(4); S.to_csv(f"{W}/results/e_{tag}_summary.csv", index=False)
    s = S[(S.alpha == 0.05) & S.method.isin(["CRC", "B5_ordinal_LAC", "B8_PAC_labelshift", "G4_CRC_K45", "G4_CP_K45"])]
    print(s.pivot_table(index=["exp", "site"], columns="method", values="miss").round(3).to_string()); print(f"{tag.upper()} DONE", flush=True)
