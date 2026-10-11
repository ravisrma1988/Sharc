import os, sys
os.environ.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
import numpy as np, pandas as pd, json, time
from multiprocessing import Pool
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e_v2 as V            # load(), methods(), thr_crc(); env BACKBONE / SEEDS / REPS
import e_v3 as V3           # gof() score-mixture test
W = V.W; ALPHAS = V.ALPHAS; KS = [20, 45, 100]; BNULL = int(os.environ.get("BNULL", "40")); DELTA = 0.1; SMOKE = V.SMOKE
# ---------- (1) prevalence-adjusted feature-level two-sample test
def dom_auc(Zs, sw, Zt, rng):
    X = np.vstack([Zs, Zt]); y = np.r_[np.zeros(len(Zs)), np.ones(len(Zt))]; w = np.r_[sw * len(Zt) / sw.sum(), np.ones(len(Zt))]
    fold = rng.integers(0, 2, len(y)); p = np.empty(len(y))
    for k in (0, 1):
        tr = fold != k; p[~tr] = LogisticRegression(C=0.1, max_iter=2000).fit(X[tr], y[tr], sample_weight=w[tr]).predict_proba(X[~tr])[:, 1]
    return roc_auc_score(y, p, sample_weight=w)
def feat_test(Zs, yS, Zt, piT, rng):
    """H0: target features = source features re-mixed to prevalence piT. Null by prevalence-shifted pseudo-targets drawn from the source."""
    piS = yS.mean(); sw = np.where(yS == 1, piT / piS, (1 - piT) / (1 - piS)); A = dom_auc(Zs, sw, Zt, rng); n = len(Zt); null = np.empty(BNULL)
    for b in range(BNULL):
        idx = rng.choice(len(Zs), n, replace=True, p=sw / sw.sum()); rest = np.setdiff1d(np.arange(len(Zs)), idx)
        null[b] = dom_auc(Zs[rest], sw[rest], Zs[idx], rng)
    return A, (1 + (null >= A).sum()) / (BNULL + 1)
# ---------- (2) exact few-label bound and label-until-K rule
def thr_cp(sp, a, delta=DELTA):
    sp = np.sort(sp); n = len(sp)
    if n == 0: return 0.0
    k = np.arange(n + 1); ucb = np.where(k < n, beta.ppf(1 - delta, k + 1, np.maximum(n - k, 1)), 1.0)
    ok = np.where(ucb <= a)[0]
    if len(ok) == 0: return 0.0                       # infeasible with this many referable cases: refer everyone
    kk = ok.max(); return sp[kk] if kk < n else 1.0
def label_until(Lab, K):
    c = np.cumsum(Lab.y.values); j = np.searchsorted(c, K); m = j + 1 if j < len(c) else len(c)
    L = Lab.iloc[:m]; return L.s.values[L.y.values == 1], m
# ---------- (3) true-risk evaluation: weighted over the full test pool
def ev(name, a, th, Tt, extra=None):
    w, st, pos = Tt.w.values, Tt.s.values, Tt.y.values == 1
    return dict(method=name, alpha=a, thr=th, miss=(w * (pos & (st < th))).sum() / max((w * pos).sum(), 1e-12),
                refer=(w * (st >= th)).sum() / w.sum(), false_refer=(w * ((st >= th) & ~pos)).sum() / w.sum(), **(extra or {}))
def w_oracle(Tt, a):
    P = Tt[Tt.y == 1]; o = np.argsort(P.s.values); s = P.s.values[o]; c = np.cumsum(P.w.values[o]) / P.w.values.sum()
    return s[min(np.searchsorted(c, a, side="right"), len(s) - 1)]
def run(C, Ta, Tt, lr, val, pca, rng, base):
    rows = V.methods(C, Ta, Tt, lr, val, pca, rng); yC = C.y.values
    _, _, p_s = V3.gof(Ta.s.values, C.s.values[yC == 1], C.s.values[yC == 0], rng)
    piB = rows[0]["piT_bbse"]; Zs = pca.transform(np.stack(C.f)); Zt = pca.transform(np.stack(Ta.f))
    neff = min(len(Zt), len(Zs) // 2); Zt = Zt[rng.choice(len(Zt), neff, replace=False)]
    A, p_f = feat_test(Zs, yC, Zt, piB, rng); flag3 = bool(p_s < 0.05); flag4 = bool(p_s < 0.025 or p_f < 0.025)
    Lab = Ta.sample(frac=1.0, random_state=int(rng.integers(1 << 30))); out = []
    for a in ALPHAS:
        th = {r["method"]: r["thr"] for r in rows if r["alpha"] == a}
        cand = dict(FIXED=th["FIXED"], CRC=th["CRC"], NAIVE=th["NAIVE"], BBSE_l1=th["BBSE_l1"], ORACLE=w_oracle(Tt, a))
        out += [ev(k, a, v, Tt) for k, v in cand.items()]
        for K in KS:
            sp, m = label_until(Lab, K); ex = dict(K=K, n_lab=m, n_lab_pos=len(sp))
            tc = V.thr_crc(sp, np.ones(len(sp)), a) if len(sp) else 0.0; tp = thr_cp(sp, a); bb = cand["BBSE_l1"]
            out += [ev(f"TGT_CRC_K{K}", a, tc, Tt, ex), ev(f"TGT_CP_K{K}", a, tp, Tt, ex), ev(f"G3_CRC_K{K}", a, tc if flag3 else bb, Tt, ex),
                    ev(f"G4_CRC_K{K}", a, tc if flag4 else bb, Tt, ex), ev(f"G4_CP_K{K}", a, tp if flag4 else bb, Tt, ex)]
    info = dict(gof_p=p_s, feat_auc=A, feat_p=p_f, flag3=flag3, flag4=flag4, piT_bbse=piB, piT_true=float((Tt.w * Tt.y).sum() / Tt.w.sum()), **base)
    return [dict(**info, **r) for r in out]
def scen_w(P, sc):
    w = np.ones(len(P))
    if "prev_high" in sc: w *= np.where(P.y == 1, 3.0, 1.0)
    if "prev_low" in sc: w *= np.where(P.y == 1, 0.4, 1.0)
    if sc == "quality": w *= np.where(P.q.fillna(-1) >= 1, 8.0, np.where(P.q.fillna(-1) == 0, 1.0, 0.0))
    return w
def as_scen(T, sc, deg):
    return deg[sc[4:].split("+")[0]].loc[T.patient].reset_index() if sc.startswith("deg:") else T.reset_index(drop=True)
def e1_task(args):
    seed, rep = args; U, deg, lr, calib, pca = V.load(seed); val = U[U.role == "val"]; rng = np.random.default_rng(50_000 * seed + rep)
    perm = rng.permutation(len(calib)); n = len(perm); C = calib.iloc[perm[:n // 2]]
    Pa = calib.iloc[perm[n // 2:3 * n // 4]].reset_index(drop=True); Pt = calib.iloc[perm[3 * n // 4:]].reset_index(drop=True)
    scen = ["none", "prev_high", "prev_low", "quality"] + ([] if SMOKE else [f"deg:{c}" for c in V.CONDS] + [f"deg:{k}2+prev_high" for k in ["blur", "contrast", "dark", "colorcast"]])
    res = []
    for sc in scen:
        wa = scen_w(Pa, sc); k = int(min(600, (wa > 0).sum())); Ta = as_scen(Pa.iloc[rng.choice(len(Pa), k, replace=False, p=wa / wa.sum())], sc, deg)
        wt = scen_w(Pt, sc); Tt = as_scen(Pt, sc, deg); Tt["w"] = wt; Tt = Tt[Tt.w > 0].reset_index(drop=True)
        res += run(C, Ta, Tt, lr, val, pca, rng, dict(exp="E1", seed=seed, rep=rep, site=sc))
    return res
def e2_task(args):
    seed, site, rep = args; U, deg, lr, calib, _ = V.load(seed); val = U[U.role == "val"]
    rng = np.random.default_rng(60_000 * seed + 100 * rep + ["messidor2", "deepdrid", "ddr", "aptos", "idrid"].index(site))
    S = U[U.ds == site].reset_index(drop=True); pca = PCA(32, random_state=0).fit(np.vstack([np.stack(calib.f), np.stack(S.f)]))
    perm = rng.permutation(len(S)); h = len(S) // 2; Tt = S.iloc[perm[h:]].reset_index(drop=True); Tt["w"] = 1.0
    return run(calib, S.iloc[perm[:h]].reset_index(drop=True), Tt, lr, val, pca, rng, dict(exp="E2", seed=seed, rep=rep, site=site))
if __name__ == "__main__":
    t0 = time.time(); sites = ["messidor2", "deepdrid", "ddr", "aptos", "idrid"]
    T1 = [(s, r) for s in V.SEEDS for r in range(V.REPS)]; T2 = [(s, si, r) for s in V.SEEDS for si in sites for r in range(V.REPS)]
    with Pool(1 if SMOKE else 36) as pool:
        R1 = [x for o in pool.imap_unordered(e1_task, T1) for x in o]; print(json.dumps(dict(stage="E1", rows=len(R1), mins=round((time.time() - t0) / 60, 1))), flush=True)
        R2 = [x for o in pool.imap_unordered(e2_task, T2) for x in o]; print(json.dumps(dict(stage="E2", rows=len(R2), mins=round((time.time() - t0) / 60, 1))), flush=True)
    tag = ("smoke4" if SMOKE else "v4") + os.environ.get("RUN_TAG", ""); R = pd.DataFrame(R1 + R2); R.to_parquet(f"{W}/results/e_{tag}_raw.parquet")
    R["viol"] = R.miss > R.alpha
    S = R.groupby(["exp", "site", "alpha", "method"]).agg(miss=("miss", "mean"), violation=("viol", "mean"), refer=("refer", "mean"), false_refer=("false_refer", "mean"),
        flag3=("flag3", "mean"), flag4=("flag4", "mean"), n_lab=("n_lab", "mean"), n_lab_pos=("n_lab_pos", "mean")).reset_index().round(4)
    S.to_csv(f"{W}/results/e_{tag}_summary.csv", index=False)
    print(R.groupby(["exp", "site"])[["flag3", "flag4"]].mean().round(3).to_string()); print(f"{tag.upper()} DONE", flush=True)
