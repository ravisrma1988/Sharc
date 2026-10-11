import os, sys, functools
os.environ.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
import numpy as np, pandas as pd, json, time
from multiprocessing import Pool
from sklearn.decomposition import PCA
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e_v2 as V, e_v3 as V3, e_v4 as V4
W = V.W; ALPHAS = V.ALPHAS; SMOKE = V.SMOKE; BB = V.BB
KMIN = {a: int(np.ceil(np.log(V4.DELTA) / np.log(1 - a))) for a in ALPHAS}          # 0.02 -> 114, 0.05 -> 45, 0.10 -> 22
GAMMAS = [0.05, 0.10, 0.20]; DEFER = [0.10, 0.20, 0.30]
@functools.lru_cache(None)
def load5(seed):
    """v2 units plus ordinal-referral score (max over eyes of max_{g>=2} p_g) and image-level LAC nonconformities per patient."""
    U, deg, lr, calib, pca = V.load(seed)
    def ordinal(d):
        d = d.assign(so=d[["p_g2", "p_g3", "p_g4"]].max(1), nc=1 - d[[f"p_g{k}" for k in range(5)]].values[np.arange(len(d)), d.grade.values])
        return d.groupby("patient").agg(so=("so", "max"), nc=("nc", list))
    O = ordinal(pd.read_parquet(f"{W}/preds/{BB}_s{seed}.parquet"))
    for T in (U, calib): T["so"] = O.so.reindex(T.patient).values; T["nc"] = O.nc.reindex(T.patient).values
    for c, D in deg.items():
        Od = ordinal(pd.read_parquet(f"{W}/preds/degraded/{BB}_s{seed}_{c}.parquet")); D["so"] = Od.so.reindex(D.index).values
    return U, deg, lr, calib, pca
def tau_hi(s_neg, g):
    s = np.sort(s_neg)[::-1]; k = int(np.floor(g * (len(s) + 1) - 1))      # allowed false auto-referrals among labelled negatives
    return 1.0 + 1e-9 if k < 0 or len(s) == 0 else (s[k] + 1e-9 if k < len(s) else 0.0)
def ev3(name, a, lo, hi, Tt, extra=None):
    w, s, pos = Tt.w.values, Tt.s.values, Tt.y.values == 1; hi = max(hi, lo); auto, nr = s >= hi, s < lo
    return dict(method=name, alpha=a, thr=lo, thr_hi=hi, miss=(w * (pos & nr)).sum() / max((w * pos).sum(), 1e-12), refer=(w * (s >= lo)).sum() / w.sum(),
                auto=(w * auto).sum() / w.sum(), defer=(w * (~auto & ~nr)).sum() / w.sum(), spec_false=(w * (auto & ~pos)).sum() / w.sum(),
                false_refer=(w * ((s >= lo) & ~pos)).sum() / w.sum(), **(extra or {}))
def b5(C, Tt, a):
    nc = np.concatenate(C.nc.values); n = len(nc); q = np.quantile(nc, min(1, np.ceil((n + 1) * (1 - a)) / n), method="higher")
    T2 = Tt.assign(s=Tt.so.values); r = V4.ev("B5_ordinal_LAC", a, 1 - q, T2); return r
def b6(Ta, Tt, fixed, a):
    H = lambda p: -(p * np.log(p + 1e-12) + (1 - p) * np.log(1 - p + 1e-12)); out = []
    for d in DEFER:
        h0 = np.quantile(H(Ta.s.values), 1 - d); w, s, pos = Tt.w.values, Tt.s.values, Tt.y.values == 1; dfr = H(s) >= h0; nr = (s < fixed) & ~dfr
        out.append(dict(method=f"B6_entropy_d{int(d * 100)}", alpha=a, thr=fixed, miss=(w * (pos & nr)).sum() / max((w * pos).sum(), 1e-12),
                        refer=(w * ((s >= fixed) & ~dfr)).sum() / w.sum(), defer=(w * dfr).sum() / w.sum(), false_refer=(w * ((s >= fixed) & ~dfr & ~pos)).sum() / w.sum()))
    return out
def run(C, Ta, Lab, Tt, lr, val, pca, rng, base):
    rows = V.methods(C, Ta, Tt, lr, val, pca, rng); yC = C.y.values
    _, _, p_s = V3.gof(Ta.s.values, C.s.values[yC == 1], C.s.values[yC == 0], rng)
    piB = rows[0]["piT_bbse"]; Zs = pca.transform(np.stack(C.f)); Zt = pca.transform(np.stack(Ta.f))
    neff = min(len(Zt), len(Zs) // 2); A, p_f = V4.feat_test(Zs, yC, Zt[rng.choice(len(Zt), neff, replace=False)], piB, rng)
    flag = bool(p_s < 0.025 or p_f < 0.025); out = []; s0src = C.s.values[yC == 0]
    for a in ALPHAS:
        th = {r["method"]: r["thr"] for r in rows if r["alpha"] == a}; bb = th["BBSE_l1"]
        out += [V4.ev(k, a, v, Tt) for k, v in dict(FIXED=th["FIXED"], CRC=th["CRC"], NAIVE=th["NAIVE"], BBSE_l1=bb, ORACLE=V4.w_oracle(Tt, a)).items()]
        out += [b5(C, Tt, a)] + b6(Ta, Tt, th["FIXED"], a)
        for K in (KMIN[a], 2 * KMIN[a]):
            sp, m = V4.label_until(Lab, K); L = Lab.iloc[:m]; s0lab = L.s.values[L.y.values == 0]; ex = dict(K=K, n_lab=m, n_lab_pos=len(sp))
            tc = V.thr_crc(sp, np.ones(len(sp)), a) if len(sp) else 0.0; tp = V4.thr_cp(sp, a)
            out += [V4.ev(f"TGT_CRC_K{K}", a, tc, Tt, ex), V4.ev(f"TGT_CP_K{K}", a, tp, Tt, ex),
                    V4.ev(f"G4_CRC_K{K}", a, tc if flag else bb, Tt, ex), V4.ev(f"G4_CP_K{K}", a, tp if flag else bb, Tt, ex)]
            if K == KMIN[a]:
                for g in GAMMAS:
                    hi = tau_hi(s0lab if flag else s0src, g)
                    out += [ev3(f"G4_CRC_K{K}_3w_g{int(g * 100)}", a, tc if flag else bb, hi, Tt, ex), ev3(f"G4_CP_K{K}_3w_g{int(g * 100)}", a, tp if flag else bb, hi, Tt, ex)]
    info = dict(gof_p=p_s, feat_auc=A, feat_p=p_f, flag=flag, piT_bbse=piB, piT_true=float((Tt.w * Tt.y).sum() / Tt.w.sum()), **base)
    return [dict(**info, **r) for r in out]
def e1_task(args):
    seed, rep = args; U, deg, lr, calib, pca = load5(seed); val = U[U.role == "val"]; rng = np.random.default_rng(70_000 * seed + rep)
    perm = rng.permutation(len(calib)); n = len(perm); C = calib.iloc[perm[:n // 2]]
    Pa = calib.iloc[perm[n // 2:3 * n // 4]].reset_index(drop=True); Pt = calib.iloc[perm[3 * n // 4:]].reset_index(drop=True)
    scen = ["none", "prev_high", "prev_low", "quality"] + ([] if SMOKE else [f"deg:{c}" for c in V.CONDS] + [f"deg:{k}2+prev_high" for k in ["blur", "contrast", "dark", "colorcast"]])
    res = []
    for sc in scen:
        wa = V4.scen_w(Pa, sc); nz = int((wa > 0).sum()); order = rng.choice(len(Pa), nz, replace=False, p=wa / wa.sum())
        Pseq = V4.as_scen(Pa.iloc[order], sc, deg); Ta = Pseq.iloc[:600]; Lab = Pseq          # unlabelled 600 for diagnostics; labels revealed in this order
        Tt = V4.as_scen(Pt, sc, deg); Tt["w"] = V4.scen_w(Pt, sc); Tt = Tt[Tt.w > 0].reset_index(drop=True)
        res += run(C, Ta, Lab, Tt, lr, val, pca, rng, dict(exp="E1", seed=seed, rep=rep, site=sc))
    return res
def e2_task(args):
    seed, site, rep = args; U, deg, lr, calib, _ = load5(seed); val = U[U.role == "val"]
    rng = np.random.default_rng(80_000 * seed + 100 * rep + ["messidor2", "deepdrid", "ddr", "aptos", "idrid"].index(site))
    S = U[U.ds == site].reset_index(drop=True); pca = PCA(32, random_state=0).fit(np.vstack([np.stack(calib.f), np.stack(S.f)]))
    perm = rng.permutation(len(S)); h = len(S) // 2; Ta = S.iloc[perm[:h]].reset_index(drop=True); Tt = S.iloc[perm[h:]].reset_index(drop=True); Tt["w"] = 1.0
    return run(calib, Ta, Ta, Tt, lr, val, pca, rng, dict(exp="E2", seed=seed, rep=rep, site=site))
if __name__ == "__main__":
    t0 = time.time(); sites = ["messidor2", "deepdrid", "ddr", "aptos", "idrid"]
    T1 = [(s, r) for s in V.SEEDS for r in range(V.REPS)]; T2 = [(s, si, r) for s in V.SEEDS for si in sites for r in range(V.REPS)]
    with Pool(1 if SMOKE else 36) as pool:
        R1 = [x for o in pool.imap_unordered(e1_task, T1) for x in o]; print(json.dumps(dict(stage="E1", rows=len(R1), mins=round((time.time() - t0) / 60, 1))), flush=True)
        R2 = [x for o in pool.imap_unordered(e2_task, T2) for x in o]; print(json.dumps(dict(stage="E2", rows=len(R2), mins=round((time.time() - t0) / 60, 1))), flush=True)
    tag = ("smoke5" if SMOKE else "v5") + os.environ.get("RUN_TAG", ""); R = pd.DataFrame(R1 + R2); R.to_parquet(f"{W}/results/e_{tag}_raw.parquet")
    R["viol"] = R.miss > R.alpha
    agg = dict(miss=("miss", "mean"), violation=("viol", "mean"), refer=("refer", "mean"), false_refer=("false_refer", "mean"), auto=("auto", "mean"),
               defer=("defer", "mean"), spec_false=("spec_false", "mean"), flag=("flag", "mean"), n_lab=("n_lab", "mean"), n_lab_pos=("n_lab_pos", "mean"))
    S = R.groupby(["exp", "site", "alpha", "method"]).agg(**agg).reset_index().round(4); S.to_csv(f"{W}/results/e_{tag}_summary.csv", index=False)
    print(R.groupby(["exp", "site"]).flag.mean().round(3).to_string())
    s = S[S.alpha == 0.05]; print(s[s.method.isin(["CRC", "B5_ordinal_LAC", "B6_entropy_d20", "G4_CRC_K45", "G4_CP_K45", "G4_CP_K45_3w_g10"])].pivot_table(index=["exp", "site"], columns="method", values="miss").round(3).to_string())
    print(f"{tag.upper()} DONE", flush=True)
