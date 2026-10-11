import os, sys
os.environ.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
import numpy as np, pandas as pd, json, time, functools
from multiprocessing import Pool
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
W = "/path/to/sharc_work"; ALPHAS = [0.02, 0.05, 0.10]; B = 200
SMOKE = len(sys.argv) > 1 and sys.argv[1] == "smoke"; REPS = 1 if SMOKE else int(os.environ.get("REPS", "100"))
BB = os.environ.get("BACKBONE", "convnext_tiny"); SEEDS = [int(s) for s in os.environ.get("SEEDS", "0" if SMOKE else "0,1,2").split(",")]
CONDS = [f"{k}{l}" for k in ["blur", "contrast", "dark", "colorcast"] for l in [1, 2, 3]]
def to_units(d, F):
    d = d.copy(); d["fi"] = np.arange(len(d))
    g = d.groupby(["ds", "role", "patient"]).agg(s=("p_ref", "max"), y=("referable", "max"), fi=("fi", list)).reset_index()
    g["f"] = [F[ix].astype(np.float32).mean(0) for ix in g.fi]; return g.drop(columns="fi")
@functools.lru_cache(None)
def load(seed):
    d = pd.read_parquet(f"{W}/preds/{BB}_s{seed}.parquet"); F = np.load(f"{W}/preds/{BB}_s{seed}_feat.npy")
    q = pd.read_parquet(f"{W}/data/proc/metadata_rows.parquet")[["row", "eyeq_quality"]]; d = d.merge(q, on="row", how="left")
    U = to_units(d, F); U["q"] = d.groupby("patient").eyeq_quality.max().reindex(U.patient).values
    deg = {}
    if not SMOKE:
        for c in CONDS:
            dd = pd.read_parquet(f"{W}/preds/degraded/{BB}_s{seed}_{c}.parquet"); FF = np.load(f"{W}/preds/degraded/{BB}_s{seed}_{c}_feat.npy")
            deg[c] = to_units(dd, FF).set_index("patient")
    val = U[U.role == "val"]; lr = LogisticRegression(C=1e6).fit(np.log(val.s / (1 - val.s + 1e-9) + 1e-9).values[:, None], val.y)
    calib = U[U.role == "calib"].reset_index(drop=True); pca_c = PCA(32, random_state=0).fit(np.stack(calib.f))
    return U, deg, lr, calib, pca_c
def cal(lr, s): return lr.predict_proba(np.log(s / (1 - s + 1e-9) + 1e-9)[:, None])[:, 1]
def thr_crc(s, w, a):
    o = np.argsort(s); s, w = s[o], w[o] / w.mean(); wm = w.max(); cum = np.concatenate([[0], np.cumsum(w)]); R = (cum + wm) / (w.sum() + wm)
    if not (R <= a).any(): return 0.0          # even zero misses cannot certify alpha: refer everyone (audit fix, 10 Oct 2026)
    k = np.max(np.where(R <= a)[0]); return s[k] if k < len(s) else 1.0
def thr_pac(s, w, a, rng, delta=0.1):
    grid = np.quantile(s, np.linspace(0, 0.5, 300)); n = len(s); w = w / w.mean(); idx = rng.integers(0, n, (B, n)); ws = w[idx]
    fnr = np.stack([(ws * (s[idx] < t)).sum(1) / ws.sum(1) for t in grid], 1); ucb = np.quantile(fnr, 1 - delta, axis=0)
    ok = np.where(ucb <= a)[0]; return grid[ok.max()] if len(ok) else 0.0
def dom(Zs, Zt, sw):
    X = np.vstack([Zs, Zt]); y = np.r_[np.zeros(len(Zs)), np.ones(len(Zt))]; w = np.r_[sw * len(Zt) / sw.sum(), np.ones(len(Zt))]
    p = LogisticRegression(C=0.1, max_iter=3000).fit(X, y, sample_weight=w).predict_proba(Zs)[:, 1]; g = p / (1 - p); return np.minimum(g, np.quantile(g, 0.99))
def em(qT, piS, cy):
    pi = 0.5
    for _ in range(200):
        r1, r0 = pi / (piS * cy[1]), (1 - pi) / ((1 - piS) * cy[0]); pi = (r1 * qT / (r1 * qT + r0 * (1 - qT))).mean()
    return pi
def bbse(qC, yC, qT):
    a, b = (qC[yC == 1] > .5).mean(), (qC[yC == 0] > .5).mean(); return float(np.clip(((qT > .5).mean() - b) / max(a - b, 1e-3), 0.01, 0.99))
def ess(w): return w.sum() ** 2 / (w ** 2).sum()
def methods(C, Ta, Tt, lr, val, pca, rng):
    piS = C.y.mean(); yC = C.y.values; pos = yC == 1; sP = C.s.values[pos]; Zs, Zt = pca.transform(np.stack(C.f)), pca.transform(np.stack(Ta.f))
    qT, qC = cal(lr, Ta.s.values), cal(lr, C.s.values)
    gN = dom(Zs, Zt, np.ones(len(C)))
    piE, cy = piS, {0: 1., 1: 1.}
    for _ in range(3):
        piE = em(qT, piS, cy); gE = dom(Zs, Zt, np.where(pos, piE / piS, (1 - piE) / (1 - piS))); cy = {1: gE[pos].mean(), 0: gE[~pos].mean()}
    piB = bbse(qC, yC, qT); gB = dom(Zs, Zt, np.where(pos, piB / piS, (1 - piB) / (1 - piS)))
    info = dict(piT_true=Tt.y.mean(), piT_em=piE, piT_bbse=piB, ess_naive=ess(gN[pos]), ess_em=ess(gE[pos]), ess_bbse=ess(gB[pos]), ess_bbse05=ess(gB[pos] ** .5), n_pos=int(pos.sum()))
    out = []; st, yt = Tt.s.values, Tt.y.values
    for a in ALPHAS:
        vp = np.sort(val[val.y == 1].s.values); tp = np.sort(st[yt == 1])
        t = dict(FIXED=vp[int(a * len(vp))], CRC=thr_crc(sP, np.ones(len(sP)), a), NAIVE=thr_crc(sP, gN[pos], a),
                 EM_l1=thr_crc(sP, gE[pos], a), EM_l05=thr_crc(sP, gE[pos] ** .5, a), BBSE_l1=thr_crc(sP, gB[pos], a), BBSE_l05=thr_crc(sP, gB[pos] ** .5, a),
                 PAC_BBSE_l1=thr_pac(sP, gB[pos], a, rng), PAC_BBSE_l05=thr_pac(sP, gB[pos] ** .5, a, rng), ORACLE=tp[int(a * len(tp))] if len(tp) else np.nan)
        for k, th in t.items():
            out.append(dict(method=k, alpha=a, thr=th, miss=((st < th) & (yt == 1)).sum() / max(1, (yt == 1).sum()), refer=(st >= th).mean(),
                            false_refer=((st >= th) & (yt == 0)).mean(), **info))
    return out
def e1_task(args):
    seed, rep = args; U, deg, lr, calib, pca = load(seed); val = U[U.role == "val"]; rng = np.random.default_rng(10_000 * seed + rep)
    perm = rng.permutation(len(calib)); n = len(perm); C = calib.iloc[perm[:n // 2]]; Pa = calib.iloc[perm[n // 2:3 * n // 4]].reset_index(drop=True); Pt = calib.iloc[perm[3 * n // 4:]].reset_index(drop=True)
    scen = ["none", "prev_high", "prev_low", "quality"] + ([] if SMOKE else [f"deg:{c}" for c in CONDS] + [f"deg:{k}2+prev_high" for k in ["blur", "contrast", "dark", "colorcast"]])
    res = []
    for sc in scen:
        def pick(P, n, repl):
            w = np.ones(len(P))
            if "prev_high" in sc: w *= np.where(P.y == 1, 3.0, 1.0)
            if "prev_low" in sc: w *= np.where(P.y == 1, 0.4, 1.0)
            if sc == "quality": w *= np.where(P.q.fillna(-1) >= 1, 8.0, np.where(P.q.fillna(-1) == 0, 1.0, 0.0))
            n = n if repl else int(min(n, (w > 0).sum())); T = P.iloc[rng.choice(len(P), n, replace=repl, p=w / w.sum())]
            if sc.startswith("deg:"):
                D = deg[sc[4:].split("+")[0]]; T = D.loc[T.patient].reset_index()
            return T
        for r in methods(C, pick(Pa, 600, False), pick(Pt, 3000, True), lr, val, pca, rng): res.append(dict(exp="E1", seed=seed, rep=rep, site=sc, **r))
    return res
def e2_task(args):
    seed, site, rep = args; U, deg, lr, calib, _ = load(seed); val = U[U.role == "val"]; rng = np.random.default_rng(20_000 * seed + 100 * rep + ["messidor2", "deepdrid", "ddr", "aptos", "idrid"].index(site))
    S = U[U.ds == site].reset_index(drop=True); pca = PCA(32, random_state=0).fit(np.vstack([np.stack(calib.f), np.stack(S.f)]))
    perm = rng.permutation(len(S)); h = len(S) // 2
    return [dict(exp="E2", seed=seed, rep=rep, site=site, **r) for r in methods(calib, S.iloc[perm[:h]], S.iloc[perm[h:]], lr, val, pca, rng)]
if __name__ == "__main__":
    t0 = time.time(); sites = ["messidor2", "deepdrid", "ddr", "aptos", "idrid"]
    tasks1 = [(s, r) for s in SEEDS for r in range(REPS)]; tasks2 = [(s, si, r) for s in SEEDS for si in sites for r in range(REPS)]
    with Pool(1 if SMOKE else 36) as pool:
        R1 = [x for out in pool.imap_unordered(e1_task, tasks1) for x in out]; print(json.dumps(dict(stage="E1", rows=len(R1), mins=round((time.time() - t0) / 60, 1))), flush=True)
        R2 = [x for out in pool.imap_unordered(e2_task, tasks2) for x in out]; print(json.dumps(dict(stage="E2", rows=len(R2), mins=round((time.time() - t0) / 60, 1))), flush=True)
    R = pd.DataFrame(R1 + R2); tag = ("smoke" if SMOKE else "v2") + os.environ.get("RUN_TAG", ""); R.to_parquet(f"{W}/results/e_{tag}_raw.parquet")
    R["viol"] = R.miss > R.alpha
    S = R.groupby(["exp", "site", "alpha", "method"]).agg(miss=("miss", "mean"), violation=("viol", "mean"), refer=("refer", "mean"), false_refer=("false_refer", "mean"),
        pi_true=("piT_true", "mean"), pi_em=("piT_em", "mean"), pi_bbse=("piT_bbse", "mean"), ess_naive=("ess_naive", "mean"), ess_em=("ess_em", "mean"),
        ess_bbse=("ess_bbse", "mean"), n_pos=("n_pos", "mean")).reset_index().round(4)
    S.to_csv(f"{W}/results/e_{tag}_summary.csv", index=False); print(S[S.alpha == 0.05].pivot_table(index=["exp", "site"], columns="method", values="miss").round(3).to_string())
    print(f"{tag.upper()} DONE", flush=True)
