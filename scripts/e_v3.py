import os, sys
os.environ.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
import numpy as np, pandas as pd, json, time
from multiprocessing import Pool
from sklearn.decomposition import PCA
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e_v2 as V            # reuses load(), methods(), thr_crc(), thr_pac(); reads BACKBONE / SEEDS / REPS from env
W = V.W; ALPHAS = V.ALPHAS; BUDGETS = [50, 100, 200, 400]; BGOF = 200; SMOKE = V.SMOKE
PIS = np.linspace(0, 1, 201)
def gof_stat(x, a1, a0):
    x = np.sort(x); n = len(x); e = np.arange(1, n + 1) / n
    F1 = np.searchsorted(np.sort(a1), x, side="right") / len(a1); F0 = np.searchsorted(np.sort(a0), x, side="right") / len(a0)
    M = PIS[:, None] * F1[None] + (1 - PIS[:, None]) * F0[None]
    d = np.maximum(np.abs(M - e[None]), np.abs(M - (e - 1 / n)[None])).max(1); j = d.argmin(); return d[j], PIS[j]
def gof(sT, s1, s0, rng):
    """Is the target score distribution a mixture of the source class-conditional score distributions? Bootstrap p-value."""
    D, pi_hat = gof_stat(sT, s1, s0); n = len(sT); Db = np.empty(BGOF)
    for b in range(BGOF):
        a1, a0 = rng.choice(s1, len(s1)), rng.choice(s0, len(s0))             # independent re-draw of the source sample
        z = rng.random(n) < pi_hat; x = np.where(z, rng.choice(s1, n), rng.choice(s0, n)); Db[b] = gof_stat(x, a1, a0)[0]   # pseudo-target from the original source; compared with the re-drawn source
    return D, pi_hat, (1 + (Db >= D).sum()) / (BGOF + 1)
def run(C, Ta, Tt, lr, val, pca, rng, base):
    rows = V.methods(C, Ta, Tt, lr, val, pca, rng)
    D, pim, p = gof(Ta.s.values, C.s.values[C.y.values == 1], C.s.values[C.y.values == 0], rng); flag = bool(p < 0.05)
    Lab = Ta.sample(frac=1.0, random_state=int(rng.integers(1 << 30)))       # labels revealed in this random order
    st, yt = Tt.s.values, Tt.y.values; out = []
    def sc(name, a, th, npos, m): return dict(method=name, alpha=a, thr=th, miss=((st < th) & (yt == 1)).sum() / max(1, (yt == 1).sum()),
                                              refer=(st >= th).mean(), false_refer=((st >= th) & (yt == 0)).mean(), n_lab=m, n_lab_pos=npos)
    for a in ALPHAS:
        bb = [r["thr"] for r in rows if r["method"] == "BBSE_l1" and r["alpha"] == a][0]
        for m in BUDGETS:
            L = Lab.iloc[:m]; sp = L.s.values[L.y.values == 1]; mm = len(L)
            tc = V.thr_crc(sp, np.ones(len(sp)), a) if len(sp) else 0.0; tp = V.thr_pac(sp, np.ones(len(sp)), a, rng) if len(sp) > 1 else 0.0
            out += [sc(f"TGT_CRC_{m}", a, tc, len(sp), mm), sc(f"TGT_PAC_{m}", a, tp, len(sp), mm),
                    sc(f"GATE_CRC_{m}", a, tc if flag else bb, len(sp), mm), sc(f"GATE_PAC_{m}", a, tp if flag else bb, len(sp), mm)]
    keep = ("method", "alpha", "thr", "miss", "refer", "false_refer", "piT_true", "piT_bbse", "ess_bbse")
    info = dict(gof_D=D, gof_pi=pim, gof_p=p, flag=flag, **base)
    return [dict(**info, **{k: r[k] for k in keep if k in r}) for r in rows] + [dict(**info, **r) for r in out]
def e1_task(args):
    seed, rep = args; U, deg, lr, calib, pca = V.load(seed); val = U[U.role == "val"]; rng = np.random.default_rng(30_000 * seed + rep)
    perm = rng.permutation(len(calib)); n = len(perm); C = calib.iloc[perm[:n // 2]]
    Pa = calib.iloc[perm[n // 2:3 * n // 4]].reset_index(drop=True); Pt = calib.iloc[perm[3 * n // 4:]].reset_index(drop=True)
    scen = ["none", "prev_high", "prev_low", "quality"] + ([] if SMOKE else [f"deg:{c}" for c in V.CONDS] + [f"deg:{k}2+prev_high" for k in ["blur", "contrast", "dark", "colorcast"]])
    res = []
    for sc in scen:
        def pick(P, k, repl):
            w = np.ones(len(P))
            if "prev_high" in sc: w *= np.where(P.y == 1, 3.0, 1.0)
            if "prev_low" in sc: w *= np.where(P.y == 1, 0.4, 1.0)
            if sc == "quality": w *= np.where(P.q.fillna(-1) >= 1, 8.0, np.where(P.q.fillna(-1) == 0, 1.0, 0.0))
            k = k if repl else int(min(k, (w > 0).sum())); T = P.iloc[rng.choice(len(P), k, replace=repl, p=w / w.sum())]
            if sc.startswith("deg:"): T = deg[sc[4:].split("+")[0]].loc[T.patient].reset_index()
            return T
        res += run(C, pick(Pa, 600, False), pick(Pt, 3000, True), lr, val, pca, rng, dict(exp="E1", seed=seed, rep=rep, site=sc))
    return res
def e2_task(args):
    seed, site, rep = args; U, deg, lr, calib, _ = V.load(seed); val = U[U.role == "val"]
    rng = np.random.default_rng(40_000 * seed + 100 * rep + ["messidor2", "deepdrid", "ddr", "aptos", "idrid"].index(site))
    S = U[U.ds == site].reset_index(drop=True); pca = PCA(32, random_state=0).fit(np.vstack([np.stack(calib.f), np.stack(S.f)]))
    perm = rng.permutation(len(S)); h = len(S) // 2
    return run(calib, S.iloc[perm[:h]], S.iloc[perm[h:]], lr, val, pca, rng, dict(exp="E2", seed=seed, rep=rep, site=site))
if __name__ == "__main__":
    t0 = time.time(); sites = ["messidor2", "deepdrid", "ddr", "aptos", "idrid"]
    T1 = [(s, r) for s in V.SEEDS for r in range(V.REPS)]; T2 = [(s, si, r) for s in V.SEEDS for si in sites for r in range(V.REPS)]
    with Pool(1 if SMOKE else 36) as pool:
        R1 = [x for o in pool.imap_unordered(e1_task, T1) for x in o]; print(json.dumps(dict(stage="E1", rows=len(R1), mins=round((time.time() - t0) / 60, 1))), flush=True)
        R2 = [x for o in pool.imap_unordered(e2_task, T2) for x in o]; print(json.dumps(dict(stage="E2", rows=len(R2), mins=round((time.time() - t0) / 60, 1))), flush=True)
    tag = ("smoke" if SMOKE else "v3") + os.environ.get("RUN_TAG", ""); R = pd.DataFrame(R1 + R2); R.to_parquet(f"{W}/results/e_{tag}_raw.parquet")
    R["viol"] = R.miss > R.alpha
    S = R.groupby(["exp", "site", "alpha", "method"]).agg(miss=("miss", "mean"), violation=("viol", "mean"), refer=("refer", "mean"),
        false_refer=("false_refer", "mean"), flag_rate=("flag", "mean"), n_lab_pos=("n_lab_pos", "mean")).reset_index().round(4)
    S.to_csv(f"{W}/results/e_{tag}_summary.csv", index=False)
    print(R.groupby(["exp", "site"]).flag.mean().round(3).to_string()); print(f"{tag.upper()} DONE", flush=True)
