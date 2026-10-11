import os, sys, functools
os.environ.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
import numpy as np, pandas as pd, json, time
from multiprocessing import Pool
from sklearn.decomposition import PCA
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e_v2 as V, e_v3 as V3, e_v4 as V4, e_v5_main as M5, e_v6 as V6
# v8 (audit response): (C4) gate and unlabelled estimation on one half of the adaptation set, target labels drawn only from
# the other half; (C2/C5) full three-way pathway accounting; (D4) concept-shift scenarios invisible to the gate;
# (B2) score-only and feature-only gates with the labelling budget actually spent.
W = V.W; ALPHAS = V.ALPHAS; SMOKE = V.SMOKE; BB = V.BB; KMIN = M5.KMIN; GAMMAS = M5.GAMMAS
SPLIT = int(os.environ.get("SPLIT", "1"))
@functools.lru_cache(None)
def gmax(seed):
    return pd.read_parquet(f"{W}/preds/{BB}_s{seed}.parquet").groupby("patient").grade.max()
def ev3x(name, a, lo, hi, Tt, extra=None):
    w, s, pos = Tt.w.values, Tt.s.values, Tt.y.values == 1; hi = max(hi, lo); auto, nr = s >= hi, s < lo; dfr = ~auto & ~nr
    P, N = (w * pos).sum(), (w * ~pos).sum()
    return dict(method=name, alpha=a, thr=lo, thr_hi=hi, miss=(w * (pos & nr)).sum() / max(P, 1e-12), pos_defer=(w * (pos & dfr)).sum() / max(P, 1e-12),
                pos_auto=(w * (pos & auto)).sum() / max(P, 1e-12), auto=(w * auto).sum() / w.sum(), defer=(w * dfr).sum() / w.sum(), clear=(w * nr).sum() / w.sum(),
                fpr_auto=(w * (auto & ~pos)).sum() / max(N, 1e-12), spec_false=(w * (auto & ~pos)).sum() / w.sum(), refer=(w * (s >= lo)).sum() / w.sum(), **(extra or {}))
def run8(C, Ta_all, Lab_pool, Tt, lr, val, pca, rng, base):
    if SPLIT:
        idx = rng.permutation(len(Ta_all)); Ta = Ta_all.iloc[idx[:len(idx) // 2]]
        Lab = Lab_pool[~Lab_pool.patient.isin(set(Ta.patient))]            # labels only from units the gate never saw
    else:
        Ta, Lab = Ta_all, Lab_pool
    rows = V.methods(C, Ta, Tt, lr, val, pca, rng); yC = C.y.values
    _, _, p_s = V3.gof(Ta.s.values, C.s.values[yC == 1], C.s.values[yC == 0], rng)
    piB = rows[0]["piT_bbse"]; Zs = pca.transform(np.stack(C.f)); Zt = pca.transform(np.stack(Ta.f))
    neff = min(len(Zt), len(Zs) // 2); A_, p_f = V4.feat_test(Zs, yC, Zt[rng.choice(len(Zt), neff, replace=False)], piB, rng)
    flags = dict(G4=bool(p_s < .025 or p_f < .025), Gscore=bool(p_s < .05), Gfeat=bool(p_f < .05)); out = []; s0src = C.s.values[yC == 0]
    for a in ALPHAS:
        th = {r["method"]: r["thr"] for r in rows if r["alpha"] == a}; bb = th["BBSE_l1"]
        out += [V4.ev(k, a, v, Tt, dict(n_lab=0)) for k, v in dict(FIXED=th["FIXED"], CRC=th["CRC"], NAIVE=th["NAIVE"], BBSE_l1=bb, ORACLE=V4.w_oracle(Tt, a)).items()]
        out += [dict(M5.b5(C, Tt, a), n_lab=0)] + [dict(r, n_lab=0) for r in M5.b6(Ta, Tt, th["FIXED"], a)] + [dict(V6.b8(C, Ta, Tt, lr, a, rng), n_lab=0)]
        K = KMIN[a]; sp, m = V4.label_until(Lab, K); L = Lab.iloc[:m]; s0lab = L.s.values[L.y.values == 0]
        tc = V.thr_crc(sp, np.ones(len(sp)), a) if len(sp) else 0.0; tp = V4.thr_cp(sp, a); ex = dict(K=K, n_lab_pos=len(sp), pool_exhausted=int(len(sp) < K))
        out += [V4.ev("TGT_CRC", a, tc, Tt, dict(ex, n_lab=m)), V4.ev("TGT_CP", a, tp, Tt, dict(ex, n_lab=m))]
        for gname, fl in flags.items():
            out += [V4.ev(f"{gname}_CRC", a, tc if fl else bb, Tt, dict(ex, n_lab=m if fl else 0)), V4.ev(f"{gname}_CP", a, tp if fl else bb, Tt, dict(ex, n_lab=m if fl else 0))]
        for g in GAMMAS:
            hi = M5.tau_hi(s0lab if flags["G4"] else s0src, g); ex3 = dict(ex, n_lab=m if flags["G4"] else 0, gamma=g)
            out += [ev3x(f"G4_CRC_3w_g{int(g * 100)}", a, tc if flags["G4"] else bb, hi, Tt, ex3), ev3x(f"G4_CP_3w_g{int(g * 100)}", a, tp if flags["G4"] else bb, hi, Tt, ex3)]
    info = dict(gof_p=p_s, feat_auc=A_, feat_p=p_f, flag=flags["G4"], flag_score=flags["Gscore"], flag_feat=flags["Gfeat"], piT_bbse=piB,
                piT_true=float((Tt.w * Tt.y).sum() / Tt.w.sum()), n_gate=len(Ta), n_label_pool=len(Lab), split=SPLIT, **base)
    return [dict(**info, **r) for r in out]
def concept(P, sc, seed, rng):
    """Concept shift invisible in the images: the target uses a stricter referral standard (mild DR, grade 1, also referable)."""
    g = gmax(seed).reindex(P.patient).values; P = P.copy()
    if sc == "concept_g1": P["y"] = (g >= 1).astype(int)
    elif sc == "concept_half": coin = rng.random(len(P)) < 0.5; P["y"] = ((g >= 2) | ((g == 1) & coin)).astype(int)
    return P
def e1_task(args):
    seed, rep = args; U, deg, lr, calib, pca = M5.load5(seed); val = U[U.role == "val"]; rng = np.random.default_rng(120_000 * seed + rep)
    perm = rng.permutation(len(calib)); n = len(perm); C = calib.iloc[perm[:n // 2]]
    Pa0 = calib.iloc[perm[n // 2:3 * n // 4]].reset_index(drop=True); Pt0 = calib.iloc[perm[3 * n // 4:]].reset_index(drop=True)
    scen = ["none", "prev_high", "prev_low", "quality", "concept_g1", "concept_half"] + ([] if SMOKE else [f"deg:{c}" for c in V.CONDS] + [f"deg:{k}2+prev_high" for k in ["blur", "contrast", "dark", "colorcast"]])
    res = []
    for sc in scen:
        Pa, Pt = (concept(Pa0, sc, seed, rng), concept(Pt0, sc, seed, rng)) if sc.startswith("concept") else (Pa0, Pt0)
        wa = V4.scen_w(Pa, sc); nz = int((wa > 0).sum()); order = rng.choice(len(Pa), nz, replace=False, p=wa / wa.sum())
        Pseq = V4.as_scen(Pa.iloc[order], sc, deg); Tt = V4.as_scen(Pt, sc, deg); Tt["w"] = V4.scen_w(Pt, sc); Tt = Tt[Tt.w > 0].reset_index(drop=True)
        res += run8(C, Pseq.iloc[:600], Pseq, Tt, lr, val, pca, rng, dict(exp="E1", seed=seed, rep=rep, site=sc))
    return res
def e2_task(args):
    seed, site, rep = args; U, deg, lr, calib, _ = M5.load5(seed); val = U[U.role == "val"]
    rng = np.random.default_rng(130_000 * seed + 100 * rep + ["messidor2", "deepdrid", "ddr", "aptos", "idrid"].index(site))
    S = U[U.ds == site].reset_index(drop=True)
    if os.environ.get("DEDUP") == "1":                                         # sensitivity: remove within-site duplicate images
        drop = set(pd.read_csv(f"{W}/results/dedup_drop_list.csv").patient); S = S[~S.patient.isin(drop)].reset_index(drop=True)
    pca = PCA(32, random_state=0).fit(np.vstack([np.stack(calib.f), np.stack(S.f)]))
    perm = rng.permutation(len(S)); h = len(S) // 2; Ta = S.iloc[perm[:h]].reset_index(drop=True); Tt = S.iloc[perm[h:]].reset_index(drop=True); Tt["w"] = 1.0
    return run8(calib, Ta, Ta, Tt, lr, val, pca, rng, dict(exp="E2", seed=seed, rep=rep, site=site))
if __name__ == "__main__":
    t0 = time.time(); sites = ["messidor2", "deepdrid", "ddr", "aptos", "idrid"]
    T1 = [] if os.environ.get("STAGES") == "E2" else [(s, r) for s in V.SEEDS for r in range(V.REPS)]; T2 = [(s, si, r) for s in V.SEEDS for si in sites for r in range(V.REPS)]
    with Pool(1 if SMOKE else 36) as pool:
        R1 = [x for o in pool.imap_unordered(e1_task, T1) for x in o]; print(json.dumps(dict(stage="E1", rows=len(R1), mins=round((time.time() - t0) / 60, 1))), flush=True)
        R2 = [x for o in pool.imap_unordered(e2_task, T2) for x in o]; print(json.dumps(dict(stage="E2", rows=len(R2), mins=round((time.time() - t0) / 60, 1))), flush=True)
    tag = ("smoke8" if SMOKE else "v8") + os.environ.get("RUN_TAG", ""); R = pd.DataFrame(R1 + R2); R.to_parquet(f"{W}/results/e_{tag}_raw.parquet")
    R["viol"] = R.miss > R.alpha; a = R[R.alpha == 0.05]
    print(a.groupby(["exp", "site"])[["flag", "flag_score", "flag_feat"]].mean().round(3).to_string())
    print(a[a.method.isin(["CRC", "BBSE_l1", "TGT_CRC", "G4_CRC", "G4_CP", "ORACLE"])].pivot_table(index=["exp", "site"], columns="method", values="miss").round(3).to_string())
    print(f"{tag.upper()} DONE", flush=True)
