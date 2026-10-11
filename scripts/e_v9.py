import os, sys, functools
os.environ.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
import numpy as np, pandas as pd, json, time
from multiprocessing import Pool
from sklearn.decomposition import PCA
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e_v2 as V, e_v3 as V3, e_v4 as V4, e_v5_main as M5, e_v6 as V6, e_v8 as V8
# v9 (theory re-audit): the source calibration set is split into two DISJOINT patient pools per run:
#   C_gate -> reference for both gate tests and the prevalence estimate used by the feature test (never used for any threshold)
#   C_crc  -> source CRC lower threshold, source negatives for the upper threshold, and all source-side baselines
# Target side unchanged from v8: gate half A, label half B (labels revealed only if flagged), disjoint test set.
# Every run asserts pairwise disjointness of all patient pools and of each pool with EyePACS train / validation.
W = V.W; ALPHAS = V.ALPHAS; SMOKE = V.SMOKE; KMIN = M5.KMIN; GAMMAS = M5.GAMMAS
LINEAGE = []
def check_pools(seed, rep, site, pools, U):
    trval = set(U[U.role.isin(["train", "val"])].patient)
    names = list(pools); sets = {k: set(v) for k, v in pools.items()}
    for i, a in enumerate(names):
        assert not (sets[a] & trval), f"{a} overlaps train/val"
        for b in names[i + 1:]: assert not (sets[a] & sets[b]), f"overlap {a} / {b}"
    return dict(seed=seed, rep=rep, site=site, **{f"n_{k}": len(v) for k, v in sets.items()})
def run9(C_gate, C_crc, Ta_all, Lab_pool, Tt, lr, val, pca, rng, base, U):
    idx = rng.permutation(len(Ta_all)); Ta = Ta_all.iloc[idx[:len(idx) // 2]]; Lab = Lab_pool[~Lab_pool.patient.isin(set(Ta.patient))]
    lin = check_pools(base["seed"], base["rep"], base["site"], dict(C_gate=C_gate.patient, C_crc=C_crc.patient, gate_A=Ta.patient,
                label_B=Lab.patient, test=Tt.patient.unique()), U)
    rows = V.methods(C_crc, Ta, Tt, lr, val, pca, rng)                     # all source-side thresholds from C_crc only
    yG = C_gate.y.values; _, _, p_s = V3.gof(Ta.s.values, C_gate.s.values[yG == 1], C_gate.s.values[yG == 0], rng)
    piG = V.bbse(V.cal(lr, C_gate.s.values), yG, V.cal(lr, Ta.s.values))     # prevalence estimate for the feature test, from C_gate
    Zs, Zt = pca.transform(np.stack(C_gate.f)), pca.transform(np.stack(Ta.f)); neff = min(len(Zt), len(Zs) // 2)
    A_, p_f = V4.feat_test(Zs, yG, Zt[rng.choice(len(Zt), neff, replace=False)], piG, rng)
    flags = dict(G4=bool(p_s < .025 or p_f < .025), Gscore=bool(p_s < .05), Gfeat=bool(p_f < .05)); out = []; s0src = C_crc.s.values[C_crc.y.values == 0]
    for a in ALPHAS:
        th = {r["method"]: r["thr"] for r in rows if r["alpha"] == a}; bb = th["BBSE_l1"]
        out += [V4.ev(k, a, v, Tt, dict(n_lab=0)) for k, v in dict(FIXED=th["FIXED"], CRC=th["CRC"], NAIVE=th["NAIVE"], BBSE_l1=bb, ORACLE=V4.w_oracle(Tt, a)).items()]
        out += [dict(M5.b5(C_crc, Tt, a), n_lab=0)] + [dict(r, n_lab=0) for r in M5.b6(Ta, Tt, th["FIXED"], a)] + [dict(V6.b8(C_crc, Ta, Tt, lr, a, rng), n_lab=0)]
        K = KMIN[a]; sp, m = V4.label_until(Lab, K); L = Lab.iloc[:m]; s0lab = L.s.values[L.y.values == 0]
        tc = V.thr_crc(sp, np.ones(len(sp)), a) if len(sp) else 0.0; tp = V4.thr_cp(sp, a)
        ex = dict(K=K, n_lab_pos=len(sp), n_lab_neg=len(s0lab), pool_exhausted=int(len(sp) < K), n_test_pos=int((Tt.y == 1).sum()))
        out += [V4.ev("TGT_CRC", a, tc, Tt, dict(ex, n_lab=m)), V4.ev("TGT_CP", a, tp, Tt, dict(ex, n_lab=m))]
        for gname, fl in flags.items():   # FINAL method = G4 branch with source CRC when unflagged
            out += [V4.ev(f"{gname}_CRC", a, tc if fl else th["CRC"], Tt, dict(ex, n_lab=m if fl else 0)), V4.ev(f"{gname}_CP", a, tp if fl else th["CRC"], Tt, dict(ex, n_lab=m if fl else 0))]
        for g in GAMMAS:
            hi = M5.tau_hi(s0lab if flags["G4"] else s0src, g); ex3 = dict(ex, n_lab=m if flags["G4"] else 0, gamma=g); lo_c = tc if flags["G4"] else th["CRC"]; lo_p = tp if flags["G4"] else th["CRC"]
            out += [V8.ev3x(f"G4_CRC_3w_g{int(g * 100)}", a, lo_c, hi, Tt, ex3), V8.ev3x(f"G4_CP_3w_g{int(g * 100)}", a, lo_p, hi, Tt, ex3)]
    info = dict(gof_p=p_s, feat_auc=A_, feat_p=p_f, flag=flags["G4"], flag_score=flags["Gscore"], flag_feat=flags["Gfeat"], piT_gate=piG,
                piT_true=float((Tt.w * Tt.y).sum() / Tt.w.sum()), n_cgate=len(C_gate), n_ccrc=len(C_crc), n_gate=len(Ta), n_label_pool=len(Lab), **base)
    return [dict(**info, **r) for r in out], lin
def split_source(C, rng):
    p = rng.permutation(len(C)); return C.iloc[p[:len(p) // 2]].reset_index(drop=True), C.iloc[p[len(p) // 2:]].reset_index(drop=True)
def e1_task(args):
    seed, rep = args; U, deg, lr, calib, pca = M5.load5(seed); val = U[U.role == "val"]; rng = np.random.default_rng(140_000 * seed + rep)
    perm = rng.permutation(len(calib)); n = len(perm); C = calib.iloc[perm[:n // 2]]; Cg, Cc = split_source(C, rng)
    Pa0 = calib.iloc[perm[n // 2:3 * n // 4]].reset_index(drop=True); Pt0 = calib.iloc[perm[3 * n // 4:]].reset_index(drop=True)
    scen = ["none", "prev_high", "prev_low", "quality", "concept_g1", "concept_half"] + ([] if SMOKE else [f"deg:{c}" for c in V.CONDS] + [f"deg:{k}2+prev_high" for k in ["blur", "contrast", "dark", "colorcast"]])
    res, lins = [], []
    for sc in scen:
        Pa, Pt = (V8.concept(Pa0, sc, seed, rng), V8.concept(Pt0, sc, seed, rng)) if sc.startswith("concept") else (Pa0, Pt0)
        wa = V4.scen_w(Pa, sc); nz = int((wa > 0).sum()); order = rng.choice(len(Pa), nz, replace=False, p=wa / wa.sum())
        Pseq = V4.as_scen(Pa.iloc[order], sc, deg); Tt = V4.as_scen(Pt, sc, deg); Tt["w"] = V4.scen_w(Pt, sc); Tt = Tt[Tt.w > 0].reset_index(drop=True)
        r, l = run9(Cg, Cc, Pseq.iloc[:600], Pseq, Tt, lr, val, pca, rng, dict(exp="E1", seed=seed, rep=rep, site=sc), U); res += r; lins.append(l)
    return res, lins
def e2_task(args):
    seed, site, rep = args; U, deg, lr, calib, _ = M5.load5(seed); val = U[U.role == "val"]
    rng = np.random.default_rng(150_000 * seed + 100 * rep + ["messidor2", "deepdrid", "ddr", "aptos", "idrid"].index(site)); Cg, Cc = split_source(calib, rng)
    S = U[U.ds == site].reset_index(drop=True)
    if os.environ.get("DEDUP") == "1": drop = set(pd.read_csv(f"{W}/results/dedup_drop_list.csv").patient); S = S[~S.patient.isin(drop)].reset_index(drop=True)
    pca = PCA(32, random_state=0).fit(np.vstack([np.stack(calib.f), np.stack(S.f)]))
    perm = rng.permutation(len(S)); h = len(S) // 2; Ta = S.iloc[perm[:h]].reset_index(drop=True); Tt = S.iloc[perm[h:]].reset_index(drop=True); Tt["w"] = 1.0
    r, l = run9(Cg, Cc, Ta, Ta, Tt, lr, val, pca, rng, dict(exp="E2", seed=seed, rep=rep, site=site), U); return r, [l]
if __name__ == "__main__":
    t0 = time.time(); sites = ["messidor2", "deepdrid", "ddr", "aptos", "idrid"]
    T1 = [] if os.environ.get("STAGES") == "E2" else [(s, r) for s in V.SEEDS for r in range(V.REPS)]; T2 = [(s, si, r) for s in V.SEEDS for si in sites for r in range(V.REPS)]
    R, LIN = [], []
    with Pool(1 if SMOKE else 36) as pool:
        for fn, T in [(e1_task, T1), (e2_task, T2)]:
            for res, lins in pool.imap_unordered(fn, T): R += res; LIN += lins
            print(json.dumps(dict(stage=fn.__name__, rows=len(R), mins=round((time.time() - t0) / 60, 1))), flush=True)
    tag = ("smoke9" if SMOKE else "v9") + os.environ.get("RUN_TAG", ""); R = pd.DataFrame(R); R.to_parquet(f"{W}/results/e_{tag}_raw.parquet")
    L = pd.DataFrame(LIN); L.to_csv(f"{W}/results/lineage_{tag}.csv", index=False)
    print("pool-disjointness assertions passed for", len(L), "runs; median pool sizes:", L.drop(columns=["seed", "rep"]).groupby("site").median().round(0).to_dict("index"))
    a = R[R.alpha == 0.05]; print(a[a.method.isin(["CRC", "TGT_CRC", "G4_CRC", "G4_CP", "ORACLE"])].pivot_table(index=["exp", "site"], columns="method", values="miss").round(3).to_string())
    print(f"{tag.upper()} DONE", flush=True)
