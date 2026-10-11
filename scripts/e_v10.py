import os, sys
os.environ.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
import numpy as np, pandas as pd, json, time
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e_v2 as V, e_v3 as V3, e_v4 as V4, e_v5_main as M5, e_v8 as V8, e_v9 as V9
# v10: preregistered same-network deployment grid (PREREG_DEPLOYMENT_GRID.md, SHA-256 0518621e...). Pools, gate and thresholds exactly as v9.
W = V.W; ALPHAS = V.ALPHAS; SMOKE = V.SMOKE; KMIN = M5.KMIN; FIXED_M = 100; GAMMA = 0.10
APPEAR = ["none"] + [f"{k}{l}" for k in ["blur", "contrast", "dark", "colorcast"] for l in (1, 2, 3)]
PREV = {"": 1.0, "+prev_low": 0.4, "+prev_high": 3.0}
def cell_name(app, p): return (p[1:] if p else "none") if app == "none" else f"deg:{app}{p}"
CELLS = [cell_name(a, p) for a in APPEAR for p in PREV] + ["concept_g1", "concept_half"]
if SMOKE: CELLS = ["none", "prev_high", "deg:blur1", "deg:dark2+prev_low", "concept_g1"]
def meta(c):
    if c.startswith("concept"): return dict(prev_mult=1.0, appearance=c, level=0)
    p = 3.0 if "prev_high" in c else (0.4 if "prev_low" in c else 1.0); a = c[4:].split("+")[0] if c.startswith("deg:") else "none"
    return dict(prev_mult=p, appearance=a.rstrip("123") if a != "none" else "none", level=int(a[-1]) if a != "none" else 0)
def run10(C_gate, C_crc, Ta_all, Lab_pool, Tt, lr, val, pca, rng, base, U):
    idx = rng.permutation(len(Ta_all)); Ta = Ta_all.iloc[idx[:len(idx) // 2]]; Lab = Lab_pool[~Lab_pool.patient.isin(set(Ta.patient))]
    V9.check_pools(base["seed"], base["rep"], base["site"], dict(C_gate=C_gate.patient, C_crc=C_crc.patient, gate_A=Ta.patient, label_B=Lab.patient, test=Tt.patient.unique()), U)
    rows = V.methods(C_crc, Ta, Tt, lr, val, pca, rng); yG = C_gate.y.values
    _, _, p_s = V3.gof(Ta.s.values, C_gate.s.values[yG == 1], C_gate.s.values[yG == 0], rng)
    piG = V.bbse(V.cal(lr, C_gate.s.values), yG, V.cal(lr, Ta.s.values)); Zs, Zt = pca.transform(np.stack(C_gate.f)), pca.transform(np.stack(Ta.f))
    _, p_f = V4.feat_test(Zs, yG, Zt[rng.choice(len(Zt), min(len(Zt), len(Zs) // 2), replace=False)], piG, rng)
    flags = dict(GATED=bool(p_s < .025 or p_f < .025), GATED_SCORE_ONLY=bool(p_s < .05), GATED_FEATURE_ONLY=bool(p_f < .05))
    s0src = C_crc.s.values[C_crc.y.values == 0]; out = []
    def two(name, a, lam, nlab, extra=None): return dict(V4.ev(name, a, lam, Tt), n_lab=nlab, refer_all=int(lam == 0.0), **(extra or {}))
    def three(name, a, lo, neg, nlab): return dict(V8.ev3x(name + "_3w", a, lo, M5.tau_hi(neg, GAMMA), Tt), n_lab=nlab, refer_all=int(lo == 0.0))
    for a in ALPHAS:
        src = [r["thr"] for r in rows if r["alpha"] == a and r["method"] == "CRC"][0]
        sp, m = V4.label_until(Lab, KMIN[a]); L = Lab.iloc[:m]; neg = L.s.values[L.y.values == 0]
        tc = V.thr_crc(sp, np.ones(len(sp)), a) if len(sp) else 0.0; tp = V4.thr_cp(sp, a)
        F = Lab.iloc[:FIXED_M]; fsp = F.s.values[F.y.values == 1]; fneg = F.s.values[F.y.values == 0]
        fc = V.thr_crc(fsp, np.ones(len(fsp)), a) if len(fsp) else 0.0; fp = V4.thr_cp(fsp, a)
        out += [two("SOURCE", a, src, 0), two("ALWAYS_TARGET_CRC", a, tc, m), two("ALWAYS_TARGET_CP", a, tp, m),
                two("FIXED_BUDGET_CRC", a, fc, len(F), dict(n_lab_pos=len(fsp))), two("FIXED_BUDGET_CP", a, fp, len(F), dict(n_lab_pos=len(fsp)))]
        for g, fl in flags.items():
            out += [two(f"{g}_CRC", a, tc if fl else src, m if fl else 0), two(f"{g}_CP", a, tp if fl else src, m if fl else 0)]
        out += [three("SOURCE", a, src, s0src, 0), three("ALWAYS_TARGET_CRC", a, tc, neg, m),
                three("GATED_CRC", a, tc if flags["GATED"] else src, neg if flags["GATED"] else s0src, m if flags["GATED"] else 0)]
    info = dict(gof_p=p_s, feat_p=p_f, flag=flags["GATED"], flag_score=flags["GATED_SCORE_ONLY"], flag_feat=flags["GATED_FEATURE_ONLY"],
                n_label_pool=len(Lab), piT_true=float((Tt.w * Tt.y).sum() / Tt.w.sum()), **base, **meta(base["site"]))
    return [dict(**info, **r) for r in out]
def task(args):
    seed, rep = args; U, deg, lr, calib, pca = M5.load5(seed); val = U[U.role == "val"]; rng = np.random.default_rng(160_000 * seed + rep)
    perm = rng.permutation(len(calib)); n = len(perm); C = calib.iloc[perm[:n // 2]]; Cg, Cc = V9.split_source(C, rng)
    Pa0 = calib.iloc[perm[n // 2:3 * n // 4]].reset_index(drop=True); Pt0 = calib.iloc[perm[3 * n // 4:]].reset_index(drop=True); res = []
    for sc in CELLS:
        Pa, Pt = (V8.concept(Pa0, sc, seed, rng), V8.concept(Pt0, sc, seed, rng)) if sc.startswith("concept") else (Pa0, Pt0)
        wa = V4.scen_w(Pa, sc); order = rng.choice(len(Pa), int((wa > 0).sum()), replace=False, p=wa / wa.sum())
        Pseq = V4.as_scen(Pa.iloc[order], sc, deg); Tt = V4.as_scen(Pt, sc, deg); Tt["w"] = V4.scen_w(Pt, sc); Tt = Tt[Tt.w > 0].reset_index(drop=True)
        res += run10(Cg, Cc, Pseq.iloc[:600], Pseq, Tt, lr, val, pca, rng, dict(exp="DEPLOY", seed=seed, rep=rep, site=sc), U)
    return res
if __name__ == "__main__":
    t0 = time.time(); T = [(s, r) for s in V.SEEDS for r in range(V.REPS)]
    with Pool(1 if SMOKE else 36) as pool: R = pd.DataFrame([x for o in pool.imap_unordered(task, T) for x in o])
    tag = ("smoke10" if SMOKE else "v10") + os.environ.get("RUN_TAG", ""); R.to_parquet(f"{W}/results/e_{tag}_raw.parquet")
    print(json.dumps(dict(stage="deploy", rows=len(R), cells=R.site.nunique(), mins=round((time.time() - t0) / 60, 1))), flush=True)
    a = R[(R.alpha == 0.05) & R.method.isin(["SOURCE", "GATED_CRC", "ALWAYS_TARGET_CRC", "FIXED_BUDGET_CRC"])]
    s = a.groupby(["site", "method"]).agg(miss=("miss", "mean"), zero_label=("n_lab", lambda v: (v == 0).mean()), n_lab=("n_lab", "mean")).unstack("method")
    print(s.round(3).to_string()); print(f"{tag.upper()} DONE", flush=True)
