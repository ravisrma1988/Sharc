import os, sys, functools
os.environ.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
import numpy as np, pandas as pd, json, time, zlib
from multiprocessing import Pool
from sklearn.decomposition import PCA
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e_v2 as V, e_v4 as V4, e_v5_main as M5
W = V.W; SMOKE = V.SMOKE; BB = V.BB; A = 0.05
NAD = [50, 100, 200, 400, 1000]; NCAL = [250, 500, 1000, 2000, 99999]
E1_SWEEP = ["none", "prev_high", "prev_low", "quality"] + ([] if SMOKE else ["deg:blur1", "deg:colorcast1", "deg:contrast1", "deg:dark1", "deg:blur2"])
SITES = ["messidor2", "deepdrid", "ddr", "aptos", "idrid"]
def keep(rows):                     # only alpha = 0.05 and the methods the ablations report
    ms = {"CRC", "BBSE_l1", "ORACLE", "G4_CRC_K45", "G4_CP_K45", "TGT_CRC_K45", "TGT_CP_K45"}
    return [r for r in rows if r["alpha"] == A and r["method"] in ms]
def e1_parts(seed, rep, sc, rng):
    U, deg, lr, calib, pca = M5.load5(seed); perm = rng.permutation(len(calib)); n = len(perm); C = calib.iloc[perm[:n // 2]]
    Pa = calib.iloc[perm[n // 2:3 * n // 4]].reset_index(drop=True); Pt = calib.iloc[perm[3 * n // 4:]].reset_index(drop=True)
    wa = V4.scen_w(Pa, sc); nz = int((wa > 0).sum()); order = rng.choice(len(Pa), nz, replace=False, p=wa / wa.sum()); Pseq = V4.as_scen(Pa.iloc[order], sc, deg)
    Tt = V4.as_scen(Pt, sc, deg); Tt["w"] = V4.scen_w(Pt, sc); Tt = Tt[Tt.w > 0].reset_index(drop=True)
    return U, lr, pca, C, Pseq, Tt
# ---------- (3) adaptation-set size sweep  and  (6) source calibration scale
def sweep_task(args):
    kind, seed, rep, where = args; rng = np.random.default_rng(90_000 * seed + 1000 * rep + zlib.crc32(f"{kind}|{where}".encode()) % 997); out = []
    if where in SITES:
        U, deg, lr, calib, _ = M5.load5(seed); S = U[U.ds == where].reset_index(drop=True); pca = PCA(32, random_state=0).fit(np.vstack([np.stack(calib.f), np.stack(S.f)]))
        perm = rng.permutation(len(S)); h = len(S) // 2; Pseq = S.iloc[perm[:h]].reset_index(drop=True); Tt = S.iloc[perm[h:]].reset_index(drop=True); Tt["w"] = 1.0; C = calib
    else:
        U, lr, pca, C, Pseq, Tt = e1_parts(seed, rep, where, rng)
    val = U[U.role == "val"]
    if kind == "nad":
        for nad in NAD:
            if nad > len(Pseq): continue
            out += [dict(**r, nad=nad) for r in keep(M5.run(C, Pseq.iloc[:nad], Pseq, Tt, lr, val, pca, rng, dict(exp=kind, seed=seed, rep=rep, site=where)))]
    else:
        for nc in NCAL:
            Cs = C.iloc[rng.permutation(len(C))[:min(nc, len(C))]]
            out += [dict(**r, ncal=len(Cs)) for r in keep(M5.run(Cs, Pseq.iloc[:600], Pseq, Tt, lr, val, pca, rng, dict(exp=kind, seed=seed, rep=rep, site=where)))]
    return out
# ---------- (4) calibration unit: patient-level (proposed) vs image-level (eye) positives; both evaluated per patient
@functools.lru_cache(None)
def eyes(seed, cond=None):
    f = f"{W}/preds/{BB}_s{seed}.parquet" if cond is None else f"{W}/preds/degraded/{BB}_s{seed}_{cond}.parquet"
    g = pd.read_parquet(f).groupby("patient"); return g.p_ref.apply(np.array).to_dict(), g.referable.apply(np.array).to_dict()
def eye_pos(pats, ES, EY):
    v = [ES[p][EY[p] == 1] for p in pats if p in ES]; return np.concatenate(v) if v else np.array([])
def units_task(args):
    seed, rep, where = args; rng = np.random.default_rng(95_000 * seed + 1000 * rep + zlib.crc32(where.encode()) % 997); ES, EY = eyes(seed)
    if where in SITES:
        U, deg, lr, calib, _ = M5.load5(seed); S = U[U.ds == where].reset_index(drop=True); perm = rng.permutation(len(S)); h = len(S) // 2
        Pseq = S.iloc[perm[:h]].reset_index(drop=True); Tt = S.iloc[perm[h:]].reset_index(drop=True); Tt["w"] = 1.0; C = calib; tES, tEY = ES, EY
    else:
        U, lr, pca, C, Pseq, Tt = e1_parts(seed, rep, where, rng); tES, tEY = eyes(seed, where[4:]) if where.startswith("deg:") else (ES, EY)
    c = np.cumsum(Pseq.y.values); m = min(int(np.searchsorted(c, 45)) + 1, len(c)); L = Pseq.iloc[:m]
    pats = dict(src_patient=C.s.values[C.y.values == 1], src_eye=eye_pos(C.patient.values, ES, EY),
                tgt_patient=L.s.values[L.y.values == 1], tgt_eye=eye_pos(L.patient.values, tES, tEY))
    out = []
    for unit in ["patient", "eye"]:
        sp, tp = pats[f"src_{unit}"], pats[f"tgt_{unit}"]
        for name, th in [("CRC", V.thr_crc(sp, np.ones(len(sp)), A)), ("TGT_CRC_K45", V.thr_crc(tp, np.ones(len(tp)), A) if len(tp) else 0.0), ("TGT_CP_K45", V4.thr_cp(tp, A))]:
            out.append(dict(V4.ev(name, A, th, Tt), unit=unit, n_pos_cal=len(sp) if name == "CRC" else len(tp), exp="units", seed=seed, rep=rep, site=where))
    return out
if __name__ == "__main__":
    t0 = time.time(); R = int(os.environ.get("REPS_ABL", "1" if SMOKE else "30"))
    Tn = [("nad", s, r, w) for s in V.SEEDS for r in range(R) for w in E1_SWEEP + SITES]
    Tc = [("ncal", s, r, w) for s in V.SEEDS for r in range(R) for w in (["none", "prev_high"] + ([] if SMOKE else ["deg:blur2"])) + SITES]
    Tu = [(s, r, w) for s in V.SEEDS for r in range(R) for w in (["none", "prev_high"] + ([] if SMOKE else ["deg:blur2"])) + ["messidor2", "deepdrid"]]
    res = {}
    with Pool(1 if SMOKE else 36) as pool:
        for name, fn, T in [("nad", sweep_task, Tn), ("ncal", sweep_task, Tc), ("units", units_task, Tu)]:
            res[name] = pd.DataFrame([x for o in pool.imap_unordered(fn, T) for x in o])
            print(json.dumps(dict(stage=name, rows=len(res[name]), mins=round((time.time() - t0) / 60, 1))), flush=True)
    tag = ("smoke5abl" if SMOKE else "v5abl") + os.environ.get("RUN_TAG", "")
    for name, D in res.items():
        D["viol"] = D.miss > D.alpha; D.to_parquet(f"{W}/results/e_{tag}_{name}_raw.parquet")
    print(res["nad"].groupby(["site", "nad"]).flag.mean().unstack().round(2).to_string())
    print(res["units"].groupby(["site", "method", "unit"]).miss.mean().unstack().round(3).to_string()); print(f"{tag.upper()} DONE", flush=True)
