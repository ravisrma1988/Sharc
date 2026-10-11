import os, sys, time, resource, json
os.environ.update(OMP_NUM_THREADS="2", BNULL="200", BACKBONE="convnext_tiny", SEEDS="0")
import numpy as np, pandas as pd
sys.path.insert(0, "/path/to/sharc_work/scripts")
import e_v2 as V, e_v3 as V3, e_v4 as V4, e_v5_main as M5, e_v8 as V8
from sklearn.decomposition import PCA
# Efficiency benchmark (single worker, 2 BLAS threads): wall-clock per pipeline component per deployment site, and peak memory.
t0 = time.time(); U, deg, lr, calib, _ = M5.load5(0); t_load = time.time() - t0; rows = []
for site in ["idrid", "deepdrid", "messidor2", "aptos", "ddr"]:
    S = U[U.ds == site].reset_index(drop=True); rng = np.random.default_rng(0); perm = rng.permutation(len(S)); h = len(S) // 2
    Ta = S.iloc[perm[:h]].reset_index(drop=True); Tt = S.iloc[perm[h:]].reset_index(drop=True); Tt["w"] = 1.0
    t = {}; s = time.time(); pca = PCA(32, random_state=0).fit(np.vstack([np.stack(calib.f), np.stack(S.f)])); t["pca"] = time.time() - s
    yC = calib.y.values; Zs, Zt = pca.transform(np.stack(calib.f)), pca.transform(np.stack(Ta.f))
    s = time.time(); piB = V.bbse(V.cal(lr, calib.s.values), yC, V.cal(lr, Ta.s.values)); t["bbse"] = time.time() - s
    s = time.time(); g = V.dom(Zs, Zt, np.where(yC == 1, piB / yC.mean(), (1 - piB) / (1 - yC.mean()))); t["domain_weights"] = time.time() - s
    s = time.time(); V.thr_crc(calib.s.values[yC == 1], g[yC == 1], 0.05); t["weighted_crc"] = time.time() - s
    s = time.time(); V3.gof(Ta.s.values, calib.s.values[yC == 1], calib.s.values[yC == 0], rng); t["score_test_B200"] = time.time() - s
    s = time.time(); neff = min(len(Zt), len(Zs) // 2); V4.feat_test(Zs, yC, Zt[rng.choice(len(Zt), neff, replace=False)], piB, rng); t["feature_test_B200"] = time.time() - s
    sp = Ta.s.values[Ta.y.values == 1][:45]; s = time.time(); V4.thr_cp(sp, 0.05); V.thr_crc(sp, np.ones(len(sp)), 0.05); t["few_label_bounds"] = time.time() - s
    s = time.time(); V8.run8(calib, Ta, Ta, Tt, lr, U[U.role == "val"], pca, rng, dict(exp="bench", seed=0, rep=0, site=site)); t["full_pipeline_all_methods_3_alphas"] = time.time() - s
    rows.append(dict(site=site, n_source_cal=len(calib), n_adapt=len(Ta), n_test=len(Tt), **{k: round(v, 3) for k, v in t.items()}))
    print(rows[-1], flush=True)
D = pd.DataFrame(rows); D.to_csv("/path/to/sharc_work/results/efficiency_benchmark.csv", index=False)
print(json.dumps(dict(load_secs=round(t_load, 1), peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2))), "BENCH DONE")
