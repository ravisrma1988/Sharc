import numpy as np, pandas as pd, sys
sys.path.insert(0, "/path/to/sharc_work/scripts"); import os; os.environ.setdefault("BACKBONE", "convnext_tiny")
import e_v4 as V4
# Does a FINITE test set explain the ~11-12% exceedance of the exact binomial rule (nominal 10%)?
# Model: positive scores ~ U(0,1) (WLOG, continuous F); calibrate thr_cp on n positives; evaluate the miss rate on m test positives.
# Exceedance against the population miss rate should be <= 0.10; against an m-sample empirical miss rate it is inflated.
rng = np.random.default_rng(7); R = 20000; rows = []
for a, K in ((0.05, 45), (0.10, 22), (0.02, 114)):
    for m in (100, 135, 160, 300, 750, 2800, 10**9):
        t = np.array([V4.thr_cp(rng.random(K), a) for _ in range(R)])         # population miss rate = t (uniform scores)
        emp = t if m == 10**9 else rng.binomial(m, t) / m
        rows.append(dict(alpha=a, K=K, test_positives="population" if m == 10**9 else m, exceed=round(float((emp > a).mean()), 4)))
D = pd.DataFrame(rows); print(D.pivot_table(index="test_positives", columns="alpha", values="exceed", sort=False).to_string())
D.to_csv("/path/to/sharc_work/results/cp_exceedance_simulation.csv", index=False)
