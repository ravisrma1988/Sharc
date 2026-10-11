import os, sys, numpy as np, pandas as pd
os.environ.setdefault("BACKBONE", "convnext_tiny"); sys.path.insert(0, "/path/to/sharc_work/scripts")
import e_v2 as V, e_v4 as V4, e_v5_main as M5
# Unit tests for the threshold rules on synthetic data with known truth. Decision rule everywhere: refer if score >= threshold.
rng = np.random.default_rng(2026); R = 4000; res = []
def fnr(t, a=5, b=2): from scipy.stats import beta; return beta.cdf(t, a, b)                  # true FNR of positives ~ Beta(5,2)
def record(test, cond, value, target, ok): res.append(dict(test=test, condition=cond, value=round(float(value), 4), target=target, PASS=bool(ok)))
# T1 unweighted CRC: E[FNR] <= alpha (and not grossly conservative: >= alpha - 2/(n+1))
for a in (0.02, 0.05, 0.10):
    for n in (45, 200, 1000):
        f = np.mean([fnr(V.thr_crc(rng.beta(5, 2, n), np.ones(n), a)) for _ in range(R)]); se = 3 / np.sqrt(R) * 0.1
        record("T1 CRC expected FNR", f"alpha={a}, n={n}", f, f"<= {a}", f <= a + se and f >= a - 2 / (n + 1) - se)
# T2 weighted CRC with exact weights under covariate shift on positives: source x~N(0,1), target x~N(0.7,1), score = Phi(x)
from scipy.stats import norm
for a in (0.05, 0.10):
    f = []
    for _ in range(R):
        xs = rng.normal(0, 1, 500); w = np.exp(0.7 * xs - 0.245); t = V.thr_crc(norm.cdf(xs), w, a)
        f.append(norm.cdf(norm.ppf(np.clip(t, 1e-12, 1 - 1e-12)) - 0.7))
    f = np.mean(f); record("T2 weighted CRC, exact weights", f"alpha={a}, n=500", f, f"<= {a}", f <= a + 0.003)
# T3 exact Clopper-Pearson: P(true FNR > alpha) <= delta = 0.1; infeasible K refers everyone
for a, K in ((0.02, 114), (0.05, 45), (0.10, 22), (0.05, 100), (0.05, 300)):
    v = np.mean([fnr(V4.thr_cp(rng.beta(5, 2, K), a)) > a for _ in range(R)])
    record("T3 CP violation probability", f"alpha={a}, K={K}", v, "<= 0.10", v <= 0.10 + 0.015)
record("T3 CP infeasible K refers all", "alpha=0.05, K=20", V4.thr_cp(rng.beta(5, 2, 20), 0.05), "== 0", V4.thr_cp(rng.beta(5, 2, 20), 0.05) == 0.0)
# T4 upper threshold: E[FPR among non-referable] <= gamma; negatives ~ Beta(2,5)
from scipy.stats import beta as Be
for g in (0.05, 0.10, 0.20):
    f = np.mean([1 - Be.cdf(M5.tau_hi(rng.beta(2, 5, 300), g), 2, 5) for _ in range(R)])
    record("T4 tau_hi expected FPR", f"gamma={g}, n=300", f, f"<= {g}", f <= g + 0.003)
# T5 prevalence-shift invariance: class-conditional scores fixed, target prevalence 0.2 -> 0.6; unweighted CRC on source positives
for pt in (0.2, 0.6):
    f = []
    for _ in range(R // 4):
        ys = rng.random(1000) < 0.25; ss = np.where(ys, rng.beta(5, 2, 1000), rng.beta(2, 5, 1000))             # source mixture, prevalence 0.25
        t = V.thr_crc(ss[ys], np.ones(ys.sum()), 0.05)
        yt = rng.random(5000) < pt; st = np.where(yt, rng.beta(5, 2, 5000), rng.beta(2, 5, 5000))               # target mixture, prevalence pt
        f.append((st[yt] < t).mean())
    f = np.mean(f); record("T5 FNR invariant to prevalence", f"source 0.25 -> target {pt}", f, "<= 0.05", f <= 0.05 + 0.003)
# T6 ties: identical scores never counted as misses at the threshold
s = np.array([0.3] * 30 + [0.8] * 30); t = V.thr_crc(s, np.ones(60), 0.05)
record("T6 ties", "60 tied positives", float((s < t).mean()), "<= 0.05 within calibration set", (s < t).mean() <= 0.05)
D = pd.DataFrame(res); print(D.to_string(index=False)); D.to_csv("/path/to/sharc_work/results/unit_tests.csv", index=False)
print("ALL PASS" if D.PASS.all() else f"FAILURES: {int((~D.PASS).sum())}")
