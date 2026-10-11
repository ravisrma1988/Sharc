import os, sys, numpy as np, pandas as pd
from scipy.stats import beta
os.environ.setdefault("BACKBONE", "convnext_tiny"); sys.path.insert(0, "/path/to/sharc_work/scripts")
import e_v2 as V, e_v4 as V4, e_v5_main as M5
# P0 theorem-implementation consistency: reference implementations written from THEORY_AND_CLAIMS.md Section 3 only,
# compared with the production functions used by e_v9.py on random inputs and edge cases.
def ref_lower(s, a):                     # k* = max{k in 0..n : (k+1)/(n+1) <= a}; lambda = (k*+1)-th smallest; refer-all (0) if none
    s = np.sort(np.asarray(s, float)); n = len(s); ks = [k for k in range(n + 1) if (k + 1) / (n + 1) <= a]
    if not ks: return 0.0
    k = max(ks); return s[k] if k < n else 1.0
def ref_cp(s, a, d=0.1):                 # k_CP = max{k : Beta^{-1}(1-d; k+1, n-k) <= a}; lambda = (k_CP+1)-th smallest; refer-all if none
    s = np.sort(np.asarray(s, float)); n = len(s)
    if n == 0: return 0.0
    ks = [k for k in range(n) if beta.ppf(1 - d, k + 1, n - k) <= a]
    return s[max(ks)] if ks else 0.0
def ref_upper(t, g, eps=1e-9):           # descending; k = floor(g(m+1)-1); no auto-referral (>1) if k<0 or m=0; else t_[k+1]+eps
    t = np.sort(np.asarray(t, float))[::-1]; m = len(t); k = int(np.floor(g * (m + 1) - 1))
    if m == 0 or k < 0: return 1.0 + eps
    return t[k] + eps if k < m else 0.0
rng = np.random.default_rng(11); fails = []; checked = {"lower": 0, "cp": 0, "upper": 0}
for trial in range(6000):
    n = int(rng.choice([1, 2, 5, 9, 10, 18, 19, 20, 22, 45, 48, 49, 50, 114, 200, 1000])); a = float(rng.choice([0.02, 0.05, 0.10]))
    s = rng.random(n) if trial % 3 else np.round(rng.random(n), 1)          # every third trial has heavy ties
    for name, ref, prod in [("lower", lambda: ref_lower(s, a), lambda: V.thr_crc(s, np.ones(n), a)), ("cp", lambda: ref_cp(s, a), lambda: V4.thr_cp(s, a)),
                            ("upper", lambda: ref_upper(s, a * 2), lambda: M5.tau_hi(s, a * 2))]:
        r, p = ref(), prod(); checked[name] += 1
        same = (r > 1 and p > 1) or abs(r - p) < 1e-12
        if not same: fails.append(dict(fn=name, n=n, alpha=a, ref=r, prod=p))
for name, ref, prod in [("cp n=0", lambda: ref_cp([], .05), lambda: V4.thr_cp(np.array([]), .05)), ("upper m=0", lambda: ref_upper([], .1), lambda: M5.tau_hi(np.array([]), .1))]:
    r, p = ref(), prod(); ok = (r > 1 and p > 1) or abs(r - p) < 1e-12
    print(f"edge {name}: ref={r} prod={p} {'OK' if ok else 'MISMATCH'}"); fails += [] if ok else [dict(fn=name, ref=r, prod=p)]
print("comparisons:", checked, "| mismatches:", len(fails))
if fails: print(pd.DataFrame(fails).head(20).to_string())
# decision-rule semantics on the implemented thresholds (cleared iff S < lambda_lo; auto-referred iff S >= lambda_hi)
s = np.array([0.1, 0.2, 0.2, 0.3]); lam = V.thr_crc(s, np.ones(4), 0.45)
print("tie check: lambda_lo =", lam, "| calibration units cleared:", int((s < lam).sum()), "(ties at lambda are referred)")
print("CONSISTENCY", "PASS" if not fails else "FAIL")
