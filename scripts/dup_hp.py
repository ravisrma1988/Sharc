import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.ndimage import uniform_filter
# Definitive duplicate check on fine detail: 128x128 grey, high-pass (minus 9x9 box blur), correlation. Duplicates: r_hp >= 0.90.
W = "/path/to/sharc_work"; P = W + "/data/proc"
SRC = [(P + "/img512_uint8.npy", P + "/metadata_rows.parquet"), (P + "/img512_eyepacs_test_uint8.npy", P + "/metadata_eyepacs_test.parquet")]
def hp(args):
    path, idx = args; X = np.load(path, mmap_mode="r"); out = []
    for i in idx:
        g = X[i].astype(np.float32).mean(2).reshape(128, 4, 128, 4).mean((1, 3)); h = (g - uniform_filter(g, 9))[32:96, 32:96]   # central retina only
        h = h.ravel(); h -= h.mean(); out.append(h / (np.linalg.norm(h) + 1e-9))
    return idx, np.array(out, np.float32)
if __name__ == "__main__":
    D = pd.read_csv(W + "/results/duplicate_audit_pairs.csv"); C = D[D.r >= 0.97].copy()
    n0 = len(pd.read_parquet(SRC[0][1])); need = np.unique(np.r_[C.i.values, C.j.values]); vec = {}
    jobs = []
    for k, (path, _) in enumerate(SRC):
        loc = need[need < n0] if k == 0 else need[need >= n0] - n0
        jobs += [(path, loc[s:s + 500], k) for s in range(0, len(loc), 500)]
    with Pool(32) as pool:
        for (idx, arr), (_, _, k) in zip(pool.imap(hp, [(p, l) for p, l, _ in jobs]), jobs):
            for a, v in zip(idx, arr): vec[int(a) + (n0 if k == 1 else 0)] = v
    C["r_hp"] = [float(vec[i] @ vec[j]) for i, j in zip(C.i, C.j)]
    C.to_csv(W + "/results/duplicate_audit_confirmed_candidates.csv", index=False)
    print("r_hp quantiles of candidates:", C.r_hp.quantile([.5, .9, .99, .999]).round(3).to_dict())
    dup = C[(C.r_hp >= 0.95) & ~C.same_patient]
    print("CONFIRMED duplicates between different patient IDs:", len(dup))
    print(dup.groupby(["ds_i", "ds_j"]).agg(pairs=("r_hp", "size"), min_r_hp=("r_hp", "min")).reset_index().to_string(index=False))
    dup.to_csv(W + "/results/duplicates_confirmed.csv", index=False); print("HP VERIFY DONE")
