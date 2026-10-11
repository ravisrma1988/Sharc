import numpy as np, pandas as pd, json
from multiprocessing import Pool
# Verify perceptual-hash candidates with pixel-level correlation of 64x64 grey thumbnails (true duplicates: r >= 0.97).
W = "/path/to/sharc_work"; P = W + "/data/proc"
SRC = [(P + "/img512_uint8.npy", P + "/metadata_rows.parquet"), (P + "/img512_eyepacs_test_uint8.npy", P + "/metadata_eyepacs_test.parquet")]
def thumbs(args):
    path, lo, hi = args; X = np.load(path, mmap_mode="r")
    t = X[lo:hi].astype(np.float32).mean(3).reshape(hi - lo, 64, 8, 64, 8).mean((2, 4)).reshape(hi - lo, -1)
    t -= t.mean(1, keepdims=True); t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9; return lo, t
if __name__ == "__main__":
    T = []
    for path, meta in SRC:
        n = len(pd.read_parquet(meta)); arr = np.empty((n, 4096), np.float32)
        with Pool(32) as pool:
            for lo, t in pool.imap_unordered(thumbs, [(path, i, min(n, i + 1000)) for i in range(0, n, 1000)]): arr[lo:lo + len(t)] = t
        T.append(arr)
    T = np.vstack(T); D = pd.read_csv(W + "/results/duplicate_audit_pairs.csv")
    D["r"] = np.einsum("ij,ij->i", T[D.i.values], T[D.j.values]); D.to_csv(W + "/results/duplicate_audit_pairs.csv", index=False)
    dup = D[(D.r >= 0.97) & ~D.same_patient]
    print("candidate pairs:", len(D), "| confirmed duplicates (r >= 0.97) between different patient IDs:", len(dup))
    print(dup.groupby(["ds_i", "ds_j"]).agg(pairs=("r", "size"), min_r=("r", "min")).reset_index().to_string(index=False))
    print("examples:"); print(dup.sort_values("r", ascending=False)[["ds_i", "id_i", "ds_j", "id_j", "hamming", "r"]].head(25).to_string(index=False))
    print("r distribution of non-confirmed candidates:", D[D.r < 0.97].r.describe().round(3).to_dict()); print("VERIFY DONE")
