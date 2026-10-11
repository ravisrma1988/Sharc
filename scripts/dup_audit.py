import os, numpy as np, pandas as pd, json, time, itertools, collections
from multiprocessing import Pool
from scipy.fft import dctn
# D2 integrity audit: perceptual hashing (64-bit DCT pHash on 32x32 grey) of every cached image, then exact and near-duplicate
# search within and across datasets and across EyePACS roles (train / val / calib / test pool). Near = Hamming distance <= 6.
W = "/path/to/sharc_work"; P = W + "/data/proc"
SRC = [("main", P + "/img512_uint8.npy", P + "/metadata_rows.parquet"), ("test", P + "/img512_eyepacs_test_uint8.npy", P + "/metadata_eyepacs_test.parquet")]
def phash_chunk(args):
    path, lo, hi = args; X = np.load(path, mmap_mode="r"); out = np.empty(hi - lo, np.uint64)
    for i in range(lo, hi):
        g = X[i].astype(np.float32).mean(2).reshape(32, 16, 32, 16).mean((1, 3)); d = dctn(g, norm="ortho")[:8, :8].ravel()[1:]
        bits = d > np.median(d); out[i - lo] = np.uint64(sum(int(b) << k for k, b in enumerate(bits)))
    return lo, out
if __name__ == "__main__":
    t0 = time.time(); frames = []
    for tag, path, meta in SRC:
        M = pd.read_parquet(meta)[["row", "ds", "img_id", "patient", "role"]]; n = len(M); H = np.empty(n, np.uint64)
        with Pool(48) as pool:
            for lo, h in pool.imap_unordered(phash_chunk, [(path, i, min(n, i + 500)) for i in range(0, n, 500)]): H[lo:lo + len(h)] = h
        M["hash"] = H; M["src"] = tag; frames.append(M); print(tag, n, "hashed", round(time.time() - t0), "s", flush=True)
    A = pd.concat(frames, ignore_index=True); A["grp"] = np.where(A.ds == "eyepacs", "eyepacs_" + A.role, A.ds)
    h = A.hash.values.astype(np.uint64); pairs = set()
    for band in range(8):                                   # pigeonhole: Hamming <= 7 implies one identical byte
        key = (h >> np.uint64(8 * band)) & np.uint64(255); buckets = collections.defaultdict(list)
        for i, k in enumerate(key): buckets[int(k)].append(i)
        for idx in buckets.values():
            if len(idx) < 2: continue
            idx = np.array(idx); x = h[idx]
            for j in range(len(idx)):
                d = np.unpackbits((x[j + 1:] ^ x[j]).view(np.uint8).reshape(-1, 8), axis=1).sum(1)
                for k in np.where(d <= 6)[0]: pairs.add((int(idx[j]), int(idx[j + 1 + k]), int(d[k])))
    D = pd.DataFrame(list(pairs), columns=["i", "j", "hamming"])
    if len(D):
        D = D.assign(ds_i=A.grp.values[D.i], ds_j=A.grp.values[D.j], id_i=A.img_id.values[D.i], id_j=A.img_id.values[D.j],
                     pat_i=A.patient.values[D.i], pat_j=A.patient.values[D.j])
        D["same_patient"] = D.pat_i == D.pat_j; D.to_csv(W + "/results/duplicate_audit_pairs.csv", index=False)
        S = D[~D.same_patient].groupby(["ds_i", "ds_j"]).agg(pairs=("i", "size"), exact=("hamming", lambda v: int((v == 0).sum()))).reset_index()
        print("near-duplicate pairs between DIFFERENT patients (Hamming <= 6):"); print(S.to_string(index=False))
        print("same-patient pairs (e.g. repeat images of one eye):", int(D.same_patient.sum()))
    else: print("no near-duplicate pairs found")
    print(json.dumps(dict(images=len(A), secs=round(time.time() - t0))), "DUP AUDIT DONE", flush=True)
