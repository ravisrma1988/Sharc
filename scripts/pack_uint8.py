import numpy as np, pandas as pd, cv2, os, time
from multiprocessing import Pool
W = "/path/to/sharc_work"; P = W + "/data/proc"
M = pd.read_parquet(P + "/metadata.parquet"); M = M[M.cached_ok == 1].reset_index(drop=True)
M["row"] = np.arange(len(M)); M.to_parquet(P + "/metadata_rows.parquet")
N = len(M); out = P + "/img512_uint8.npy"
mm = np.lib.format.open_memmap(out, mode="w+", dtype=np.uint8, shape=(N, 512, 512, 3)); del mm
def work(chunk):
    mm = np.load(out, mmap_mode="r+")
    for i, p in chunk: mm[i] = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
    mm.flush(); return len(chunk)
items = list(zip(M.row, M.cache)); chunks = [items[i:i + 256] for i in range(0, N, 256)]
t = time.time()
with Pool(64) as pool: done = sum(pool.map(work, chunks))
print("packed", done, "of", N, "images into", out, "in", round(time.time() - t), "s", flush=True)
print("PACK DONE", flush=True)
