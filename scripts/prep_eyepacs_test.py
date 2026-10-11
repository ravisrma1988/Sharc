import os, glob, zipfile, shutil, numpy as np, pandas as pd, cv2, json, time
from multiprocessing import Pool
# EyePACS test set: unwrap Kaggle parts, join, extract, keep only EyeQ-labelled images (official academic DR grades), cache at 512 px.
R = "/path/to/sharc_work/data/raw/eyepacs_test"; X = "/path/to/sharc_work/data/extracted/eyepacs_test"; P = "/path/to/sharc_work/data/proc"
os.makedirs(X, exist_ok=True); t0 = time.time(); parts = []
for w in sorted(glob.glob(R + "/test.zip.00*.zip")):
    z = zipfile.ZipFile(w); nm = z.namelist()
    if len(nm) == 1 and nm[0].startswith("test.zip.00"):
        if not os.path.exists(f"{X}/{nm[0]}"): z.extract(nm[0], X)
        parts.append(f"{X}/{nm[0]}"); os.remove(w)                     # free space as we go
    else: parts.append(w)
print("parts", [os.path.basename(p) for p in parts], flush=True)
with open(X + "/test.zip", "wb") as out:
    for p in parts:
        with open(p, "rb") as f: shutil.copyfileobj(f, out, 1 << 26)
        os.remove(p)
zipfile.ZipFile(X + "/test.zip").extractall(X); os.remove(X + "/test.zip")
ims = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(X + "/**/*.jpeg", recursive=True)}
print(json.dumps(dict(event="extracted", images=len(ims), expected=53576, mins=round((time.time() - t0) / 60, 1))), flush=True)
eq = pd.read_csv(glob.glob("/path/to/sharc_work/data/raw/eyeq/*test*.csv")[0]); eq["stem"] = eq.image.str.replace(".jpeg", "", regex=False)
eq = eq[eq.stem.isin(ims)].reset_index(drop=True)
eq["patient"] = "eyepacs_test:" + eq.stem.str.split("_").str[0]; eq["eye"] = eq.stem.str.split("_").str[1].str[0].str.upper()
eq["grade"] = eq.DR_grade.astype(int); eq["referable"] = (eq.grade >= 2).astype(int); eq["src"] = eq.stem.map(ims); eq["row"] = np.arange(len(eq))
eq = eq.rename(columns={"quality": "eyeq_quality"}); eq["ds"] = "eyepacs_test"; eq["img_id"] = eq.stem; eq["role"] = "test_pool"
mm = np.lib.format.open_memmap(P + "/img512_eyepacs_test_uint8.npy", mode="w+", dtype=np.uint8, shape=(len(eq), 512, 512, 3)); del mm
def cache(chunk):
    mm = np.load(P + "/img512_eyepacs_test_uint8.npy", mmap_mode="r+")
    for i, src in chunk:
        im = cv2.imread(src); g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY); ys, xs = np.where(g > max(10, 0.04 * g.max()))
        if len(ys) > 1000: im = im[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        h, w = im.shape[:2]; s = max(h, w); pad = np.zeros((s, s, 3), np.uint8); pad[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = im
        mm[i] = cv2.cvtColor(cv2.resize(pad, (512, 512), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    mm.flush(); return len(chunk)
items = list(zip(eq.row, eq.src)); 
with Pool(64) as pool: n = sum(pool.map(cache, [items[i:i + 128] for i in range(0, len(items), 128)]))
eq[["row", "ds", "img_id", "patient", "eye", "grade", "referable", "role", "eyeq_quality", "src"]].to_parquet(P + "/metadata_eyepacs_test.parquet")
print(json.dumps(dict(event="cached", images=n, patients=eq.patient.nunique(), referable=round(eq.referable.mean(), 3),
                      both_eyes=int((eq.groupby("patient").size() == 2).sum()), mins=round((time.time() - t0) / 60, 1))), flush=True)
print("EYEPACS TEST DONE", flush=True)
