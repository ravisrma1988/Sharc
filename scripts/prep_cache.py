import os, glob, json, numpy as np, pandas as pd, cv2
from multiprocessing import Pool
X = "/path/to/sharc_work/data/extracted"; C = "/path/to/sharc_work/data/cache512"; P = "/path/to/sharc_work/data/proc"
os.makedirs(C, exist_ok=True); os.makedirs(P, exist_ok=True)
rows = []
def add(ds, img_id, path, patient, eye, grade, **kw): rows.append(dict(ds=ds, img_id=img_id, src=path, patient=f"{ds}:{patient}", eye=eye, grade=grade, **kw))
# EyePACS
lab = pd.read_csv(glob.glob(X + "/eyepacs/**/trainLabels.csv", recursive=True)[0])
ep = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(X + "/eyepacs/**/*.jpeg", recursive=True)}
eq = pd.concat([pd.read_csv(f) for f in glob.glob("/path/to/sharc_work/data/raw/eyeq/*.csv")])
eq["stem"] = eq.image.str.replace(".jpeg", "", regex=False); eqq = dict(zip(eq.stem, eq.quality))
for _, r in lab.iterrows():
    pid, eye = r.image.split("_"); add("eyepacs", r.image, ep[r.image], pid, eye[0].upper(), int(r.level), eyeq_quality=eqq.get(r.image, np.nan))
# Messidor-2 (Google Brain adjudicated grades; patients from official exam pairs)
mi = {os.path.splitext(os.path.basename(p))[0].lower(): p for p in glob.glob(X + "/messidor2/**/*.*", recursive=True) if p.lower().endswith((".png", ".jpg", ".tif"))}
g = pd.read_csv("/path/to/sharc_work/data/raw/messidor2/messidor2_googlebrain_grades_from_repack.csv")
gg = {os.path.splitext(s)[0].lower(): int(v) for s, v in zip(g.id_code, g.diagnosis)}
pr = pd.read_csv("/path/to/sharc_work/data/raw/messidor2/messidor-2.csv", sep=None, engine="python"); pr.columns = [c.strip() for c in pr.columns]
for i, r in pr.iterrows():
    for eye, col in [("L", "left"), ("R", "right")]:
        s = os.path.splitext(str(r[col]).strip())[0].lower()
        if s in gg: add("messidor2", s, mi[s], i, eye, gg[s])
# DeepDRiD (labelled training + validation; hospital source recorded)
for part in ["training", "validation"]:
    base = glob.glob(X + f"/deepdrid/**/regular-fundus-{part}", recursive=True)[0]
    d = pd.read_csv(f"{base}/regular-fundus-{part}.csv"); s = pd.read_csv(f"{base}/regular-fundus-source-{part}.csv")[["image_id", "Source"]]
    d = d.merge(s, on="image_id", how="left")
    for _, r in d.iterrows():
        eye = "L" if "_l" in r.image_id else "R"; gr = r.left_eye_DR_Level if eye == "L" else r.right_eye_DR_Level
        p = os.path.join(base, str(r.patient_id), r.image_id + ".jpg")
        if pd.notna(gr) and os.path.exists(p): add("deepdrid", r.image_id, p, r.patient_id, eye, int(gr), hospital=r.Source, quality=r["Overall quality"])
# DDR (image-level; grade 5 = ungradable, excluded)
for sp in ["train", "valid", "test"]:
    t = glob.glob(X + f"/ddr/**/DR_grading/{sp}.txt", recursive=True)[0]
    for line in open(t):
        if not line.strip(): continue
        n, gr = line.split(); gr = int(gr)
        if gr < 5: add("ddr", n, os.path.join(os.path.dirname(t), sp, n), n, "U", gr, ddr_split=sp)
# APTOS (image-level, labelled train only)
a = pd.read_csv(X + "/aptos/train.csv")
for _, r in a.iterrows(): add("aptos", r.id_code, X + f"/aptos/train_images/{r.id_code}.png", r.id_code, "U", int(r.diagnosis))
# IDRiD (image-level, both official sets)
for sp, lf in [("a. Training Set", "a. IDRiD_Disease Grading_Training Labels.csv"), ("b. Testing Set", "b. IDRiD_Disease Grading_Testing Labels.csv")]:
    d = pd.read_csv(glob.glob(X + f"/idrid/**/{lf}", recursive=True)[0])
    for _, r in d.iterrows():
        p = glob.glob(X + f"/idrid/**/{sp}/{r['Image name']}.jpg", recursive=True)[0]; tag = sp[0]
        add("idrid", f"{tag}_{r['Image name']}", p, f"{tag}_{r['Image name']}", "U", int(r["Retinopathy grade"]))
M = pd.DataFrame(rows); M["referable"] = (M.grade >= 2).astype(int)
M["cache"] = C + "/" + M.ds + "/" + M.img_id.astype(str).str.replace("/", "_") + ".jpg"
# EyePACS patient split 60/10/30 (seed 2026)
rng = np.random.default_rng(2026); pts = M[M.ds == "eyepacs"].patient.unique(); rng.shuffle(pts)
n = len(pts); role = {p: "train" for p in pts[:int(.6 * n)]}; role.update({p: "val" for p in pts[int(.6 * n):int(.7 * n)]}); role.update({p: "calib" for p in pts[int(.7 * n):]})
M["role"] = np.where(M.ds == "eyepacs", M.patient.map(role), "target")
print(M.groupby("ds").agg(images=("img_id", "size"), patients=("patient", "nunique"), referable=("referable", "mean")).round(3).to_string(), flush=True)
print(M[M.ds == "eyepacs"].groupby("role").agg(images=("img_id", "size"), patients=("patient", "nunique")).to_string(), flush=True)
def crop_resize(args):
    src, dst = args
    if os.path.exists(dst): return 1
    im = cv2.imread(src, cv2.IMREAD_COLOR)
    if im is None: return 0
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY); m = g > max(10, 0.04 * g.max())
    ys, xs = np.where(m)
    if len(ys) > 1000: im = im[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = im.shape[:2]; s = max(h, w); pad = np.zeros((s, s, 3), np.uint8); pad[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = im
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    return int(cv2.imwrite(dst, cv2.resize(pad, (512, 512), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 95]))
with Pool(64) as pool: ok = pool.map(crop_resize, list(zip(M.src, M.cache)), chunksize=64)
M["cached_ok"] = ok
M.to_parquet(P + "/metadata.parquet"); print("cached", sum(ok), "of", len(M), flush=True)
print("CACHE DONE", flush=True)
