import os, glob, time, subprocess, numpy as np, pandas as pd, cv2
X = "/path/to/sharc_work/data/extracted"; C = "/path/to/sharc_work/data/cache512"; P = "/path/to/sharc_work/data/proc"
while subprocess.run(["pgrep", "-f", "scripts/prep_cache.py"], capture_output=True).returncode == 0: time.sleep(20)
M = pd.read_parquet(P + "/metadata.parquet"); M = M[M.ds != "deepdrid"]
rows = []
for part in ["training", "validation"]:
    base = glob.glob(X + f"/deepdrid/**/regular-fundus-{part}", recursive=True)[0]
    d = pd.read_csv(f"{base}/regular-fundus-{part}.csv").merge(pd.read_csv(f"{base}/regular-fundus-source-{part}.csv")[["image_id", "Source"]], on="image_id", how="left")
    for _, r in d.iterrows():
        eye = "L" if "_l" in r.image_id else "R"; gr = r.left_eye_DR_Level if eye == "L" else r.right_eye_DR_Level
        p = os.path.join(base, "Images", str(r.patient_id), r.image_id + ".jpg")
        if pd.notna(gr) and os.path.exists(p):
            rows.append(dict(ds="deepdrid", img_id=r.image_id, src=p, patient=f"deepdrid:{r.patient_id}", eye=eye, grade=int(gr), hospital=r.Source,
                             quality=r["Overall quality"], referable=int(gr >= 2), cache=f"{C}/deepdrid/{r.image_id}.jpg", role="target"))
D = pd.DataFrame(rows)
def crop_resize(src, dst):
    im = cv2.imread(src, cv2.IMREAD_COLOR)
    if im is None: return 0
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY); m = g > max(10, 0.04 * g.max()); ys, xs = np.where(m)
    if len(ys) > 1000: im = im[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = im.shape[:2]; s = max(h, w); pad = np.zeros((s, s, 3), np.uint8); pad[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = im
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    return int(cv2.imwrite(dst, cv2.resize(pad, (512, 512), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 95]))
D["cached_ok"] = [crop_resize(s, d) for s, d in zip(D.src, D.cache)]
M = pd.concat([M, D], ignore_index=True); M.to_parquet(P + "/metadata.parquet")
print(M.groupby("ds").agg(images=("img_id", "size"), patients=("patient", "nunique"), referable=("referable", "mean"), cached=("cached_ok", "sum")).round(3).to_string())
print("DeepDRiD hospitals:", D.hospital.value_counts().to_dict())
print("PATCH DONE")
