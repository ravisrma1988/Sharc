import zipfile, os, glob, shutil, json, re, time, datetime, collections
import pandas as pd
RAW = "/path/to/sharc_work/data/raw"; EX = "/path/to/sharc_work/data/extracted"; os.makedirs(EX, exist_ok=True)
IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
rep = {}
def log(*a): print(datetime.datetime.now(IST).strftime("%H:%M:%S"), *a, flush=True)
def cat(parts, out):
    with open(out, "wb") as w:
        for p in parts:
            with open(p, "rb") as r: shutil.copyfileobj(r, w, 1 << 26)
def unzip(z, dest):
    os.makedirs(dest, exist_ok=True); zipfile.ZipFile(z).extractall(dest)
def imgs(d): return [p for p in glob.glob(d + "/**/*", recursive=True) if p.lower().endswith((".jpeg", ".jpg", ".png", ".tif", ".tiff"))]

# ---------------- EyePACS
try:
    d = EX + "/eyepacs"; os.makedirs(d, exist_ok=True); parts = []
    for w in sorted(glob.glob(RAW + "/eyepacs/train.zip.00*.zip")):
        z = zipfile.ZipFile(w); names = z.namelist()
        if len(names) == 1 and re.match(r"train\.zip\.00\d$", names[0]):   # Kaggle wrapper around a split part
            if not os.path.exists(d + "/" + names[0]): z.extract(names[0], d)
            parts.append(d + "/" + names[0])
        else: parts.append(w)
    log("eyepacs parts", [os.path.basename(p) for p in parts])
    cat(parts, d + "/train.zip")
    for p in parts:
        if p.startswith(d): os.remove(p)
    unzip(d + "/train.zip", d); os.remove(d + "/train.zip")
    zipfile.ZipFile(RAW + "/eyepacs/trainLabels.csv.zip").extractall(d)
    lab = pd.read_csv(glob.glob(d + "/**/trainLabels.csv", recursive=True)[0]); ims = imgs(d)
    stems = {os.path.splitext(os.path.basename(p))[0] for p in ims}
    lab["patient"] = lab.image.str.split("_").str[0]
    rep["eyepacs"] = dict(images=len(ims), labels=len(lab), labels_with_image=int(lab.image.isin(stems).sum()), patients=lab.patient.nunique(),
                          patients_with_both_eyes=int((lab.groupby("patient").size() == 2).sum()), grade_counts=lab.level.value_counts().sort_index().to_dict(),
                          expected="35,126 images, 17,563 patients")
    log("eyepacs", rep["eyepacs"])
except Exception as e: rep["eyepacs"] = "ERROR " + repr(e); log(rep["eyepacs"])

# ---------------- APTOS
try:
    d = EX + "/aptos"; unzip(RAW + "/aptos/aptos2019-blindness-detection.zip", d)
    tr = pd.read_csv(d + "/train.csv"); ti = imgs(d + "/train_images"); te = imgs(d + "/test_images")
    rep["aptos"] = dict(train_images=len(ti), train_labels=len(tr), test_images_unlabelled=len(te), grade_counts=tr.diagnosis.value_counts().sort_index().to_dict(),
                        patient_ids="none provided", expected="3,662 labelled train, 1,928 test")
    log("aptos", rep["aptos"])
except Exception as e: rep["aptos"] = "ERROR " + repr(e); log(rep["aptos"])

# ---------------- Messidor-2
try:
    d = EX + "/messidor2"; os.makedirs(d, exist_ok=True)
    cat(sorted(glob.glob(RAW + "/messidor2/IMAGES.zip.00*")), d + "/IMAGES.zip"); unzip(d + "/IMAGES.zip", d); os.remove(d + "/IMAGES.zip")
    pairs = pd.read_csv(RAW + "/messidor2/messidor-2.csv", sep=None, engine="python")
    g = pd.read_csv(RAW + "/messidor2/messidor2_googlebrain_grades_from_repack.csv")
    ims = imgs(d); base = {os.path.basename(p).lower() for p in ims}
    gstem = g.id_code.str.replace("_PP.png", "", regex=False).str.lower()
    matched = sum(1 for s in gstem if any(b.startswith(s) for b in [s + ".jpg", s + ".png", s + ".tif"]) and (s + ".jpg" in base or s + ".png" in base or s + ".tif" in base))
    rep["messidor2"] = dict(images=len(ims), pairing_file_shape=list(pairs.shape), pairing_columns=list(pairs.columns)[:6], grades=len(g), grades_matched_to_images=matched,
                            grade_counts=g.diagnosis.value_counts().sort_index().to_dict(), expected="1,748 images (874 exams x 2 eyes), 1,744 graded")
    log("messidor2", rep["messidor2"])
except Exception as e: rep["messidor2"] = "ERROR " + repr(e); log(rep["messidor2"])

# ---------------- IDRiD (Disease Grading)
try:
    d = EX + "/idrid"; unzip(RAW + "/idrid/B. Disease Grading.zip", d)
    csvs = glob.glob(d + "/**/*.csv", recursive=True); ims = imgs(d)
    info = {os.path.relpath(c, d): pd.read_csv(c).shape for c in csvs}
    rep["idrid"] = dict(images=len(ims), csvs={k: list(v) for k, v in info.items()}, expected="516 images (413 train, 103 test)")
    log("idrid", rep["idrid"])
except Exception as e: rep["idrid"] = "ERROR " + repr(e); log(rep["idrid"])

# ---------------- DeepDRiD
try:
    d = EX + "/deepdrid"; unzip(RAW + "/deepdrid/DeepDRiD-master.zip", d)
    rf = [p for p in imgs(d) if "regular" in p.lower()]; uw = [p for p in imgs(d) if "ultra" in p.lower()]
    csvs = [c for c in glob.glob(d + "/**/*.csv", recursive=True) if "regular" in c.lower()]
    rep["deepdrid"] = dict(regular_images=len(rf), ultrawidefield_images=len(uw), regular_csvs={os.path.relpath(c, d): list(pd.read_csv(c).shape) for c in csvs},
                           patient_dirs=len({p.split("/")[-2] for p in rf}), expected="2,000 regular images from 500 patients")
    log("deepdrid", rep["deepdrid"])
except Exception as e: rep["deepdrid"] = "ERROR " + repr(e); log(rep["deepdrid"])

json.dump(rep, open("/path/to/sharc_work/results/data_verification.json", "w"), indent=1, default=str)
log("VERIFY DONE")
