import os, json, time, math, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, timm
# Overnight GPU job: synthetic appearance shifts on held-out EyePACS calibration images, inference with all ConvNeXt seeds.
W = "/path/to/sharc_work"; dev = "cuda"; ARCH = os.environ.get("ARCH", "convnext_tiny.fb_in22k"); SHORT = ARCH.split(".")[0]; torch.set_num_threads(4)
M = pd.read_parquet(W + "/data/proc/metadata_rows.parquet"); X = np.load(W + "/data/proc/img512_uint8.npy", mmap_mode="r")
C = M[M.role == "calib"].sort_values("row").reset_index(drop=True); rows = C.row.values
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
def gblur(x, sigma):
    k = int(2 * math.ceil(3 * sigma) + 1); t = torch.arange(k, device=dev) - k // 2; g = torch.exp(-t.float() ** 2 / (2 * sigma ** 2)); g = g / g.sum()
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), g.view(1, 1, 1, k).repeat(3, 1, 1, 1), groups=3)
    return F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), g.view(1, 1, k, 1).repeat(3, 1, 1, 1), groups=3)
def degrade(x, kind, lvl):
    mask = (x.mean(1, keepdim=True) > 0.02).float()                       # keep the black surround black
    if kind == "blur": y = gblur(x, [1.5, 3.0, 5.0][lvl - 1])
    elif kind == "contrast": c = [0.7, 0.5, 0.3][lvl - 1]; m = (x * mask).sum((2, 3), keepdim=True) / mask.sum((2, 3), keepdim=True); y = m + c * (x - m)
    elif kind == "dark": y = x ** [1.4, 1.8, 2.4][lvl - 1]
    elif kind == "colorcast": s = torch.tensor([[1.15, 0.95, 0.80], [1.30, 0.90, 0.65], [1.45, 0.85, 0.50]][lvl - 1], device=dev).view(1, 3, 1, 1); y = x * s
    return (y.clamp(0, 1) * mask)
class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.b = timm.create_model(ARCH, pretrained=False, num_classes=0); d = s.b.num_features; s.ref = nn.Linear(d, 1); s.grade = nn.Linear(d, 5)
    def forward(s, x): f = s.b(x); return s.ref(f).squeeze(1), s.grade(f), f
conds = [(k, l) for k in ["blur", "contrast", "dark", "colorcast"] for l in [1, 2, 3]]
import sys
SANITY = len(sys.argv) > 1 and sys.argv[1] == "sanity"; SEEDS = [int(s) for s in os.environ.get("SEEDS", "0" if SANITY else "0,1,2").split(",")]
if SANITY: conds = [(k, 3) for k in ["blur", "contrast", "dark", "colorcast"]]
OUTD = W + ("/preds/degraded_sanity" if SANITY else "/preds/degraded")
Xc = torch.from_numpy(np.ascontiguousarray(X[rows])).to(dev)          # ~8 GB uint8 on GPU
os.makedirs(OUTD, exist_ok=True); t0 = time.time()
for seed in SEEDS:
    net = Net().to(dev).to(memory_format=torch.channels_last); net.load_state_dict(torch.load(f"{W}/models/{SHORT}_s{seed}.pt")); net.eval(); fwd = torch.compile(net)
    for kind, lvl in conds:
        P, G, Fe = [], [], []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(Xc), 128):
                x = degrade(Xc[i:i + 128].permute(0, 3, 1, 2).float().div(255.), kind, lvl)
                r, g, f = fwd(((x - MEAN) / STD).contiguous(memory_format=torch.channels_last)); P.append(torch.sigmoid(r.float())); G.append(torch.softmax(g.float(), 1)); Fe.append(f.half())
        out = C[["row", "ds", "img_id", "patient", "eye", "grade", "referable", "role"]].copy(); out["p_ref"] = torch.cat(P).cpu().numpy()
        Gm = torch.cat(G).cpu().numpy()
        for k in range(5): out[f"p_g{k}"] = Gm[:, k]
        tag = f"{SHORT}_s{seed}_{kind}{lvl}"; out.to_parquet(f"{OUTD}/{tag}.parquet"); np.save(f"{OUTD}/{tag}_feat.npy", torch.cat(Fe).cpu().numpy())
        auc_like = float(out[out.referable == 1].p_ref.mean() - out[out.referable == 0].p_ref.mean())
        print(json.dumps(dict(tag=tag, mean_p=round(float(out.p_ref.mean()), 4), sep_pos_minus_neg=round(auc_like, 4), mins=round((time.time() - t0) / 60, 1))), flush=True)
print("DEGRADE DONE", flush=True)
