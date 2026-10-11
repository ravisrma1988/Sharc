import os, sys, json, time, math, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, timm
from sklearn.metrics import roc_auc_score
# DINOv2 (frozen) third backbone: refit + SAVE linear heads per seed, clean predictions, and degraded-image predictions.
W = "/path/to/sharc_work"; dev = "cuda"; ARCH = "vit_small_patch14_dinov2.lvd142m"; BB = "dinov2s_frozen"; torch.set_num_threads(4)
SMOKE = len(sys.argv) > 1 and sys.argv[1] == "smoke"; SEEDS = [0] if SMOKE else [0, 1, 2]
M = pd.read_parquet(W + "/data/proc/metadata_rows.parquet"); X = np.load(W + "/data/proc/img512_uint8.npy", mmap_mode="r")
Fe = torch.from_numpy(np.load(f"{W}/preds/{BB}_s0_feat.npy").astype(np.float32)).to(dev)       # clean features (backbone frozen)
tr = torch.tensor(M.role.values == "train", device=dev); va = torch.tensor(M.role.values == "val", device=dev)
y = torch.tensor(M.referable.values, device=dev).float(); g = torch.tensor(M.grade.values, device=dev).long()
mu, sd = Fe[tr].mean(0), Fe[tr].std(0) + 1e-6
net = timm.create_model(ARCH, pretrained=True, num_classes=0, img_size=518).to(dev).eval(); cfg = net.pretrained_cfg; fwd = torch.compile(net)
MEAN = torch.tensor(cfg["mean"], device=dev).view(1, 3, 1, 1); STD = torch.tensor(cfg["std"], device=dev).view(1, 3, 1, 1)
def gblur(x, sigma):
    k = int(2 * math.ceil(3 * sigma) + 1); t = torch.arange(k, device=dev) - k // 2; gg = torch.exp(-t.float() ** 2 / (2 * sigma ** 2)); gg = gg / gg.sum()
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), gg.view(1, 1, 1, k).repeat(3, 1, 1, 1), groups=3)
    return F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), gg.view(1, 1, k, 1).repeat(3, 1, 1, 1), groups=3)
def degrade(x, kind, lvl):
    mask = (x.mean(1, keepdim=True) > 0.02).float()
    if kind == "blur": yy = gblur(x, [1.5, 3.0, 5.0][lvl - 1])
    elif kind == "contrast": c = [0.7, 0.5, 0.3][lvl - 1]; m = (x * mask).sum((2, 3), keepdim=True) / mask.sum((2, 3), keepdim=True); yy = m + c * (x - m)
    elif kind == "dark": yy = x ** [1.4, 1.8, 2.4][lvl - 1]
    elif kind == "colorcast": s = torch.tensor([[1.15, 0.95, 0.80], [1.30, 0.90, 0.65], [1.45, 0.85, 0.50]][lvl - 1], device=dev).view(1, 3, 1, 1); yy = x * s
    return yy.clamp(0, 1) * mask
Z = (Fe - mu) / sd; heads = {}; os.makedirs(W + "/models", exist_ok=True); os.makedirs(W + "/preds/degraded", exist_ok=True)
for seed in SEEDS:
    best = (-1, None, None)
    for wd in [1e-4, 1e-3, 1e-2, 1e-1]:
        torch.manual_seed(seed); h = nn.Linear(Z.shape[1], 6).to(dev); opt = torch.optim.AdamW(h.parameters(), lr=1e-3, weight_decay=wd); idx = torch.where(tr)[0]
        for ep in range(60):
            perm = idx[torch.randperm(len(idx), device=dev)]
            for k in range(0, len(perm), 512):
                b = perm[k:k + 512]; o = h(Z[b]); loss = F.binary_cross_entropy_with_logits(o[:, 0], y[b]) + 0.5 * F.cross_entropy(o[:, 1:], g[b]); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad(): auc = roc_auc_score(y[va].cpu().numpy(), torch.sigmoid(h(Z[va])[:, 0]).cpu().numpy())
        if auc > best[0]: best = (auc, {k: v.clone() for k, v in h.state_dict().items()}, wd)
    h.load_state_dict(best[1]); heads[seed] = h; torch.save(dict(head=best[1], mu=mu, sd=sd, wd=best[2]), f"{W}/models/{BB}_s{seed}_head.pt")
    with torch.no_grad(): o = h(Z); p = torch.sigmoid(o[:, 0]).cpu().numpy(); pg = torch.softmax(o[:, 1:], 1).cpu().numpy()
    out = M[["row", "ds", "img_id", "patient", "eye", "grade", "referable", "role"]].copy(); out["p_ref"] = p
    for k in range(5): out[f"p_g{k}"] = pg[:, k]
    out.to_parquet(f"{W}/preds/{BB}_s{seed}.parquet"); np.save(f"{W}/preds/{BB}_s{seed}_feat.npy", Fe.cpu().numpy().astype(np.float16))
    print(json.dumps(dict(seed=seed, val_auc=round(best[0], 4), wd=best[2])), flush=True)
C = M[M.role == "calib"].sort_values("row").reset_index(drop=True); Xc = torch.from_numpy(np.ascontiguousarray(X[C.row.values])).to(dev)
conds = [(k, 3) for k in ["blur", "contrast", "dark", "colorcast"]] if SMOKE else [(k, l) for k in ["blur", "contrast", "dark", "colorcast"] for l in [1, 2, 3]]
OUT = W + ("/preds/degraded_sanity" if SMOKE else "/preds/degraded"); os.makedirs(OUT, exist_ok=True); t0 = time.time()
for kind, lvl in conds:
    feats = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(Xc), 128):
            x = F.interpolate(degrade(Xc[i:i + 128].permute(0, 3, 1, 2).float().div(255.), kind, lvl), size=(518, 518), mode="bilinear", align_corners=False)
            feats.append(fwd((x - MEAN) / STD).float())
    Fd = torch.cat(feats); Zd = (Fd - mu) / sd
    for seed in SEEDS:
        with torch.no_grad(): o = heads[seed](Zd); p = torch.sigmoid(o[:, 0]).cpu().numpy(); pg = torch.softmax(o[:, 1:], 1).cpu().numpy()
        out = C[["row", "ds", "img_id", "patient", "eye", "grade", "referable", "role"]].copy(); out["p_ref"] = p
        for k in range(5): out[f"p_g{k}"] = pg[:, k]
        tag = f"{BB}_s{seed}_{kind}{lvl}"; out.to_parquet(f"{OUT}/{tag}.parquet"); np.save(f"{OUT}/{tag}_feat.npy", Fd.cpu().numpy().astype(np.float16))
        auc = roc_auc_score(out.referable, out.p_ref); print(json.dumps(dict(tag=tag, auc=round(auc, 4), mins=round((time.time() - t0) / 60, 1))), flush=True)
print("DINO DEGRADE DONE", flush=True)
