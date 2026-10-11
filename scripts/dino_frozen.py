import os, json, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, timm
from sklearn.metrics import roc_auc_score
# Second backbone: frozen DINOv2 ViT-S/14 (Apache-2.0) + linear heads. Same 512-px cache, resized on GPU to 518 (= 37 x 14 patches).
W = "/path/to/sharc_work"; dev = "cuda"; ARCH = "vit_small_patch14_dinov2.lvd142m"
torch.set_num_threads(4)  # leave CPU cores to the concurrent E1/E2 job
M = pd.read_parquet(W + "/data/proc/metadata_rows.parquet"); X = np.load(W + "/data/proc/img512_uint8.npy", mmap_mode="r")
net = timm.create_model(ARCH, pretrained=True, num_classes=0, img_size=518).to(dev).eval()
cfg = net.pretrained_cfg; MEAN = torch.tensor(cfg["mean"], device=dev).view(1, 3, 1, 1); STD = torch.tensor(cfg["std"], device=dev).view(1, 3, 1, 1)
fwd = torch.compile(net)
t0 = time.time(); feats = []
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    for i in range(0, len(M), 2048):
        xb = torch.from_numpy(np.ascontiguousarray(X[i:i + 2048])).to(dev)
        for j in range(0, len(xb), 128):
            x = xb[j:j + 128].permute(0, 3, 1, 2).float().div_(255.)
            x = F.interpolate(x, size=(518, 518), mode="bilinear", align_corners=False)
            feats.append(fwd((x - MEAN) / STD).float())
Fe = torch.cat(feats); print(json.dumps(dict(event="features", n=len(Fe), dim=Fe.shape[1], secs=round(time.time() - t0))), flush=True)
tr = torch.tensor(M.role.values == "train", device=dev); va = torch.tensor(M.role.values == "val", device=dev)
y = torch.tensor(M.referable.values, device=dev).float(); g = torch.tensor(M.grade.values, device=dev).long()
mu, sd = Fe[tr].mean(0), Fe[tr].std(0) + 1e-6; Z = (Fe - mu) / sd
os.makedirs(W + "/preds", exist_ok=True)
for seed in [0, 1, 2]:
    best = (-1, None)
    for wd in [1e-4, 1e-3, 1e-2, 1e-1]:
        torch.manual_seed(seed); head = nn.Linear(Z.shape[1], 6).to(dev); opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=wd)
        idx = torch.where(tr)[0]
        for ep in range(60):
            perm = idx[torch.randperm(len(idx), device=dev)]
            for k in range(0, len(perm), 512):
                b = perm[k:k + 512]; o = head(Z[b]); loss = F.binary_cross_entropy_with_logits(o[:, 0], y[b]) + 0.5 * F.cross_entropy(o[:, 1:], g[b])
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad(): auc = roc_auc_score(y[va].cpu().numpy(), torch.sigmoid(head(Z[va])[:, 0]).cpu().numpy())
        print(json.dumps(dict(seed=seed, wd=wd, val_auc=round(auc, 4))), flush=True)
        if auc > best[0]: best = (auc, {k: v.clone() for k, v in head.state_dict().items()}, wd)
    head.load_state_dict(best[1])
    with torch.no_grad(): o = head(Z); p = torch.sigmoid(o[:, 0]).cpu().numpy(); pg = torch.softmax(o[:, 1:], 1).cpu().numpy()
    tag = f"dinov2s_frozen_s{seed}"; out = M[["row", "ds", "img_id", "patient", "eye", "grade", "referable", "role"]].copy(); out["p_ref"] = p
    for k in range(5): out[f"p_g{k}"] = pg[:, k]
    out.to_parquet(f"{W}/preds/{tag}.parquet"); np.save(f"{W}/preds/{tag}_feat.npy", Fe.cpu().numpy().astype(np.float16))
    print("DONE", tag, "best val AUC", round(best[0], 4), "wd", best[2], flush=True)
