import os, sys, time, json, math, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, timm
from sklearn.metrics import roc_auc_score
# GPU-resident pipeline: uint8 images live in GPU memory, augmentation runs on the GPU, no CPU data workers.
W = "/path/to/sharc_work"; seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
ARCH = sys.argv[2] if len(sys.argv) > 2 else "convnext_tiny.fb_in22k"; EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 15; BS = 64
torch.manual_seed(seed); np.random.seed(seed); dev = "cuda"
M = pd.read_parquet(W + "/data/proc/metadata_rows.parquet"); X = np.load(W + "/data/proc/img512_uint8.npy", mmap_mode="r")
tr = M[M.role == "train"].row.values; va = M[M.role == "val"].row.values
t0 = time.time()
Xtr = torch.from_numpy(np.ascontiguousarray(X[np.sort(tr)])).to(dev); Ytr = torch.tensor(M.set_index("row").loc[np.sort(tr)].referable.values, device=dev).float()
Gtr = torch.tensor(M.set_index("row").loc[np.sort(tr)].grade.values, device=dev).long()
Xva = torch.from_numpy(np.ascontiguousarray(X[np.sort(va)])).to(dev); Yva = M.set_index("row").loc[np.sort(va)].referable.values
print(json.dumps(dict(event="loaded_to_gpu", train=len(tr), val=len(va), secs=round(time.time() - t0),
                      gpu_gb=round(torch.cuda.memory_allocated() / 1e9, 1))), flush=True)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
def prep(xb, aug):
    x = xb.permute(0, 3, 1, 2).float().div_(255.)
    if aug:
        B = x.shape[0]
        fx = torch.where(torch.rand(B, device=dev) < .5, -1., 1.); fy = torch.where(torch.rand(B, device=dev) < .5, -1., 1.)
        a = (torch.rand(B, device=dev) * 30 - 15) * math.pi / 180; s = 1 / (torch.rand(B, device=dev) * .2 + .9)
        th = torch.zeros(B, 2, 3, device=dev)
        th[:, 0, 0] = s * torch.cos(a) * fx; th[:, 0, 1] = -s * torch.sin(a); th[:, 1, 0] = s * torch.sin(a); th[:, 1, 1] = s * torch.cos(a) * fy
        x = F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False), mode="bilinear", padding_mode="zeros", align_corners=False)
        x = (x * (torch.rand(B, 1, 1, 1, device=dev) * .3 + .85) + (torch.rand(B, 1, 1, 1, device=dev) * .16 - .08)).clamp_(0, 1)
    return ((x - MEAN) / STD).contiguous(memory_format=torch.channels_last)
class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.b = timm.create_model(ARCH, pretrained=True, num_classes=0); d = s.b.num_features
        s.ref = nn.Linear(d, 1); s.grade = nn.Linear(d, 5)
    def forward(s, x): f = s.b(x); return s.ref(f).squeeze(1), s.grade(f), f
net = Net().to(dev).to(memory_format=torch.channels_last)
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", W + "/.inductor_cache")
fwd = torch.compile(net)  # same weights as net; compiled graph for speed only
print(json.dumps(dict(event="torch_compile_enabled", torch=torch.__version__)), flush=True)
steps = EPOCHS * (len(tr) // BS)
opt = torch.optim.AdamW([{"params": net.b.parameters(), "lr": 1e-4}, {"params": list(net.ref.parameters()) + list(net.grade.parameters()), "lr": 1e-3}], weight_decay=0.05)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[1e-4, 1e-3], total_steps=steps, pct_start=0.1)
bce, ce = nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss()
tag = f"{ARCH.split('.')[0]}_s{seed}"; best = -1; os.makedirs(W + "/models", exist_ok=True)
@torch.no_grad()
def predict_tensor(Xg, bs=128):
    net.eval(); P, G, Fe = [], [], []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(Xg), bs):
            r, g, f = fwd(prep(Xg[i:i + bs], False)); P.append(torch.sigmoid(r.float())); G.append(torch.softmax(g.float(), 1)); Fe.append(f.half())
    return torch.cat(P).cpu().numpy(), torch.cat(G).cpu().numpy(), torch.cat(Fe).cpu().numpy()
for ep in range(EPOCHS):
    net.train(); torch.cuda.synchronize(); t0 = time.time(); L = 0.; perm = torch.randperm(len(tr), device=dev); n = 0
    for i in range(0, len(tr) - BS + 1, BS):
        idx = perm[i:i + BS]; x = prep(Xtr[idx], True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            r, gl, _ = fwd(x); loss = bce(r.float(), Ytr[idx]) + 0.5 * ce(gl.float(), Gtr[idx])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); L += loss.detach(); n += 1
    torch.cuda.synchronize(); tt = time.time() - t0
    p, _, _ = predict_tensor(Xva); auc = roc_auc_score(Yva, p)
    print(json.dumps(dict(tag=tag, epoch=ep, loss=round(float(L) / n, 4), val_auc=round(auc, 4), train_secs=round(tt), img_per_s=round(n * BS / tt))), flush=True)
    if auc > best: best = auc; torch.save(net.state_dict(), f"{W}/models/{tag}.pt")
del Xtr; torch.cuda.empty_cache()
net.load_state_dict(torch.load(f"{W}/models/{tag}.pt"))
P_, G_, F_ = [], [], []
for i in range(0, len(M), 4096):
    p, g, f = predict_tensor(torch.from_numpy(np.ascontiguousarray(X[i:i + 4096])).to(dev)); P_.append(p); G_.append(g); F_.append(f)
p, g, f = np.concatenate(P_), np.concatenate(G_), np.concatenate(F_)
os.makedirs(W + "/preds", exist_ok=True)
out = M[["row", "ds", "img_id", "patient", "eye", "grade", "referable", "role"]].copy(); out["p_ref"] = p
for k in range(5): out[f"p_g{k}"] = g[:, k]
out.to_parquet(f"{W}/preds/{tag}.parquet"); np.save(f"{W}/preds/{tag}_feat.npy", f)
print("DONE", tag, "best val AUC", round(best, 4), flush=True)
