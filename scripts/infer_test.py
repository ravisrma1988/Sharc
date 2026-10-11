import os, sys, json, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, timm
# Inference on the cached EyeQ-labelled EyePACS test pool for all trained models (ConvNeXt x5, ResNet-50 x3, DINOv2-frozen x3).
W = "/path/to/sharc_work"; dev = "cuda"; torch.set_num_threads(4)
SMOKE = len(sys.argv) > 1 and sys.argv[1] == "smoke"
M = pd.read_parquet(W + "/data/proc/metadata_eyepacs_test.parquet"); X = np.load(W + "/data/proc/img512_eyepacs_test_uint8.npy", mmap_mode="r")
if SMOKE: M = M.iloc[:512]
OUT = W + "/preds/eyepacs_test" + ("_smoke" if SMOKE else ""); os.makedirs(OUT, exist_ok=True)
IMN = (torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1), torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1))
class Net(nn.Module):
    def __init__(s, arch):
        super().__init__(); s.b = timm.create_model(arch, pretrained=False, num_classes=0); d = s.b.num_features; s.ref = nn.Linear(d, 1); s.grade = nn.Linear(d, 5)
    def forward(s, x): f = s.b(x); return s.ref(f).squeeze(1), s.grade(f), f
def batches(fn, bs=128):
    P, G, Fe = [], [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(M), 2048):
            xb = torch.from_numpy(np.ascontiguousarray(X[M.row.values[i:i + 2048]])).to(dev)
            for j in range(0, len(xb), bs):
                r, g, f = fn(xb[j:j + bs].permute(0, 3, 1, 2).float().div(255.)); P.append(torch.sigmoid(r.float())); G.append(torch.softmax(g.float(), 1)); Fe.append(f.float().half())
    return torch.cat(P).cpu().numpy(), torch.cat(G).cpu().numpy(), torch.cat(Fe).cpu().numpy()
def save(tag, p, g, f):
    out = M[["row", "ds", "img_id", "patient", "eye", "grade", "referable", "role", "eyeq_quality"]].copy(); out["p_ref"] = p
    for k in range(5): out[f"p_g{k}"] = g[:, k]
    out.to_parquet(f"{OUT}/{tag}.parquet"); np.save(f"{OUT}/{tag}_feat.npy", f)
    from sklearn.metrics import roc_auc_score; print(json.dumps(dict(tag=tag, n=len(out), auc=round(roc_auc_score(out.referable, p), 4))), flush=True)
jobs = [("convnext_tiny.fb_in22k", s) for s in ([0] if SMOKE else [0, 1, 2, 3, 4])] + [("resnet50.a1_in1k", s) for s in ([] if SMOKE else [0, 1, 2])]
for arch, s in jobs:
    net = Net(arch).to(dev).to(memory_format=torch.channels_last); short = arch.split(".")[0]; net.load_state_dict(torch.load(f"{W}/models/{short}_s{s}.pt")); net.eval(); fwd = torch.compile(net)
    save(f"{short}_s{s}", *batches(lambda x: fwd(((x - IMN[0]) / IMN[1]).contiguous(memory_format=torch.channels_last))))
dn = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=518).to(dev).eval(); cfg = dn.pretrained_cfg; dfwd = torch.compile(dn)
DM, DS = torch.tensor(cfg["mean"], device=dev).view(1, 3, 1, 1), torch.tensor(cfg["std"], device=dev).view(1, 3, 1, 1)
for s in ([0] if SMOKE else [0, 1, 2]):
    H = torch.load(f"{W}/models/dinov2s_frozen_s{s}_head.pt"); head = nn.Linear(384, 6).to(dev); head.load_state_dict(H["head"]); mu, sd = H["mu"].to(dev), H["sd"].to(dev)
    def fn(x):
        f = dfwd((F.interpolate(x, size=(518, 518), mode="bilinear", align_corners=False) - DM) / DS).float(); o = head((f - mu) / sd); return o[:, 0], o[:, 1:], f
    save(f"dinov2s_frozen_s{s}", *batches(fn))
print("INFER TEST DONE", flush=True)
