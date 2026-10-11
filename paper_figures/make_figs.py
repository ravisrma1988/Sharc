import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from figdata import *
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.8, "pdf.fonttype": 42, "savefig.dpi": 300})
def save(fig, name): fig.savefig(name + ".pdf", bbox_inches="tight"); fig.savefig(name + ".png", bbox_inches="tight", dpi=200); plt.close(fig)

# Figure 2: worst-scenario miss rate per method, alpha 0.05 (horizontal bars, methods on the y-axis)
fig, axs = plt.subplots(1, 3, figsize=(7.4, 4.2), sharey=True)
y = np.arange(len(METHODS))[::-1]; h = 0.26; greys = ["0.15", "0.5", "0.8"]
for ax, g in zip(axs, ["Prevalence shift", "Appearance shift", "Real sites"]):
    for i, b in enumerate(BB):
        ax.barh(y + (1 - i) * h, 100 * np.array(WORST[g][b]), h, color=greys[i], edgecolor="black", linewidth=0.3, label=b)
    ax.axvline(5, color="red", ls="--", lw=1)
    ax.set_xscale("log"); ax.set_xlim(1, 100); ax.set_xticks([1, 2, 5, 10, 20, 50, 100]); ax.set_xticklabels(["1", "2", "5", "10", "20", "50", "100"])
    ax.set_title(g, fontsize=9.5); ax.set_xlabel("Worst-scenario miss rate (%)", fontsize=8.5)
    ax.tick_params(axis="y", length=0)
axs[0].set_yticks(y); axs[0].set_yticklabels(METHODS, fontsize=9)
for k in (8, 9): axs[0].get_yticklabels()[k].set_fontweight("bold")
hd, lb = axs[0].get_legend_handles_labels()
fig.legend(hd + [plt.Line2D([], [], color="red", ls="--")], lb + ["Target miss rate (5%)"], loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.55, 1.03), fontsize=8.5)
fig.tight_layout(rect=(0, 0, 1, 0.95)); save(fig, "fig2_worst_case")

# Figure 3: miss rate against severity, ConvNeXt
cols = {"Source CRC": "black", "Weighted CRC": "orange", "Unlabelled correction": "green", "SHARC (CRC)": "blue", "SHARC (exact)": "red"}
marks = {"Source CRC": "o", "Weighted CRC": "s", "Unlabelled correction": "^", "SHARC (CRC)": "D", "SHARC (exact)": "v"}
fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.5), sharey=True)
for ax, (kind, d) in zip(axs, SEV.items()):
    for m, v in d.items():
        ax.plot([0, 1, 2, 3], 100 * np.array(v), color=cols[m], marker=marks[m], ms=3.5, lw=1.2, label=m)
    ax.axhline(5, color="red", ls="--", lw=0.8); ax.set_title(kind, fontsize=9); ax.set_xticks([0, 1, 2, 3]); ax.set_xlabel("Severity level")
    ax.set_yscale("log"); ax.set_ylim(0.08, 80); ax.set_yticks([0.1, 1, 5, 10, 50]); ax.set_yticklabels(["0.1", "1", "5", "10", "50"])
axs[0].set_ylabel("Miss rate (%)")
axs[-1].legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0)); fig.tight_layout(); save(fig, "fig3_severity")

# Figure 4: preregistered deployment grid, ConvNeXt
rows = [0.4, 1.0, 3.0]; colnames = ["none", "B1", "B2", "B3", "C1", "C2", "C3", "D1", "D2", "D3", "K1", "K2", "K3"]
src = np.array([[100 * SRC_MISS[c] for c in COLS] for _ in rows])
flag = np.array([[100 * (FLAG[r]["none"] if c == "none" else 1.0) for c in COLS] for r in rows])
lab = np.array([[LABELS[r][c] for c in COLS] for r in rows])
fig, axs = plt.subplots(3, 1, figsize=(7.4, 5.6))
panels = [(src, "(a) Miss rate of source calibration (%), target 5%", "Greys", 0, 30, "{:.1f}"),
          (flag, "(b) Deployments in which SHARC acquired labels (%)", "Blues", 0, 100, "{:.0f}"),
          (lab, "(c) Mean number of target units inspected by SHARC", "Oranges", 0, 400, "{:.0f}")]
for ax, (M, title, cmap, lo, hi, fmt) in zip(axs, panels):
    ax.imshow(M, cmap=cmap, vmin=lo, vmax=hi, aspect="auto")
    ax.set_yticks(range(3)); ax.set_yticklabels(["0.4x", "1x", "3x"]); ax.set_ylabel("Prevalence")
    ax.set_xticks(range(len(COLS))); ax.set_xticklabels(colnames, fontsize=9); ax.set_title(title, fontsize=9.5, loc="left")
    for i in range(3):
        for j in range(len(COLS)):
            v = M[i, j]; dark = (v - lo) / (hi - lo) > 0.55
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8, color="white" if dark else "black",
                    fontweight="bold" if (title.startswith("(a)") and v > 5) else "normal")
    ax.set_xticks(np.arange(-0.5, len(COLS)), minor=True); ax.set_yticks(np.arange(-0.5, 3), minor=True)
    ax.grid(which="minor", color="white", lw=1); ax.tick_params(which="minor", length=0)
    for s in ax.spines.values(): s.set_visible(False)
axs[-1].set_xlabel("Appearance condition (B blur, C contrast loss, D darkening, K colour cast, levels 1 to 3)")
fig.tight_layout(); save(fig, "fig4_deployment_grid")

# Figure 5: annotation and safety by setting, ConvNeXt
pc = {"Source only": "black", "SHARC": "blue", "Always label": "0.6", "Fixed 100 labels": "orange"}
lab5 = ["No shift", "Prevalence only", "Appearance", "Mixed", "Concept"]
fig, axs = plt.subplots(2, 1, figsize=(7.0, 4.9), sharex=True)
x = np.arange(len(SETTINGS)); w = 0.2
for i, (p, d) in enumerate(POL.items()):
    axs[0].bar(x + (i - 1.5) * w, d["labels"], w, color=pc[p], edgecolor="black", linewidth=0.3, label=p)
    axs[1].bar(x + (i - 1.5) * w, 100 * np.array(d["miss"]), w, color=pc[p], edgecolor="black", linewidth=0.3)
axs[1].axhline(5, color="red", ls="--", lw=1)
for ax, yl in zip(axs, ["Target units inspected (mean)", "Miss rate (%)"]):
    ax.set_xticks(x); ax.set_xticklabels(lab5, fontsize=9); ax.set_ylabel(yl, fontsize=9.5); ax.tick_params(axis="y", labelsize=9)
axs[0].set_title("(a) Annotation per deployment", fontsize=10, loc="left"); axs[1].set_title("(b) Missed referable units", fontsize=10, loc="left")
hd, lb = axs[0].get_legend_handles_labels()
fig.legend(hd + [plt.Line2D([], [], color="red", ls="--")], lb + ["Target (5%)"], loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.03), fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.95)); save(fig, "fig5_settings")

# Figure 6: annotation budget against prevalence
fig, ax = plt.subplots(figsize=(4.2, 2.9)); pr = np.linspace(0.08, 0.7, 200)
for a, K, c in [(0.02, 114, "red"), (0.05, 45, "blue"), (0.10, 22, "black")]:
    ax.plot(pr, K / pr, color=c, lw=1.1, label=fr"$K/\pi$ at $\alpha={a:g}$ (K={K})")
    p, n = zip(*BUDGET[a]); ax.scatter(p, n, s=14, color=c, edgecolor="black", linewidth=0.3, zorder=3)
ax.set_yscale("log"); ax.set_xlabel(r"Referable prevalence at the site, $\pi$"); ax.set_ylabel("Target units inspected")
ax.set_yticks([20, 50, 100, 200, 500, 1000]); ax.set_yticklabels(["20", "50", "100", "200", "500", "1000"])
ax.legend(fontsize=7, frameon=False); fig.tight_layout(); save(fig, "fig6_label_budget")
print("figures done")
