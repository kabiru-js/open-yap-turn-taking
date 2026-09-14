"""Figures for EXP-002 annotation/segmentation robustness.

Reads results/exp002_arm_a.json and exp002_arm_b.json; writes
figures/exp002_{arm_a,arm_b}.{png,pdf}. Expects matplotlib; backend Agg.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

CLEAN = (0.7148, 0.2808)
SHUFFLE = (0.5007, 0.1316)
UNIFORM = (0.4956, 0.1499)
CHANCE = 0.5


def main():
    a = json.loads((RES / "exp002_arm_a.json").read_text())
    conds = a["conditions"]

    sigmas = sorted({c for c in conds if "sigma" in c and c.startswith("jitter")})
    by_s = {}
    for c in sigmas:
        head, sig = c.split("sigma", 1)
        seed, s = int(head.rstrip("_").rsplit("_", 1)[1][1:]), int(sig)
        ev = conds[c]["evals"]["label_1000"]
        by_s.setdefault(s, []).append(ev["lr"]["roc_auc"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    # Panel 1: ROC vs onset jitter sigma (LR)
    ax = axes[0]
    xs = sorted(by_s)
    ax.plot(xs, [CLEAN[0]] * len(xs), "--", color="#888", lw=1.2,
            label=f"clean ({CLEAN[0]:.4f})")
    for s, rocs in by_s.items():
        for r in rocs:
            ax.scatter([s], [r], s=26, color="C0", zorder=3)
    ax.plot(xs, [sum(by_s[x]) / len(by_s[x]) for x in xs], "-o", color="C0",
            lw=1.6, label="LR mean over label seeds")
    ax.axhline(SHUFFLE[0], color="#d62728", ls=":", lw=1.4,
               label=f"label-shuffle ({SHUFFLE[0]:.4f})")
    ax.axhline(UNIFORM[0], color="#ff7f0e", ls=":", lw=1.4,
               label=f"uniform ({UNIFORM[0]:.4f})")
    ax.axhline(CHANCE, color="#222", ls="--", lw=1.0, alpha=.7,
               label="chance (0.50)")
    ax.set_xlabel("label onset jitter sigma (ms)")
    ax.set_ylabel("ROC-AUC @1000ms (LOCO, LR)")
    ax.set_title("Arm A: annotation onset jitter — robustness")
    ax.legend(fontsize=7, loc="lower left")
    ax.set_ylim(0.45, 0.78)

    # Panel 2: PR vs sigma + controls
    ax = axes[1]
    prs = {s: [conds[c]["evals"]["label_1000"]["lr"]["pr_auc"] for c in conds
               if c == f"jitter_s101_sigma{s}"] for s in xs}
    ax.plot(xs, [CLEAN[1]] * len(xs), "--", color="#888", lw=1.2,
            label=f"clean ({CLEAN[1]:.4f})")
    ax.plot(xs, [prs[s][0] if prs[s] else float("nan") for s in xs], "-o", color="C0",
            lw=1.6, label="LR (seed 101)")
    ax.axhline(SHUFFLE[1], color="#d62728", ls=":", lw=1.4,
               label=f"label-shuffle ({SHUFFLE[1]:.4f})")
    ax.axhline(UNIFORM[1], color="#ff7f0e", ls=":", lw=1.4,
               label=f"uniform ({UNIFORM[1]:.4f})")
    ax.set_xlabel("label onset jitter sigma (ms)")
    ax.set_ylabel("PR-AUC @1000ms (LOCO, LR)")
    ax.set_title("Arm A: PR curve (seed 101)")
    ax.legend(fontsize=7, loc="lower left")
    ax.set_ylim(0.05, 0.35)
    plt.suptitle("EXP-002 — annotation/segmentation robustness (LR, 16-fold LOCO, @1000ms)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG / "exp002_arm_a.png", dpi=200)
    fig.savefig(FIG / "exp002_arm_a.pdf")
    plt.close(fig)

    # Panel 3: Arm B — ROC vs margin delta, consistent vs clean-label drift
    b = json.loads((RES / "exp002_arm_b.json").read_text())
    d = b["datasets"]
    fig, ax = plt.subplots(figsize=(7, 5))
    deltas, cons, drift = [], [], []
    for tag in sorted(d):
        dv = [float(c) for c in tag[1:].split("_")]  # e.g. "m-3" -> [-3.0], "m+3" -> [3.0]
        delta = dv[0]
        cell = d[tag]["cells"]
        if "consistent" in cell:
            deltas.append(delta)
            cons.append(cell["consistent"]["lr"]["roc_auc"])
        if "drift_labels_from_clean" in cell:
            drift.append((delta, cell["drift_labels_from_clean"]["lr"]["roc_auc"]))
    ax.axhline(CLEAN[0], color="#888", ls="--", lw=1.2,
               label=f"anchor (delta=0, =clean {CLEAN[0]:.4f})")
    if deltas:
        ax.plot(deltas, cons, "-o", color="C0", lw=1.6, label="consistent VAD re-segment")
    if drift:
        ds, rs = zip(*drift)
        ax.plot(ds, rs, "-o", color="C2", lw=1.6,
                label="features-at-margin, clean labels")
    ax.set_xlabel("VAD threshold margin delta (dB)")
    ax.set_ylabel("ROC-AUC @1000ms (LOCO, LR)")
    ax.set_title("Arm B: VAD segmentation margin robustness")
    ax.legend(fontsize=8)
    ax.set_ylim(0.68, 0.74)
    fig.tight_layout()
    fig.savefig(FIG / "exp002_arm_b.png", dpi=200)
    fig.savefig(FIG / "exp002_arm_b.pdf")
    plt.close(fig)
    print("figures written:", sorted(p.name for p in FIG.iterdir()))


if __name__ == "__main__":
    main()