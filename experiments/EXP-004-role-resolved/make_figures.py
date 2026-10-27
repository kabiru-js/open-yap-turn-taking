"""Generate figures for EXP-004: Role-resolved feature representation."""

from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

EXP_DIR = Path(__file__).resolve().parent
RESULTS_FILE = EXP_DIR / "results" / "exp004_role_resolved.json"
FIG_DIR = EXP_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_primary_benchmark(data: dict):
    paired = data["primary_benchmark"]["paired_comparison"]
    models = ["LR", "GBDT", "RF"]
    channel_rocs = [paired[m.lower()]["channel_roc_mean"] for m in models]
    channel_sds = [paired[m.lower()]["channel_roc_sd"] for m in models]
    role_rocs = [paired[m.lower()]["role_roc_mean"] for m in models]
    role_sds = [paired[m.lower()]["role_roc_sd"] for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=200)
    rects1 = ax.bar(x - width/2, channel_rocs, width, yerr=channel_sds, capsize=4,
                    label="Channel-Arbitrary (a/b)", color="#4A90E2", alpha=0.85)
    rects2 = ax.bar(x + width/2, role_rocs, width, yerr=role_sds, capsize=4,
                    label="Role-Resolved (floor/listener)", color="#50E3C2", alpha=0.85)

    ax.axhline(0.50, color="gray", linestyle="--", linewidth=1, label="Chance floor (0.50)")
    ax.set_ylabel("ROC-AUC (conv-LOCO mean ± sd)")
    ax.set_title("EXP-004: Channel-Arbitrary vs. Role-Resolved Turn-Onset Prediction (h=1000ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight="bold")
    ax.set_ylim(0.45, 0.85)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(frameon=True, facecolor="white", edgecolor="none")

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out_path = FIG_DIR / "role_resolved_benchmark.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


def plot_ablations(data: dict):
    abl = data["ablations"]
    categories = [
        ("temporal_only", "Temporal Only (3)"),
        ("floor_acoustics_only", "Floor Acoustics Only (35)"),
        ("listener_acoustics_only", "Listener Acoustics Only (35)"),
        ("both_role_acoustics", "Both Role Acoustics (70)"),
        ("both_channel_acoustics", "Both Channel Acoustics (70)"),
        ("role_f0_only", "Role F0 Only (8)"),
        ("full_role", "Full Role-Resolved (73)"),
    ]

    labels = [c[1] for c in categories]
    lr_vals = [abl[c[0]]["lr_roc"] for c in categories]
    gbdt_vals = [abl[c[0]]["gbdt_roc"] for c in categories]

    y = np.arange(len(labels))
    height = 0.35

    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    ax.barh(y - height/2, lr_vals, height, label="Logistic Regression", color="#5B9BD5")
    ax.barh(y + height/2, gbdt_vals, height, label="GBDT", color="#ED7D31")

    ax.axvline(0.50, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("ROC-AUC (conv-LOCO mean)")
    ax.set_title("EXP-004: Feature Decomposition & Incremental Predictive Value")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0.45, 0.80)
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    ax.legend(frameon=True, loc="lower right")

    plt.tight_layout()
    out_path = FIG_DIR / "role_resolved_ablation.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


def plot_horizon_sweep(data: dict):
    h_data = data["horizon_sweep"]
    horizons = sorted([int(k) for k in h_data.keys()])

    ch_lr = [h_data[str(h)]["channel_lr_roc"] for h in horizons]
    ro_lr = [h_data[str(h)]["role_lr_roc"] for h in horizons]
    ch_gbdt = [h_data[str(h)]["channel_gbdt_roc"] for h in horizons]
    ro_gbdt = [h_data[str(h)]["role_gbdt_roc"] for h in horizons]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=200)
    ax.plot(horizons, ch_lr, "o--", label="LR (Channel-Arbitrary)", color="#5B9BD5", alpha=0.8)
    ax.plot(horizons, ro_lr, "o-", label="LR (Role-Resolved)", color="#2E75B6", linewidth=2)
    ax.plot(horizons, ch_gbdt, "s--", label="GBDT (Channel-Arbitrary)", color="#ED7D31", alpha=0.8)
    ax.plot(horizons, ro_gbdt, "s-", label="GBDT (Role-Resolved)", color="#C55A11", linewidth=2)

    ax.set_xlabel("Forecast Horizon Δt (ms)")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("EXP-004: Lead-Time Trajectory across Forecast Horizons")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True)

    plt.tight_layout()
    out_path = FIG_DIR / "horizon_sweep_role.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


def main():
    if not RESULTS_FILE.exists():
        print(f"Results file {RESULTS_FILE} not found!")
        return
    with open(RESULTS_FILE) as f:
        data = json.load(f)
    plot_primary_benchmark(data)
    plot_ablations(data)
    plot_horizon_sweep(data)
    print("All figures successfully generated.")


if __name__ == "__main__":
    main()
