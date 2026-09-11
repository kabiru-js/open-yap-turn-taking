"""Collect held-out predictions and produce all figures.

Run:  python scripts/07_figures.py
"""
import sys, time, json
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

from research.config import Config
from research.events import build_segments, classify_transitions, merge_asr_event_type
from research.dataset import load_all_metadata
from research.window_features import FEAT_COLS

plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight"})

PAL = {"lr": "#1f77b4", "gbdt": "#d62728", "majority": "#7f7f7f", "base": "#9e9e9e"}


def _held_out(df, grp, seed):
    fl_tr = df[df["conv_id"].astype(str) != grp]
    fl_te = df[df["conv_id"].astype(str) == grp]
    if len(fl_tr) > 90000:
        fl_tr = fl_tr.sample(90000, random_state=seed)
    return fl_tr, fl_te


def collect_lr(df, horizons, groups, seed):
    preds = {h: {"y": [], "p": []} for h in horizons}
    for h in horizons:
        ycol = f"label_{h}"
        for grp in groups:
            fl_tr, fl_te = _held_out(df, grp, seed)
            imp = SimpleImputer(strategy="median")
            Xi = imp.fit_transform(fl_tr[FEAT_COLS])
            Xe = imp.transform(fl_te[FEAT_COLS])
            sc = StandardScaler().fit(Xi)
            Xi, Xe = sc.transform(Xi), sc.transform(Xe)
            lr = LogisticRegression(max_iter=2000, C=1.0).fit(Xi, fl_tr[ycol])
            p = lr.predict_proba(Xe)[:, 1]
            preds[h]["y"].append(fl_te[ycol].to_numpy())
            preds[h]["p"].append(p)
        print(f"LR {h}ms collected", flush=True)
    return preds


def collect_tree(df, ycol, groups, seed):
    preds = {"y": [], "p": []}
    for grp in groups:
        fl_tr, fl_te = _held_out(df, grp, seed)
        imp = SimpleImputer(strategy="median")
        Xi = imp.fit_transform(fl_tr[FEAT_COLS])
        Xe = imp.transform(fl_te[FEAT_COLS])
        model = HistGradientBoostingClassifier(max_iter=160, learning_rate=0.12,
                                               random_state=seed)
        p = model.fit(Xi, fl_tr[ycol]).predict_proba(Xe)[:, 1]
        preds["y"].append(fl_te[ycol].to_numpy())
        preds["p"].append(p)
        print(f"  gbdt fold {grp} done", flush=True)
    return preds


def main():
    cfg = Config()
    df = pd.read_parquet(cfg.paths.interim_dir / "dataset.parquet")
    groups = sorted(df["conv_id"].astype(str).unique())
    horizons = cfg.prediction.horizons_ms
    fig_dir = cfg.paths.figures_dir
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. LR across horizons ────────────────────────────────────────────────
    t0 = time.time()
    lr_preds = collect_lr(df, horizons, groups, cfg.model.seed)

    # ── 2. GBDT at 1000ms ───────────────────────────────────────────────────
    gb_preds = collect_tree(df, "label_1000", groups, cfg.model.seed)

    # ── fig: AUC vs horizon (LR) ────────────────────────────────────────────
    rocs, prs, base_rates = [], [], []
    for h in horizons:
        y = np.concatenate(lr_preds[h]["y"])
        p = np.concatenate(lr_preds[h]["p"])
        rocs.append(roc_auc_score(y, p))
        prs.append(average_precision_score(y, p))
        base_rates.append(y.mean())
    gb_roc_1000 = roc_auc_score(np.concatenate(gb_preds["y"]), np.concatenate(gb_preds["p"]))
    gb_pr_1000 = average_precision_score(np.concatenate(gb_preds["y"]), np.concatenate(gb_preds["p"]))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(horizons, rocs, "o-", color=PAL["lr"], label="LR ROC-AUC")
    ax.plot(horizons, prs, "s--", color=PAL["lr"], alpha=0.6, label="LR PR-AUC")
    ax.plot([1000], [gb_roc_1000], "o", color=PAL["gbdt"], ms=9, label=f"GBDT ROC-AUC@1000 ({gb_roc_1000:.3f})")
    ax.plot([1000], [gb_pr_1000], "s", color=PAL["gbdt"], ms=9, label=f"GBDT PR-AUC@1000 ({gb_pr_1000:.3f})")
    ax.plot(horizons, base_rates, ":", color=PAL["base"], label="base rate")
    ax.set_xlabel("horizon Δt (ms)")
    ax.set_ylabel("AUC")
    ax.set_title("Turn-onset prediction vs horizon (conversation-LOCO)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(fig_dir / "fig_auc_vs_horizon.png")

    # ── fig: ROC & PR at 1000ms ──────────────────────────────────────────────
    y1000 = np.concatenate(gb_preds["y"])
    p_lr = np.concatenate(lr_preds[1000]["p"])
    p_gb = np.concatenate(gb_preds["p"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for p, name, c in [(p_lr, "LR", PAL["lr"]), (p_gb, "GBDT", PAL["gbdt"])]:
        fpr, tpr, _ = roc_curve(y1000, p)
        axes[0].plot(fpr, tpr, color=c, label=f"{name} (AUC={roc_auc_score(y1000,p):.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color=PAL["majority"], label="chance")
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[0].set_title("ROC @ 1000ms"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    for p, name, c in [(p_lr, "LR", PAL["lr"]), (p_gb, "GBDT", PAL["gbdt"])]:
        prec, rec, _ = precision_recall_curve(y1000, p)
        axes[1].plot(rec, prec, color=c, label=f"{name} (PR={average_precision_score(y1000,p):.3f})")
    axes[1].axhline(y1000.mean(), ls=":", color=PAL["base"], label=f"base rate={y1000.mean():.3f}")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("PR @ 1000ms"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    fig.savefig(fig_dir / "fig_roc_pr_1000.png")

    # ── fig: calibration at 1000ms (GBDT, held-out) ─────────────────────────
    from sklearn.calibration import calibration_curve
    true_prob, pred_prob = calibration_curve(y1000, p_gb, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color=PAL["majority"], label="perfect")
    ax.plot(pred_prob, true_prob, "o-", color=PAL["gbdt"], label="GBDT @1000ms")
    ax.set_xlabel("predicted probability"); ax.set_ylabel("observed frequency")
    ax.set_title("Reliability (conversation-LOCO)"); ax.legend(); ax.grid(alpha=0.3)
    fig.savefig(fig_dir / "fig_calibration_1000.png")

    # ── fig: per-conversation GBDT ROC @ 1000 ───────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    for (y, p) in zip(gb_preds["y"], gb_preds["p"]):
        try:
            r = roc_auc_score(y, p)
        except Exception:
            r = np.nan
        ax.scatter(y.mean(), r, s=34, color=PAL["gbdt"], alpha=0.7)
    ax.axhline(gb_roc_1000, color=PAL["gbdt"], ls="--", lw=1, label=f"pooled {gb_roc_1000:.3f}")
    ax.set_xlabel("per-conv base rate"); ax.set_ylabel("per-conv ROC-AUC")
    ax.set_title("GBDT @1000ms per-conversation"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.savefig(fig_dir / "fig_per_conv.png")

    # ── fig: taxonomy + gap distribution ────────────────────────────────────
    ev_all = pd.read_csv(cfg.paths.interim_dir / "events" / "all_events.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    vc = ev_all["type"].value_counts()
    axes[0].bar(vc.index, vc.values, color="#4c72b0")
    axes[0].set_title("Transition taxonomy"); axes[0].set_yscale("log")
    axes[0].tick_params(axis="x", rotation=30)
    clean_gaps = ev_all.loc[ev_all["type"] == "clean", "gap_ms"].dropna()
    axes[1].hist(clean_gaps.clip(-50, 1500), bins=60, color="#c44e52", alpha=0.8)
    axes[1].set_title("Clean-transition gaps (ms)"); axes[1].set_xlabel("gap to next onset")
    fig.savefig(fig_dir / "fig_taxonomy.png")

    # ── fig: top-15 feature importance ──────────────────────────────────────
    with open(cfg.paths.results_dir / "analysis_1000.json") as f:
        an = json.load(f)
    top = list(an["top20_features"].items())[:15]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([t[0] for t in top][::-1], [t[1] for t in top][::-1], color="#55a868")
    ax.set_xlabel("RandomForest importance (full data, interpretability only)")
    ax.set_title("Feature importance @1000ms")
    fig.savefig(fig_dir / "fig_importance.png")

    # ── persist a summary ───────────────────────────────────────────────────
    summary = {
        "lr_roc_vs_horizon": {str(h): rocs[i] for i, h in enumerate(horizons)},
        "lr_pr_vs_horizon": {str(h): prs[i] for i, h in enumerate(horizons)},
        "base_rates": {str(h): base_rates[i] for i, h in enumerate(horizons)},
        "gbdt_roc_1000": gb_roc_1000,
        "gbdt_pr_1000": gb_pr_1000,
        "n_decision_points": len(df),
        "n_events": int(len(ev_all)),
        "event_type_counts": vc.to_dict(),
    }
    with open(cfg.paths.results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    np.savez(cfg.paths.results_dir / "predictions_1000.npz",
             y=y1000, p_lr=p_lr, p_gb=p_gb)
    print(f"\ndone in {time.time()-t0:.0f}s; figures in {fig_dir}")


if __name__ == "__main__":
    main()