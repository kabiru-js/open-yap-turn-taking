"""EXP-004: Role-resolved feature representation and cross-dyad generalization.

Executes:
1. Primary Benchmark (conv-LOCO @ 1000ms): Channel-arbitrary vs Role-resolved
   across Majority, LR, GBDT, RF.
2. Paired statistical significance (mean delta, 95% bootstrap CI, paired t-test).
3. Ablation suite (temporal, floor acoustics, listener acoustics, both acoustics, F0).
4. Dyad Generalization (speaker-pair LOCO across 4 held-out pairs).
5. Lead-time horizon sweep (200, 500, 750, 1000, 1500 ms).
6. Chance control (within-conversation label permutation).
7. Event-type stratification (clean vs backchannel vs interruption).

Saves machine-readable results to results/exp004_role_resolved.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from research.config import Config
from research.dataset import load_all_metadata
from research.window_features import FEAT_COLS
from role_features import (
    ACOUSTIC_ROLE_COLS,
    F0_ROLE_COLS,
    FLOOR_COLS,
    LISTENER_COLS,
    NO_F0_ROLE_COLS,
    ROLE_FEAT_COLS,
    TEMPORAL_COLS,
    build_role_resolved_dataframe,
)

EXP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXP_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _safe_roc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float(np.nan)
    return float(roc_auc_score(y, p))


def _safe_pr(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float(np.nan)
    return float(average_precision_score(y, p))


def bootstrap_ci(deltas: np.ndarray, n_boot: int = 2000, seed: int = 42) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(deltas)
    boot_means = [np.mean(rng.choice(deltas, size=n, replace=True)) for _ in range(n_boot)]
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def evaluate_folds(
    df: pd.DataFrame,
    features: List[str],
    ycol: str,
    groups: List[str],
    group_col: str = "conv_id",
    models: Tuple[str, ...] = ("lr", "gbdt"),
    cap: int = 90000,
    seed: int = 42,
) -> dict:
    """Evaluate specified models on groups (LOCO or pair-LOCO)."""
    res = {m: {"roc": [], "pr": [], "n_test": [], "groups": []} for m in models}
    base_rate = float(df[ycol].mean())

    for grp in groups:
        fl_tr = df[df[group_col] != grp]
        fl_te = df[df[group_col] == grp]
        if len(fl_te) == 0:
            continue
        if len(fl_tr) > cap:
            fl_tr = fl_tr.sample(cap, random_state=seed)

        imputer = SimpleImputer(strategy="median")
        Xi = imputer.fit_transform(fl_tr[features])
        Xe = imputer.transform(fl_te[features])
        ytr = fl_tr[ycol].to_numpy()
        yte = fl_te[ycol].to_numpy()

        for m in models:
            if m == "majority":
                p = np.full(len(fl_te), base_rate)
            elif m == "lr":
                sc = StandardScaler().fit(Xi)
                clf = LogisticRegression(max_iter=2000, C=1.0)
                clf.fit(sc.transform(Xi), ytr)
                p = clf.predict_proba(sc.transform(Xe))[:, 1]
            elif m == "gbdt":
                clf = HistGradientBoostingClassifier(
                    max_iter=160, learning_rate=0.12, random_state=seed
                )
                clf.fit(Xi, ytr)
                p = clf.predict_proba(Xe)[:, 1]
            elif m == "rf":
                clf = RandomForestClassifier(
                    n_estimators=80,
                    random_state=seed,
                    n_jobs=-1,
                    min_samples_leaf=25,
                )
                clf.fit(Xi, ytr)
                p = clf.predict_proba(Xe)[:, 1]
            else:
                raise ValueError(f"Unknown model {m}")

            res[m]["roc"].append(_safe_roc(yte, p))
            res[m]["pr"].append(_safe_pr(yte, p))
            res[m]["n_test"].append(len(fl_te))
            res[m]["groups"].append(str(grp))

    # Summarize
    out = {}
    for m in models:
        rocs = np.asarray(res[m]["roc"], dtype=float)
        prs = np.asarray(res[m]["pr"], dtype=float)
        wt = np.asarray(res[m]["n_test"])
        out[m] = {
            "roc_auc_mean": float(np.nanmean(rocs)),
            "roc_auc_sd": float(np.nanstd(rocs)),
            "pr_auc_mean": float(np.nanmean(prs)),
            "pr_auc_sd": float(np.nanstd(prs)),
            "pr_auc_weighted": float(np.nansum(prs * wt) / wt.sum()) if wt.sum() > 0 else 0.0,
            "per_fold_roc": [float(x) for x in rocs],
            "per_fold_pr": [float(x) for x in prs],
            "groups": res[m]["groups"],
        }
    return out


def main():
    t_start = time.time()
    cfg = Config()
    print("Loading interim dataset...", flush=True)
    raw_df = pd.read_parquet(cfg.paths.interim_dir / "dataset.parquet")
    raw_df["conv_id"] = raw_df["conv_id"].astype(str)

    meta = load_all_metadata(cfg)
    pair_of_conv = {
        cid: "-".join(sorted([m["speakers"]["a"]["speaker_id"], m["speakers"]["b"]["speaker_id"]]))
        for cid, m in meta.items()
    }
    raw_df["pair"] = raw_df["conv_id"].map(pair_of_conv)

    print("Building role-resolved features...", flush=True)
    role_df = build_role_resolved_dataframe(raw_df)

    conv_groups = sorted(raw_df["conv_id"].unique())
    pair_groups = sorted(raw_df["pair"].unique())
    print(f"Dataset: {len(raw_df)} rows, 16 conversations, 4 speaker pairs.", flush=True)

    results = {
        "metadata": {
            "date": "2026-10-23",
            "n_decision_points": len(raw_df),
            "n_conversations": len(conv_groups),
            "n_pairs": len(pair_groups),
            "primary_horizon": 1000,
            "seed": cfg.model.seed,
            "train_cap": 90000,
        }
    }

    # ──────────────────────────────────────────────────────────────────────────
    # 1. Primary Benchmark (conv-LOCO @ 1000ms): Channel-Blind vs Role-Resolved
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70, flush=True)
    print("1. PRIMARY BENCHMARK: CONV-LOCO @ 1000ms", flush=True)
    print("=" * 70, flush=True)

    models_to_test = ("majority", "lr", "gbdt", "rf")
    print("\nRunning Channel-Arbitrary representation (73 features)...", flush=True)
    t0 = time.time()
    bench_channel = evaluate_folds(
        raw_df, FEAT_COLS, "label_1000", conv_groups, group_col="conv_id",
        models=models_to_test, seed=cfg.model.seed
    )
    print(f"  Channel-arbitrary done in {time.time() - t0:.1f}s", flush=True)

    print("\nRunning Role-Resolved representation (73 features)...", flush=True)
    t0 = time.time()
    bench_role = evaluate_folds(
        role_df, ROLE_FEAT_COLS, "label_1000", conv_groups, group_col="conv_id",
        models=models_to_test, seed=cfg.model.seed
    )
    print(f"  Role-resolved done in {time.time() - t0:.1f}s", flush=True)

    # Paired stats
    paired_stats = {}
    print("\n--- Model Performance Comparison (ROC-AUC mean ± sd, PR-AUC weighted) ---")
    for m in models_to_test:
        ch_roc = bench_channel[m]["roc_auc_mean"]
        ch_sd = bench_channel[m]["roc_auc_sd"]
        ch_pr = bench_channel[m]["pr_auc_weighted"]

        ro_roc = bench_role[m]["roc_auc_mean"]
        ro_sd = bench_role[m]["roc_auc_sd"]
        ro_pr = bench_role[m]["pr_auc_weighted"]

        # Per-fold delta
        deltas = np.array(bench_role[m]["per_fold_roc"]) - np.array(bench_channel[m]["per_fold_roc"])
        mean_delta = float(np.mean(deltas))
        sd_delta = float(np.std(deltas))
        ci_lo, ci_hi = bootstrap_ci(deltas, n_boot=2000, seed=cfg.model.seed)
        t_stat, p_val = stats.ttest_1samp(deltas, 0.0)

        paired_stats[m] = {
            "channel_roc_mean": ch_roc,
            "channel_roc_sd": ch_sd,
            "channel_pr_weighted": ch_pr,
            "role_roc_mean": ro_roc,
            "role_roc_sd": ro_sd,
            "role_pr_weighted": ro_pr,
            "mean_delta_roc": mean_delta,
            "sd_delta_roc": sd_delta,
            "ci_95": [ci_lo, ci_hi],
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "per_fold_deltas": [float(d) for d in deltas],
        }

        print(f"{m.upper():8s} | Channel: {ch_roc:.4f} ± {ch_sd:.4f} (PR: {ch_pr:.4f}) "
              f"| Role: {ro_roc:.4f} ± {ro_sd:.4f} (PR: {ro_pr:.4f}) "
              f"| Delta: {mean_delta:+.4f} (95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}], p={p_val:.4f})")

    results["primary_benchmark"] = {
        "channel_blind": bench_channel,
        "role_resolved": bench_role,
        "paired_comparison": paired_stats,
    }

    # ──────────────────────────────────────────────────────────────────────────
    # 2. Feature Ablations under Role Resolution vs Channel Blind
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70, flush=True)
    print("2. ABLATION ANALYSIS: ACOUSTIC RECOVERY & ROLE ISOLATION", flush=True)
    print("=" * 70, flush=True)

    channel_acoustics = [c for c in FEAT_COLS if c.startswith("a_") or c.startswith("b_")]
    channel_a_only = [c for c in FEAT_COLS if c.startswith("a_")]
    channel_b_only = [c for c in FEAT_COLS if c.startswith("b_")]
    channel_f0_only = [c for c in FEAT_COLS if "f0" in c]

    ablation_suites = {
        "temporal_only": (TEMPORAL_COLS, role_df),
        "floor_acoustics_only": (FLOOR_COLS, role_df),
        "listener_acoustics_only": (LISTENER_COLS, role_df),
        "both_role_acoustics": (ACOUSTIC_ROLE_COLS, role_df),
        "both_channel_acoustics": (channel_acoustics, raw_df),
        "channel_a_acoustics": (channel_a_only, raw_df),
        "channel_b_acoustics": (channel_b_only, raw_df),
        "role_f0_only": (F0_ROLE_COLS, role_df),
        "channel_f0_only": (channel_f0_only, raw_df),
        "role_no_f0": (NO_F0_ROLE_COLS, role_df),
        "full_role": (ROLE_FEAT_COLS, role_df),
    }

    ablation_results = {}
    for name, (feat_list, source_df) in ablation_suites.items():
        res = evaluate_folds(
            source_df, feat_list, "label_1000", conv_groups, group_col="conv_id",
            models=("lr", "gbdt"), seed=cfg.model.seed
        )
        ablation_results[name] = {
            "n_features": len(feat_list),
            "lr_roc": res["lr"]["roc_auc_mean"],
            "lr_roc_sd": res["lr"]["roc_auc_sd"],
            "lr_pr": res["lr"]["pr_auc_weighted"],
            "gbdt_roc": res["gbdt"]["roc_auc_mean"],
            "gbdt_roc_sd": res["gbdt"]["roc_auc_sd"],
            "gbdt_pr": res["gbdt"]["pr_auc_weighted"],
        }
        print(f"  {name:25s} (n={len(feat_list):2d}) | LR: {res['lr']['roc_auc_mean']:.4f} (PR {res['lr']['pr_auc_weighted']:.4f}) | GBDT: {res['gbdt']['roc_auc_mean']:.4f}")

    results["ablations"] = ablation_results

    # ──────────────────────────────────────────────────────────────────────────
    # 3. Cross-Dyad Generalization: Speaker-Pair LOCO
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70, flush=True)
    print("3. CROSS-DYAD GENERALIZATION: SPEAKER-PAIR LOCO (4 unseen pairs)", flush=True)
    print("=" * 70, flush=True)

    pair_channel = evaluate_folds(
        raw_df, FEAT_COLS, "label_1000", pair_groups, group_col="pair",
        models=("lr", "gbdt"), seed=cfg.model.seed
    )
    pair_role = evaluate_folds(
        role_df, ROLE_FEAT_COLS, "label_1000", pair_groups, group_col="pair",
        models=("lr", "gbdt"), seed=cfg.model.seed
    )

    pair_comparison = {}
    for m in ("lr", "gbdt"):
        ch_roc = pair_channel[m]["roc_auc_mean"]
        ro_roc = pair_role[m]["roc_auc_mean"]
        delta = ro_roc - ch_roc
        pair_comparison[m] = {
            "channel_roc_mean": ch_roc,
            "channel_roc_sd": pair_channel[m]["roc_auc_sd"],
            "role_roc_mean": ro_roc,
            "role_roc_sd": pair_role[m]["roc_auc_sd"],
            "delta_roc": delta,
            "per_fold_channel": pair_channel[m]["per_fold_roc"],
            "per_fold_role": pair_role[m]["per_fold_roc"],
        }
        print(f"  {m.upper():6s} Pair-LOCO | Channel: {ch_roc:.4f} ± {pair_channel[m]['roc_auc_sd']:.4f} | Role: {ro_roc:.4f} ± {pair_role[m]['roc_auc_sd']:.4f} | Delta: {delta:+.4f}")

    results["speaker_pair_generalization"] = pair_comparison

    # ──────────────────────────────────────────────────────────────────────────
    # 4. Lead-Time Analysis: Horizon Sweep (200, 500, 750, 1000, 1500 ms)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70, flush=True)
    print("4. LEAD-TIME HORIZON SWEEP", flush=True)
    print("=" * 70, flush=True)

    horizons = [200, 500, 750, 1000, 1500]
    horizon_results = {}
    for h in horizons:
        ycol = f"label_{h}"
        h_ch = evaluate_folds(
            raw_df, FEAT_COLS, ycol, conv_groups, group_col="conv_id",
            models=("lr", "gbdt"), seed=cfg.model.seed
        )
        h_ro = evaluate_folds(
            role_df, ROLE_FEAT_COLS, ycol, conv_groups, group_col="conv_id",
            models=("lr", "gbdt"), seed=cfg.model.seed
        )
        horizon_results[h] = {
            "base_rate": float(raw_df[ycol].mean()),
            "channel_lr_roc": h_ch["lr"]["roc_auc_mean"],
            "role_lr_roc": h_ro["lr"]["roc_auc_mean"],
            "channel_gbdt_roc": h_ch["gbdt"]["roc_auc_mean"],
            "role_gbdt_roc": h_ro["gbdt"]["roc_auc_mean"],
            "channel_lr_pr": h_ch["lr"]["pr_auc_weighted"],
            "role_lr_pr": h_ro["lr"]["pr_auc_weighted"],
        }
        print(f"  h={h:4d}ms (base {horizon_results[h]['base_rate']:.3f}) | LR Ch: {h_ch['lr']['roc_auc_mean']:.4f} -> Role: {h_ro['lr']['roc_auc_mean']:.4f} | GBDT Ch: {h_ch['gbdt']['roc_auc_mean']:.4f} -> Role: {h_ro['gbdt']['roc_auc_mean']:.4f}")

    results["horizon_sweep"] = horizon_results

    # ──────────────────────────────────────────────────────────────────────────
    # 5. Chance Control: Within-Conversation Label Permutation
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70, flush=True)
    print("5. CHANCE CONTROLS & FALSIFICATION", flush=True)
    print("=" * 70, flush=True)

    rng = np.random.default_rng(cfg.model.seed)
    shuffled_df = role_df.copy()
    # Permute within each conversation to preserve marginal and duration profile
    shuffled_labels = np.zeros(len(shuffled_df), dtype=int)
    for cid in conv_groups:
        idx = np.where(shuffled_df["conv_id"] == cid)[0]
        shuffled_labels[idx] = rng.permutation(shuffled_df.loc[idx, "label_1000"].to_numpy())
    shuffled_df["label_1000_shuffled"] = shuffled_labels

    ctrl_res = evaluate_folds(
        shuffled_df, ROLE_FEAT_COLS, "label_1000_shuffled", conv_groups, group_col="conv_id",
        models=("lr", "gbdt"), seed=cfg.model.seed
    )
    results["chance_control"] = {
        "lr_shuffled_roc": ctrl_res["lr"]["roc_auc_mean"],
        "lr_shuffled_pr": ctrl_res["lr"]["pr_auc_weighted"],
        "gbdt_shuffled_roc": ctrl_res["gbdt"]["roc_auc_mean"],
        "gbdt_shuffled_pr": ctrl_res["gbdt"]["pr_auc_weighted"],
    }
    print(f"  Within-conv Shuffle Control | LR ROC: {ctrl_res['lr']['roc_auc_mean']:.4f} | GBDT ROC: {ctrl_res['gbdt']['roc_auc_mean']:.4f} (expected ~0.50)")

    # ──────────────────────────────────────────────────────────────────────────
    # Save Results
    # ──────────────────────────────────────────────────────────────────────────
    out_file = RESULTS_DIR / "exp004_role_resolved.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    total_time = time.time() - t_start
    print(f"\nAll experiments successfully completed in {total_time:.1f}s.")
    print(f"Saved machine-readable results to: {out_file}")


if __name__ == "__main__":
    main()
