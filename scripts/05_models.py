"""LOCO model evaluation (baselines, LR, HistGB, RF) on decision points.

Run:  python scripts/05_models.py --horizons 1000 --loco
      python scripts/05_models.py --horizons 1000 --pairs
"""
import sys, time, json, argparse
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

from research.config import Config
from research.dataset import load_all_metadata
from research.window_features import FEAT_COLS


def safe_roc(y, p):
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, p)


def safe_pr(y, p):
    if len(np.unique(y)) < 2:
        return np.nan
    return average_precision_score(y, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", default="1000", help="comma list, e.g. 200,500,1000,1500")
    ap.add_argument("--loco", action="store_true", help="leave-one-conversation-out")
    ap.add_argument("--pairs", action="store_true", help="leave-one-speaker-pair-out")
    ap.add_argument("--models", default="lr,gbdt",
                    help="comma list of majority,lr,rf,gbdt")
    args = ap.parse_args()
    assert args.loco or args.pairs, "choose --loco or --pairs"
    cfg = Config()
    horizons = [int(x) for x in args.horizons.split(",")]
    want = set(x.strip() for x in args.models.split(","))

    df = pd.read_parquet(cfg.paths.interim_dir / "dataset.parquet")
    print(f"dataset {df.shape}", flush=True)

    meta = load_all_metadata(cfg)
    pair_of_conv = {
        cid: tuple(sorted([m["speakers"]["a"]["speaker_id"], m["speakers"]["b"]["speaker_id"]]))
        for cid, m in meta.items()
    }
    df["pair"] = df["conv_id"].astype(str).map(pair_of_conv)

    if args.pairs:
        groups = sorted(df["pair"].unique())
        fold_name = "speaker-pair"
    else:
        groups = sorted(df["conv_id"].astype(str).unique())
        fold_name = "conversation"

    models = {
        "majority": None,
        "lr": (LogisticRegression(max_iter=2000, C=1.0), True, True),
        "rf": (RandomForestClassifier(n_estimators=80, random_state=cfg.model.seed,
                                      n_jobs=-1, min_samples_leaf=25), True, False),
        "gbdt": (HistGradientBoostingClassifier(max_iter=160, learning_rate=0.12,
                                                random_state=cfg.model.seed), True, False),
    }
    models = {k: v for k, v in models.items() if k in want}
    assert models, "no models selected"

    all_metrics = {}
    for h in horizons:
        ycol = f"label_{h}"
        base = float(df[ycol].mean())
        res = {name: {"roc": [], "pr": [], "n_test": [], "groups": []} for name in models}

        t0 = time.time()
        for grp in groups:
            if args.pairs:
                fl_tr = df[df["pair"] != grp]
                fl_te = df[df["pair"] == grp]
            else:
                fl_tr = df[df["conv_id"].astype(str) != grp]
                fl_te = df[df["conv_id"].astype(str) == grp]
            if len(fl_te) == 0:
                continue
            # cap training rows to reduce cost (decision points are
            # heavily autocorrelated within a conversation window)
            if len(fl_tr) > 90000:
                fl_tr = fl_tr.sample(90000, random_state=cfg.model.seed)
            importer = SimpleImputer(strategy="median")
            Xi = importer.fit_transform(fl_tr[FEAT_COLS])
            Xe = importer.transform(fl_te[FEAT_COLS])
            for name, val in models.items():
                if name == "majority":
                    p = np.full(len(fl_te), base)
                else:
                    model, imp, scl = val
                    x_in = Xi
                    x_eval = Xe
                    if scl:
                        sc = StandardScaler().fit(Xi)
                        x_in, x_eval = sc.transform(Xi), sc.transform(Xe)
                    model.fit(x_in, fl_tr[ycol])
                    p = model.predict_proba(x_eval)[:, 1]
                res[name]["roc"].append(safe_roc(fl_te[ycol], p))
                res[name]["pr"].append(safe_pr(fl_te[ycol], p))
                res[name]["n_test"].append(len(fl_te))
                res[name]["groups"].append(grp)
            print(f"  fold {grp} done ({time.time()-t0:.0f}s elapsed)", flush=True)

        print(f"\n=== horizon {h}ms  base_rate={base:.3f}  ({fold_name}-LOCO) ===", flush=True)
        out = {"base_rate": base, "fold_unit": fold_name, "models": {}}
        for name, r in res.items():
            rocs = np.array(r["roc"], dtype=float)
            prs = np.array(r["pr"], dtype=float)
            wt = np.array(r["n_test"])
            finite = np.isfinite(rocs)
            out["models"][name] = {
                "roc_auc": float(np.nanmean(rocs)),
                "roc_auc_sd": float(np.nanstd(rocs)),
                "pr_auc": float(np.nanmean(prs)),
                "pr_auc_sd": float(np.nanstd(prs)),
                "pr_auc_weighted": float(np.nansum(prs * wt) / wt.sum()),
                "n_folds_ok": int(finite.sum()),
            }
            print(f"  {name:8s} ROC-AUC={out['models'][name]['roc_auc']:.4f} "
                  f"(sd {out['models'][name]['roc_auc_sd']:.4f})  "
                  f"PR-AUC={out['models'][name]['pr_auc']:.4f} (w {out['models'][name]['pr_auc_weighted']:.4f})",
                  flush=True)
        all_metrics[h] = out

    outdir = cfg.paths.results_dir
    outdir.mkdir(parents=True, exist_ok=True)
    tag = "pairs" if args.pairs else "loco"
    path = outdir / f"models_{tag}_{'_'.join(str(h) for h in horizons)}.json"
    with open(path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()