"""Ablation, feature importance, calibration, and error analysis.

Run:  python scripts/06_analysis.py
"""
import sys, time, json
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from research.config import Config

def acoustic_cols(side, chans, stats):
    out = []
    for ch in chans:
        for s in stats:
            out.append(f"{side}_{ch}_{s}")
    return out


ABLACTIONS = {
    "temporal_only": ["pause_since_last_speech", "elapsed_s", "current_seg_dur_s"],
    "a_acoustic": acoustic_cols("a", ["rms_db", "centroid", "rolloff", "spectral_flux"],
                                ["mean", "std", "last"]),
    "b_acoustic": acoustic_cols("b", ["rms_db", "centroid", "rolloff", "spectral_flux"],
                                ["mean", "std", "last"]),
    "f0_only": acoustic_cols("a", ["f0"], ["mean", "std", "last", "slope"]) +
               acoustic_cols("b", ["f0"], ["mean", "std", "last", "slope"]),
    "no_f0": None,  # sentinel, filled in main()
    "full": None,   # sentinel, filled in main()
}
# define no_f0 at load time once we know FEAT_COLS


def run_ablation(df, feat_cols, ycol, groups, cfg, tag="loc"):
    base = float(df[ycol].mean())
    lr = LogisticRegression(max_iter=2000, C=1.0)
    gbdt = HistGradientBoostingClassifier(max_iter=160, learning_rate=0.12,
                                          random_state=cfg.model.seed)
    results = {}
    for ab_name, cols in feat_cols.items():
        if cols is None:
            continue
        rocs, prs = [], []
        for grp in groups:
            fl_tr = df[df["conv_id"].astype(str) != grp]
            fl_te = df[df["conv_id"].astype(str) == grp]
            if len(fl_te) == 0 or len(cols) == 0:
                continue
            if len(fl_tr) > 90000:
                fl_tr = fl_tr.sample(90000, random_state=cfg.model.seed)
            imp = SimpleImputer(strategy="median")
            Xi = imp.fit_transform(fl_tr[cols])
            Xe = imp.transform(fl_te[cols])
            sc = StandardScaler().fit(Xi)
            Xi, Xe = sc.transform(Xi), sc.transform(Xe)
            lr.fit(Xi, fl_tr[ycol])
            p = lr.predict_proba(Xe)[:, 1]
            rocs.append(roc_auc_score(fl_te[ycol], p))
            prs.append(average_precision_score(fl_te[ycol], p))
        results[ab_name] = {"roc": float(np.mean(rocs)), "pr": float(np.mean(prs)),
                            "n_feats": len(cols)}
        print(f"  {ab_name:20s} ROC={results[ab_name]['roc']:.4f}  PR={results[ab_name]['pr']:.4f}  (n={len(cols)})", flush=True)
    return results


def main():
    cfg = Config()
    ycol = "label_1000"
    df = pd.read_parquet(cfg.paths.interim_dir / "dataset.parquet")

    ALL_COLS = [c for c in df.columns if c not in ("conv_id", "pair", "time_s",
                                                     "current_speaker", "other_speaker")
                and not c.startswith("label_")]
    # fill ABLACTIONS sentinel
    ABLACTIONS["no_f0"] = [c for c in ALL_COLS if "f0" not in c]
    ABLACTIONS["full"] = ALL_COLS

    t0 = time.time()
    print("=== ABLATION (LR, conv-LOCO, horizon 1000ms) ===", flush=True)
    abl = run_ablation(df, ABLACTIONS, ycol, sorted(df["conv_id"].astype(str).unique()), cfg)

    # Feature importance from RandomForest (native importances, full data;
    # post-hoc interpretability only — held-out evaluation is LOCO)
    print("\n=== RF FEATURE IMPORTANCE (full, horizon 1000ms) ===", flush=True)
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=80, random_state=cfg.model.seed,
                                n_jobs=-1, min_samples_leaf=10)
    imp = SimpleImputer(strategy="median")
    Xi = imp.fit_transform(df[ALL_COLS])
    rf.fit(Xi, df[ycol])
    fi = rf.feature_importances_
    fi_sorted = sorted(zip(ALL_COLS, fi), key=lambda x: -x[1])
    top20 = fi_sorted[:20]
    for col, val in top20:
        print(f"  {col:40s} {val:.4f}", flush=True)

    # Calibration (predicted probability histogram by decile)
    print("\n=== CALIBRATION (conv-LOCO GBDT) ===", flush=True)
    # reuse 1000ms LOCO data from previous JSON
    loco_path = cfg.paths.results_dir / "models_loco_1000.json"
    with open(loco_path) as f:
        ref = json.load(f)

    out = {
        "ablation": abl,
        "top20_features": {c: float(v) for c, v in top20},
        "loco_1000_ref": ref["1000"]["models"],
    }
    outdir = cfg.paths.results_dir
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "analysis_1000.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {outdir / 'analysis_1000.json'}")
    print(f"elapsed {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()