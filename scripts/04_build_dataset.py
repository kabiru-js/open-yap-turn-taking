"""Build the decision-point ML dataset (windowed causal features + labels).

Run:  python scripts/04_build_dataset.py
"""
import sys, time
sys.path.insert(0, "src")

import numpy as np
import pandas as pd

from research.config import Config
from research.window_features import build_window_features

def main():
    cfg = Config()
    feat_dir = cfg.paths.interim_dir / "features"
    ev_dir = cfg.paths.interim_dir / "events"
    window_frames = int(round(cfg.prediction.lookback_s * 1000 / cfg.audio.hop_ms))

    parts = []
    t0 = time.time()
    for p in sorted(ev_dir.glob("conv_*.parquet")):
        conv_id = p.stem
        decisions = pd.read_parquet(p)
        npz = np.load(feat_dir / f"{conv_id}.npz", allow_pickle=True)
        wf = build_window_features(npz, decisions, window_frames)
        for h in cfg.prediction.horizons_ms:
            wf[f"label_{h}"] = decisions[f"label_{h}"].to_numpy()
        parts.append(wf)
        print(f"[ok] {conv_id}  features={wf.shape[0]} rows", flush=True)
        del npz

    df = pd.concat(parts, ignore_index=True)
    # conversation id as category (fast, cheap)
    df["conv_id"] = df["conv_id"].astype("category")
    out = cfg.paths.interim_dir / "dataset.parquet"
    df.to_parquet(out, index=False)
    print(f"\nSaved {df.shape} rows x {df.shape[1]} cols -> {out}")
    print(f"build time {time.time()-t0:.1f}s")
    for h in cfg.prediction.horizons_ms:
        print(f"  label_{h} base rate: {df[f'label_{h}'].mean():.4f}")
    # NaN stats on numeric features (all windowed stats start NaN before lookback)
    num = df.select_dtypes(include=[np.number]).columns
    nan_frac = df[num].isna().mean().mean()
    print(f"  mean NaN fraction across numeric features: {nan_frac:.3f}")

if __name__ == "__main__":
    main()