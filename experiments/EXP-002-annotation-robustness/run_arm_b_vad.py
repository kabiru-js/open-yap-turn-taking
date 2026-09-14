"""Arm B: VAD/segmentation sensitivity (plan §10), from cached RMS only.

Re-derives the VAD at absolute thresholds ``20 + delta`` (delta in dB), rebuilds
segments -> decisions -> every decision-window features/labels from the cached
frame features, then runs LOCO LR + GBDT. Delta = 0 must reproduce the stored
``dataset.parquet``; the consistent cell (features+labels jointly re-derived)
and the drift cell (features from delta, labels from delta = 0) are reported.

Usage (resume-safe):
  python run_arm_b_vad.py --margins -3,0,3
  python run_arm_b_vad.py --drift --margins 3
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from research.config import Config
from common import (
    RESULT_DIR, build_dataset_at_margin, eval_loco, load_config,
    load_dataset_frame_idx, load_noise_floor_metrics, save_json, stamp_meta,
    summarize_eval,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--margins", default=None, help="comma list of deltas in dB")
    ap.add_argument("--no-drift", action="store_true", help="skip drift cells")
    ap.add_argument("--models", default="lr", help="comma list of lr,gbdt (default lr)")
    args = ap.parse_args()

    cfg = load_config()
    cfg_obj = Config()
    tcfg = cfg["train"]
    b = cfg["arm_b"]
    base_margin = cfg_obj.features.vad_margin_above_noise_db  # 20 dB
    want_models = [m.strip() for m in args.models.split(",")]

    margins = [int(x) for x in args.margins.split(",")] if args.margins else b["margins_db"]
    do_drift = not args.no_drift
    horizons = b["horizons_ms"]
    window_frames = int(round(cfg_obj.prediction.lookback_s * 1000 / cfg_obj.audio.hop_ms))

    metrics = load_noise_floor_metrics(cfg_obj)

    stored = load_dataset_frame_idx(cfg_obj)
    stored_labels = stored.sort_values(["conv_id", "frame_idx"]).reset_index(drop=True)

    out_file = RESULT_DIR / "exp002_arm_b.json"
    results = json.loads(out_file.read_text()) if out_file.exists() else {"meta": None}
    datasets = results.setdefault("datasets", {})
    if not results.get("meta"):
        results["meta"] = {**stamp_meta(), "base_margin_db": float(base_margin)}

    for delta in sorted(margins):
        margin_db = base_margin + delta
        tag = f"m{'+' if delta >= 0 else ''}{delta}"
        if tag in datasets and "cells" in datasets[tag]:
            print(f"[skip] consistent {tag}", flush=True)
            continue
        print(f"\n=== ARM B margin delta={delta:+d}dB (absolute {margin_db:.0f}dB) ===", flush=True)
        dset = build_dataset_at_margin(cfg_obj, metrics, margin_db, horizons, window_frames)
        dset["frame_idx"] = (dset["time_s"] / 0.01).round().astype("int64")
        for h in horizons:
            ycol = f"label_{h}"
            entry = {
                "delta_db": delta, "base_rate": float(dset[ycol].mean()),
                "n_decision_points": int(len(dset)),
                "cells": {},
            }
            t0 = time.time()
            eval_res = eval_loco(dset, ycol, models=want_models,
                                 seed=tcfg["seed"], cap=tcfg["cap"])
            entry["cells"]["consistent"] = summarize_eval(eval_res)
            if do_drift and delta != 0:
                # features from this dataset, labels from the clean (delta=0) dataset
                dr = dset.assign(conv_id=dset["conv_id"].astype(str)).merge(
                    stored_labels[["conv_id", "frame_idx", ycol]],
                    on=["conv_id", "frame_idx"], how="inner",
                    suffixes=("", "_clean"), validate="one_to_one",
                )
                dr2 = (dr.drop(columns=[ycol])
                       .rename(columns={ycol + "_clean": ycol})
                       .reset_index(drop=True))
                dr2["conv_id"] = dr2["conv_id"].astype("category")
                print(f"    drift cells rows: {len(dr2)} (of {len(dset)})", flush=True)
                eval_d = eval_loco(dr2, ycol, models=want_models,
                                   seed=tcfg["seed"], cap=tcfg["cap"])
                entry["cells"]["drift_labels_from_clean"] = summarize_eval(eval_d)
                entry["cells"]["drift_labels_from_clean"]["n_join_rows"] = int(len(dr2))
            datasets[tag] = entry
            save_json(out_file, results)
            print(f"  [{tag}] consistent saved in {time.time()-t0:.0f}s", flush=True)
    print(f"\narm_b complete: {sorted(datasets)}")


if __name__ == "__main__":
    main()