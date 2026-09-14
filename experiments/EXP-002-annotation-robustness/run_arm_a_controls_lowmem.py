"""Low-memory runner for the Arm A control conditions (shuffle, uniform).

Uses the per-fold evaluation path (one fold's imputed matrices in memory at a
time) so it survives the machine's memory pressure; results are bit-consistent
with ``eval_loco`` (the same code path that produced the published anchors).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from research.config import Config
from common import (
    RESULT_DIR, eval_loco, labels_from_onsets, load_config,
    load_dataset_frame_idx, merge_onsets, perturb_onsets, save_json,
    stamp_meta, summarize_eval,
)


def main():
    cfg = load_config()
    cfg_obj = Config()
    tcfg = cfg["train"]
    which = sys.argv[1].split(",") if len(sys.argv) > 1 else ["shuffle", "uniform"]

    rows = merge_onsets(cfg_obj).sort_values(["conv_id", "frame_idx"]).reset_index(drop=True)
    dset = load_dataset_frame_idx(cfg_obj).sort_values(["conv_id", "frame_idx"]).reset_index(drop=True)
    dset = dset.drop(columns=["frame_idx"])
    assert len(dset) == len(rows)
    base_rate = float(rows["label_1000"].mean())
    time_s = rows["time_s"].to_numpy()
    onsets = rows["next_other_onset_s"].to_numpy()

    out_file = RESULT_DIR / "exp002_arm_a.json"
    results = json.loads(out_file.read_text())
    conds = results.setdefault("conditions", {})

    for name in which:
        if name in conds:
            print(f"[skip] {name}", flush=True)
            continue
        print(f"\n=== {name} ===", flush=True)
        t_cond = time.time()
        ef = {"evals": {}}

        if name == "shuffle":
            # Within-conversation permutation of the *labels* (not onsets):
            # preserves each conversation's marginal rate and its time profile,
            # destroying only the feature<->label alignment. Any AUC above 0.5
            # here is an annotation-alignment artifact, not a real signal.
            rng = np.random.default_rng(cfg["arm_a"]["shuffle_seed"])
            y = labels_from_onsets(time_s, onsets, 1000).astype(np.int8)
            conv = dset["conv_id"].astype(str).to_numpy()
            for grp in np.unique(conv):
                m = conv == grp
                y[m] = rng.permutation(y[m])
            spec = {"kind": "shuffle", "seed": cfg["arm_a"]["shuffle_seed"]}
        elif name == "uniform":
            # Bernoulli labels at the clean base rate (chance floor).
            rng = np.random.default_rng(cfg["arm_a"]["uniform_seed"])
            y = (rng.random(len(dset)) < base_rate).astype(np.int8)
            spec = {"kind": "uniform", "seed": cfg["arm_a"]["uniform_seed"]}
        else:
            raise SystemExit(f"unknown control {name!r}")

        br = float(y.mean())
        dr = dset.copy()
        dr["label_1000"] = y
        t_m = time.time()
        ev = eval_loco(dr, "label_1000", models=["lr"],
                       seed=tcfg["seed"], cap=tcfg["cap"])
        summ = summarize_eval(ev)["lr"]
        ef["evals"]["label_1000"] = {"base_rate": br, "lr": summ}
        print(f"    label_1000 ({time.time()-t_m:.0f}s): "
              f"base={br:.4f} ROC={summ['roc_auc']:.4f} PR={summ['pr_auc']:.4f}", flush=True)
        conds[name] = {**spec, **ef}
        save_json(out_file, results)
        print(f"  [{name}] total {time.time()-t_cond:.0f}s -> saved", flush=True)


if __name__ == "__main__":
    main()