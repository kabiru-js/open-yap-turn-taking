"""Arm A: label/annotation perturbation robustness (features frozen).

Precomputes the 16 LOCO fold splits + imputed feature matrices once (they
are identical across conditions because the feature block is frozen), then
each condition only needs fresh model fits on the perturbed labels.

Usage (all runs resume-safe; completed conditions are skipped):
  python run_arm_a_labels.py                                    # all pending (LR only)
  python run_arm_a_labels.py --with-gbdt                        # add GBDT@1000 per cond
  python run_arm_a_labels.py --seeds 101 --sigmas 25,50,100,200,400
  python run_arm_a_labels.py --max-conds 3 --no-controls
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from research.config import Config
from common import (
    RESULT_DIR, eval_loco_fast, labels_from_onsets, load_config,
    load_dataset_frame_idx, merge_onsets, prep_loco_folds,
    perturb_onsets, save_json, stamp_meta, summarize_eval,
)


def build_condition_specs(cfg, seeds, sigmas):
    """Build jitter + shift condition specs.

    The shuffle and uniform chance-controls are NOT generated here — they live
    in ``run_arm_a_controls_lowmem.py`` which computes the correct
    within-conversation label-permutation control. The global onset-shuffle
    produced by ``perturb_onsets(..."shuffle"...)`` is confounded by the
    ``elapsed_s`` feature and MUST NOT be used as the reported control.
    """
    specs = [{"name": "clean", "kind": "clean", "sigma_ms": 0, "seed": None}]
    for s in sigmas:
        for seed in seeds:
            specs.append({"name": f"jitter_s{seed}_sigma{s}",
                          "kind": "jitter", "sigma_ms": s, "seed": seed})
    for shift in cfg["arm_a"]["shifts_ms"]:
        tag = ("p" if shift >= 0 else "m") + str(abs(shift))
        specs.append({"name": f"shift_{tag}", "kind": "shift",
                      "shift_ms": shift, "seed": cfg["arm_a"]["shift_seed"]})
    return specs


def make_labels(row_arr, spec, base_rate, kind_rng):
    time_s = row_arr["time_s"]
    if spec["kind"] == "uniform":
        y = (kind_rng.random(len(row_arr)) < base_rate).astype(np.int8)
        return {"1000": y, "500": y}
    onsets = perturb_onsets(row_arr["next_other_onset_s"], spec["kind"],
                            spec.get("sigma_ms", 0.0), spec.get("shift_ms", 0.0),
                            kind_rng)
    return {"1000": labels_from_onsets(time_s, onsets, 1000),
            "500": labels_from_onsets(time_s, onsets, 500)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=None, help="label seeds (comma list)")
    ap.add_argument("--sigmas", default=None, help="jitter sigmas ms (comma list)")
    ap.add_argument("--no-controls", action="store_true", help="(deprecated, ignored; controls live in run_arm_a_controls_lowmem.py)")
    ap.add_argument("--max-conds", type=int, default=0)
    ap.add_argument("--with-gbdt", action="store_true",
                    help="add expensive GBDT@1000 per condition")
    args = ap.parse_args()

    cfg = load_config()
    cfg_obj = Config()
    tcfg = cfg["train"]
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else cfg["arm_a"]["label_seeds"]
    sigmas = [int(x) for x in args.sigmas.split(",")] if args.sigmas else cfg["arm_a"]["jitter_sigmas_ms"]
    specs = build_condition_specs(cfg, sorted(set(seeds)), sorted(set(sigmas)))

    t0 = time.time()
    rows = merge_onsets(cfg_obj).sort_values(["conv_id", "frame_idx"]).reset_index(drop=True)
    base_rate = float(rows["label_1000"].mean())
    dset = load_dataset_frame_idx(cfg_obj).sort_values(["conv_id", "frame_idx"]).reset_index(drop=True)
    assert len(dset) == len(rows), (len(dset), len(rows))
    print(f"[prep] join+dataset in {time.time() - t0:.0f}s  n={len(dset)}  base={base_rate:.4f}",
          flush=True)

    n = len(dset)
    row_arr = np.zeros(n, dtype=[("time_s", "f8"), ("next_other_onset_s", "f8")])
    row_arr["time_s"] = rows["time_s"].to_numpy()
    row_arr["next_other_onset_s"] = rows["next_other_onset_s"].to_numpy()

    # precompute the 16 LOCO folds once (imputed feature matrices)
    t0 = time.time()
    folds = prep_loco_folds(dset, seed=tcfg["seed"], cap=tcfg["cap"])
    print(f"[prep] folds in {time.time() - t0:.0f}s  groups={len(folds)}", flush=True)

    out_file = RESULT_DIR / "exp002_arm_a.json"
    results = json.loads(out_file.read_text()) if out_file.exists() else {"meta": None}
    conds = results.setdefault("conditions", {})
    if not results.get("meta"):
        results["meta"] = {**stamp_meta(), "n_decision_points": int(n),
                           "base_rate_1000_clean": float(base_rate)}

    done = 0
    for spec in specs:
        name = spec["name"]
        if name in conds:
            print(f"[skip] {name}", flush=True)
            continue
        if args.max_conds and done >= args.max_conds:
            break
        y = make_labels(row_arr, spec, base_rate,
                        np.random.default_rng(spec["seed"] or 101))
        lr_models = ["lr"]
        gb_models = ["gbdt"] if args.with_gbdt else []
        all_models = lr_models + gb_models
        print(f"\n=== {name}  kind={spec['kind']}  "
              f"sigma={spec.get('sigma_ms','-')}ms  seed={spec['seed']}", flush=True)
        cond_out = {"kind": spec["kind"], "sigma_ms": spec.get("sigma_ms"),
                    "seed": spec.get("seed"), "evals": {}}
        t_cond = time.time()
        for h in ("1000", "500"):
            yh = y[h]
            cond_out["evals"][f"label_{h}"] = {"base_rate": float(yh.mean())}
            eval_models = lr_models if h == "500" else all_models
            if not eval_models:
                continue
            t_m = time.time()
            ev = eval_loco_fast(folds, yh,
                                models=eval_models, seed=tcfg["seed"])
            cond_out["evals"][f"label_{h}"] = {
                **summarize_eval(ev), **cond_out["evals"][f"label_{h}"]}
            tag = ", ".join(f"{m} {round(summarize_eval(ev)[m]['roc_auc'],4)}"
                           for m in eval_models if summarize_eval(ev)[m]["roc_auc"])
            print(f"    label_{h} ({time.time()-t_m:.0f}s): {tag}", flush=True)
        conds[name] = cond_out
        save_json(out_file, results)
        done += 1
        print(f"  [{name}] total {time.time()-t_cond:.0f}s -> saved", flush=True)

    print(f"\ncompleted={len(conds)}/{len(specs)}  total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
