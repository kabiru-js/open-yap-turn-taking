"""Build event tables and decision-point data frames.

Run:  python scripts/02_define_events.py
"""
import sys
sys.path.insert(0, "src")

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from research.config import Config
from research.dataset import load_all_metadata
from research.events import (
    build_segments,
    classify_transitions,
    build_decision_frame,
    merge_asr_event_type,
)


def _words_for(meta: dict) -> dict:
    out = {}
    for side in ("a", "b"):
        out[side] = [
            (w["word"], w["start"], w["end"])
            for w in meta["transcripts"][side]["words"]
        ]
    return out


def main():
    cfg = Config()
    meta = load_all_metadata(cfg)
    feat_dir = cfg.paths.interim_dir / "features"
    out_dir = cfg.paths.interim_dir / "events"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_events = []
    for conv_id in sorted(meta):
        npz = np.load(feat_dir / f"{conv_id}.npz", allow_pickle=True)
        vad_a, vad_b = npz["vad_a"], npz["vad_b"]
        n_frames = len(vad_a)

        segments = build_segments(
            vad_a, vad_b, cfg.audio.hop_ms,
            max_gap_ms=cfg.audio.max_gap_within_speaker_ms,
        )
        events = classify_transitions(segments)
        events = merge_asr_event_type(events, _words_for(meta[conv_id]))
        events["conv_id"] = conv_id
        all_events.append(events)

        decisions = build_decision_frame(
            conv_id,
            segments,
            n_frames,
            hop_ms=cfg.audio.hop_ms,
            decision_grid_ms=cfg.prediction.decision_grid_ms,
            horizons_ms=cfg.prediction.horizons_ms,
        )
        # downcast dtypes for storage
        decisions["current_speaker"] = decisions["current_speaker"].astype("category")
        decisions["other_speaker"] = decisions["other_speaker"].astype("category")
        decisions.to_parquet(out_dir / f"{conv_id}.parquet", index=False)
        events.to_csv(out_dir / f"{conv_id}_events.csv", index=False)
        types = ", ".join(f"{t}:{n}" for t, n in events["type"].value_counts().items())
        print(f"[ok] {conv_id}  segs={len(segments)} ev={len(events)}  ({types})", flush=True)

    ev_all = pd.concat(all_events, ignore_index=True)
    ev_out = cfg.paths.interim_dir / "events" / "all_events.csv"
    ev_all.to_csv(ev_out, index=False)

    dec_all = pd.concat(
        [
            pd.read_parquet(p)
            for p in sorted((cfg.paths.interim_dir / "events").glob("conv_*.parquet"))
        ],
        ignore_index=True,
    )
    dec_out = cfg.paths.interim_dir / "events" / "all_decisions.parquet"
    dec_all.to_parquet(dec_out, index=False)

    # ── Quick summary ────────────────────────────────────────────────────────
    print(f"Events (transitions): {len(ev_all)}")
    print(ev_all["type"].value_counts())
    print()
    print(f"Decision points: {len(dec_all)}")
    for h in cfg.prediction.horizons_ms:
        rate = dec_all[f"label_{h}"].mean()
        print(f"  base rate label_{h}: {rate:.4f}  (n_pos={int(rate*len(dec_all))})")
    print(f"\nSaved to {cfg.paths.interim_dir / 'events'}")


if __name__ == "__main__":
    main()