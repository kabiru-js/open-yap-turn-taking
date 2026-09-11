"""Build and cache frame-level features for all conversations.

Run:  python scripts/01_build_features.py
"""
import sys, time
sys.path.insert(0, "src")

import numpy as np

from research.config import Config
from research.dataset import iter_records
from research.audio import compute_all_frame_features_wrapper


def main():
    cfg = Config()
    cache = cfg.paths.interim_dir / "features"
    cache.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for data, a_audio, b_audio, conv_id in iter_records(cfg):
        out_path = cache / f"{conv_id}.npz"
        if out_path.exists():
            print(f"[skip] {conv_id} already cached")
            continue
        tc = time.time()
        nf_a = data["speakers"]["a"]["audio_metrics"]["noise_floor_dbfs"] or -70.0
        nf_b = data["speakers"]["b"]["audio_metrics"]["noise_floor_dbfs"] or -70.0
        feats = compute_all_frame_features_wrapper(
            a_audio, b_audio, cfg.audio.sample_rate, cfg,
            noise_floor_a_db=nf_a, noise_floor_b_db=nf_b,
        )
        flat = {"vad_a": feats["vad_a"], "vad_b": feats["vad_b"]}
        for side in ("a", "b"):
            for k, v in feats[side].items():
                flat[f"{side}_{k}"] = v.astype(np.float32)
        np.savez(out_path, **flat)
        print(
            f"[ok] {conv_id}  {len(feats['a']['time_s'])*cfg.audio.hop_ms/1000:.0f}s "
            f"({time.time()-tc:.1f}s)"
        )
    print(f"\nDone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()