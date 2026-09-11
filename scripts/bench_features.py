"""Time feature extraction on a single conversation."""
import sys, time
sys.path.insert(0, "src")
import numpy as np
from research.config import Config
from research.dataset import iter_records
from research.audio import compute_all_frame_features_wrapper

cfg = Config()
t_start = time.time()
for data, a_audio, b_audio, conv_id in iter_records(cfg):
    if conv_id != "conv_31bedfc99b53":
        continue
    nf_a = data["speakers"]["a"]["audio_metrics"]["noise_floor_dbfs"] or -70.0
    nf_b = data["speakers"]["b"]["audio_metrics"]["noise_floor_dbfs"] or -70.0
    t0 = time.time()
    feats = compute_all_frame_features_wrapper(
        a_audio, b_audio, cfg.audio.sample_rate, cfg,
        noise_floor_a_db=nf_a, noise_floor_b_db=nf_b,
    )
    print(f"{conv_id}: {time.time()-t0:.1f}s  dur={data['duration_seconds']:.0f}s")
    for side in ("a", "b"):
        for k, v in feats[side].items():
            print(f"  {side}_{k}: shape={v.shape} nan={np.isnan(v).mean():.3f}")
    break
print(f"total {time.time()-t_start:.1f}s")