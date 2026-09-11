"""Profile event construction on one conversation."""
import sys, time
sys.path.insert(0, "src")
import numpy as np
from research.config import Config
from research.dataset import load_all_metadata
from research.events import build_segments, classify_transitions, build_decision_frame

cfg = Config()
meta = load_all_metadata(cfg)
conv_id = "conv_0c917fc63b3b"
npz = np.load(cfg.paths.interim_dir / "features" / f"{conv_id}.npz", allow_pickle=True)

t0 = time.time()
vad_a, vad_b = npz["vad_a"], npz["vad_b"]
print("vad shapes", vad_a.shape, time.time() - t0)
t1 = time.time()
segments = build_segments(vad_a, vad_b, cfg.audio.hop_ms)
print(f"segments: {len(segments)} ({time.time()-t1:.1f}s)")
t2 = time.time()
events = classify_transitions(segments)
print(f"transitions: {len(events)} ({time.time()-t2:.1f}s)")
t3 = time.time()
decisions = build_decision_frame(
    conv_id, segments, len(vad_a), hop_ms=cfg.audio.hop_ms,
    decision_grid_ms=cfg.prediction.decision_grid_ms,
    horizons_ms=cfg.prediction.horizons_ms,
)
print(f"decisions: {len(decisions)} ({time.time()-t3:.1f}s)")
print(events["type"].value_counts())
print(decisions[[c for c in decisions.columns if c.startswith('label')]].mean().round(4))
print(f"total {time.time()-t0:.1f}s")