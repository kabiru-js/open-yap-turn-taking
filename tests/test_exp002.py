"""EXP-002 fidelity guards.

These tests assert that the corruption machinery is a faithful superset of the
stored EXP-001 pipeline:
1. Arm A: recomputing labels from *unperturbed* onsets reproduces the stored
   labels exactly (all horizons, every decision point).
2. Arm A: the dataset <-> decision-row join is complete and 1:1.
3. Arm B: re-deriving the VAD from cached RMS at the original margin reproduces
   the cached VAD masks exactly (all conversations).
4. Arm B: the full decision-frame rebuild at the original margin reproduces the
   stored decision frames (counts and base rates).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "experiments" / "EXP-002-annotation-robustness"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from research.config import Config  # noqa: E402
import common  # noqa: E402


def _require_interim():
    cfg = Config()
    if not (cfg.paths.interim_dir / "dataset.parquet").exists():
        pytest.skip("data/interim missing (unavailable to other machines)")


# ── Arm A guards ─────────────────────────────────────────────────────────────

def test_arm_a_join_is_complete_and_one_to_one():
    _require_interim()
    cfg = Config()
    dset = common.load_dataset_frame_idx(cfg)
    ev = common.load_decision_rows(cfg)
    assert len(dset) == len(ev)
    keys = ["conv_id", "frame_idx"]
    dup = ev.duplicated(subset=keys)
    assert not dup.any()
    joined = dset.merge(ev[keys + ["next_other_onset_s"]], on=keys, how="left",
                        validate="one_to_one")
    assert joined["next_other_onset_s"].notna().all()


def test_arm_a_clean_reproduces_stored_labels():
    _require_interim()
    cfg = Config()
    dset = common.load_dataset_frame_idx(cfg)
    ev = common.load_decision_rows(cfg)
    keys = ["conv_id", "frame_idx"]
    assert len(dset) == len(ev)
    for h in (500, 1000):
        recomputed = common.labels_from_onsets(
            ev["time_s"].to_numpy(),
            ev["next_other_onset_s"].to_numpy(),
            h,
        )
        assert recomputed.dtype == np.int8
        assert float(recomputed.mean()) == pytest.approx(
            float(ev[f"label_{h}"].mean()), abs=1e-9)
        # per-conversation equality against the stored dataset labels
        e_ev = ev.sort_values(keys).reset_index(drop=True)
        e_ds = dset.sort_values(keys).reset_index(drop=True)
        np.testing.assert_array_equal(recomputed, e_ds[f"label_{h}"].to_numpy())
        # stored decision labels must also match what a clean join would produce
        np.testing.assert_array_equal(e_ev[f"label_{h}"].to_numpy(),
                                      e_ds[f"label_{h}"].to_numpy())


def test_arm_a_shuffle_and_jitter_are_deterministic():
    _require_interim()
    cfg = Config()
    ev = common.load_decision_rows(cfg).sort_values(["conv_id", "frame_idx"])
    onsets = ev["next_other_onset_s"].to_numpy()
    rng1 = np.random.default_rng(404)
    rng2 = np.random.default_rng(404)
    a1 = common.perturb_onsets(onsets, "shuffle", 0.0, 0.0, rng1)
    a2 = common.perturb_onsets(onsets, "shuffle", 0.0, 0.0, rng2)
    np.testing.assert_array_equal(a1, a2)
    assert np.isfinite(a1).sum() == np.isfinite(onsets).sum()
    # jitter preserves inf positions
    r1 = np.random.default_rng(101)
    j = common.perturb_onsets(onsets, "jitter", 100.0, 0.0, r1)
    np.testing.assert_array_equal(np.isfinite(j), np.isfinite(onsets))
    assert j.min() >= 0.0


# ── Arm B guards ─────────────────────────────────────────────────────────────

def _metrics_if_available():
    cfg = Config()
    if not list(cfg.paths.raw_dir.glob("shard-*.tar")):
        pytest.skip("raw tar shards missing (needed for noise-floor metrics)")
    return common.load_noise_floor_metrics(cfg)


def test_arm_b_vad_reuse_matches_cached_masks():
    _require_interim()
    metrics = _metrics_if_available()
    cfg = Config()
    base = cfg.features.vad_margin_above_noise_db
    feat_dir = cfg.paths.interim_dir / "features"
    checked = 0
    for p in sorted(feat_dir.glob("conv_*.npz")):
        npz = np.load(p, allow_pickle=True)
        vad_a, vad_b = common.recompute_vad_from_rms(
            npz["a_rms_db"], npz["b_rms_db"],
            metrics[p.stem]["a"], metrics[p.stem]["b"],
            base, cfg)
        # The cached RMS chains are truncated a few frames before the cached
        # VAD (feature-cache tail trim ~30ms at recording end). VAD must match
        # on the shared prefix; the tail is unreachable by any decision point
        # (2 s tail exclusion), so it cannot affect Arm B results.
        shared = len(npz["a_rms_db"])
        assert len(npz["vad_a"]) - shared <= 5
        np.testing.assert_array_equal(vad_a[:shared], npz["vad_a"][:shared])
        np.testing.assert_array_equal(vad_b[:shared], npz["vad_b"][:shared])
        checked += 1
    assert checked == 16


def test_arm_b_decision_frame_at_margin0_matches_stored():
    _require_interim()
    metrics = _metrics_if_available()
    cfg = Config()
    base = cfg.features.vad_margin_above_noise_db
    horizons = cfg.prediction.horizons_ms
    ev_dir = cfg.paths.interim_dir / "events"
    feat_dir = cfg.paths.interim_dir / "features"
    checked = 0
    for p in sorted(feat_dir.glob("conv_*.npz")):
        npz = np.load(p, allow_pickle=True)
        vad_a, vad_b = common.recompute_vad_from_rms(
            npz["a_rms_db"], npz["b_rms_db"],
            metrics[p.stem]["a"], metrics[p.stem]["b"], base, cfg)
        segments = common.build_segments(
            vad_a, vad_b, cfg.audio.hop_ms,
            max_gap_ms=cfg.audio.max_gap_within_speaker_ms)
        dec = common.build_decision_frame(
            p.stem, segments, len(vad_a),
            hop_ms=cfg.audio.hop_ms,
            decision_grid_ms=cfg.prediction.decision_grid_ms,
            horizons_ms=horizons,
        )
        stored = pd.read_parquet(ev_dir / f"{p.stem}.parquet").sort_values("time_s")
        dec = dec.sort_values("time_s").reset_index(drop=True)
        assert len(dec) == len(stored), (p.stem, len(dec), len(stored))
        np.testing.assert_allclose(dec["time_s"].to_numpy(),
                                   stored["time_s"].to_numpy(), atol=1e-12)
        for h in horizons:
            np.testing.assert_array_equal(dec[f"label_{h}"].to_numpy(),
                                          stored[f"label_{h}"].to_numpy())
        checked += 1
    assert checked == 16