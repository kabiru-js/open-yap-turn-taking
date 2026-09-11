"""Causality and leak-safety tests for the feature stack.

Run:  python -m pytest tests -q
"""
import numpy as np
import pandas as pd

from research.window_features import FEAT_COLS, _rolling_sums
from research.audio import energy_slope


def test_feat_cols_is_73_and_label_free():
    assert len(FEAT_COLS) == 73
    assert not any(c.startswith("label_") for c in FEAT_COLS)
    assert "pause_since_last_speech" in FEAT_COLS
    assert "current_seg_dur_s" in FEAT_COLS


def test_rolling_last_is_strictly_past():
    w = 4
    v = np.zeros(12)
    v[5] = 1.0
    stats = _rolling_sums(v, w)
    # last[f] = v[f-1]: frame f is still in progress at time f*hop
    np.testing.assert_array_equal(stats["last"][1:], v[:-1])
    assert stats["last"][0] != stats["last"][0]  # NaN at first frame

    # mean[f] covers [f-w, f-1]; the pulse at index 5 shows up at f=9
    mean = stats["mean"]
    assert np.isnan(mean[3])
    assert mean[9] == 0.25
    # the in-flight frame (f=9) must not contribute at decision f=9
    assert mean[9] == 0.25


def test_energy_slope_is_trailing_causal():
    rng = np.random.RandomState(0)
    rms = 10 * np.abs(rng.randn(300)) - 50
    w = 30
    e = energy_slope(rms, w)
    for i in (w, w + 10, 150):
        prefix = energy_slope(rms[: i + 1], w)
        np.testing.assert_allclose(e[i], prefix[-1], atol=1e-12)