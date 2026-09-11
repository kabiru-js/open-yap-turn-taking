"""Causal windowed feature extraction for decision points.

For each decision time t we describe the recent past [t - L, t] using
only information available at or before t (no future leakage). All
aggregates over the frame grid are computed vectorized via cumulative
sums; min/max via pandas rolling windows.
"""

from __future__ import annotations
from typing import Dict

import numpy as np
import pandas as pd

CHANNELS = [
    "rms_db", "zcr", "centroid", "bandwidth", "rolloff",
    "spectral_flux", "f0", "energy_slope",
]

# Canonical feature list used at decision time (all causal).
_FEAT_STATS = ["mean", "std", "slope", "last", "min", "max"]
_FEAT_STATS_SLIM = ["mean", "std", "slope", "last"]
FEAT_COLS = (
    sum([[f"{side}_{ch}_{st}" for st in _FEAT_STATS]
         for side in ("a", "b")
         for ch in ("rms_db",)], [])
    + sum([[f"{side}_{ch}_{st}" for st in _FEAT_STATS_SLIM]
           for side in ("a", "b")
           for ch in ("zcr", "centroid", "bandwidth", "rolloff", "spectral_flux",
                      "f0", "energy_slope")], [])
    + ["a_voiced_frac_mean", "b_voiced_frac_mean"]
    + ["pause_since_last_speech", "elapsed_s", "current_seg_dur_s"]
)


def _rolling_sums(v: np.ndarray, w: int) -> Dict[str, np.ndarray]:
    """Cumsum-based rolling aggregates over causal windows.

    Returned arrays have the same length as ``v`` and are indexed by the
    END frame ``f``: ``stat[f]`` = aggregate over frames ``[f-w, f-1]``
    (strictly past; frame ``f`` is still in progress at time ``f*hop``).
    Statistics near the array start are NaN (window not full).
    """
    n = len(v)
    if n < w:
        return {k: np.full(n, np.nan)
                for k in ("mean", "std", "slope", "last", "min", "max")}

    vf = np.nan_to_num(v, nan=0.0)
    t = np.arange(n, dtype=np.float64)
    c = np.cumsum(np.ones(n))
    S1 = np.cumsum(vf)
    S2 = np.cumsum(vf * vf)
    St = np.cumsum(t)
    Stf = np.cumsum(t * vf)
    St2 = np.cumsum(t * t)

    def win(A):
        A = np.concatenate([[0.0], A])
        return A[w:] - A[:-w]  # length n-w+1; win[j] = sum over [j, j+w)

    cnt = win(c)          # number of frames in [j, j+w)
    Ts = win(St)
    T2s = win(St2)
    Tfs = win(Stf)
    S1w = win(S1)
    S2w = win(S2)

    valid = cnt > 0
    mean = np.full(n, np.nan)
    std = np.full(n, np.nan)
    slope = np.full(n, np.nan)
    m = mean
    # map: stat[f] = value[j] with j = f-w  (window [f-w, f-1])
    out_idx = np.arange(w, n)          # frames f in [w, n) have a full window
    j = out_idx - w
    mean[out_idx] = S1w[j] / cnt[j]
    var = np.maximum(S2w[j] / cnt[j] - mean[out_idx] ** 2, 0.0)
    std[out_idx] = np.sqrt(var)
    denom = cnt[j] * T2s[j] - Ts[j] ** 2
    ok = denom > 1e-9
    slope[out_idx[ok]] = (cnt[j[ok]] * Tfs[j[ok]] - Ts[j[ok]] * S1w[j[ok]]) / denom[ok]

    # last complete value (frame f-1)
    lastval = np.full(n, np.nan)
    lastval[1:] = v[:-1]

    # min / max over [f-w, f-1] via pandas rolling with shift(1)
    s = pd.Series(v).shift(1)
    out_min = s.rolling(w, min_periods=1).min().to_numpy()
    out_max = s.rolling(w, min_periods=1).max().to_numpy()

    return {"mean": mean, "std": std, "slope": slope,
            "last": lastval, "min": out_min, "max": out_max}


def build_window_features(
    npz: dict,
    decisions: pd.DataFrame,
    window_frames: int,
    channels: list = CHANNELS,
) -> pd.DataFrame:
    """Return a DataFrame with one row per decision point.

    Columns: ``a_<ch>_<stat>``, ``b_<ch>_<stat>`` for every channel x
    stat; plus ``a_voiced_frac`` and ``b_voiced_frac`` (fraction of
    non-NaN f0 frames in the window).
    """
    rows = {}
    for side in ("a", "b"):
        voiced = (~np.isnan(npz[f"{side}_f0"].ravel())).astype(np.float64)
        rows[(side, "voiced_frac", "mean")] = _rolling_sums(voiced, window_frames)["mean"]
        for ch in channels:
            v = npz[f"{side}_{ch}"]
            if v.ndim != 1:
                v = v.ravel()
            if v.shape[0] <= window_frames:
                stats = {k: np.full(v.shape[0], np.nan) for k in
                         ("mean", "std", "slope", "last", "min", "max")}
            else:
                stats = _rolling_sums(v, window_frames)
            for stat, arr in stats.items():
                rows[(side, ch, stat)] = arr

    base_idx = decisions["frame_idx"].to_numpy()
    data = {
        "conv_id": decisions["conv_id"].to_numpy(),
        "time_s": decisions["time_s"].to_numpy(),
        "current_speaker": decisions["current_speaker"].astype(str).to_numpy(),
        "other_speaker": decisions["other_speaker"].astype(str).to_numpy(),
        "pause_since_last_speech": decisions["pause_since_last_speech"].to_numpy(),
        "elapsed_s": decisions["elapsed_s"].to_numpy(),
        "current_seg_dur_s": decisions["current_seg_dur_s"].to_numpy(),
    }
    for (side, ch, stat), arr in rows.items():
        # stat arrays are indexed by END frame f (window [f-w, f-1]);
        # frame_idx is the first frame still in progress at decision time,
        # so window ending at frame_idx-1 is stat[frame_idx].
        if base_idx.max() < arr.shape[0]:
            data[f"{side}_{ch}_{stat}"] = arr[base_idx]
        else:
            data[f"{side}_{ch}_{stat}"] = np.full(len(base_idx), np.nan)
    df = pd.DataFrame(data)
    return df