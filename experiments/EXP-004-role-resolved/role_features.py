"""Role-resolved feature transformation for conversational turn-taking.

Transforms channel-arbitrary acoustic features (a_*, b_*) into conversational
role-aligned features:
- floor_*    : acoustic features of the speaker who held the floor at time t
               (most recent segment with start_s <= t)
- listener_* : acoustic features of the partner / incoming speaker
- temporal   : pause_since_last_speech, elapsed_s, current_seg_dur_s

Strictly causal: 'current_speaker' is determined solely by past/present segment
starts (start_s <= t), with zero peeking into future speech onsets or labels.
"""

from __future__ import annotations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from research.window_features import FEAT_COLS

TEMPORAL_COLS: List[str] = [
    "pause_since_last_speech",
    "elapsed_s",
    "current_seg_dur_s",
]

# 35 per-speaker acoustic features
STAT_NAMES: List[str] = [c[2:] for c in FEAT_COLS if c.startswith("a_")]

FLOOR_COLS: List[str] = [f"floor_{name}" for name in STAT_NAMES]
LISTENER_COLS: List[str] = [f"listener_{name}" for name in STAT_NAMES]
ACOUSTIC_ROLE_COLS: List[str] = FLOOR_COLS + LISTENER_COLS
ROLE_FEAT_COLS: List[str] = ACOUSTIC_ROLE_COLS + TEMPORAL_COLS

F0_ROLE_COLS: List[str] = [c for c in ROLE_FEAT_COLS if "f0" in c]
NO_F0_ROLE_COLS: List[str] = [c for c in ROLE_FEAT_COLS if "f0" not in c]


def build_role_resolved_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Construct role-resolved features from a dataset dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing FEAT_COLS and 'current_speaker' ('a' or 'b').

    Returns
    -------
    pd.DataFrame
        DataFrame with ROLE_FEAT_COLS plus metadata ('conv_id', 'time_s', 'pair', labels).
    """
    assert "current_speaker" in df.columns, "'current_speaker' must be in df"
    is_a = (df["current_speaker"].astype(str) == "a").to_numpy()

    cols_a = [f"a_{name}" for name in STAT_NAMES]
    cols_b = [f"b_{name}" for name in STAT_NAMES]

    role_dict: Dict[str, np.ndarray] = {}
    for ca, cb in zip(cols_a, cols_b):
        stat_name = ca[2:]
        arr_a = df[ca].to_numpy()
        arr_b = df[cb].to_numpy()
        role_dict[f"floor_{stat_name}"] = np.where(is_a, arr_a, arr_b)
        role_dict[f"listener_{stat_name}"] = np.where(is_a, arr_b, arr_a)

    for t_col in TEMPORAL_COLS:
        role_dict[t_col] = df[t_col].to_numpy()

    role_df = pd.DataFrame(role_dict, index=df.index)

    # Copy identifiers and label columns for downstream evaluation
    for col in ["conv_id", "time_s", "current_speaker", "other_speaker"]:
        if col in df.columns:
            role_df[col] = df[col]
    if "pair" in df.columns:
        role_df["pair"] = df["pair"]

    for col in df.columns:
        if col.startswith("label_"):
            role_df[col] = df[col]

    return role_df
