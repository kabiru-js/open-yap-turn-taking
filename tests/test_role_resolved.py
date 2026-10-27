"""Tests for role-resolved feature transformation and causal validity."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "experiments" / "EXP-004-role-resolved"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from research.config import Config
from research.window_features import FEAT_COLS
from role_features import (
    ROLE_FEAT_COLS,
    FLOOR_COLS,
    LISTENER_COLS,
    TEMPORAL_COLS,
    ACOUSTIC_ROLE_COLS,
    build_role_resolved_dataframe,
)

DATA = ROOT / "data" / "interim" / "dataset.parquet"


@pytest.fixture(scope="module")
def raw_df():
    if not DATA.exists():
        pytest.skip("dataset.parquet not built")
    return pd.read_parquet(DATA)


@pytest.fixture(scope="module")
def role_df(raw_df):
    return build_role_resolved_dataframe(raw_df)


def test_role_features_shape(raw_df, role_df):
    assert len(role_df) == len(raw_df)
    assert len(ROLE_FEAT_COLS) == len(FEAT_COLS) == 73
    assert len(FLOOR_COLS) == 35
    assert len(LISTENER_COLS) == 35
    assert len(TEMPORAL_COLS) == 3
    assert len(ACOUSTIC_ROLE_COLS) == 70
    for col in ROLE_FEAT_COLS:
        assert col in role_df.columns, f"Missing {col} in role_df"


def test_role_alignment_strict(raw_df, role_df):
    # When current_speaker == 'a', floor must equal 'a' and listener must equal 'b'
    mask_a = (raw_df["current_speaker"] == "a").to_numpy()
    np.testing.assert_array_equal(
        role_df.loc[mask_a, "floor_rms_db_mean"].to_numpy(),
        raw_df.loc[mask_a, "a_rms_db_mean"].to_numpy(),
    )
    np.testing.assert_array_equal(
        role_df.loc[mask_a, "listener_rms_db_mean"].to_numpy(),
        raw_df.loc[mask_a, "b_rms_db_mean"].to_numpy(),
    )

    # When current_speaker == 'b', floor must equal 'b' and listener must equal 'a'
    mask_b = (raw_df["current_speaker"] == "b").to_numpy()
    np.testing.assert_array_equal(
        role_df.loc[mask_b, "floor_rms_db_mean"].to_numpy(),
        raw_df.loc[mask_b, "b_rms_db_mean"].to_numpy(),
    )
    np.testing.assert_array_equal(
        role_df.loc[mask_b, "listener_rms_db_mean"].to_numpy(),
        raw_df.loc[mask_b, "a_rms_db_mean"].to_numpy(),
    )


def test_temporal_features_preserved(raw_df, role_df):
    for col in TEMPORAL_COLS:
        np.testing.assert_array_equal(
            role_df[col].to_numpy(),
            raw_df[col].to_numpy(),
        )


def test_no_leakage_and_clean_columns(role_df):
    for col in ROLE_FEAT_COLS:
        assert not col.startswith("label_")
        assert not col.startswith("next_")
        assert col not in ("conv_id", "time_s", "pair", "current_speaker", "other_speaker")


def test_floor_energy_dominates_listener(role_df):
    # Floor holder must have substantially higher RMS energy on average than listener
    mean_floor = np.nanmean(role_df["floor_rms_db_mean"].to_numpy())
    mean_listener = np.nanmean(role_df["listener_rms_db_mean"].to_numpy())
    # Expect ~60 dB difference
    assert mean_floor > mean_listener + 30.0, f"Expected floor > listener + 30dB, got {mean_floor} vs {mean_listener}"
