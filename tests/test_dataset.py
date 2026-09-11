"""Dataset invariants (skipped until `scripts/04_build_dataset.py` has run).

Run:  python -m pytest tests -q
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.window_features import FEAT_COLS

DATA = Path(__file__).resolve().parent.parent / "data" / "interim" / "dataset.parquet"


@pytest.fixture(scope="module")
def df():
    if not DATA.exists():
        pytest.skip("dataset.parquet not built; run scripts/04_build_dataset.py first")
    return pd.read_parquet(DATA)


def test_dataset_shape(df):
    assert df.shape[1] == 110
    assert 160000 <= len(df) <= 161000


def test_decision_grid(df):
    times = df["time_s"].to_numpy()
    d = np.round(np.diff(np.sort(np.unique(times))), 3)
    assert (d == 0.2).all()


def test_labels_are_binary(df):
    for h in (200, 500, 750, 1000, 1500):
        assert set(df[f"label_{h}"].dropna().unique()) <= {0, 1}


def test_base_rate_1000(df):
    assert abs(df["label_1000"].mean() - 0.15008) < 0.005


def test_loco_unit_is_conversation(df):
    assert df["conv_id"].nunique() == 16
    assert "current_speaker" in df.columns
    assert "other_speaker" in df.columns


def test_all_feat_cols_present(df):
    missing = [c for c in FEAT_COLS if c not in df.columns]
    assert missing == []


def test_no_label_leak_into_feature_columns(df):
    feat = set(df.columns) & set(FEAT_COLS)
    assert feat == set(FEAT_COLS)
    assert not any(c.startswith("label_") for c in feat)


def test_results_match_readme():
    """Guard the headline claims against the committed results."""
    root = Path(__file__).resolve().parent.parent
    s = json.load(open(root / "results" / "summary.json"))
    assert abs(s["gbdt_roc_1000"] - 0.750) < 0.02
    assert abs(s["base_rates"]["1000"] - 0.15008) < 0.005
    loco = json.load(open(root / "results" / "models_loco_1000.json"))
    rf = loco["1000"]["models"]["rf"]["roc_auc"]
    assert abs(rf - 0.762) < 0.02