"""Turn training, gap merging, and transition taxonomy tests.

Run:  python -m pytest tests -q
"""
import numpy as np
import pandas as pd

from research.audio import get_segments
from research.events import build_segments, classify_transitions


def test_build_segments_merges_short_gaps_only():
    vad_a = np.zeros(80, dtype=bool)
    vad_a[0:5] = True        # (0.00, 0.05)
    vad_a[50:55] = True      # (0.50, 0.55) gap 0.45 > 0.40 -> no merge
    vad_a[60:65] = True      # (0.60, 0.65) gap 0.05 < 0.40 -> merge w/ prev
    segs = build_segments(vad_a, np.zeros_like(vad_a), hop_ms=10.0, max_gap_ms=400.0)
    a = segs[segs.speaker == "a"].sort_values("start_s")
    assert list(np.round(a["start_s"].to_numpy(), 2)) == [0.0, 0.5]
    assert list(np.round(a["end_s"].to_numpy(), 2)) == [0.05, 0.65]


def _df(rows):
    return pd.DataFrame(rows, columns=["speaker", "start_s", "end_s", "dur_s"])


def test_taxonomy_clean():
    segs = _df([
        ("a", 0.0, 1.0, 1.0),
        ("b", 1.5, 2.0, 0.5),
    ])
    ev = classify_transitions(segs)
    assert (ev["type"] == ["first_segment", "clean"]).all()


def test_taxonomy_backchannel():
    segs = _df([
        ("a", 0.0, 5.0, 5.0),
        ("b", 1.0, 1.3, 0.3),   # short new turn, overlapping
    ])
    ev = classify_transitions(segs)
    assert list(ev["type"]) == ["first_segment", "backchannel"]


def test_taxonomy_interruption():
    segs = _df([
        ("a", 0.0, 1.2, 1.2),
        ("b", 1.0, 2.5, 1.5),   # prior speaker stops < 800ms after onset
    ])
    ev = classify_transitions(segs)
    assert list(ev["type"]) == ["first_segment", "interruption"]


def test_taxonomy_overlap_no_floor():
    segs = _df([
        ("a", 0.0, 4.0, 4.0),
        ("b", 1.0, 2.0, 1.0),   # overlap, prior keeps talking > 800ms
    ])
    ev = classify_transitions(segs)
    assert list(ev["type"]) == ["first_segment", "overlap_no_floor"]