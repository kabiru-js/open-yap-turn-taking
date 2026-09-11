"""Event construction and turn-taxonomy classification.

A *turn segment* is a maximal run of voice activity in one speaker's
channel (energy VAD with hangover). We classify every speaker change
into a transition type, and we build a per-decision-point data frame
that pairs causal features with a label of "other speaker begins within
Δt".
"""

from __future__ import annotations
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .audio import get_segments


# ── Segment helpers ──────────────────────────────────────────────────────────

def build_segments(
    vad_a: np.ndarray,
    vad_b: np.ndarray,
    hop_ms: float = 10.0,
    max_gap_ms: float = 400.0,
) -> pd.DataFrame:
    """Return a segment table for both speakers.

    VAD frames are grouped into utterances, then within-speaker
    utterances separated by less than ``max_gap_ms`` of silence are
    merged into turns.

    Columns: ``speaker, start_s, end_s, dur_s``.
    """
    rows = []
    for speaker, vad in (("a", vad_a), ("b", vad_b)):
        segs = list(get_segments(vad, hop_ms))
        if max_gap_ms > 0:
            segs = _merge_gaps(segs, max_gap_ms / 1000.0)
        for s, e in segs:
            rows.append({"speaker": speaker, "start_s": s, "end_s": e, "dur_s": e - s})
    df = pd.DataFrame(rows).sort_values("start_s").reset_index(drop=True)
    return df


def _merge_gaps(segs, max_gap: float):
    """Merge consecutive segments of a single speaker separated by <= max_gap."""
    if not segs:
        return segs
    merged = [list(segs[0])]
    for s, e in segs[1:]:
        if s - merged[-1][1] <= max_gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def classify_transitions(segments: pd.DataFrame) -> pd.DataFrame:
    """Classify every speaker-change into the turn-taking taxonomy.

    For each segment i (new), find the previous segment of the *other*
    speaker whose start is at/before the new onset; tag the change.

    Event types:
    - ``clean``            : prior segment ended before the new one began
    - ``backchannel``      : overlap >= 30ms and new segment <= 800ms
    - ``interruption``     : overlap >= 30ms and prior speaker stops <= 800ms
                             after the new onset (floor taken)
    - ``overlap_no_floor`` : overlap >= 30ms and prior speaker keeps talking
    - ``first_segment``    : opening of the conversation for a speaker

    The energy VAD has limited overlap sensitivity; ``n_words`` from ASR
    is merged later in ``merge_asr_event_type`` to refine the taxonomy.
    """
    segs = segments.sort_values(["speaker", "start_s"]).reset_index(drop=True)
    rows = []
    speakers = ["a", "b"]

    # Precompute per-speaker segment lists
    by_spk = {s: segs[segs.speaker == s].sort_values("start_s") for s in speakers}

    for idx, seg in segs.iterrows():
        new_spk = seg["speaker"]
        old_spk = "b" if new_spk == "a" else "a"
        # most recent segment of the other speaker whose START is at/before
        # the new onset (includes the other speaker's still-active segment,
        # enabling overlap / interruption detection)
        cand = by_spk[old_spk]
        cand = cand[cand["start_s"] <= seg["start_s"] + 0.03]
        if len(cand) == 0:
            # no prior segment of the other speaker (conversation opening)
            rows.append({"seg_idx": idx, "type": "first_segment"})
            continue
        prev = cand.iloc[-1]
        overlap_dur = max(0.0, prev["end_s"] - seg["start_s"])

        if overlap_dur <= 0.03:
            event_type = "clean"
        elif seg["dur_s"] <= 0.80:
            event_type = "backchannel"
        elif prev["end_s"] - seg["start_s"] <= 0.80:
            event_type = "interruption"
        else:
            event_type = "overlap_no_floor"

        rows.append(
            {
                "seg_idx": idx,
                "speaker": new_spk,
                "start_s": seg["start_s"],
                "end_s": seg["end_s"],
                "dur_s": seg["dur_s"],
                "prev_speaker": old_spk,
                "prev_start_s": prev["start_s"],
                "prev_end_s": prev["end_s"],
                "prev_dur_s": prev["end_s"] - prev["start_s"],
                "gap_ms": (seg["start_s"] - prev["end_s"]) * 1000.0,
                "overlap_ms": overlap_dur * 1000.0,
                "type": event_type,
            }
        )

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("seg_idx").reset_index(drop=True)
    return out


# ── Word-level enrichment ────────────────────────────────────────────────────

def merge_asr_event_type(events: pd.DataFrame, words: dict) -> pd.DataFrame:
    """Refine ``backchannel`` using ASR word counts.

    words: {speaker: [(word, start, end), ...]} for a conversation.
    Counts the number of words of the new segment starting within the
    segment interval. If segment type was ``backchannel`` but the new
    segment has many words, downgrade to ``overlapping`` (or ``clean``);
    if it is short but with many words, upgrade to ``interruption`` when
    overlapping.
    """
    if len(events) == 0:
        return events
    n_words = np.zeros(len(events), dtype=int)
    first_word = np.array([None] * len(events), dtype=object)

    for s in ("a", "b"):
        ws = words.get(s)
        if not ws:
            continue
        starts = np.array([w[1] for w in ws])
        order = np.argsort(starts)
        ws_sorted = [ws[i] for i in order]
        st_sorted = starts[order]

        mask = events["speaker"] == s
        idx = np.flatnonzero(mask.to_numpy() if hasattr(mask, "to_numpy") else mask)
        if len(idx) == 0:
            continue
        lo = events["start_s"].to_numpy()[idx] - 0.10
        hi = events["end_s"].to_numpy()[idx] + 0.10
        i_lo = np.searchsorted(st_sorted, lo, side="left")
        i_hi = np.searchsorted(st_sorted, hi, side="right")
        cnt = i_hi - i_lo
        n_words[idx] = cnt
        for pos, (j, c) in enumerate(zip(idx, cnt)):
            if c > 0:
                first_word[j] = ws_sorted[i_lo[pos]][0]

    events["n_words"] = n_words
    events["first_word"] = first_word

    mask = events["type"] == "backchannel"
    many_words = events["n_words"] > 4
    events.loc[mask & many_words, "type"] = "overlap_no_floor"
    return events


# ── Decision-point dataset ───────────────────────────────────────────────────

def build_decision_frame(
    conv_id: str,
    segments: pd.DataFrame,
    n_frames: int,
    hop_ms: float = 10.0,
    decision_grid_ms: int = 200,
    horizons_ms: List[int] = None,
    exclude_tail_s: float = 2.0,
) -> pd.DataFrame:
    """Build the per-decision-point dataset for one conversation.

    For every decision time ``t`` on the grid:
    - ``current_speaker``: speaker of the most recent segment (by start)
      overlapping or ending before ``t``.
    - ``last_segment_end``: end of that segment.
    - ``pause_since_last_speech``: t - last speech end (0 if someone is
      speaking at t).
    - For each horizon, ``label_{horizon}`` = 1 if the OTHER speaker has
      a segment onset ``o`` with ``t < o <= t + horizon``.

    The ``current_speaker`` can be e.g. 'a'; the ``other_speaker`` column
    is the complementary speaker.

    Decision points where no one has ever spoken up to ``t`` are dropped.
    """
    horizons = horizons_ms or [200, 500, 750, 1000, 1500]
    t_seconds = np.arange(0, n_frames) * (hop_ms / 1000.0)
    grid = np.arange(
        0, n_frames * hop_ms / 1000.0,
        decision_grid_ms / 1000.0,
    )

    by_spk = {s: segments[segments.speaker == s].sort_values("start_s").values
              for s in ("a", "b")}

    # Precompute sorted onset lists per speaker
    onsets = {
        s: by_spk[s][:, 1]  # start_s column
        for s in ("a", "b")
    }

    rows = []
    for t in grid:
        # current speaker: most recent segment with start <= t and (end >= t OR end < t)
        # i.e., the last segment whose interval starts at or before t.
        best = None
        best_start = -np.inf
        for s in ("a", "b"):
            arr = by_spk[s]
            mask = arr[:, 1] <= t  # start_s <= t
            if mask.any():
                cand = arr[mask][-1]
                if cand[1] >= best_start:
                    best_start = cand[1]
                    best = (s, cand)
        if best is None:
            continue
        cur_spk, seg = best
        other = "b" if cur_spk == "a" else "a"
        seg_end = seg[2]
        pause = max(0.0, t - seg_end)

        row = {
            "conv_id": conv_id,
            "time_s": round(float(t), 3),
            "current_speaker": cur_spk,
            "other_speaker": other,
            "current_seg_start_s": float(seg[1]),
            "current_seg_end_s": float(seg_end),
            "current_seg_dur_s": float(seg[3]),
            "pause_since_last_speech": round(pause, 3),
            "elapsed_s": round(float(t), 3),
        }

        # Labels for each horizon
        o_onsets = onsets[other]
        o_after = o_onsets[o_onsets > t]
        next_other_onset = o_after[0] if len(o_after) else np.inf
        for h in horizons:
            row[f"label_{h}"] = int(next_other_onset <= t + h / 1000.0)

        # Also store time to next other onset (for later calibration /
        # thresholding experiments and event alignment)
        row["next_other_onset_s"] = (
            float(next_other_onset) if np.isfinite(next_other_onset) else np.inf
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    # drop decision points within the last ~2s of the recording (no room
    # for the largest horizon) — handled via inf label as 0, but cleaner to drop
    if len(df):
        df = df[df["time_s"] <= (n_frames - 1) * hop_ms / 1000.0 - exclude_tail_s]
        df = df.reset_index(drop=True)
        df["frame_idx"] = (df["time_s"] / (hop_ms / 1000.0)).round().astype(int)
    return df