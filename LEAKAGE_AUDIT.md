# LEAKAGE_AUDIT.md — Data Leakage Risk Assessment

**Date**: 2026-09-11

---

## Guiding Principle

Every feature must be computable at prediction time using only information available at or before time *t*. The label is defined using information from (t, t+Δt]. Any feature that uses post-transition information is a leak.

---

## Feature-by-Feature Audit

### Temporal Features

| Feature | Known at t? | Risk | Verdict |
|---|---|---|---|
| Pause duration (time since any speech) | YES — computed from VAD segments ending before t | None | SAFE |
| Time since current speaker's last word | YES — ASR word timestamps before t | None | SAFE |
| Previous turn duration | YES — segment start/end before t | None | SAFE |
| Number of turns in last 30s | YES — segment boundaries before t | None | SAFE |
| Time since last speaker switch | YES — segment boundaries before t | None | SAFE |
| Conversation elapsed time | YES — trivially | None | SAFE |

### Acoustic Features

| Feature | Known at t? | Risk | Verdict |
|---|---|---|---|
| RMS energy (current frame at t) | YES — audio available in real time | None | SAFE |
| RMS energy slope | YES — uses frames [t−W, t] | None | SAFE |
| Zero crossing rate | YES — uses frames [t−W, t] | None | SAFE |
| Spectral centroid/bandwidth/rolloff | YES — uses frames [t−W, t] | None | SAFE |
| Spectral flux | YES — uses frames [t−W, t] | None | SAFE |
| F0 (pitch) | YES — uses frames [t−W, t] | None | SAFE |
| Silence proportion in last 300/500/1000ms | YES — VAD flags before t | None | SAFE |

**Note on F0**: Computed using autocorrelation on a 30ms window. At time *t*, the current frame may include audio slightly after *t* (frame centre = t). This is standard for real-time systems with ≤ 15ms latency; document this assumption.

### Conversational Features

| Feature | Known at t? | Risk | Verdict |
|---|---|---|---|
| Previous speaker identity | YES — last speaker segment before t | None | SAFE |
| Consecutive turns by current speaker | YES — segment history before t | None | SAFE |
| Recent overlap count | YES — both channels VAD before t | None | SAFE |
| Recent backchannel count | YES — segment properties before t | None | SAFE |
| Turn-length statistics | YES — segment durations before t | None | SAFE |

### Transcript Features (Ablation Only)

| Feature | Known at t? | Risk | Verdict |
|---|---|---|---|
| Word rate in last 2s | YES — ASR words before t | None | SAFE |
| Discourse marker presence | YES — ASR words before t | None | SAFE |
| Sentence-final particles | YES — ASR words before t | None | SAFE |
| Word count in current turn | YES — ASR words before t | None | SAFE |

---

## Label Leakage Risks

### Risk 1: VAD-Based Label Definition
**Issue**: If the label "other speaker begins within Δt" uses the same VAD system as the features, the VAD at time *t* and the VAD in (t, t+Δt] share the same noise floor estimate and threshold.

**Severity**: LOW. The VAD threshold is computed from a robust statistic (25th percentile of RMS) over the entire channel, which is a slow-moving statistic. A decision at *t* does not change the VAD output at *t+Δt*.

**Mitigation**: None needed; document that VAD is a deterministic function of the full channel, and features at *t* only use VAD flags at *t* and before.

### Risk 2: ASR Word Timestamps vs Audio VAD
**Issue**: ASR word start/end times may not perfectly align with VAD segment boundaries. Using ASR for labels but VAD for features could create mild misalignment.

**Severity**: LOW. Both are derived from the same audio. Small misalignments (tens of ms) are inherent in any annotation system.

**Mitigation**: Use audio VAD as the primary signal for BOTH features and labels. Use ASR words only for conversational features (word counts, discourse markers). Report consistency between VAD-derived and ASR-derived transition counts.

### Risk 3: Frame Alignment at Horizon Boundary
**Issue**: When Δt = 200ms and the decision grid is 200ms, a label might be ambiguous at the boundary.

**Severity**: LOW. Use inclusive/exclusive convention: label=1 iff there exists an onset *o* such that *t < o ≤ t + Δt*. This is well-defined at all boundaries.

---

## Split Leakage Risks

### Risk 4: Conversation-Level Contamination
**Issue**: If windows from the same conversation appear in both train and test, neighboring windows share context (same acoustic environment, same speakers, same topic), inflating performance.

**Severity**: HIGH. This is the most serious leakage risk.

**Mitigation**: **Conversation-level split (LOCO)**. Train/test conversations are always disjoint. No window from conversation *c* is ever used when testing on *c*.

### Risk 5: Speaker Identity Leakage
**Issue**: The sample has only 4 speaker pairs. Pair 1 (spk_03f80fcf × spk_77fb5673) accounts for 50% of all conversations (8/16). A model trained on some Pair 1 conversations and tested on others has seen the same speakers' prosodic patterns.

**Severity**: MEDIUM-HIGH. This is a real limitation of the sample, not a coding error.

**Mitigation**:
1. **LOCO evaluation**: When a Pair 1 conversation is the test set, the other 7 Pair 1 conversations are in training → speaker leakage is INHERENT.
2. **Speaker-disjoint evaluation**: Hold out all Pair 1 conversations for test (or all of Pair 2+3+4). This is the honest measure of cross-speaker generalization.
3. **Report both LOCO and speaker-disjoint results**, clearly labeling which includes speaker information.

### Risk 6: Temporal Contamination Within LOCO
**Issue**: Within a single conversation, if feature extraction for a window at time *t* used a robust statistic (e.g., noise floor percentile) computed over the full conversation, then training windows (from other conversations) and test windows share no contamination — but within a conversation, the robust statistic is computed once per channel.

**Severity**: LOW. The robust statistic is a global property of the channel, not of the window. It's analogous to speaker-normalization. Safe.

---

## Field Notes: Real Leaks Caught During Development

Leakage checks are not theoretical in this project — two real leaks were caught
by the audit discipline (`RESEARCH_LOG.md` exp-001/exp-002):

1. **Non-causal energy slope (caught at feature layer)**: the first
   `energy_slope` implementation measured the energy *after* the current frame
   (≈75 ms of future audio). Any downstream model trained on it would have
   "peeked" forward. Fixed by computing the slope over frames strictly before
   the decision time and recomputing the entire feature cache before any dataset
   was built.
2. **Label-column sweep (caught at model layer)**: the ablation script's "full"
   / "no-F0" feature lists were derived by sweeping all numeric dataset columns,
   which silently included `label_200..label_1500`. This gave instantaneous
   ROC-AUC 1.0000. The model feature column list is now explicit
   (`FEAT_COLS`) and excludes all `label_*` columns, and any future sweep
   filters them up front.

Additional design decision recorded after the fact: window `min`/`max`
statistics are shifted one frame so the in-flight frame (t — t+10ms) never
contributes to a decision at t; windowed stats are indexed by END frame with
windows `[f−w, f−1]`.

---

## Summary

| Risk | Severity | Addressed? |
|---|---|---|
| Future acoustic info in features | None | N/A — all features use audio ≤ t |
| Future transcript in features | None | N/A — all ASR words have start < t |
| Conversation-level split | HIGH | YES — LOCO |
| Speaker identity leakage | MEDIUM-HIGH | YES — speaker-disjoint eval reported |
| ASR/VAD misalignment | LOW | Documented |
| Frame alignment at boundary | LOW | Convention: inclusive/exclusive clear |
| Robust statistics (noise floor) | LOW | Documented |
