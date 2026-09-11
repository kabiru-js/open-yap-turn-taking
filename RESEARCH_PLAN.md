# RESEARCH_PLAN.md — Turn-Taking Prediction

**Date**: 2026-09-11  
**Dataset**: Open Yap 1K sample (16 conversations, 8 speakers, 4 speaker pairs, 8.92h)

---

## 1. Research Question

**Primary:** Can we predict an upcoming conversational turn transition from the acoustic and temporal information available immediately before the new speaker begins talking?

**Formally:** Given audio history up to time *t*, estimate

$$P(\text{other speaker begins within } \Delta t \mid \text{history up to } t)$$

where *t* is sampled at 200ms intervals across the conversation timeline.

---

## 2. Competing Hypotheses

### H1: Acoustic Signal Hypothesis (PRIMARY)
Immediately before a speaker begins, the acoustic environment changes measurably: the preceding speaker's energy trails off, pitch may fall or plateau, spectral flux increases, and the silence duration exceeds a characteristic threshold. These acoustic cues carry predictive information beyond what pure timing provides.

**Prediction:** Acoustic features achieve ROC-AUC > 0.60 at 500–1000ms horizons, significantly above timing-only baselines.

### H2: Timing-Only Hypothesis
The strongest signal is simply how long the preceding speaker has been silent. The acoustic details matter less than the timing structure. Pause duration is sufficient for short-horizon prediction.

**Prediction:** Timing-only features achieve ROC-AUC > 0.55 at 200–500ms horizons, but performance degrades rapidly at longer horizons.

### H3: Conversational Structure Hypothesis
Recent turn-taking history (switch frequency, turn durations, overlap patterns) carries independent information beyond what acoustic and timing features capture, because conversation has a rhythm that encodes expectations.

**Prediction:** Adding conversational features improves AUC by ≥0.02 over acoustic+timing combined, and the improvement is concentrated at 750–1500ms horizons.

### H4: Per-Speaker Signature Hypothesis (NEGATIVE EXPECTATION)
Given that the sample contains only 4 speaker pairs (two of which dominate), much of the apparent predictive signal may reflect speaker-specific prosodic patterns rather than general turn-taking cues. The model may be learning speaker identity rather than conversational structure.

**Prediction:** Speaker-disjoint evaluation shows substantially lower AUC than conversation-disjoint evaluation.

### H5: No Free Lunch Hypothesis (NULL)
The sample is too small (16 conversations, 4 speaker pairs) and the ASR timestamps too coarse (40ms grid) to learn a generalizable turn-taking predictor. Simple baselines may outperform ML models due to overfitting.

**Prediction:** No ML model consistently and meaningfully outperforms the timing-only baseline across conversations in leave-one-out evaluation.

---

## 3. Event Taxonomy

Based on audio VAD segments + ASR word timestamps, classify transitions into:

| Type | Definition |
|---|---|
| **Clean transition** | Speaker A segment ends → gap ≥ 0ms → Speaker B segment starts; no temporal overlap |
| **Overlapping transition** | B starts while A is still active (overlap ≥ 30ms); both may continue |
| **Interruption** | B starts during A's turn, and A's segment ends within 800ms of B's start |
| **Backchannel** | B's segment is short (≤ 800ms) and contains ≤ 3 ASR words; typically overlaps with A |
| **Same-speaker continuation** | Next segment by same speaker after a gap (no transition) |
| **Ambiguous** | Both speakers begin within a short window (≤ 200ms gap), or VAD-derived segments disagree with ASR |

For prediction: **binary label** y=1 if other-speaker onset occurs within Δt of time *t*; y=0 otherwise. Event subtype is used for error analysis, not as prediction target.

---

## 4. Feature Layers (in dependency order)

### Layer T: Temporal
- Pause duration (time since previous speech ended in ANY channel)
- Time since current speaker's last word
- Previous turn duration
- Number of turns in last 30s
- Time-since-last-speaker-switch
- Conversation elapsed time

### Layer A: Acoustic
All computed from the **current speaker's channel** (known at prediction time):
- RMS energy (dB, current value + slope)
- Zero crossing rate
- Spectral centroid, bandwidth, rolloff
- Spectral flux
- F0 (pitch) — voiced frames only, interpolated
- Energy slope (short-term trend)
- Proportion of silence in last 300ms, 500ms, 1000ms

### Layer C: Conversational
- Previous speaker identity (binary: A or B)
- Number of consecutive turns by current speaker
- Recent overlap count (last 5 turns)
- Recent backchannel count (last 5 turns)
- Turn-length statistics (mean, variance of last N turns)

### Layer X: Transcript (where available, for ablation only)
- Word rate in last 2s
- Discourse marker presence ("uh", "um", "you know", "right")
- Sentence-final particles ("so", "and then", "but")
- Word count in current unfinished turn

---

## 5. Baseline Models

| ID | Features | Model | Description |
|---|---|---|---|
| B-A | None | Majority | Predict most common class (always 0 or always 1) |
| B-B | Layer T | LogisticRegression | Timing only |
| B-C | Layer A | LogisticRegression | Acoustic only |
| B-D | T + A | LogisticRegression | Acoustic + timing |
| B-E | T + A + C | LogisticRegression | All features |
| M-1 | T + A | RandomForest(200) | Non-linear acoustic + timing |
| M-2 | T + A | GradientBoostedTrees(100) | Non-linear acoustic + timing |
| M-3 | T + A + C | GradientBoostedTrees(100) | All features, non-linear |

If any model achieves ROC-AUC > 0.65 with meaningful PR-AUC, consider:
- M-4: Small MLP on T+A features
- M-5: Temporal CNN on feature sequences (1.5s lookback)

---

## 6. Evaluation Protocol

### Primary: Leave-One-Conversation-Out (LOCO)
For each conversation *c*:
1. Train on all other 15 conversations
2. Predict on *c*
3. Pool predictions across all test conversations

Report: ROC-AUC, PR-AUC, calibration (reliability diagram), F1, precision/recall at fixed thresholds.

### Secondary: Speaker-Disjoint
Hold out all conversations by one pair:
- **Test on Pair 1** (8 convs): most data, likely easiest for speaker-specific patterns
- **Test on Pairs 2-4** (7 convs): reveals whether model generalizes to unseen speaker pairs

### Horizons
Evaluate at Δt ∈ {200ms, 500ms, 750ms, 1000ms, 1500ms}. Report AUC × horizon curve.

### Metrics
- ROC-AUC (primary ranking metric)
- PR-AUC (sensitive to class imbalance)
- Precision, Recall, F1 at optimal threshold (Youden's J)
- Calibration: Expected Calibration Error (ECE), Brier score
- Confusion matrix at base rate

---

## 7. Feature Importance & Ablation

1. **Ablation by layer**: Remove each feature layer (T, A, C) and measure AUC delta
2. **Per-feature importance**: RF feature importances, SHAP values for top features
3. **Feature correlations**: Identify collinear features; report VIF
4. **Feature stability**: Do the same features matter across conversations?

---

## 8. Pre-Turn Signature Analysis

Align all transition events at t=0 (onset of new speaker). Compute mean/median trajectories over [-1.5s, +0.5s] for:
- RMS energy (both channels)
- F0 (both channels)
- Spectral flux
- Silence proportion
- Pause duration (cumulative)

Plot with 95% bootstrap CI. Compare:
- Clean transitions vs interruptions vs backchannels
- Positive examples (transition imminent) vs negative examples (matched duration, no transition)

This figure addresses: "Is there a consistent acoustic signature immediately preceding turn onset?"

---

## 9. Leakage Risks (Summary)

See LEAKAGE_AUDIT.md for full audit. Key risks:
1. **Future feature contamination**: Must not use post-onset information in features
2. **Conversation-level leakage**: Must not train and test on windows from the same conversation
3. **Speaker identity leakage**: Must not rely on speaker-specific patterns for generalization
4. **ASR word leakage**: ASR words at time *t* use a 40ms lookback; the word ending at *t* started at *t−0.04s*, so only using words with start < t is safe
5. **VAD leakage**: VAD is frame-level on audio; using only VAD frames at or before *t* is safe

---

## 10. Expected Failure Modes

| Failure mode | Mitigation |
|---|---|
| Sample too small → overfitting | LOCO evaluation, report variance |
| Speaker-specific patterns → poor generalization | Speaker-disjoint evaluation |
| ASR word timestamps too coarse | Use audio VAD for primary event timing |
| Bluetooth bandwidth variation | Report per-track bandwidth; exclude narrowband from spectral analyses |
| Class imbalance (transitions are sparse) | Report PR-AUC alongside ROC-AUC; use class weights |
| Energy threshold sensitivity in VAD | Report VAD sensitivity analysis (±3dB threshold) |
| Noisy ASR → false word counts | Use audio segment lengths as primary proxy |

---

## 11. Compute & Storage Requirements

| Item | Size | Notes |
|---|---|---|
| Raw tar archives | 1.9 GB | Already downloaded |
| Feature cache (per-conversation) | ~200 MB | Float32, 10ms grid, both channels |
| Cached model predictions | ~1 MB | CSV per experiment |
| Figures | ~5 MB | 7 publication figures |
| RAM for feature extraction | ~1.5 GB | One conversation at a time |
| RAM for model training | ~2 GB | One LOCO fold at a time |
| Total disk footprint | ~2.2 GB | Including raw data |

---

## 12. Experiment Sequence

| # | Script | Description | Status |
|---|---|---|---|
| 0 | `00_dataset_audit.py` | Dataset statistics and validation | DONE |
| 1 | `01_build_features.py` | Compute + cache frame features for all conversations | TODO |
| 2 | `02_define_events.py` | Detect transitions, classify into taxonomy, save events | TODO |
| 3 | `03_descriptive.py` | Descriptive statistics: turn durations, gaps, overlap | TODO |
| 4 | `04_signature.py` | Pre-turn acoustic signature (central figure) | TODO |
| 5 | `05_build_dataset.py` | Construct labeled windows from events + features | TODO |
| 6 | `06_baselines.py` | Baseline models B-A through B-E | TODO |
| 7 | `07_models.py` | ML models M-1 through M-3 + horizon comparison | TODO |
| 8 | `08_ablation.py` | Feature layer ablation study | TODO |
| 9 | `09_error_analysis.py` | Error analysis: failure modes by event type | TODO |
| 10 | `10_figures.py` | Generate all publication figures | TODO |
| 11 | `11_report.py` | Generate final paper (turn-taking-analysis.md) | TODO |
