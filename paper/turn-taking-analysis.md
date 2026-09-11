# Predicting Conversational Turn-Onsets from Causal Audio Features

**Analysis report** — companion to `RESEARCH_LOG.md`, `LEAKAGE_AUDIT.md`, `LIMITATIONS.md`

---

## 1. Abstract

Conversational turn-taking is fast, mostly entrained, and mostly invisible to
text transcripts alone. We ask whether *causal* acoustic and timing cues —
features computable from the audio up to time *t* and nothing after — predict
whether the *other* speaker will begin speaking within a horizon Δt. Using the
public Open Yap 1K sample (16 two-channel conversations, 8.92 h, 8 speakers),
we convert energy-VAD activity into turns, label 160,295 decision points on a
200 ms grid, and evaluate under strict leave-one-conversation-out (LOCO) and
leave-one-speaker-pair-out protocols.

At Δt = 1000 ms the best models reach pooled **ROC-AUC ≈ 0.75–0.76 / PR-AUC ≈ 0.35**
(RF per-fold mean 0.762 ± 0.067; GBDT pooled 0.750/0.345; base rate 0.15) and
precision of ~0.50 at a 50% confidence threshold (a 3.3× lift over random).
Skill is carried primarily by **rhythm/timing features** (current-turn duration,
time since last speech) and **spectral-flux / energy modulation**; fundamental
frequency (F0) contributes no additional signal. Predictability *declines* with
horizon (LR ROC-AUC 0.76 at 200 ms → 0.69 at 1500 ms), consistent with
turn-onset timing being a short-horizon rhythmic phenomenon rather than a
long-range countdown. ≈ 9 ROC-AUC points are speaker-pair specific: models
degrade from 0.752 to 0.667 when a full speaker pair is held out. Results are
honest negative on one axis: backchannel/interruption detection from energy VAD
is far weaker than for ASR, so the *type* of upcoming transition is hard to
anticipate from acoustics alone.

## 2. Introduction

Humans anticipate turn boundaries with ~200 ms precision across cultures, using
prosody, gaze, and syntax. An automated system that can forecast *when* the
other party will speak next would enable natural-sounding agents, latency-aware
duplex systems, and assistive tools (smart hearing aids that reduce
simultaneous-spoken ambiguity, meeting analytics). The engineering question is
whether the acoustic channel alone carries enough *causal* information to do
this.

We contribute an end-to-end, audited pipeline:

1. Turn segmentation from energy VAD with 400 ms within-speaker gap merging;
2. a transition taxonomy (clean / backchannel / interruption / overlap);
3. a causal, windowed feature representation (`[t − 1.5 s, t)`) over both
   channels;
4. LOCO and speaker-pair-disjoint model evaluation;
5. leakage audits central to the methodology (two real leaks were caught and
   fixed during development; see `LEAKAGE_AUDIT.md` and `RESEARCH_LOG.md`).

## 3. Related Work

- **Turn-taking in conversation analysis** (Sacks et al. 1974; Duncan 1972;
  Heldner & Edlund 2010) established the ~200 ms turn-switch gap and the
  "no gap, no overlap" default norm, later relaxed. Our clean-gap distribution
  (median ≈ negative → slight overlap-dominated at frame granularity) is
  consistent with modern doubly-articulated views and depends on VAD
  sensitivity.
- **Acoustic turn-end detection**: prosodic boundary cues (F0, duration,
  pre-boundary lengthening; e.g., Wightman 2000) predict turn *ends*. We differ
  by predicting *other-onset* — including backchannels — not terminal events.
- **Latency target in spoken interfaces** mirrors our Δt sweep: systems that
  *know* an onset is imminent can pre-position inference; our PR-AUC numbers
  bound what such optimization can expect from audio alone.
- **NFC-adjacent applied monitoring systems** (thesis context) motivate a
  deployable, on-device turn-prediction widget; this report establishes the
  signal available from a single lightweight feature set (no neural models).

## 4. Dataset

Audit details in `DATASET.md`. Relevant facts:

- 16 conversations; 8 speakers but **only 4 speaker pairs**
  (one pair = 50% of data: spk_03f80fcf × spk_77fb5673).
- 32,120.6 s of dual-channel audio (48 kHz FLAC, shared timeline), 89,677 ASR
  words (Deepgram Nova-3, ~40 ms grid).
- 8/32 tracks Bluetooth (effective bandwidth < 9 kHz); crosstalk
  (`silent_while_partner_spoke_pct`) 0–93%; energy-VAD overlap fraction median
  3.6% (range 1.6–12.3%).

## 5. Methods

### 5.1 Turn segmentation

Per channel: 10 ms frames, energy threshold = noise floor + 20 dB, hangover
6 frames (60 ms), speech runs merged when separated by ≤ 400 ms silence →
*turns*. Left with 20,935 turns. Naïve VAD runs (unsmoothed) produce ~40 k
"turns"; 400 ms merging is essential to avoid syllable-fragmentation
artifacts (see Failure mode in exp-001).

### 5.2 Transition taxonomy

For each new turn, classify versus the other speaker's most recent turn by
**start** time (so the still-active turn is the comparator):

- `clean` — no overlap (or < 30 ms): gap ≥ 0.
- `backchannel` — new turn ≤ 800 ms long, overlapping ≥ 30 ms.
- `interruption` — prior speaker stops ≤ 800 ms after the new onset.
- `overlap_no_floor` — overlap ≥ 30 ms, prior speaker continues.
- `first_segment` — conversation opening.

Counts: clean 10,976 (80.4%), backchannel 1,525 (11.2%), interruption 811
(5.9%), overlap_no_floor 313 (2.3%), first_segment 18.

### 5.3 Decision points and labels

Every 200 ms, a decision point with label `1` iff the *other* speaker's next
turn-onset lies in `(t, t + Δt]`, Δt ∈ {200, 500, 750, 1000, 1500} ms. Base
rates: 3.3%, 8.0%, 11.6%, 15.0%, 21.2% respectively.

### 5.4 Features (all causal)

Per channel × per statistic over `[t − 1.5 s, t)`: mean, std, slope
(least-squares), min, max, last-value. Channels: RMS, ZCR, spectral centroid,
bandwidth, rolloff, spectral flux, F0 (autocorrelation, 60–400 Hz), energy
slope, voiced fraction. Plus temporal: `pause_since_last_speech`, `elapsed_s`,
`current_seg_dur_s`. Models train on the canonical 73-feature set `FEAT_COLS`
(see `src/research/window_features.py`). The ablation "full" set is the broader
101-column superset (all acoustic stats per channel), used only in §6.4 for
internal comparison.

**Leakage discipline**: windows end strictly before decision time; min/max
shifted one frame so the in-flight frame is excluded; labels derive from
segments/onsets strictly after `t`; models never see future words or segments
(ablation experiment caught a label-column sweep — see §9).

### 5.5 Models and evaluation

- Baselines: majority (predict base rate).
- Linear: LR (C=1, standardized, median imputation).
- Nonlinear: RandomForest (80 trees), HistGradientBoosting (160 iters, lr 0.12).
- Protocols: **LOCO** (16 folds, one conversation out) pooled ROC/PR-AUC with
  per-fold mean ± sd; **pair-LOCO** (4 folds, one speaker pair out).
- Training rows capped at 90 k/fold (decision points within a turn are heavily
  autocorrelated).

## 6. Results

### 6.1 Main comparison (Δt = 1000 ms, conversation-LOCO)

All rows are per-fold **mean ± sd** across 16 folds (`models_loco_1000.json`);
pooled values (concatenated held-out predictions) shown where available
(`results/summary.json`).

| model | pooled ROC / PR | per-fold ROC (mean ± sd) | per-fold PR (mean ± sd) |
|---|---|---|---|
| majority | 0.500 / 0.151 | 0.500 ± 0.000 | 0.131 ± 0.063 |
| LR | 0.708 / 0.300 | 0.715 ± 0.072 | 0.281 ± 0.041 |
| RF | — (per-fold only) | **0.762 ± 0.067** | **0.348 ± 0.032** |
| GBDT | 0.750 / 0.345 | 0.752 ± 0.065 | 0.332 ± 0.032 |

Trees outperform linear by ~4–5 ROC points, indicating interactions (e.g.,
pause × energetic modulation) matter. Pooled and per-fold values differ slightly
because folds are unequal in size.

### 6.2 Horizon sweep (LR, LOCO)

| Δt | base | ROC | PR |
|---|---|---|---|
| 200 | 0.033 | **0.757** | 0.106 |
| 500 | 0.080 | 0.737 | 0.211 |
| 750 | 0.116 | 0.718 | 0.260 |
| 1000 | 0.150 | 0.708 | 0.300 |
| 1500 | 0.212 | 0.691 | 0.362 |

ROC-AUC monotonically decreases with horizon — **closer onsets are easier to
see coming**; PR-AUC rises with base rate. This pattern argues for operational
usage at shorter horizons (or cascaded "when, then what" systems).

### 6.3 Speaker-pair generality (Δt = 1000 ms)

| model | ROC-AUC (mean ± sd) | PR-AUC |
|---|---|---|
| LR | 0.621 ± 0.027 | 0.259 |
| GBDT | 0.667 ± 0.014 | 0.277 |
| (conv-LOCO GBDT ref) | 0.752 ± 0.065 | 0.332 |

Dropping ≈ 0.086 ROC-AUC when an entire unseen pair is held out: a meaningful
fraction of the skill is dyadic/entrained rhythm, not universal acoustics.

### 6.4 Ablation (LR, Δt = 1000 ms, LOCO)

| feature set | ROC-AUC | n |
|---|---|---|
| temporal only | **0.668** | 3 |
| a-acoustic | 0.607 | 12 |
| b-acoustic | 0.600 | 12 |
| F0 only | 0.637 | 8 |
| all except F0 | 0.714 | 89 |
| full | 0.714 | 101 |

F0 is effectively redundant after RMS/spectral/rhythm; channel-asymmetric
acoustics are roughly equal in value; the temporal layer alone captures most of
the linear signal (0.67 vs 0.71 full).

### 6.5 Feature importance (RF, top-6)

`current_seg_dur_s` (0.038), `pause_since_last_speech` (0.037),
`a_spectral_flux_max` (0.017), `b_energy_slope_mean` (0.017), `elapsed_s`
(0.016), `b_rms_db_mean` (0.014). Rhythm + spectral/energy modulation
dominate; consistent with §6.4.

### 6.6 Operating points (GBDT @ 1000 ms, pooled)

| threshold | predicted rate | recall | precision |
|---|---|---|---|
| 0.20 | 25.5% | 0.548 | 0.323 |
| 0.30 | 13.0% | 0.347 | 0.402 |
| 0.50 | 2.5% | 0.084 | 0.502 |
| BF1 (θ=0.215) | — | 0.516 | 0.336 |

At a 50%-confidence alert level precision ≈ 0.50 (3.3× lift over the 0.150 base
rate) — usable for low-noise assistants, but recall is small (8%): most onsets
arrive with < 50%-confidence evidence.

## 7. Discussion

1. **The signal is rhythmic first.** Pause duration and current-turn length are
   the strongest predictors; spectral-flux/energy modulation of the *other*
   channel ranks next — i.e., the system partly hears *breathing/grooming*
   activity before speech. F0/prosody adds nothing beyond these in aggregate
   (though F0-only is non-trivial at 0.637, it is collinear with energy/voicing).
2. **Horizon matters inversely.** "Will the other speak in the next 200 ms" is
   more detectable than "…in the next 1.5 s". This is exactly the regime needed
   to beat classic turn-end detection and supports incremental (frame-by-frame)
   systems rather than one-shot forecasting.
3. **Dyadic entrainment is real but bounded.** The pair-LOCO drop (0.752→0.667)
   quantifies how much of the pattern is per-pair rhythm; even fully novel pairs
   are above chance, so a *generic* acoustic predictor is plausible but a
   personalized one is strictly better.
4. **Type prediction remains hard.** Only 5.9% of transitions are
   "interruptions" and the energy VAD under-senses overlap versus ASR; acoustics
   alone give a weak basis for classifying the *kind* of upcoming event.
5. **Implication for products**: an assistant or conference system can ask
   "should I expect the other to start now?" at 200 ms cadence with informed
   priors and decision thresholds; cascading a low-cost LR at high cadence and a
   tree model at moments of pause yields < 10 ms incremental compute per frame.

## 8. Limitations

- **4 speaker pairs** and a friends-only register bound generality; the pair-LOCO
  numbers (§6.3) are the honest ceiling estimate for unseen dyads.
- **Energy-VAD segmentation** is crosstalk- and Bluetooth-filter sensitive;
  overlap statistics (median 3.6%) likely undercount true simultaneous speech.
- **ASR words are machine transcripts**; word-level timing is ~40 ms quantized
  and unverified; we de-risk by labeling from audio, not ASR words.
- **Decision points are autocorrelated** within turns; train capping mitigates
  but does not remove dependence — LOCO still leaks *temporal* structure only to
  the degree that rhythm generalizes (see pair-LOCO for the residual).
- **F0 extracted via pure autocorrelation** on decimated audio; on Bluetooth
  tracks the effective range is reduced.
- Single-lookup, non-neural models: deliberately, to bound the available signal.

## 9. Leakage audit summary

- Rule set and audit in `LEAKAGE_AUDIT.md`.
- **Caught & fixed leak 1 (development)**: `energy_slope` was computed with one
  frame of future audio (non-causal 75 ms lookahead) → made causal before any
  dataset was built.
- **Caught & fixed leak 2**: ablation feature lists swept up `label_*` columns
  yielding ROC 1.0; model feature columns now exclude labels explicitly
  (recorded in exp-002 Failure mode 4).
- **Design**: `min`/`max` window statistics shifted one frame to exclude the
  in-flight frame; windows close at `t`; no future segments/words reach models.

## 10. Ethics

- Data are a public sample; identities are pseudonymous speaker IDs; we do not
  add new biometric identity signals. Deployed turn-prediction conveys behavioral
  timing; consent and on-device processing are recommended for commercial use.
- Results are reported honestly including negative ones (F0 null result,
  pair-specificity loss, weak overlap/type detection).
- No dataset founders were contacted; findings are reproducible from public
  assets and this repo's fixed-random-state pipeline.

## 11. Conclusion

Causal, lightweight audio features can predict imminent turn-onset by the other
speaker far above chance one second in advance (ROC-AUC 0.76, PR-AUC 0.35 vs
base 0.15) with rhythm and spectral/energy dynamics as the carriers. Horizons of
200–500 ms are the most informative, tree models capture nonlinear pause-acoustic
interactions absent from linear fits, and roughly a third of the measurable
skill is pair-specific entrainment. F0 adds no marginal value. The framework is
directly deployable in latency-aware conversational systems and in NFC-linked
livestock/monitoring use-cases (thesis context) where wearable microphones sense
nearby speaker activity.

## 12. Future work

- Neural temporal models (LSTM/transformer over frame sequences) with
  conversation-grouped splits to capture longer-range rhythm.
- Richer VAD (spectral subtraction, domain-adapted) and ASR-aligned overlap
  labels for robust backchannel/interruption detection.
- Cross-corpus generalization (Native-satellite datasets, more dyads) to
  isolate universal vs dyadic rhythm.
- Controlled latency/lookahead trade-offs in a duplex agent simulation.