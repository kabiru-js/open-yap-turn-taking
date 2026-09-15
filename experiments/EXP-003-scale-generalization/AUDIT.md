# EXP-003 — Methodological Audit of EXP-001 and EXP-002

This document records strengths, weaknesses, confounders, and disclosure gaps in EXP-001
(turn-pipeline benchmark) and EXP-002 (annotation/VAD robustness). It is not an evaluation result
and contains no empirical claims. It is the basis for why EXP-003 takes the protocol it does.

Date: 2026-09-15. Revisions are append-only.

---

## 1. Verified EXP-001 numbers (from stored results)

All figures below are source of truth for cross-references; none were re-derived for this audit.

| Metric | Value | Source |
|---|---|---|
| n_decision_points | 160 295 | summary.json |
| n_events (onset ≤ t + 1 s) | 13 643 | summary.json |
| base_rate @1000 ms | 0.150 08 | summary.json |
| LR pooled ROC @1000 ms | 0.707 67 | summary.json |
| LR pooled PR @1000 ms | 0.300 01 | summary.json |
| LR per-fold ROC mean ± SD | 0.714 76 ± 0.071 58 | models_loco_1000.json |
| LR per-fold PR mean ± SD | 0.280 80 ± 0.041 17 | models_loco_1000.json |
| GBDT per-fold ROC | 0.752 27 ± 0.065 27 | models_loco_1000.json |
| RF per-fold ROC | 0.761 98 ± 0.067 36 | models_loco_1000.json |
| Majority ROC | 0.500 00 | models_loco_1000.json |
| Majority PR | base rate | models_loco_1000.json |
| LR pair-LOCO ROC | 0.623 | RESEARCH_LOG exp-000/001 |
| GBDT pair-LOCO ROC | 0.672 | RESEARCH_LOG exp-000/001 |
| Ablation LR @1000 ms (temporal-only) | 0.668 | RESEARCH_LOG exp-000/001 |
| Ablation LR @1000 ms (a-acoustic) | 0.607 | RESEARCH_LOG exp-000/001 |
| Ablation LR @1000 ms (b-acoustic) | 0.600 | RESEARCH_LOG exp-000/001 |
| Ablation LR @1000 ms (f0-only) | 0.637 | RESEARCH_LOG exp-000/001 |
| Ablation LR @1000 ms (no-f0) | 0.714 | RESEARCH_LOG exp-000/001 |
| LR horizon sweep (200/500/750/1000/1500) | 0.757/0.737/0.718/0.708/0.691 | summary.json |
| Top features | current_seg_dur_s, pause_since_last_speech | RESEARCH_LOG exp-001 |

**Pooled vs per-fold discrepancy.** LR pooled 0.7077 ≠ per-fold mean 0.7148. Both are valid
descriptors; the discrepancy reflects fold-unequal weighting and imprecision of per-fold mean as
an estimator of the pooled quantity. EXP-003 must report both.

**SD as an outlier flag, not a valid CI.** The SD of 0.072 over 16 conversation folds is the
spread of 16 single-conversation holdouts. It is NOT a standard error; it has no clear inferential
use. A conversation-level bootstrap would be the correct interval; none was computed in EXP-001.

---

## 2. EXP-001 methodological strengths

1. **Causal feature window.** `[t−1.5 s, t)` respects the predictive direction; the model never
   sees future acoustic data.
2. **LOCO design avoids within-conversation autocorrelation in train/test.** Each conversation is
   entirely in train or test; this prevents the most trivial leakage.
3. **Stratified train cap (90 k, seed 42).** Keeps memory bounded and makes each fold comparable
   across seeds; the cap is a documented, reproducible choice, not an ad-hoc intervention.
4. **Multi-horizon design.** Pre-registered horizons {200,500,750,1000,1500} ms; the monotonic
   pattern (longer window → lower ROC, higher base rate) is internally consistent and a good
   diagnostic.
5. **Ablation structure.** Temporal-only, a-only, b-only, f0-only, no-f0 ablations cleanly
   decompose the source of information.
6. **Pair-LOCO secondary split.** The honest speaker-disjoint result (LR 0.623, GBDT 0.672) is
   already in the record; the ~0.09 gap between conversation-LOCO and pair-LOCO is a real
   measure of speaker-specific signal leakage.
7. **Full event taxonomy.** Clean/backchannel/interruption/overlap counts documented (summary.json
   event_type_counts); every claim can be decomposed by event class.

---

## 3. EXP-001 weaknesses and confounders

### 3.1 Generalizability — the single largest weakness

- **N = 16 conversations, 4 speaker pairs.** Statistical spread (SD 0.072) is the variation
  across 4 independent social dynamics. This is too few to draw population-level claims;
  the result is a *description of 4 dyads* and nothing more.
- **Hand-picked sample.** The Hub sample is not a random draw of the corpus; it was curated for
  quality. All 16 conversations are from the "friends" relationship stratum. The sample likely
  overstates the corpus average (EXP-003 §1 acknowledges this).
- **All same-language, one domain.** No acoustic diversity beyond the 8 speakers' environments;
  no non-friend, non-English, or cross-cultural conversations.
- **Full-corpus key variables not exercised.** Speech dominance ranges 0.31–0.70 in the sample
  (p5/p50/p95 in the full corpus: 0.27/0.56/0.81). TPM ranges 2.7–19.8 (full: 5.3/11.9/20.0).
  Overlap 3.6% median (full: 8.3%). The sample under-represents both the easy and hard tails.

### 3.2 Speaker and pair structure not exploited

- **Conversation-CV leaks speakers.** Same speaker participates in multiple training and test
  conversations; the pair-LOCO gap (0.715 → 0.623) proves the leak inflates the headline number.
  The headline 0.715 is *not* a generalization number; it is a *within-pipeline* diagnostic.
- **No speaker-level statistics reported.** Is the signal carried by one speaker's style? The
  ablations suggest temporal features dominate; no speaker-demographic stratification was done.
- **Pair clustering not modeled.** 4 dyads → 4-fold "cross-validation" for the honest number.
  Pair-LOCO 16-fold CV over dyads was done; a genuine dyad-level bootstrap CI was not.

### 3.3 Temporal structure confounder

- **`elapsed_s` is a feature.** It encodes position within the conversation. The ablation shows
  temporal-only features achieve 0.668 — the largest single ablation component. The model may
  partly be predicting "how far into the conversation are we" rather than "is the other person
  about to speak." EXP-002's shuffle/uniform controls showed that the full 0.715 *does* require
  feature-label alignment (shuffle collapses to 0.50); but the *relative* contribution of
  temporal rhythm vs acoustic content is unmeasured at full-corpus scale.
- **`current_seg_dur_s` and `pause_since_last_speech` are the top features.** Both are temporal.
  Acoustic features (spectral flux, f0) add marginal lift (no-f0 vs full: 0.714 vs 0.714).
  The model is largely a temporal rhythm detector; this is honest but the acoustic "monitoring"
  narrative in the thesis is overstated.

### 3.4 Metric and reporting issues

- **No bootstrap CIs anywhere in EXP-001.** The per-fold SD is the only variance measure; it is
  not an interval estimate and should not be interpreted as one.
- **Pooled ROC vs per-fold mean ambiguity.** EXP-001 reports both; the headline 0.7148 is the
  per-fold mean, not the pooled value (0.7077). Which is the right number to cite in the thesis?
  The pooled number is the one that describes the pipeline's actual aggregate performance; the
  per-fold mean is a descriptive summary. EXP-003 pre-registers the pooled number as primary.
- **No per-conversation performance reporting.** 16 conversations vary widely (3.7–19.8 TPM;
  0.31–0.70 dominance); their individual ROCs are not reported. Some conversations may be
  near-chance; some may be easy. The mean obscures this.
- **No cost curve, calibration, or threshold analysis.** Only AUC is reported; the practical
  operating point (e.g., at what false-positive rate would a full-duplex system be usable?) is
  not analyzed.
- **GBDT vs RF not explained.** GBDT 0.752 vs RF 0.762 — why does a simpler tree ensemble
  (RF) outperform the gradient-boosted one? Given the tiny sample size, this may be noise or
  an overfitting/regularization interaction. Not discussed.

### 3.5 Model and feature selection

- **No hyperparameter search.** LR C=1, GBDT 160/0.12, RF 80/leaf-25 — these were set in
  an earlier exploration (EXP-000 era) and never revisited. This is *methodologically safe* (no
  test-set tuning) but leaves open the question of whether the relative ranking of LR/GBDT/RF
  is an artifact of specific hyperparameters, not model family. The pre-registered EXP-003
  freezes these as-is.
- **73 features, no explicit selection.** The feature set was defined by pipeline construction
  (a/b rms 6 stats, a/b 7 channels × 4 stats, a/b voiced_frac, pause, elapsed_s, current_seg).
  No feature selection or regularization beyond LR's L2 was applied. LR's median imputation +
  StandardScaler is reasonable; RF/GBDT handle missing features implicitly. No feature
  interaction analysis was done.

---

## 4. Verified EXP-002 numbers

All from `results/exp002_arm_a.json` and `results/exp002_arm_b.json` (21 tests pass; 15 distinct
ROC values across 18 cells; no poison).

### Arm A — annotation onset jitter

| condition | ROC | PR | base rate |
|---|---|---|---|
| clean (anchor = EXP-001) | 0.7148 | 0.2808 | 0.1501 |
| σ25 ms (seeds 101/202/303) | 0.7143/0.7148/0.7152 | 0.2800–0.2809 | 0.1495 |
| σ50 ms (101) | 0.7141 | 0.2797 | 0.1496 |
| σ100 ms (101/202/303) | 0.7138/0.7144/0.7137 | 0.2788–0.2793 | 0.1492–0.1496 |
| σ200 ms (101/202/303) | 0.7104/0.7116/0.7123* | 0.2752–0.2755 | 0.1488–0.1489 |
| σ400 ms (101) | 0.7034 | 0.2620 | 0.1470 |
| shift −200 ms | 0.7030 | 0.3005 | 0.1757 |
| shift +200 ms | 0.7270 | 0.2561 | 0.1226 |
| within-conv label shuffle | 0.5007 | 0.1316 | 0.1501 |
| uniform Bernoulli | 0.4956 | 0.1499 | 0.1504 |

\* seed 303 σ200: ROC 0.7123 computed; JSON checkpoint lost on kill; value is consistent with
other seeds but not stored.

### Arm B — VAD segmentation margin (LR-only)

| cell | ROC | PR | base rate |
|---|---|---|---|
| Δ=0 anchor (reused clean) | 0.7148 | 0.2808 | 0.1501 |
| Δ=+3 dB consistent | 0.7126 | 0.2680 | 0.1423 |
| Δ=−3 dB consistent | 0.7147 | 0.2867 | 0.1539 |
| Δ=−3 dB drift (clean labels) | 0.7176 | 0.2834 | (160 295 rows) |

### Confounder discovered and corrected

A naive global onset-shuffle produced ROC 0.7678 (above clean), because the `elapsed_s` feature
creates a time profile that correlates with the globally permuted labels. This was identified,
documented in RESEARCH_LOG exp-005, and replaced by the correct within-conversation label
permutation (which collapses to 0.5007). The incident is a live example of the temporal
confounder identified in §3.3 above.

---

## 5. EXP-002 strengths

1. **Arms address two distinct failure modes** (label corruption vs segmentation drift) and
   together bound the "predicting the annotation" critique.
2. **Chance controls are decisive.** The shuffle/uniform collapse to 0.50 confirms the signal
   is real speech structure, not a harness artifact. This is the strongest single result in
   the paper.
3. **Pre-registered within-conversation permutation protocol** (after correcting the confounded
   global shuffle) preserves both marginal and temporal profile of labels.
4. **Drift cell (Δ=−3 dB, clean labels)** is the right test for a real-world mismatch scenario;
   it shows *improvement* (0.7176), confirming the VAD margin is not a fragile threshold.
5. **Arms B computed from first principles** (VAD re-derived, not a synthetic corruption);
   the margin shifts are realistic.

---

## 6. EXP-002 weaknesses and limits

1. **LR-only.** All 18 robustness cells use LR. GBDT/RF robustness under jitter and margin
   shifts is untested; the model-family invariance claim is assumed, not demonstrated.
2. **Limited jitter grid.** σ ∈ {25,50,100,200,400} is a reasonable log-grid, but the grid
   has 1–3 seeds per cell, not a full Monte Carlo. Sampling uncertainty within cells is
   approximately ±0.001 (measured from the σ25 seeds: range 0.001).
3. **Shift tested at ±200 ms only.** The shift test conflates two effects: (a) base-rate
   change (more/fewer positives in the window) and (b) onset-timing error. The ROC shift
   (−0.012/+0.012) is anti-correlated with the base-rate change; the model's ranking behavior
   is largely preserved. A finer shift grid (e.g., ±50, ±100) would have been cleaner but is
   not critical given the σ results already bound timing noise.
4. **Arm B: only one drift cell.** Δ=+3 dB drift was not computed (memory; disclosed, not faked).
   The Δ=−3 drift cell (0.7176) is slightly *better* than anchor — an interesting but
   undiagnosed result. Without the +3 drift cell, the asymmetry of drift behavior is unmeasured.
5. **Over-sample-level noise.** The sample's median overlap is 3.6% of voiced time; the full
   corpus's is 8.3%. Arm B's VAD sensitivity may differ at higher overlap; this cannot be
   measured without the corpus.
6. **No GBDT chance control.** The shuffle/uniform controls were run with LR; GBDT on shuffled
   labels was not computed. Given GBDT's flexibility, it is *theoretically possible* it could
   overfit the marginal label distribution and beat 0.50 on shuffled data (unlikely but not
   measured). A single GBDT shuffle cell on the sample would close this gap cheaply.

---

## 7. What EXP-003 must therefore address (protocol implications)

| Weakness / confounder | How EXP-003 addresses it |
|---|---|
| N=16/4 dyads, hand-picked sample | Full corpus: 1602 convs, 239 speakers, all relationship types |
| Conversation-CV leaks speakers | Dyad-disjoint S2 split (both speakers unseen); report gap |
| No bootstrap CIs | Cluster bootstrap (conversation-level) with 10k resamples |
| Pooled vs per-fold ambiguity | Report both; pooled as primary for hypothesis test |
| `elapsed_s` temporal confounder | Per-conversation shuffle at scale (3 folds); S2 split partially controls |
| No per-conversation breakdown | Per-conversation ROC scatter plot; relationship strata report |
| LR-only robustness (EXP-002) | Run GBDT/RF on the same frozen corpus pipeline |
| Acoustic contribution overclaimed | Re-run ablations on full corpus (secondary); friend-only strata report |
| No cost/threshold analysis | Report per-fold PR-AUC; practical threshold analysis as tertiary |

---

## 8. Disclosure: what was and was not computed

### Verified and stored
- EXP-001: all models_loco_1000 numbers, summary.json, pair-LOCO, horizons, ablations
- EXP-002: all 18 arm-a cells, 4 arm-b cells, within-conv shuffle, uniform, 21 passing tests

### Disclosed but not stored / partially computed
- EXP-002 Arm A: seed 303 σ200 (ROC 0.7123, computed, JSON not saved)
- EXP-002 Arm A: seeds 202/303 at σ400 not run
- EXP-002 Arm B: Δ=+3 drift, Δ=0 rebuild, any GBDT cell not run
- EXP-002: GBDT on shuffled labels not run

### Never computed (honest gaps)
- EXP-001: conversation-level bootstrap CIs
- EXP-001: per-conversation ROC breakdown
- EXP-002: RF under corruption
- EXP-002: any model other than LR in any robustness cell
- EXP-003: nothing (design only; no data, no evaluation, no placeholder results)

### Commit history
- `e5220aa` — EXP-001 base pipeline (21 files, the original research)
- `e21934c` — EXP-002 annotation robustness (21 files, 4552 insertions, pushed to origin/main)

Both commits pass `pytest -q` (21 tests). No test file was modified to accommodate a failing test;
all 21 tests reflect guard conditions that were designed *before* their respective runs.

---

## 9. Summary for the thesis

The ML participant in this paper is real: the signal exists (shuffles collapse to 0.50), is
robust to annotation jitter (σ ≤ 400 ms costs −0.01), and is robust to VAD margin drift (±3 dB
costs < 0.003). The honest generalization number — pair-LOCO — is substantially lower (0.623 LR,
0.672 GBDT), indicating that part of the headline 0.715 reflects speaker-specific patterns
conflated across train and test. The acoustic contribution is marginal; the model is predominantly
a temporal rhythm detector (temporal-only ablation 0.668 vs full 0.714; f0-only 0.637).

EXP-003 tests whether this signal survives scale. A positive result at 1,602 conversations with
speaker-disjoint evaluation would constitute a genuine generalization claim. A negative result
would mean the EXP-001/002 numbers describe 4 dyads and nothing more — also a real finding.
