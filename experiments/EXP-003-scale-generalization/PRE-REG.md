# EXP-003 — Pre-Registered Design: Signal Survival at Scale (full Open Yap 1K)

- **Status.** DESIGN ONLY. No code, no data, no evaluation has been run for EXP-003.
- **Research log entry.** `exp-006` (reserved).
- **Date.** 2026-09-15. Signatures/commits made after this date are pre-registered decisions only.
- **Relation to prior experiments.** EXP-001 built the 73-feature LR/RF/GBDT turn-onset pipeline on the
  hand-picked 16-conversation sample (dubbed *the sample*). EXP-002 bounded its annotation/VAD robustness.
  EXP-003 tests whether the signal **survives the full corpus** (~1,602 conversations, ~239 speakers,
  ~1,000 h, all relationships) — the scale and speaker-mix the study ultimately claims.

---

## 1. Research question and hypotheses

**RQ.** Does the turn-onset predictive signal measured in EXP-001 (LR per-fold mean ROC 0.7148,
pooled 0.7077; GBDT 0.7523; RF 0.7620 at h=1000 ms) persist — meaningfully above chance — when the
identical pipeline is evaluated on the full Open Yap 1K corpus (~112× the sample's hours, ~30× the
speaker count, all relationship types)?

- **Primary hypothesis (H1).** The full-corpus *pooled* ROC-AUC at h=1000 ms (LR) satisfies
  `cluster-bootstrap 95% CI lower bound > 0.60`. The 0.60 threshold is the pre-registered "meaningful
  survival" floor: 10 ROC points above the 0.50 chance floor, and beyond the EXP-002 shuffle/uniform
  controls (0.5007 / 0.4956). Selection is deliberately effect-size based, not p-value based:
  with ≈18 M decision points *any* residual signal is statistically significant, and significance
  alone would validate a worthless model.
- **Null hypothesis (H0).** Full-corpus pooled ROC-AUC @1000 ms (LR) has CI upper bound ≤ 0.60, i.e.
  the signal is not meaningfully above chance at scale and the sample result did not survive
  generalization.
- **Degradation / non-inferiority (H2).** H1 *and* `lower bound of CI(Δ) ≥ −0.10`, where
  `Δ = full-corpus pooled ROC − 0.7077 (EXP-001 pooled LR)`. Margin 0.10 is pre-registered from:
  (a) EXP-001 per-fold SD ≈ 0.072, (b) the pair-LOCO drop (LR 0.7148→0.623, ≈ −0.09) as the measured
  cost of a strict speaker-honest split. If Δ CI does not justify equivalence, the result is reported
  as **persistent but materially degraded**, not as "persist".
- **Secondary predictions (exploratory, stated now, no peeking).**
  - GBDT and RF survival: does the ensemble advantage hold at scale (EXP-001: GBDT +0.038 over LR)?
  - Dyad-disjoint strict-generalization ROC (see §4): the honest upper bound of the claim.
  - Horizon monotonicity (LR): longer windows → larger ROC, as in EXP-001 (0.757→0.691).
  - Base-rate drift report (sample 0.1501 @1000 ms vs full corpus).
  - Top-feature stability (GBDT importance ordering) vs EXP-001.

**The answer is allowed to be no.** A flat-lining, degraded, or chance-level result at scale is a
valid, publishable finding and will be reported as such. Nothing in this design margin-shares for a
positive result.

---

## 2. Data

- **Full corpus.** Open Yap 1K v1.0 (The Agentic Data Company): ~1,000 h, **1,602 conversations**,
  **239 speakers**, 48 kHz, channel-separated (one file per speaker), word-level Deepgram Nova-3
  transcripts, JSON metadata identical in schema to the sample (verified against both sample shards;
  sample profile in `RESEARCH_LOG.md` exp-000/exp-006). Relationships span friends 70.9%,
  colleagues 10.8%, romantic partners 9.8%, family 8.5%. Per-track `effective_bandwidth_hz`,
  DNSMOS BAK, loudness and clipping metadata, plus conversation-level `turns_per_minute`,
  `turn_taking_gap_ms`, `speech_dominance`. Access under the Open Yap 1K Data Use Agreement.
- **Access status.** Full-corpus download is not present in `data/`; must be obtained by the author
  via the DUA request. Until then, EXP-003 stays DESIGN ONLY.
- **Sample vs corpus caveat (pre-registered).** The Hub sample is *hand-picked*, not a random draw
  of the corpus (all friends, high-quality). The sample therefore likely overstates the corpus
  average. This design treats "worse at scale" as an expected, reportable outcome, not a bug.
- **Sample self-check (unit test before corpus runs).** Ingestion + pipeline on the known sample must
  reproduce EXP-001's stored parquet row count and pooled ROC (0.7077) to within ±0.005. If it does
  not, the port is broken and must be fixed *before* any corpus evaluation; no corpus run may proceed
  on a failing repro.
- **Concentration check (reported).** Top-1 contributor ≈1.8 %, top-10 ≈17.9 % of the corpus. The
  cluster bootstrap (§8) treats this explicitly; the concentration figure is reported alongside CIs.

---

## 3. Preprocessing and feature extraction (frozen, byte-identical intent)

Everything below is **reused unchanged** from the production pipeline; the port may only wrap I/O
(corpus tar/PCM layout → the exact per-conversation npz/parquet schema). No DSP or feature change is
permitted to improve results. `FEAT_COLS` (73) is a frozen symbol.

- Audio: 48 kHz; frame features at 10 ms hop, 30 ms frames, causal window `[t−1.5 s, t)`.
- VAD: energy threshold `max(noise_floor_metric, p25(rms)) + 20 dB`, hangover 6 frames,
  min-speech 200 ms, max gap within speaker 400 ms (identical to EXP-001/EXP-002).
- Events/labels: `label_h(t) = next_other_onset_s <= t + h/1000`, `h ∈ {200,500,750,1000,1500}`;
  decision grid 200 ms; 2 s tail censoring preserved.
- Dialectic model fit: `LogisticRegression(C=1.0, max_iter=2000)` on `StandardScaler` + median
  imputation; `HistGradientBoosting(max_iter=160, learning_rate=0.12)`; `RandomForestClassifier(
  n_estimators=80, min_samples_leaf=25)`; majority baseline. No tuning — parameters are the EXP-001
  constants.
- Train cap: stratified per-fold cap `n=90,000`, `seed=42` — identical rule to EXP-001/002 so the
  per-fold numbers remain comparable. The cap's influence is reported as `n_used_per_fold`.

---

## 4. Splits and unit of independence

**Unit of independence = conversation** for all within-corpus statistics: decision points are
temporally autocorrelated (10 ms grid, 1.5 s overlapping windows) and are nested inside
conversations, which are nested inside speaker dyads, which repeat. **All aggregation, bootstrap and
variance estimates must be at the conversation (or dyad) level, never at the decision-point level.**

Conversation- and speaker-level *grouping* is therefore **required**, both to (a) build valid folds
and (b) draw valid inference.

| Split | Purpose | Construction |
|---|---|---|
| **S1 — primary: conversation-grouped 10-fold CV** | Comparative + inferential backbone (mirrors EXP-001's conversation-level independence, but with 1,602 conversations → 10 folds, not LOCO) | Fold assignment of `conversation_id` deterministic `seed=42`, stratified on coarse bins: relationship (5 classes) × bandwidth class (`effective_bandwidth_hz < 9k` share: none / some) × duration quartiles. No conversation appears in two folds. Speaker overlap across folds is permitted and *expected* — this split deliberately inherits SpeakerLeak as the EXP-001-analogous number. |
| **S2 — secondary: dyad-disjoint strict generalization** | The honest "unseen people" claim: no speaker ever seen in training may appear in test | Unordered speaker pairs (dyads) are split by deterministic `seed=42` into 20 % hold-out / 80 % train. Test = conversations whose **both** speakers belong to hold-out dyads; train = conversations both ∈ train; conversations with one speaker on each side are **excluded from both and counted** in the report. Expected test ≈ 60–90 conversations (≈ (0.20)² of 1,602). Evaluated with a model fit on full corpus at default cap. |
| **S3 — diagnostic subset** | Isolate scale vs speaker-mix vs hand-picking | (a) friends-only strata of S1 reported separately vs others; (b) per-conversation ROC scatter vs EXP-001 sample conversations. |

Fold construction, dyad assignment, cap seeds, and all random state are frozen at `seed=42` unless a
pre-registered amendment changes them. **No split is regenerated after seeing any EXP-003 test result.**

---

## 5. Baselines and chance controls

- **Majority baseline.** Predict the constant positive-rate; ROC 0.50, PR-AUC = observed base rate.
  Reported per split so base-rate drift is visible.
- **Shuffle control (within-conversation label permutation).** The EXP-002 lesson applies verbatim:
  only *within-conversation* label permutation is a valid chance floor. A naive *global*
  onset permutation reintroduces the `elapsed_s` t-profile and inflates ROC (measured 0.77 in
  EXP-002 and discarded as confounded). The at-scale shuffle control is computed on a pre-registered
  subset (3 folds of S1) to keep compute bounded.
- **Uniform control.** Bernoulli labels at the observed base rate (same subset as shuffle).
- Both controls must land ≈0.50; if not, the harness is broken and the corpus run stops.

---

## 6. Models, seeds, and evaluation protocol

- Models: LR (primary, = EXP-001 headline model), GBDT (secondary), RF (secondary), majority
  (baseline). All frozen hyperparameters as §3.
- Seeds: split `42`, cap `42`, GBDT/RF `random_state=42`. Optional secondary robustness sweep:
  seeds {41,42,43} *may* be run for LR only if budget allows (any such run is reported, never
  cherry-picked).
- Evaluation points: **all held-out decision points at @1000 ms** from the fold being evaluated;
  pooled-across-folds ROC/PR and per-fold mean ± SD both reported (EXP-001 conventions emit both:
  pooled 0.7077 vs per-fold mean 0.7148 — both must be given).
- Primary metric for the hypothesis tests = **pooled ROC-AUC @1000 ms (LR)**.
- No test-labels peeking, no early stopping on a favorable looking checkpoint. Evaluation of a given
  status only after the full pre-registered budget for that status is consumed. The pipeline runs to
  completion per splitting schedule; partial output is stored in append-only JSON (`exp003_*.json`)
  with per-component timestamps.

---

## 7. Compute feasibility and the pre-registered budget rule

Full corpus ≈ 18 M decision points and ~1,000 h of feature computation. The local box (7.9 GB RAM)
will not ingest that in one shot. Rather than silently changing the question, the budget and any
hardware decision are **pre-registered here**:

- **B1 Target run:** full corpus on adequate hardware (feature build fold-parallel, fold evaluation
  streaming, never load > 1 conv of frames into RAM at once, identical to the production per-fold
  low-memory path).
- **B2 Fallback staged protocol (if B1 hardware/budget is unavailable):** draw pre-registered
  contiguous conversation slices S1..Sk (each ~160–200 conversations, ~100–125 h) built *before any
  evaluation*; evaluate each stage on its own S1 CV; never tune anything between stages against a
  prior stage's test numbers; final stage aggregates all held-out predictions via the §8 bootstrap.
  Each stage is a pre-committed unit, not an opportunistically extended run.
- **Stopping rule.** Stop exactly when the pre-registered budget (data slices or compute-hours
  authorized in the amendment) is consumed, or all stages complete — whichever is committed first.
  No optional stopping on a favorable trend. Any deviation requires a timestamped pre-registration
  amendment in this file *before* it affects any evaluation.

---

## 8. Confidence intervals and statistical tests

- **Primary inference tool: conversation-level cluster bootstrap** — resample *conversations* with
  replacement from the held-out conversation pool (each conversation pushes its full (label,
  predicted-score) block), recompute pooled ROC-AUC on the resampled blocks, 10,000 replicates,
  percentile CI. This respects autocorrelation and the 17.9 % top-contributor concentration. If the
  bootstrap fails numerically, fall back to 1,000-replicate bootstrap **and say so**.
- **H1 test.** Is the 95 % CI lower bound > 0.60?
- **H2 test.** CI on Δ vs the fixed EXP-001 pooled reference (0.7077): is lower bound ≥ −0.10?
  (Difference is a single-population Δ, *within* the full corpus; EXP-001 acts as a fixed threshold,
  not a resampled population, because the sample is hand-picked and not a random subset.)
- Per-fold mean ± SD reported as the EXP-001-style descriptive counterpart.
- **No DeLong/point-level tests** anywhere — they are invalid under clustering.

---

## 9. Comparisons vs EXP-001 (exact table contract)

| Quantity | EXP-001 (sample, h=1000) | EXP-003 (full corpus) |
|---|---|---|
| LR pooled ROC | 0.7077 | S1 (primary) |
| LR per-fold mean (SD) | 0.7148 (0.0716) | S1 |
| GBDT per-fold mean | 0.7523 | S1 |
| RF per-fold mean | 0.7620 | S1 |
| LR pair/strict split | 0.623 | S2 (dyad-disjoint, 20 %) |
| base rate @1000 ms | 0.1501 | S1 + S2 |
| LR horizon 200/500/750/1000/1500 | 0.757/0.737/0.718/0.708/0.691 (pooled) | S1 pooled |

Keep the same table shape and metric ordering; no new metrics invented to flatter the result.

---

## 10. The strongest ways EXP-003 could mislead (and how the design catches each)

1. **Point-level significance inflation.** 18 M points make every residual signal "significant".
   → Hypotheses are **effect-size gated** (§1) and all intervals are **conversation-cluster
   bootstrap** (§8). No p-values on the decision path.
2. **Speaker leakage making "generalization" fake.** Conversation-CV lets the same speaker appear in
   train and test; EXP-001's pair-LOCO already showed the −0.09 cost. → S2 dyad-disjoint split is
   mandatory reporting, and H2's margin encodes the expected honest drop.
3. **A "positive" port artifact.** Changing the I/O wrapper silently changing DSP is the #1 risk.
   → Sample repro unit test (§2) must reproduce EXP-001 (0.7077 ± 0.005) before any corpus run;
   FEAT_COLS frozen; any deviation stops the experiment.
4. **Harness flooring at 0.50 being confused with occlusion.** A chance-floor on shuffle/uniform
   (≈0.50) is required (§5); a *naive global* escalated to per-conversation permutation so the floor
   stays honest.
5. **Optional stopping / peeking.** → Pre-registered budget + stopping rule (§7); any amendment is
   timestamped and precedes affected runs; results appended to `.json`, not edited.
6. **Base-rate drift misread as model change.** Corpus gaps are longer (p50 580 ms vs sample ≈440 ms)
   and overlap higher (p50 8.3 % vs 3.6 %). → base rate is reported in every table; comparisons are
   ROC/PR relative to their own base rate.
7. **Hand-picked sample bias masquerading as "scale effect".** The sample is friends-only and edited.
   → S3 reports friends-only strata and per-conversation scatter; "worse at scale" is an expected
   outcome, not signaled as failure.
8. **Top-contributor concentration.** One contributor ≈1.8 %, ten ≈17.9 %. → cluster bootstrap +
   concentration reported beside every CI.
9. **ASR/annotation jitter at scale.** Word timing degrades on noisier real rooms. → EXP-002 Arm A
   showed ≤σ100 ms jitter is free for LR; the corpus-view report includes the per-track
   `effective_bandwidth_hz`/DNSMOS profile as context. No re-tuning is permitted to compensate.
10. **Equating within-corpus ROC with honest generalization.** → the S2 number is the claim that
    trains the thesis; where S1 ≫ S2 the gap is stated in the abstract, not explained away.

---

## 11. Deliverables and non-deliverables

- **Will produce.** `experiments/EXP-003-scale-generalization/` (this design; later: runner(s) that
  reuse production modules read-only, a `results/exp003_*.json` append-only log, `exp003_*.png/pdf`
  figures, and a findings `README.md` exactly following EXP-002's audit style).
- **Will NOT produce or do** (binding). No edits to EXP-001/EXP-002 code or stored results. No
  hyperparameter or feature search. No model search beyond the frozen LR/GBDT/RF. No test-set-driven
  iteration. No deleting or editing of recorded numbers. No cache warm-ups tuned after seeing
  results. Full-corpus data kept out of `git` (adapter + metadata summarized, audio never committed).

---

## 12. Decision rule for the thesis

| Outcome | Statement justified by data |
|---|---|
| H1+H2 pass | Signal **persists** at scale and is within 0.10 of the sample estimate |
| H1 pass, H2 fail | Signal persists, **materially degraded** at scale (strength lost, existence retained) |
| H1 fail, control checks ok | Signal does **not** survive scale; EXP-001 was sample-specific (a real, publishable negative) |
| Control checks fail | Experiment invalid; no thesis claim until the harness is fixed and re-run under a new pre-registration |

---

## 13. Verification checklist — items to confirm once corpus access is granted (`PENDING_ACCESS`)

Every item below is checked exactly once when the full corpus arrives. Any mismatch that changes the
design is recorded as a pre-registration amendment with a new timestamp. Items that already hold for
the sample (marked `[sample-verified]`) require only a same/different confirmation.

### 13.1 Data format

- [ ] **PENDING_ACCESS** Full corpus shard layout: confirm files are `shard-*.tar` with per-member
  `a.*` / `b.*` audio + a `json` metadata member (sample-verified: FLAC, 48 kHz, dual-track).
- [ ] **PENDING_ACCESS** Audio container: confirm PCM WAV (not FLAC) in the full corpus; verify
  `soundfile.read()` handles the format without resampling; both tracks identical length per
  conversation.
- [ ] **PENDING_ACCESS** Sample rate: confirm 48 000 Hz everywhere; if any conversation is a
  different rate, resample to 48 000 and flag in the report.

### 13.2 Metadata schema

- [ ] **PENDING_ACCESS** Top-level JSON keys present in every conversation: `conversation_id`,
  `duration_seconds`, `language`, `relationship`, `sample_rate`, `speakers` (list of 2),
  `speech_dominance`, `topics`, `transcripts`, `turn_taking_gap_ms`, `turns_per_minute`.
- [ ] **PENDING_ACCESS** Per-speaker keys present in every speaker object: `speaker_id`,
  `gender`, `age_range`, `country`, `accent`, `native_language`, `education_level`, `device`,
  `echo_cancellation`, `headphones`, `avg_wpm`, `audio_metrics`, `recording`, `transcript`.
- [ ] **PENDING_ACCESS** `audio_metrics` sub-keys present: `noise_floor_dbfs`,
  `silent_while_partner_spoke_pct`, `effective_bandwidth_hz`, `dnsmos` (BAK sub-field).
- [ ] **PENDING_ACCESS** `transcripts.words[]` present for both speakers: each word contains
  `word`, `start`, `end`, `type` (word/filler/laugh/cough/noise).
- [ ] **PENDING_ACCESS** `type` field values: confirm presence of `{word, filler, laugh, cough,
  noise}` — if `filler`/`laugh`/`cough` are absent, that is acceptable (sample has all five but
  full corpus may not).

### 13.3 Speaker and conversation identifiers

- [ ] **PENDING_ACCESS** `speaker_id` is stable and pseudonymous across conversations (same speaker
  in multiple conversations has the same ID).
- [ ] **PENDING_ACCESS** `conversation_id` is unique and stable (each conversation has exactly one).
- [ ] **PENDING_ACCESS** Each conversation contains exactly 2 speakers (no 3+ person conversations
  leaked into the corpus).

### 13.4 Distribution and concentration

- [ ] **PENDING_ACCESS** Count total conversations and speakers; confirm consistency with published
  claims (~1 602 conversations, ~239 speakers). Any deviation > 5% is flagged.
- [ ] **PENDING_ACCESS** Relationship distribution: compute from metadata; compare to published
  (friends 70.9%, colleagues 10.8%, romantic 9.8%, family 8.5%). Any deviation > 3pp is flagged.
- [ ] **PENDING_ACCESS** Top-contributor concentration: compute top-1 and top-10 speaker %
  contribution (by hours). Confirm top-1 ≈ 1.8%, top-10 ≈ 17.9%.
- [ ] **PENDING_ACCESS** Conversation duration distribution: compute min/median/max. Compare to
  published (median 30 min, mean 37.5 min, long tail past 100 min).

### 13.5 Speaker dyad structure

- [ ] **PENDING_ACCESS** Build the dyad graph: map each unordered `(speaker_a, speaker_b)` pair to
  its set of conversations. Report: number of unique dyads, conversations per dyad distribution
  (min/median/max), number of single-conversation dyads.
- [ ] **PENDING_ACCESS** Confirm dyad structure supports the S2 dyad-disjoint split: if any dyad
  has > 50 conversations, flag as a potential fold-imbalance issue.

### 13.6 Channel integrity

- [ ] **PENDING_ACCESS** Verify the `[t, t + duration]` anchor (both tracks share a timeline):
  for a random sample of 10 conversations, confirm `len(a_audio) == len(b_audio)`.
  The published claim is 100% conformance.
- [ ] **PENDING_ACCESS** Verify effective bandwidth: for a random sample of 10 conversations,
  confirm `effective_bandwidth_hz` is present and > 0 for both tracks.

### 13.7 Feature pipeline self-check

- [ ] **PENDING_ACCESS** Ingest 2 full-corpus conversations (randomly selected) through the
  production pipeline (scripts 01→02→04); confirm: feature parquet row count, `FEAT_COLS` (73)
  present, no NaN-column collapse, `pause_since_last_speech` and `elapsed_s` columns present.
- [ ] **PENDING_ACCESS** Ingest the known sample through the same pipeline path; confirm the output
  parquet matches the existing stored parquet row-for-row (or within tolerance if any format
  difference exists). If mismatch: stop, diagnose, fix before proceeding.

### 13.8 Sample repro unit test (mandatory before any corpus evaluation)

- [ ] **PENDING_ACCESS** Run the EXP-003 pipeline on the known 16-conv sample with all splits and
  models as specified in `config_exp003.yaml` (using conversation-grouped 10-fold on the sample's
  16 conversations). Must reproduce:
  - LR per-fold ROC mean within ±0.005 of 0.7148
  - GBDT per-fold ROC mean within ±0.005 of 0.7523
  - RF per-fold ROC mean within ±0.7620 ±0.005
  - Pooled LR ROC within ±0.005 of 0.7077
  If any model deviates by > 0.005, the port is broken and corpus runs are prohibited.

### 13.9 On completion

Each checked item above is stamped with the date it was verified and the outcome (same/different/NA).
The completed checklist is appended to this PRE-REG.md as a block-quoted section so that the full
audit trail is in one file. Any amendment arising from a mismatch is timestamped in the pre-register
header section above.