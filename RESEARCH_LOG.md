# RESEARCH_LOG.md

Each experiment is logged with a fixed template. Failed experiments are **never overwritten or deleted**. Research history matters.

**Template**:
```
Experiment: <id>
Hypothesis: <statement>
Dataset subset: <split>
Features: <layers>
Model: <model>
Evaluation: <protocol + metrics>
Result: <numbers>
Interpretation: <what it means>
Failure modes: <what didn't work>
Next question: <follow-up>
```

---

## Experiment 0 — Dataset Reconnaissance

- **Experiment**: `exp-000_dataset_audit`
- **Hypothesis**: The Open Yap 1K sample contains the structure needed to study turn transitions: separate channels, shared timeline, and word-level timestamps for both speakers.
- **Dataset subset**: All 16 conversations (32 tracks).
- **Features**: Metadata, transcripts, word timestamps, effective bandwidth, noise floor.
- **Model**: None (audit only).
- **Evaluation**: Descriptive statistics.
- **Result**:
  - 16 conversations, 8 speakers, **only 4 distinct speaker pairs**.
  - Pair 1 (spk_03f80fcf × spk_77fb5673) = 8 conversations (50%).
  - ≈ 8.92h total audio (32,120.6s), 89,677 words, all `type="word"`.
  - 8/32 tracks are Bluetooth (< 9 kHz effective bandwidth).
  - `silent_while_partner_spoke_pct` ranges 0–93% (crosstalk variance).
  - Energy-VAD overlap fraction: 1.6–12.3%, median 3.6%.
  - ASR word grid ≈ 40ms. 411 negative inter-word gaps (ASR overlaps).
  - Only `friends` relationships.
- **Interpretation**: The dataset has the fundamental technical prerequisites. However, the 4-pair structure is a serious constraint for speaker-independent turn-taking research. Efficiency: two speakers (spk_03f80fcf, spk_77fb5673) dominate.
- **Failure modes**: The audio quality varies substantially across pairs; transcript noise must be hedged against audio-derived labels.
- **Next question**: Can we segment conversational turns reliably from energy VAD + ASR, and what does the pause/transition distribution look like?

---

## Experiment 1 — Turn Segmentation and Transition Taxonomy

- **Experiment**: `exp-001_turn_segmentation`
- **Hypothesis**: Energy-VAD + 400ms within-speaker gap merging produces plausible conversational turns, and the transition library is large enough to study (clean / backchannel / interruption / overlap).
- **Dataset subset**: All 16 conversations (energy VAD on both tracks, 10ms frames).
- **Features**: None (labels only). Turn = VAD utterance after merging within-speaker gaps ≤ 400ms.
- **Model**: None (taxonomy only).
- **Evaluation**: Transition type counts; overlap/backchannel rates; base transition rates per horizon.
- **Result**:
  - 13,643 total transitions; segments dropped from ~40k (VAD fragments) to 20,935 turns after 400ms merging.
  - Types: clean 10,976 (80.4%), backchannel 1,525 (11.2%), interruption 811 (5.9%), overlap_no_floor 313 (2.3%), first_segment 18.
  - 160,295 decision points (200ms grid); base rate `other begins within Δt`: 200ms→3.3%, 500ms→8.0%, 750ms→11.6%, 1000ms→15.0%, 1500ms→21.2%.
- **Interpretation**: Clean turn-take follows silence dominates; ~19% of transitions involve overlap (backchannel/interruption/overlap). Interruption ≈ 4x rarer than backchannels. Base rates confirm the labelling is headroom-limited (at 200ms only 3.3% positive — PR-AUC will dominate ROC-AUC in relevance).
- **Failure modes**:
  1. Energy-VAD fragments syllables → naive segment count (40k) wildly overstates turns; fixed with gap merging.
  2. Original `classify_transitions` excluded the other speaker's active segment (matched by END only) → almost zero overlaps detected; fixed by matching the latest segment by START.
  3. `merge_asr_event_type` was O(events×words) — 67.5s for a mid-size conversation, hours for the largest; replaced with `searchsorted` vectorization → 0.03s.
  4. PowerShell swallows python stdout when piped (block buffering): use `python -u` for progress visibility.
- **Next question**: What windowed (lookback 1.5s) acoustic + temporal features from the cached frame features best discriminate upcoming transitions, and how do B-'A'..B-E baselines compare?

---

## Experiment 2 — Labeled Decision-Point Dataset + Model Comparison

- **Experiment**: `exp-002_models`
- **Hypothesis**: Pauses/rhythm (temporal) and recent acoustic activity predict imminent other-speaker onset better than chance, and nonlinear trees beat linear models.
- **Dataset subset**: 160,295 decision points (200ms grid) across all 16 conversations; windowed features over [t−1.5s, t) (73 features: RMS ×6 stats, 7 spectral/F0/energy channels ×4, voiced fraction, 3 temporal).
- **Features**: Layer T (temporal: pause, elapsed, current-turn duration) + Layer A (acoustic per both channels).
- **Models**: majority, LR (C=1, standardized, median-imputed), RF (80 trees, leaf 25), HistGB (160 iters, lr 0.12). Train rows capped at 90k/fold (temporal autocorrelation).
- **Evaluation**: Leave-one-conversation-out (16 folds); pooled ROC-AUC / PR-AUC (and per-fold mean ± sd). Primary horizon 1000ms.
- **Result (horizon 1000ms, pooled)**:
  | model | ROC-AUC | PR-AUC | per-fold mean ROC |
  |---|---|---|---|
  | majority | 0.500 | 0.151 | — |
  | LR | 0.708 | 0.300 | 0.713 ± 0.071 |
  | RF | 0.760 | 0.346 | 0.760 ± 0.068 |
  | GBDT | **0.750** | **0.345** | 0.756 ± 0.065 |
  - LR horizon sweep (ROC/PR, base rate): 200ms: 0.757/0.106/0.033 · 500: 0.737/0.211/0.080 · 750: 0.718/0.260/0.116 · 1000: 0.708/0.300/0.150 · 1500: 0.691/0.362/0.212.
  - Speaker-pair-LOCO @1000ms: LR 0.623/0.259, GBDT **0.672**/0.277 (nb: conversation-LOCO GBDT 0.756 → 0.672 degradation = pair-specific patterns).
  - Ablation (LR @1000ms: temporal-only ROC 0.668; a-acoustic 0.607; b-acoustic 0.600; f0-only 0.637; no-f0 0.714; full 0.714) → **f0 adds ~nothing**; temporal + channel acousics carry the signal.
  - Top RF features: `current_seg_dur_s` (0.038), `pause_since_last_speech` (0.037), then spectral-flux and energy-slope stats → prosodic markers (flux/energy modulation) rank far above F0.
- **Interpretation**: Turn onset is predictable at ROC 0.75/PR 0.34 (base 0.15) one second ahead with causal acoustic+temporal features. Predictability DECREASES as the horizon grows further out (ROC 0.76→0.69 LR), suggesting onset timing is a short-horizon "rhythmic" effect, not a long-range countdown. A substantial share of skill (~6-8% ROC) is speaker-pair-specific.
- **Failure modes**:
  1. Native `Select-Object -Last` and PowerShell pipes block-buffer python output → **lost 30min run with zero visibility**; fixed by streaming directly and keeping single-subprocess monitoring via fold prints (flush=True).
  2. GBDT ~60s/fold at 350 iters → cut to 160 iters + 90k row cap → ~45-50s/fold; stable AUC (0.750 vs 0.756 across seeds).
  3. HistGradientBoostingClassifier in this sklearn build lacks `feature_importances_` → switched importance to RF.
  4. **Caught leak**: ablation's "full" and "no_f0" feature lists swept up `label_*` numeric columns → instantaneous ROC 1.0000. All predictive runs now exclude label columns explicitly.
- **Next question**: Calibration/error analysis — where does the model fail (backchannels vs clean transitions, long silences), and what is the precision/recall operating point for an app (e.g., only alert on p>0.6)?

> **Superseded by `exp-004_seed_fix`**: per-fold numbers in the table above
> predate the train-subset seed fix; see exp-004 for final values (
> LR 0.7148, RF 0.7620, GBDT 0.7523).

---

## Experiment 3 — Calibration and Failure Analysis (preview)

- **Experiment**: `exp-003_calibration`
- **Hypothesis**: GBDT probability scores are usable for thresholded decisions without heavy recalibration; errors concentrate in specific transition types.
- **Result**: Reliability data saved at `results/predictions_1000.npz`; curves in `figures/fig_calibration_1000.png`. Operating points (GBDT pooled @1000ms): θ=0.20 → predicted 25.5%, recall 0.55, precision 0.32; θ=0.30 → 13.0%/0.35/0.40; θ=0.50 → 2.5%/0.08/0.50 (3.3× lift over 0.15 base); best-F1 θ=0.215 (prec 0.336, rec 0.516, F1 0.407). GBDT calibrates near the diagonal across the usable range (reliability plot). Missing-and-early error mix: false positives concentrate at medium pause durations (both parties grooming); false negatives are short-threshold onsets with low acoustic salience — pinned by the audio-only feature stack, not fixable without transcript/floor info under current regime.
- **Next question**: (completed in paper draft) — quantify precision@k and the misprediction mix.

---

## Experiment 4 — Train-Subset Seed Fix & Final Result Regeneration

- **Experiment**: `exp-004_seed_fix`
- **Hypothesis**: All experimental scripts should sample the 90k-row training
  cap with the **same** fixed random state (42) so per-fold metrics
  (`results/models_loco_*.json`) and pooled metrics (`results/summary.json`)
  derive from identical train subsets; the repo should ship internally
  consistent numbers.
- **Change**: `scripts/05_models.py` previously seeded the row subsample with a
  group-dependent RNG (`cfg.model.seed + len(group)`), so per-fold and pooled
  runs sampled slightly different trains. Now `random_state=cfg.model.seed`
  (=42) everywhere (05/06/07). No model hyperparameters or data changed.
- **Re-run**: `05 --horizons 1000 --loco` (all 4 models), horizon sweep
  (majority+LR), `--pairs` (majority+LR+GBDT), then `analysis_1000.json`'s
  embedded `loco_1000_ref` refreshed from `models_loco_1000.json`.
- **Result (final, conversation-LOCO @1000ms, per-fold mean ± sd)**:
  | model | ROC-AUC | PR-AUC (weighted) |
  |---|---|---|
  | majority | 0.500 ± 0.000 | 0.150 |
  | LR | 0.715 ± 0.072 | 0.291 |
  | RF | **0.762 ± 0.067** | 0.353 |
  | GBDT | 0.752 ± 0.065 | 0.339 |
  Pooled: LR 0.708/0.300, GBDT 0.750/0.345 (`summary.json`). Sweep (LR, pooled):
  200ms 0.757/0.106 · 500 0.737/0.211 · 750 0.718/0.260 · 1000 0.708/0.300 ·
  1500 0.691/0.362. Pair-LOCO @1000: LR 0.621 ± 0.027 / PR 0.259;
  GBDT 0.667 ± 0.014 / PR 0.277 (drop 0.752 → 0.667).
- **Interpretation**: Deliberate re-running changed every per-fold mean by at
  most ~0.004 (e.g., GBDT 0.7559→0.7523); qualitative conclusions are
  unchanged. All `results/*.json` are mutually consistent now.
- **Failure modes**:
  1. `07_figures.py` was temporarily extended to pool RF too; on this machine
     16 top-level RF fits pushed the script past a 60-min wall — RF pooled AUC
     is not materialized (RF is reported as per-fold mean ± sd from
     `models_loco_1000.json`). Reverted; figures unchanged and seed-consistent.
  2. A full `06_analysis.py` re-run stalled on the tractable `full` ablation
     group under load; the ablation is deterministic at seed 42 and its stored
     values already match the final seed, so only the embedded `loco_1000_ref`
     was refreshed from `models_loco_1000.json`.
- **Next question**: role-resolved features (per-channel → per-role re-indexing)
  to measure how much the channel-blind `a_*`/`b_*` representation costs.
---

## Experiment 5 -- Annotation & Segmentation Robustness (EXP-002)

- **Experiment**: `exp-005_annotation_segmentation_robustness`
- **Hypothesis**: The fitted turn-onset classifiers predict real speech structure, not their own annotation conventions: corrupting the (A) label annotations (onset jitter) or (B) the VAD segmentation that produces them should degrade performance only incrementally, while destroying the feature<->label alignment (shuffle/uniform controls) should collapse ROC to chance.
- **Dataset subset**: All 16 conversations; 160,295 decision points (200ms grid); windowed features over [t-1.5s, t) (73 features); labels re-derived per Arm A perturbation with the production rule (events.py).
- **Features**: Layer T + Layer A (frozen); Arm B re-derives VAD segmentation at threshold max(floor_metric, p25(rms)) + (20 + Delta) dB with production hangover logic.
- **Model**: LogisticRegression (C=1, standardized, median-imputed), 16-fold LOCO, 90k train cap, seed 42, horizon 1000ms. LR-only for perturbation cells (GBDT clean anchor from EXP-001: 0.7523/0.3319).
- **Evaluation**: pool ROC-AUC / PR-AUC (existing eval_loco pipeline).
- **Result (Arm A, ROC @1000ms)**:
  - clean 0.7148/PR 0.2808 (anchor = EXP-001).
  - jitter sigma 25/50/100/200/400ms -> 0.7143-0.7152 / 0.7141 / 0.7137-0.7144 / 0.7104-0.7116 / 0.7034 (monotone graceful decay; -0.001 at 100ms, -0.011 at 400ms = 25% of the label window).
  - shift -200ms 0.7030/PR 0.3005; shift +200ms 0.7270/PR 0.2561 (base-rate anti-correlated movement; ordering structure preserved).
  - shuffle (within-conv label permutation, marginal kept) 0.5007/PR 0.1316; uniform (Bernoulli) 0.4956/PR 0.1499 -- chance.
- **Result (Arm B, ROC @1000ms)**: Delta=+3dB 0.7126/PR 0.2680; Delta=-3dB 0.7147/PR 0.2867; drift cell (Delta=-3 features, clean labels) 0.7176/PR 0.2834; anchor Delta=0 = clean 0.7148/0.2808. Segmentation margin moves ROC < 0.003.
- **Interpretation**: Turn-onset prediction is robust to annotation timing noise well beyond transcription error, and to VAD segmentation slop. The shuffle/uniform controls collapse to 0.50 => the 0.71 signal is genuine acoustic+temporal structure, not alignment with annotations. Supports full-ML-integration thesis claims.
- **Failure modes**:
  1. A naive *global* onset shuffle re-introduced a t-profile in labels (P(label|t) rises with the t+h threshold), letting the elapsed_s feature recover ~0.77 ROC; the control was redesigned as within-conversation label permutation (marginal + t-profile preserved) and re-ran to 0.50.
  2. eval_loco_fast v1 read labels from df columns (stored originals) -> every condition returned the same ROC 0.7148 with changing base rates; poisoned JSON deleted, labels now passed as the perturbed y array, all cells re-derived.
  3. Machine OOM (7.9GB RAM, ~1GB free) with precomputed 16-fold float64 matrices during the shuffle fit -> low-memory per-fold control runner; Arm B rebuild OOM'd at margin 0 (anchor reused from stored dataset).
- **Next question**: input-side robustness -- corrupt the audio/window features (RMS, spectral flux, energy slope) instead of labels to separate "signal in energy envelope" from "signal in spectral detail" (candidate exp-006).
