# LIMITATIONS.md

**Date**: 2026-09-11

This document records the limitations of the Open Yap 1K sample for turn-taking prediction research, and the limitations of this study specifically.

---

## Dataset Limitations

1. **Only 4 speaker pairs**. The 16-conversation sample contains exactly 4 distinct speaker pairs:
   - Pair 1 (spk_03f80fcf × spk_77fb5673): **8 conversations** (50% of the sample)
   - Pair 2 (spk_3c647261 × spk_209a8771): 4 conversations
   - Pair 3 (spk_0e73db57 × spk_5dd2763d): 3 conversations
   - Pair 4 (spk_72908577 × spk_b6ef3dc4): 1 conversation
   
   This is the single most important limitation. Any model trained on this sample is dominated by Pair 1's conversational style (two young adult men). Cross-speaker generalizations cannot be reliably estimated.

2. **Hand-picked, not random**. The sample was hand-curated by the dataset creators to showcase the corpus. It is explicitly stated in the README that the sample's distribution should not be assumed to generalize to the full corpus.

3. **Machine-generated transcripts**. Phrases, word timings, word boundaries, and the 40ms ASR grid are all from Deepgram Nova-3, not human-verified. Overlap regions are likely to have degraded ASR quality.

4. **No explicit event labels**. The sample contains no annotations for overlaps, backchannels, pauses, or turn boundaries. The `word[].type` field is uniformly "word" (no filler/laugh/cough tags despite the README describing them). All event types in this study are inferred from audio energy VAD + ASR word timing.

5. **Bluetooth tracks**. 8 of 32 tracks carry no energy above ~8-9 kHz (per `effective_bandwidth_hz`), meaning high-frequency acoustic features (spectral centroid above ~9 kHz) are unreliable on those tracks.

6. **Channel crosstalk**. `silent_while_partner_spoke_pct` ranges from 0% to 93% across tracks. Some channels leak significant partner audio into the other channel, which could corrupt VAD and pitch estimates, especially during overlap.

7. **Conversation imbalance**. Turn rate varies from 2.7 to 19.8 TPM across conversations. A model trained on fast conversationalists may behave differently on slow speakers.

8. **All-friends, all-English**. Sample is homogeneous: all relationships are "friends", language is English. No code-switching, no variety in relationship types.

9. **Sampling imbalance in speech dominance**. Dominance ranges 0.31–0.70, so some conversations are highly unbalanced (one person talks more, one is mostly passive — potentially producing many backchannel-like events).

---

## Methodological Limitations

1. **Small-held-out test size**. LOCO evaluation on 16 conversations yields a small test pool per fold. Aggregated estimates are dominated by the 8 Pair-1 conversations, meaning the "test" distribution is heavily influenced by one pair.

2. **VAD threshold sensitivity**. All event labels depend on a VAD threshold (noise floor + 20dB margin). VAD has been validated against ASR word boundaries on a sample, but some transitions may be mis-segmented.

3. **Horizon label definition**. The label "other speaker begins within Δt" is well-defined but sensitive to the duration of the current speaker's current segment. If the current speaker talks for another 30 seconds, the other speaker cannot begin unless interrupting (creating label y=0 for long stretches even when a transition is plausible later).

4. **Backchannel detection thresholds**. The 800ms/3-word thresholds for backchannels are heuristic. Real backchannels can be longer or shorter.

5. **No neural baseline for comparison**. Given the small sample, deep models were not expected to beat well-regularized trees. A neural baseline on the full 1000h corpus might behave differently.

6. **Temporal granularity**. Features are computed at 10ms hop. The ASR word grid is 40ms. Precision of event onset is ±20ms at best.

7. **Causality window**. "Immediately before" means features derived from a 1.5s lookback window ending at t. Longer lookback windows (e.g., 5s) might carry different information but are not explored in v1.

---

## Statistical Limitations

1. **No population-level claims**: All results in this study describe *this sample*. We explicitly avoid claiming "humans universally..." Anything phrased as "within this sample..." is safe.

2. **Multiple comparison concern**: We evaluate 5 horizon values × 3 model classes × feature ablations. We do not correct for multiple comparisons; p-values, where reported, are descriptive rather than inferential. We report effect sizes (AUC deltas) and their uncertainty via bootstrap.

3. **Speaker-disjoint test has concentrated power**: A single test protocol (hold out Pair 1) is the only genuinely speaker-disjoint check, and it collapses to one "experiment" with 8 held-out conversations.

4. **Class imbalance**: Transitions within 200ms are a small fraction of decision points, and within 1500ms a larger fraction. PR-AUC must be reported alongside ROC-AUC to interpret.

---

## What This Study Does NOT Claim

- We do NOT claim a deployable turn-prediction system.
- We do NOT claim the results generalize to Open Yap 1K full corpus (1000h).
- We do NOT claim the "pre-turn signature" is universal across pairs.
- We do NOT claim ASR word labels are ground truth.

---

## Measured Constraint-Lifting (added 2026-09-11, exp-002/003)

Empirical consequences of the above, now quantified:

1. **Pair-specificity cost is ≈ 0.086 ROC-AUC**: leave-one-speaker-pair-out GBDT
   @1000ms = 0.667 ± 0.014 ROC vs 0.752 ± 0.065 conversation-LOCO (per-fold means).
   A universal (non-personalized) turn-predictor from audio is plausible but
   strictly weaker than a dyad-entrained one.
2. **F0 contributes nothing beyond energy/spectral/rhythm** (LR ablation: 0.714
   with vs 0.714 without F0; F0-only 0.637). Pitch-based "intonation" cues do not
   improve onset prediction in this feature family.
3. **Class-imbalance consequence**: PR-AUC vs ROC-AUC diverge strongly (e.g.
   @200ms PR-AUC 0.106 at base 0.033 vs @1500ms PR-AUC 0.362 at base 0.212);
   "Rising" ROC must not be read as "rising signal strength" — horizon-sweep
   interpretation must pair ROC with base-rate-corrected PR.
4. **VAD-threshold sensitivity confirmed**: turning off 400ms within-speaker
   merging inflated transition counts 3x (40k vs 13.6k) and washed out
   backchannel/interruption structure (9 vs 2,336).
5. **Robustness bounds are LR-only at the extreme**: the EXP-002 perturbation
   cells (annotation onset jitter, VAD margin) were run with LR; GBDT (the
   paper's best, 0.7523) was only anchored at the clean condition. Robustness
   of nonlinear models at 400 ms jitter is not directly measured (disclosed in
   `experiments/EXP-002-annotation-robustness/README.md`).

---

## What Additional Data Would Help

1. **Full corpus (1000h)**: enough pairs for meaningful speaker-disjoint splits, enough diversity of relationships, and enough occurrences of each event type.
2. **Human-verified transcripts**: to validate ASR-derived event timing.
3. **Explicit tags**: backchannel/laughter/filler tags (available in full corpus per README).
4. **Longer recordings per pair**: to study same-pair stability over time.
5. **Multi-pair diversity**: more than 4 pairs per sample.
6. **Ground-truth VAD or manual annotation of a validation subset**: to calibrate the VAD threshold.