# EXP-004 — Role-Resolved Conversational Feature Representation & Generalization

An empirical investigation of conversational asymmetry in causal turn-onset prediction.

- **Research question.** How much predictive power is lost by evaluating conversational audio under an arbitrary channel mapping ($A$ vs. $B$) compared to an aligned, conversational **role-resolved representation** ($\text{floor-holder}$ vs. $\text{listener/partner}$)?
- **Hypothesis.** Arbitrary channel multiplexing forces linear models (LR) to learn symmetric averages and forces trees (GBDT/RF) to waste capacity routing between channels. Aligning causal features to speaker role will:
  1. Recover the true predictive value of causal acoustics (which appeared weak in EXP-001).
  2. Increase ROC-AUC and PR-AUC across all models with statistically significant paired deltas.
  3. Improve cross-dyad generalization in strict dyad-disjoint evaluation (Pair-LOCO).
- **Leakage controls.** Strictly causal. At time $t$, `current_speaker` is determined solely by the speaker whose turn segment started at or before $t$ (last segment with $\text{start\_s} \le t$), using past/present VAD only. No future onsets, words, or labels are accessed. Chance control (within-conversation label permutation) verifies complete collapse to 0.50.

---

## 1. Primary Benchmark: Leave-One-Conversation-Out (conv-LOCO @ 1000ms)

16-fold LOCO evaluation on all 160,295 decision points ($N=16$ conversations), comparing the 73 channel-arbitrary features against the 73 role-resolved features.

| Model | Channel-Arbitrary ROC (mean ± sd) | Channel PR (weighted) | Role-Resolved ROC (mean ± sd) | Role PR (weighted) | Paired Mean $\Delta$ ROC (95% Bootstrap CI) | $p$-value |
|---|---|---|---|---|---|---|
| **Majority** | 0.5000 ± 0.0000 | 0.1501 | 0.5000 ± 0.0000 | 0.1501 | +0.0000 [+0.0000, +0.0000] | — |
| **LR** | 0.7148 ± 0.0716 | 0.2906 | **0.7291 ± 0.0705** | **0.3208** | **+0.0144** [+0.0053, +0.0246] | $p = 0.0106$ |
| **GBDT** | 0.7523 ± 0.0653 | 0.3390 | **0.7709 ± 0.0641** | **0.3787** | **+0.0186** [+0.0130, +0.0246] | $p < 0.0001$ |
| **RF** | 0.7620 ± 0.0674 | 0.3528 | **0.7707 ± 0.0658** | **0.3835** | **+0.0088** [+0.0044, +0.0136] | $p = 0.0019$ |

### Key Findings
1. **Statistically Significant Improvement Across All Models**: Role resolution produces significant gains across LR ($+0.0144$, $p=0.011$), GBDT ($+0.0186$, $p<0.0001$), and RF ($+0.0088$, $p=0.002$).
2. **Substantial PR-AUC Lift**: For GBDT, weighted PR-AUC increases from $0.3390$ to **$0.3787$** ($+0.0397$ lift over base rate $0.1501$), demonstrating sharper confidence on true transition points.
3. **New State-of-the-Art for the Repository**: GBDT reaches **$0.7709$ ROC-AUC** and RF reaches **$0.7707$**, establishing the highest predictive accuracy reported on this dataset without adding any new data or external models.

---

## 2. Feature Ablation: True Acoustic Recovery

In EXP-001, acoustic features appeared weak ($\text{ROC} \approx 0.607$) and temporal features seemed to dominate ($\text{ROC} = 0.668$). Role-resolved ablation reveals this was an artifact of channel blindness.

| Feature Subset | $N$ Features | LR ROC (mean) | LR PR (weighted) | GBDT ROC (mean) | GBDT PR (weighted) |
|---|---|---|---|---|---|
| Temporal only | 3 | 0.6676 | 0.2269 | 0.6651 | 0.2242 |
| Channel A acoustics only | 35 | 0.6380 | 0.2212 | 0.6701 | 0.2458 |
| Channel B acoustics only | 35 | 0.6263 | 0.2143 | 0.6530 | 0.2338 |
| **Both channel acoustics** | 70 | 0.6944 | 0.2748 | 0.7297 | 0.3090 |
| **Floor acoustics only** | 35 | **0.6814** | **0.2362** | **0.7015** | **0.2647** |
| **Listener acoustics only** | 35 | **0.6663** | **0.2636** | **0.6985** | **0.2851** |
| **Both role acoustics** | 70 | **0.7194** | **0.3113** | **0.7575** | **0.3547** |
| Channel F0 only | 8 | 0.6369 | 0.2110 | 0.6886 | 0.2526 |
| Role F0 only | 8 | **0.6669** | **0.2348** | **0.7016** | **0.2687** |
| Role without F0 | 65 | 0.7297 | 0.3220 | 0.7680 | 0.3735 |
| **Full Role-Resolved** | 73 | **0.7291** | **0.3208** | **0.7709** | **0.3787** |

### Crucial Scientific Insight: Acoustics Alone Outperform Full Channel-Blind Models
- When acoustics are role-resolved, **acoustics alone** (without pause duration, elapsed time, or current segment duration) achieve **$0.7194$ ROC in LR** and **$0.7575$ ROC in GBDT**.
- This exceeds the **full 73-feature channel-blind baseline** ($0.7148$ in LR and $0.7523$ in GBDT).
- Both floor-holder prosody ($0.6814$) and incoming listener activity ($0.6663$) provide independent, complementary signals. When combined, they jump to $0.7194$ (LR) and $0.7575$ (GBDT).

---

## 3. Cross-Dyad Generalization: Speaker-Pair LOCO (4 unseen dyads)

To assess whether role resolution overfits to speaker pairs or learns universal dyadic turn-taking dynamics, we evaluated models under 4-fold Speaker-Pair LOCO (holding out entire unseen dyads).

| Model | Channel-Arbitrary Pair-LOCO ROC | Role-Resolved Pair-LOCO ROC | Paired $\Delta$ ROC |
|---|---|---|---|
| **LR** | 0.6213 ± 0.0270 | **0.6600 ± 0.0256** | **+0.0386** |
| **GBDT** | 0.6666 ± 0.0143 | **0.6722 ± 0.0171** | **+0.0056** |

- For linear models (LR), role resolution closes almost **half of the generalization gap** observed in prior work (recovering $+0.0386$ ROC points on novel dyads).
- For GBDT, performance improves to $0.6722$, maintaining stability across dyad holdouts.

---

## 4. Lead-Time Analysis: Horizon Sweep

Forecast horizon sweep $\Delta t \in \{200, 500, 750, 1000, 1500\}\,\text{ms}$:

| Horizon $\Delta t$ | Base Rate | LR Channel | LR Role | GBDT Channel | GBDT Role |
|---|---|---|---|---|---|
| **200 ms** | 0.033 | 0.7673 | **0.7860** (+0.019) | 0.7929 | **0.8191** (+0.026) |
| **500 ms** | 0.080 | 0.7472 | **0.7638** (+0.017) | 0.7847 | **0.8064** (+0.022) |
| **750 ms** | 0.116 | 0.7293 | **0.7433** (+0.014) | 0.7643 | **0.7829** (+0.019) |
| **1000 ms** | 0.150 | 0.7148 | **0.7291** (+0.014) | 0.7523 | **0.7709** (+0.019) |
| **1500 ms** | 0.212 | 0.6882 | **0.7046** (+0.016) | 0.7275 | **0.7460** (+0.019) |

- Across **every single horizon**, role resolution delivers consistent $+0.014$ to $+0.026$ ROC gains.
- At short horizons ($h \le 500\text{ms}$), GBDT surpasses **$0.80$ ROC-AUC** ($0.8191$ at $200\text{ms}$, $0.8064$ at $500\text{ms}$), demonstrating high predictive precision as floor transition approaches.

---

## 5. Chance Control and Falsification

- **Within-Conversation Label Permutation**:
  - LR: **$0.5039$ ROC-AUC** (chance)
  - GBDT: **$0.5024$ ROC-AUC** (chance)
- Confirming that without temporal alignment between causal features and turn onsets, the representation provides zero predictive advantage.

---

## 6. Reproduction

```bash
# Run unit tests
python -m pytest open-yap-turn-taking/tests/test_role_resolved.py -v

# Run full experiment
python open-yap-turn-taking/experiments/EXP-004-role-resolved/run_experiment.py

# Generate figures
python open-yap-turn-taking/experiments/EXP-004-role-resolved/make_figures.py
```
