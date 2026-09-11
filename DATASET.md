# DATASET.md — Open Yap 1K Sample Audit

**Date of audit**: 2026-09-11

**Dataset**: Open Yap 1K sample (The Agentic Data Company)

**License**: CC-BY-4.0

## High-Level Summary

| Metric | Value |
|---|---|
| Conversations | 16 |
| Unique speakers | 8 |
| Total duration | 32120.6s (8.92h) |
| Duration range | 1002.4s – 4419.8s |
| Duration mean | 2007.5s |
| Duration median | 1488.8s |
| Audio format | FLAC, 48kHz, 16-bit mono per speaker |
| Total word count (A+B) | 89677 |
| Relationships | [('friends', 16)] |

## Per-Conversation Summary

| conv_id | dur(s) | TPM | gap_ms | dom | nA | nB | topics |
|---|---|---|---|---|---|---|---|
| conv_04692967d40… | 1015 | 3.9 | 760 | 0.70 | 1653 | 718 | formula one; motorsport; sports viewing |
| conv_0c917fc63b3… | 1183 | 2.7 | 460 | 0.31 | 827 | 1812 | career changes; software startups; space travel |
| conv_10cf7f6ff08… | 1377 | 3.7 | 640 | 0.31 | 900 | 2025 | evolution; human origins; biology |
| conv_14f897f7501… | 1402 | 4.1 | 640 | 0.44 | 1461 | 1852 | home appliances; formula one; auto racing training |
| conv_2e4aec4666d… | 1002 | 5.0 | 860 | 0.69 | 1723 | 776 | football; sports history; athletes |
| conv_31982da49c7… | 2725 | 16.3 | 380 | 0.56 | 4709 | 3738 | online shopping; thrifting; fashion |
| conv_31bedfc99b5… | 1384 | 4.5 | 640 | 0.45 | 1461 | 1779 | soccer; world cup; sports predictions |
| conv_5a798c657bb… | 1273 | 9.3 | 380 | 0.36 | 1505 | 2676 | family relationships; weather; introversion vs extroversion |
| conv_756ee444d82… | 4420 | 9.2 | 340 | 0.40 | 5861 | 8925 | music and concerts; relationships; communication habits |
| conv_aa4777220a1… | 3605 | 7.9 | 360 | 0.44 | 5023 | 6313 | horror movies; 1980s nostalgia; first jobs |
| conv_b98b64206ae… | 1015 | 5.1 | 560 | 0.68 | 1721 | 799 | books and reading; self-improvement; artificial intelligence |
| conv_bcff0c2a0be… | 2163 | 9.8 | 400 | 0.40 | 2886 | 4390 | halloween; horror movies; childhood memories |
| conv_cb37687ec91… | 2558 | 14.9 | 440 | 0.49 | 3525 | 3686 | food and recipes; weather; relocation |
| conv_d4005da6db9… | 1717 | 7.2 | 440 | 0.51 | 2220 | 2169 | daily routines; travel planning; visa and paperwork |
| conv_e11edcca852… | 1576 | 16.1 | 420 | 0.49 | 2127 | 2183 | prescription medication; celebrity cosmetic surgery; perfume and fragrances |
| conv_f5c0b02f01e… | 3706 | 19.8 | 320 | 0.41 | 3367 | 4867 | trivia quizzes; geography; world history |

## Speaker Demographics

| speaker_id | age | gender | country | #convs |
|---|---|---|---|---|
| spk_03f80fcf | 16-24 | male | GB | 8 |
| spk_0e73db57 | 55-64 | female | US | 3 |
| spk_209a8771 | 35-44 | male | US | 4 |
| spk_3c647261 | 45-54 | female | US | 4 |
| spk_5dd2763d | 35-44 | female | US | 3 |
| spk_72908577 | 45-54 | male | US | 1 |
| spk_77fb5673 | 16-24 | male | FR | 8 |
| spk_b6ef3dc4 | 55-64 | male | TH | 1 |

## Audio Quality & Effective Bandwidth

**Effective bandwidth** (Hz, both tracks):
- Mean: 11909
- Median: 12192
- Min: 7541
- Max: 15413
- Tracks below 9 kHz (Bluetooth): 8/32

## Echo Cancellation & Silence Profiles

Echo cancellation: A=[None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], B=[None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]

| conv_id | A_noise_dbfs | B_noise_dbfs | A_silent_pct | B_silent_pct |
|---|---|---|---|---|
| conv_04692967d40… | -80.8 | -53.8 | 7.01 | 93.36 |
| conv_0c917fc63b3… | -50.0 | -79.5 | 94.87 | 8.01 |
| conv_10cf7f6ff08… | -53.2 | -82.3 | 93.34 | 17.51 |
| conv_14f897f7501… | -59.5 | -82.0 | 86.01 | 48.6 |
| conv_2e4aec4666d… | -75.6 | -57.5 | 33.17 | 91.14 |
| conv_31982da49c7… | -65.8 | -51.0 | 0.0 | 63.35 |
| conv_31bedfc99b5… | -84.2 | -51.5 | 50.09 | 90.6 |
| conv_5a798c657bb… | -74.0 | -85.9 | 0.0 | 7.18 |
| conv_756ee444d82… | -73.8 | -86.0 | 0.0 | 6.79 |
| conv_aa4777220a1… | -73.9 | -85.9 | 0.0 | 0.88 |
| conv_b98b64206ae… | -77.7 | -47.7 | 44.78 | 91.63 |
| conv_bcff0c2a0be… | -72.6 | -83.0 | 0.0 | 0.56 |
| conv_cb37687ec91… | -66.0 | -52.7 | 0.0 | 64.04 |
| conv_d4005da6db9… | -82.0 | -56.5 | 4.38 | 78.92 |
| conv_e11edcca852… | -66.6 | -50.8 | 0.0 | 62.23 |
| conv_f5c0b02f01e… | -72.0 | -59.9 | 0.0 | 27.46 |

## Transcript Statistics

Total words: 89677
Word types: {'word': 89677}
Duration of speaker-a transcripts (speech-time): mean 2001s, range 999–4412s
Duration of speaker-b transcripts (speech-time): mean 2003s, range 998–4416s

## Word-Level Quality Checks

Tracks with non-monotonic word starts: 0/32
Negative gaps (overlapping words in transcript): 411
Max negative gap: 7.905s
Positive inter-word gaps: mean 3.156s, median 0.720s, p99 35.090s

## ASR Timing Resolution

Fractional start-time mod 0.1 (sampled): [0.0, 0.004999, 0.005, 0.005001, 0.014999, 0.015, 0.019999, 0.02, 0.024999, 0.025]
→ Resolution appears to be 40ms (25 Hz frame rate)

## Missing Values & Malformed Records

| Field | Checks |
|---|---|
| language | all = 'en' |
| transcripts.a | present in all |
| transcripts.b | present in all |
| speakers.a | present in all |
| speakers.b | present in all |
| word[].type | all = 'word' (no explicit filler/laugh tags in sample) |
| echo_cancellation | None for all tracks |
| headphones | None for all tracks |
| device | None for all tracks |

## Key Observations for Turn-Taking Research

1. **Separate-channel audio**: both speakers share one timeline but each channel is that speaker's own microphone. This is the ideal format for studying turn-taking (no diarisation needed).

2. **Word-level ASR**: Deepgram Nova-3, 40ms grid. No explicit backchannel/laughter tags in the sample (all type='word'). Word timestamps are **not human-verified**.

3. **Bluetooth tracks**: 8/32 tracks have effective bandwidth < 9 kHz. Must be considered when computing spectral features.

4. **Turn-taking gap**: median 400ms across conversations in the sample. Range 320–860ms. Comparable to the full corpus (p50 = 580ms).

5. **Full-duplex behaviour preserved**: overlap % is not explicitly computed for the sample, but the blog reports 8.3% across the corpus. The dual-channel format allows direct overlap detection.

6. **Small sample**: 16 conversations, 8 speakers. Generalisation to the full corpus is not guaranteed.

7. **Conversation structure**: all 'friends' relationships. Self-selected pairs, free-talk topics. Different conversational styles reflected in TPM (2.7–19.8) and dominance (0.31–0.70).

## Direct Overlap Measurement

Computing speech overlaps using energy-based VAD...

Total voiced time (A∪B): 20777.5s (5.77h)
Total overlap time: 1187.9s
Overall overlap fraction: 5.7%
Per-conversation overlap fractions: [np.float64(0.015528325599381283), np.float64(0.0173379884054223), np.float64(0.019718017831225378), np.float64(0.020207576543850546), np.float64(0.023147769693997112), np.float64(0.027244784513766267), np.float64(0.03136652774731301), np.float64(0.0323738996786363), np.float64(0.040226006748258034), np.float64(0.04385626793438691), np.float64(0.048854053976601065), np.float64(0.04920725527428618), np.float64(0.09526829338598793), np.float64(0.10041913761984997), np.float64(0.12073194555992099), np.float64(0.12306805586344491)]
Mean: 5.1% Median: 3.6% Range: 1.6%–12.3%
