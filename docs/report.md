# Vigie: technical report

This report records how every number in the README was produced, including the ones that did
not go well. Design rationale is in [`design.md`](design.md).

## 1. Data audit

| Split | Source | Videos | Content | Used for |
|---|---|---|---|---|
| `retails_fit` | RetailS train | 868 | ~8.4M person-frames of normal shopping | normality model |
| `retails_holdout` | RetailS train | 74 | ~0.7M person-frames, held out by video | thirds: head negatives / calibration / false alarms |
| `retails_staged` | RetailS staged test | 624 | staged thefts, frame labels, 50% theft frames | supervised heads |
| `poselift_train` | PoseLift | 104 | 1.7 h normal | normality model, published setting |
| `poselift_test` = `retails_real` | PoseLift / RetailS | 47 | real thefts, 43 events, 32% theft frames | **test only** |

Findings before any modelling:

1. **The PoseLift test set and the RetailS real-world test set are the same 47 clips.** Names,
   keypoints and labels match exactly. They are evaluated once.
2. **No footage leaks between splits.** Every 8-frame window of every track was fingerprinted
   (pixel-rounded keypoints, BLAKE2 hash); no fingerprint is shared between any training split and
   the test set (`scripts/check_leakage.py`). Exact hashing would miss re-extracted duplicates, so
   this rules out copy leakage, not near-duplicates.
3. **The cameras are mounted at about 90 degrees.** In RetailS the median hip-to-shoulder direction
   is close to the image x axis, and differs per camera. A model fed raw coordinates would learn
   camera orientation. Vigie estimates "up" per person from the torso (running circular mean over
   2 s) instead of assuming it.
4. **Test clips are short**: median 32 frames (2.1 s) per track, cut around each event.

## 2. Protocol

- Windows of 24 frames (1.6 s) at stride 6; short tracks (at least 8 frames) are edge-padded.
- Frame score per person = mean of the windows covering the frame; video frame score = maximum
  over people; empty frames get the minimum score. This matches the PoseLift / STG-NF protocol.
- Metrics: frame-level AUC-ROC (comparable with the published table), AUC-PR, EER.
  AUC-PR is **not** compared with the published numbers: their STG-NF AUC-PR (84.1) is far above the
  other published rows and above anything seen here at similar AUC-ROC, which suggests a different
  convention (for example, normal frames as the positive class).
- 95% intervals: 500 bootstrap resamples of **whole videos** (frames within a clip are not
  independent).
- Event recall: an event counts as caught when an alert fires within 1 s of the labelled interval.
- False alarms: alerts per hour on 13.7 h of held-out normal footage, using the same alert policy
  (EMA smoothing 0.3, at least 8 frames above threshold, re-arm below 80% of it).
- Threshold choice: the lowest quantile of fused scores on the calibration third that keeps false
  alarms at or under 1 per camera-hour. The test set is never used to choose anything.

## 3. Results on the real test set

| Model | AUC-ROC | 95% CI | AUC-PR | EER |
|---|---|---|---|---|
| Normality, trained on PoseLift normal (published setting) | 63.5 | 43.1 to 76.5 | 37.2 | 0.41 |
| Normality, trained on RetailS normal | 63.4 | 43.7 to 76.5 | 36.9 | 0.41 |
| Cues + boosted trees | 74.1 | 54.3 to 85.8 | 50.8 | 0.32 |
| Fine-tuned transformer | 74.4 | 53.1 to 85.5 | 50.0 | 0.33 |
| Trees + transformer | 74.6 | 53.3 to 86.1 | 49.3 | 0.32 |
| Three-way fusion (shipped, pre-registered) | 71.8 | 50.8 to 84.2 | 47.5 | 0.35 |

Published (PoseLift paper, same test set): STG-NF 67.5, TSGAD 63.4, GEPC 60.6; Shopformer 69.2.

Temporal smoothing (Gaussian, sigma 5 frames) adds 0.1 to 1.9 points; numbers above are unsmoothed.

Cross-validation of the trees head (5 folds grouped by video): 0.84 AUC on staged + normal windows,
but only **0.61 among staged clips** (theft vs non-theft frames of the same staged clip). Telling
"someone who steals" from "someone who shops" is easier than finding the exact second of
concealment.

## 4. Alerts

| Threshold (quantile of normal fused scores) | False alerts / camera-hour | Events caught |
|---|---|---|
| 0.95 (review mode) | 16.3 | 34.9% |
| 0.98 | 5.8 | 20.9% |
| 0.99 | 2.6 | 11.6% |
| 0.995 | 1.2 | 11.6% |
| 0.998 (shipped) | 0.6 | 2.3% |
| 0.999 and above | 0.1 to 0.4 | 0% |

Frame-level ranking is decent; event-level alerting at strict false-alarm budgets is weak. Two causes
are visible in the replays: the risk often rises when a person *approaches and reaches* rather than
during the labelled concealment itself, and the clips are too short for a policy that wants 0.5 s of
sustained evidence. Both point to training on whole events rather than windows.

## 5. Negative results worth keeping

- **Masked pose prediction alone does not beat STG-NF.** The normality transformer learns normal
  motion well (validation error 0.24 torso lengths) but theft is not "unpredictable" motion; it is
  careful, slow and ordinary-looking.
- **More normal data did not help the unsupervised model**: 200k windows from RetailS and 11k from
  PoseLift give the same AUC (63.4 vs 63.5).
- **Fusing a weaker scorer hurts.** The pre-registered three-way fusion is 2.8 points below the
  two supervised heads. Percentile calibration puts all scorers on one scale but does not weight them.

## 6. Speed (Intel i5-1135G7, 8 GB RAM, no GPU, idle CPU)

| Stage | Time |
|---|---|
| YOLO11n-pose, OpenVINO, 1080p frame, alone | ~30 ms |
| Same model with ByteTrack, alone | ~50 ms |
| Inside the full pipeline, busy aisle (~8 people) | detect + track ~180 ms, scoring ~16 ms, rendering ~36 ms per frame |
| End to end | 7 to 8 fps |

Two fixes found by profiling: blurring the full frame for the privacy view cost 117 ms per frame
(now done at 1/24 scale, a few ms), and PyTorch's default thread pool starved OpenVINO's (the
scorers now use 2 threads). The remaining gap between "detect + track alone" (~50 ms) and inside
the pipeline (~180 ms) is not explained yet and is the first item for reaching 15 fps.

## 7. Reproduce

```bash
python scripts/download_poselift.py
python scripts/check_leakage.py poselift_train retails_holdout retails_staged retails_real
python scripts/train_normality.py --train retails_fit --windows 200000 --epochs 10
python scripts/train_normality.py --train poselift_train --epochs 15 --tests retails_real
python scripts/train_heads.py --clf-epochs 6
python scripts/make_figures.py
python scripts/render_replay.py --clips 5
```

Seeds are fixed; CPU nondeterminism in PyTorch can still move AUCs by a few tenths.
