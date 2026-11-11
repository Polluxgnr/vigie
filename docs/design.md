# Vigie: design

*Privacy-first shoplifting detection from human pose, real-time on a laptop CPU.*

## 1. Why this project

Most public shoplifting demos are a pretrained detector plus a handful of hand-written rules,
shown on one clip, with no numbers. Commercial systems exist, but in France the CNIL found the
leading one (Veesion, 2024) non-compliant, largely because it continuously analyses customers'
images. A bill to allow a supervised experiment was passed by the Assemblée nationale in
February 2026.

Vigie takes the opposite stance on every axis:

| Typical demo | Vigie |
|---|---|
| Stores or zooms on faces | Converts video to **skeletons** at the edge, discards pixels, no identity |
| Hand-tuned rule weights | **Learned** models, benchmarked on public real-world datasets |
| "It works on my clip" | Frame-level AUC, **false alarms per hour**, confidence intervals, cross-dataset tests |
| Black-box score | Every alert carries a **reason** ("item to waistband 1.8 s after reaching a shelf") |
| Needs a GPU | Runs **real-time on an Intel i5 laptop CPU** (OpenVINO) |

It is a portfolio project. It is not a certified product and makes no claim of legal compliance.

## 2. Goals, non-goals, anti-goals

**Goals**
1. A pose-only shoplifting detector that beats the published baselines on PoseLift
   (STG-NF: 67.5 AUC-ROC; Shopformer: 69.2), using an honest, documented protocol.
2. An explainable risk score per person, with a human-readable reason for each alert.
3. A live pipeline: video file or webcam → skeletons → tracks → risk → overlay and event log,
   at at least 15 fps on the author's laptop (i5-1135G7, no discrete GPU).
4. A demo video and a README a recruiter can understand in 60 seconds, plus a technical report
   an ML engineer can check.

**Non-goals**
- Face recognition, re-identification across cameras, demographic attributes.
- Detecting theft that pose cannot see (an item slid off a shelf by a foot, a swapped label).
- Production deployment, a cloud backend, or multi-store management.

**Anti-goals** (would make it a failure even if the numbers are good)
- Any feature that singles out or profiles a person by appearance.
- Results reported only on data the model was tuned on, or leakage between train and test.
- Accuracy numbers without the false-alarm rate next to them.

## 3. Data

| Dataset | What it is | Role |
|---|---|---|
| **PoseLift** (WACV 2025) | 153 real CCTV videos, 6 cameras, US store; 43 contain real shoplifting. COCO-17 keypoints (YOLOv8 + ByteTrack + HRNet), 15 fps, frame-level labels. Pose only. | Main benchmark, comparable with published numbers |
| **RetailS** (IEEE IoT-J 2026) | Same research group and store. About 20M normal training frames, a real-world test set and a staged test set. COCO-17 pose only. | Large normal-only training data; second test set; source of labelled staged thefts |
| **Own footage** | Staged scenes filmed by the author with consenting friends | Demo video; check the live pipeline end to end, including camera-to-dataset domain shift |

Raw video for the demo can only come from footage with clear reuse rights (own recordings, or
royalty-free stock). Public datasets are used for numbers, never redistributed.

**Leakage check.** PoseLift and RetailS come from the same store and authors, so their videos
may overlap. Before any cross-dataset claim, overlapping sequences are detected by hashing
normalised keypoint windows and removed from training.

## 4. Method

### 4.1 Representation
Each tracked person becomes a sequence of COCO-17 keypoints `(x, y, confidence)` at 15 fps.
Windows of `T = 24` frames (1.6 s) with stride 6.

Normalisation, per window:
- translate to the mid-hip point of the window's first frame (keeps motion inside the window);
- scale by the median torso length (shoulder-mid to hip-mid), so camera distance does not matter;
- low-confidence joints are masked rather than set to zero, and the mask is fed to the model.

### 4.2 Three complementary scorers

1. **Normality model (unsupervised).** A small spatio-temporal transformer trained only on
   normal behaviour to reconstruct masked joints and predict the next frames. Its anomaly score
   is the reconstruction and prediction error. It catches "unusual" without ever seeing a theft,
   which is what the literature benchmarks. Its known weakness: shoplifting looks a lot like
   shopping.

2. **Concealment cues (interpretable).** Body-relative kinematic signals, computed per frame and
   aggregated over windows:
   - wrist-to-pocket / waistband / torso-centre distances (in torso units) and how long a hand
     stays there;
   - "reach then conceal": a hand extended away from the body (reaching) followed within a few
     seconds by that hand moving to a hidden body zone;
   - hand hidden behind the torso (wrist confidence drops while the elbow stays visible);
   - head scanning: frequency of left-right head yaw changes from the nose/ear geometry;
   - crouching and body-turn-away relative to the camera.
   These cues give the alert its reason.

3. **Supervised theft head (weak supervision).** A gradient-boosted classifier on window
   features (cues and normality score) trained on labelled thefts from one dataset and tested
   on the other, never on the same videos.

### 4.3 Fusion and decision
- Scores are calibrated (isotonic regression on a validation split) and combined.
- Per-track temporal smoothing (exponential moving average), then a **hysteresis** state machine
  (`normal → watch → alert`) so one noisy frame cannot raise an alert, and an alert needs
  sustained evidence.
- The operating threshold is chosen for a target false-alarm budget (for example at most one
  false alert per camera per hour), not for the best F1.

## 5. Evaluation

- **Frame-level AUC-ROC, AUC-PR, EER** on PoseLift, computed the same way as the published table.
- **Bootstrap confidence intervals over videos.** With 43 theft videos, a two-point gap can be
  noise, so the README reports intervals.
- **Event-level metrics**: share of theft events caught, time from concealment to alert, and
  **false alarms per hour** of normal footage.
- **Cross-dataset**: train on RetailS, test on PoseLift, and the reverse, after the leakage check.
- **Ablations**: each scorer alone, each pair, all three; with and without normalisation.
- **Robustness**: performance by camera view; keypoint noise and dropped joints injected at test
  time; frame-rate changes.
- **Speed**: frames per second and latency per stage on the target laptop.

## 6. Live pipeline

```
video / webcam ─► YOLO11-pose (OpenVINO, CPU) ─► ByteTrack ─► per-track buffers (15 fps)
     │                                                              │
     └── pixels dropped here (privacy view)                         ▼
                                                    normalise ─► 3 scorers ─► fusion ─► hysteresis
                                                                                         │
                                       overlay video + side panel + JSON event log ◄─────┘
```

- Same keypoint format as the datasets (COCO-17), resampled to 15 fps so models transfer.
- Two render modes: **privacy view** (skeletons on a blurred or blank background, the default)
  and debug view.
- Event log: one JSON line per alert with track id, time, score, top reasons, and a short
  skeleton-only clip. No face crops, ever.

## 7. Deliverables

- `src/vigie`: data loaders, features, models, evaluation, live pipeline, CLI.
- `scripts/`: download, train, evaluate, demo rendering.
- `docs/report.md`: method, results tables and figures, limitations.
- `README.md`: one GIF, the headline numbers, how to run, ethics and limitations.
- `docs/linkedin.md`: post text in French and English.
- Tests for loaders, normalisation, metrics and the state machine.

## 8. Constraints and decisions

- **Cost: zero.** Everything trains and runs on the author's laptop CPU (Intel i5-1135G7,
  8 GB RAM). Models stay small; RetailS is streamed and subsampled.
- **Licence: AGPL-3.0**, because the live pipeline uses Ultralytics YOLO11 (AGPL). The models in
  `vigie.models` do not depend on it.
- Datasets are never committed; scripts download them from the authors' links.
- Results are reported as measured, including where Vigie does not beat a baseline.

## 9. Known limitations, stated up front

- One store, US, a few hundred videos: generalisation to French stores is unproven.
- Pose-only means it is blind to small hand-object details.
- Staged thefts differ from real ones; the staged-versus-real gap is measured, not assumed away.
- Privacy by design reduces risk but does not make deployment legal; that depends on the law.
