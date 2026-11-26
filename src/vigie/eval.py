"""Evaluation: from window scores to frame scores, benchmark metrics, and false alarms per hour.

Frame-level protocol (same as the PoseLift / STG-NF benchmark):
  1. each person's frame score = mean of the scores of the windows covering that frame;
  2. a video frame's score = the highest score among the people in it (the most suspicious
     person decides); frames with nobody get the lowest score;
  3. optional Gaussian smoothing over time within each video;
  4. all frames of all videos are concatenated and compared with the frame labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from .data import Video
from .windows import WIN, WindowSet

FPS = 15


def frame_scores(
    videos: list[Video],
    ws: WindowSet,
    win_scores: np.ndarray,
    per_frame: np.ndarray | None = None,
    sigma: float = 0.0,
) -> list[np.ndarray]:
    """Video-level frame scores, one array of length n_frames per video.

    ``per_frame`` (W, WIN) optionally gives a score per frame inside each window; otherwise the
    window score is spread over its frames.
    """
    vals = per_frame if per_frame is not None else np.repeat(win_scores[:, None], WIN, axis=1)
    floor = float(np.min(vals)) if len(vals) else 0.0
    out = []
    by_video: dict[int, list[int]] = {}
    for i, v in enumerate(ws.video):
        by_video.setdefault(int(v), []).append(i)
    for vi, v in enumerate(videos):
        n = v.n_frames
        best = np.full(n, -np.inf)
        rows = np.array(by_video.get(vi, []), dtype=np.int64)
        for ti in np.unique(ws.track[rows]) if len(rows) else []:
            r = rows[ws.track[rows] == ti]
            acc, cnt = np.zeros(n), np.zeros(n)
            idx = ws.frame0[r][:, None] + np.arange(WIN)[None]
            ok = (idx >= 0) & (idx < n)
            np.add.at(acc, idx[ok], vals[r][ok])
            np.add.at(cnt, idx[ok], 1)
            has = cnt > 0
            best[has] = np.maximum(best[has], acc[has] / cnt[has])
        best[~np.isfinite(best)] = floor
        if sigma > 0:
            best = gaussian_filter1d(best, sigma)
        out.append(best)
    return out


@dataclass
class Metrics:
    auc_roc: float
    auc_pr: float
    eer: float
    pos_rate: float

    def row(self) -> str:
        return f"AUC-ROC {100 * self.auc_roc:5.2f} | AUC-PR {100 * self.auc_pr:5.2f} | EER {self.eer:.3f}"


def _eer(y: np.ndarray, s: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2)


def metrics(videos: list[Video], scores: list[np.ndarray]) -> Metrics:
    y = np.concatenate([v.labels[: len(s)] for v, s in zip(videos, scores)])
    s = np.concatenate([s[: len(v.labels)] for v, s in zip(videos, scores)])
    return Metrics(roc_auc_score(y, s), average_precision_score(y, s), _eer(y, s), float(y.mean()))


def bootstrap_ci(
    videos: list[Video], scores: list[np.ndarray], n: int = 1000, seed: int = 0, alpha: float = 0.05
) -> dict[str, tuple[float, float]]:
    """Confidence intervals by resampling whole videos (frames of one video are not independent)."""
    rng = np.random.default_rng(seed)
    k = len(videos)
    vals = {"auc_roc": [], "auc_pr": []}
    for _ in range(n):
        pick = rng.integers(0, k, k)
        y = np.concatenate([videos[i].labels[: len(scores[i])] for i in pick])
        if y.min() == y.max():
            continue
        s = np.concatenate([scores[i][: len(videos[i].labels)] for i in pick])
        vals["auc_roc"].append(roc_auc_score(y, s))
        vals["auc_pr"].append(average_precision_score(y, s))
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        k_: (float(np.percentile(v, lo)), float(np.percentile(v, hi))) for k_, v in vals.items()
    }


# ---------------------------------------------------------------- alerts and false alarms


@dataclass
class AlertPolicy:
    """Per-person hysteresis: an alert needs sustained evidence, and re-arms only after calm."""

    threshold: float
    min_frames: int = 8  # about 0.5 s above threshold before alerting
    release: float = 0.8  # fall below threshold * release to re-arm
    ema: float = 0.3  # smoothing of the score (weight of the newest frame)

    def run(self, scores: np.ndarray) -> list[int]:
        """Frame indices at which alerts fire for one person's score sequence."""
        alerts, above, armed, s = [], 0, True, None
        for i, x in enumerate(scores):
            s = x if s is None else self.ema * x + (1 - self.ema) * s
            if armed:
                above = above + 1 if s >= self.threshold else 0
                if above >= self.min_frames:
                    alerts.append(i)
                    armed, above = False, 0
            elif s < self.threshold * self.release:
                armed = True
        return alerts


def person_frame_scores(ws: WindowSet, vals: np.ndarray, vi: int, ti: int) -> np.ndarray:
    """Contiguous per-frame score for one track (frames between windows are interpolated)."""
    r = np.flatnonzero((ws.video == vi) & (ws.track == ti))
    if len(r) == 0:
        return np.zeros(0)
    idx = ws.frame0[r][:, None] + np.arange(WIN)[None]
    lo, hi = idx.min(), idx.max() + 1
    acc, cnt = np.zeros(hi - lo), np.zeros(hi - lo)
    np.add.at(acc, idx - lo, vals[r])
    np.add.at(cnt, idx - lo, 1)
    has = cnt > 0
    return np.interp(np.arange(hi - lo), np.flatnonzero(has), acc[has] / cnt[has])


def false_alarms_per_hour(
    videos: list[Video], ws: WindowSet, per_frame: np.ndarray, policy: AlertPolicy
) -> tuple[float, int, float]:
    """Alerts raised on normal-only footage, per camera-hour. Returns (rate, alerts, hours)."""
    alerts = 0
    for vi in np.unique(ws.video):
        for ti in np.unique(ws.track[ws.video == vi]):
            alerts += len(policy.run(person_frame_scores(ws, per_frame, vi, ti)))
    hours = sum(v.n_frames for v in videos) / FPS / 3600
    return alerts / max(hours, 1e-9), alerts, hours


def event_recall(
    videos: list[Video], ws: WindowSet, per_frame: np.ndarray, policy: AlertPolicy, slack: int = 15
) -> tuple[float, int, list[float]]:
    """Share of labelled theft events caught by at least one alert (within ``slack`` frames),
    and the delay from event start to the alert in seconds."""
    caught, total, delays = 0, 0, []
    for vi, v in enumerate(videos):
        if v.labels is None or v.labels.max() == 0:
            continue
        lab = v.labels
        edges = np.flatnonzero(np.diff(np.r_[0, lab, 0]))
        events = list(zip(edges[::2], edges[1::2]))
        fired = []
        for ti in np.unique(ws.track[ws.video == vi]):
            r = np.flatnonzero((ws.video == vi) & (ws.track == ti))
            start = ws.frame0[r].min()
            fired += [start + a for a in policy.run(person_frame_scores(ws, per_frame, vi, ti))]
        for s, e in events:
            total += 1
            hits = [f for f in fired if s - slack <= f <= e + slack]
            if hits:
                caught += 1
                delays.append((min(hits) - s) / FPS)
    return caught / max(total, 1), total, delays
