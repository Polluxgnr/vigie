"""Pose geometry: gap filling, body-centric normalisation and windowing.

Conventions after normalisation:
  * unit length = the person's torso length (mid-shoulder to mid-hip), so camera distance cancels;
  * "up" is -y, estimated from the person's own median torso direction, so the camera's
    mounting angle cancels (RetailS cameras are rotated by about 90 degrees, for example);
  * low-confidence joints are kept but flagged through the confidence channel.
"""

from __future__ import annotations

import numpy as np

from .data import Track

L_SH, R_SH, L_HIP, R_HIP = 5, 6, 11, 12
CONF_MIN = 0.2  # below this a joint is treated as unreliable
LEFT = [1, 3, 5, 7, 9, 11, 13, 15]
RIGHT = [2, 4, 6, 8, 10, 12, 14, 16]
FLIP = np.arange(17)
FLIP[LEFT], FLIP[RIGHT] = RIGHT, LEFT


def fill_gaps(track: Track, max_gap: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return contiguous segments (frames, kps), linearly interpolating gaps up to ``max_gap``.

    Interpolated joints get confidence 0 so models know they were not observed.
    """
    f, k = track.frames, track.kps
    if len(f) == 0:
        return []
    breaks = np.flatnonzero(np.diff(f) > max_gap + 1) + 1
    segs = []
    for s, e in zip(np.r_[0, breaks], np.r_[breaks, len(f)]):
        fs, ks = f[s:e], k[s:e].astype(np.float32)
        full = np.arange(fs[0], fs[-1] + 1, dtype=np.int32)
        if len(full) == len(fs):
            segs.append((fs, ks))
            continue
        out = np.zeros((len(full), 17, 3), dtype=np.float32)
        for j in range(17):
            for c in range(2):
                out[:, j, c] = np.interp(full, fs, ks[:, j, c])
        observed = np.isin(full, fs)
        out[observed, :, 2] = ks[:, :, 2]
        segs.append((full, out))
    return segs


def _mid(kps: np.ndarray, a: int, b: int) -> np.ndarray:
    return (kps[..., a, :2] + kps[..., b, :2]) / 2.0


def torso_vectors(kps: np.ndarray) -> np.ndarray:
    """Mid-hip to mid-shoulder vectors, (..., 2)."""
    return _mid(kps, L_SH, R_SH) - _mid(kps, L_HIP, R_HIP)


def body_frame(kps: np.ndarray, smooth: int = 31) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame rotation angle and scale for a contiguous sequence (N, 17, 3).

    Both are running medians over ``smooth`` frames (about 2 s at 15 fps): stable enough to
    ignore a single bad detection, local enough to follow a person who turns or walks closer.
    """
    tv = torso_vectors(kps)
    ok = (kps[:, [L_SH, R_SH, L_HIP, R_HIP], 2] > CONF_MIN).all(axis=1)
    length = np.linalg.norm(tv, axis=1)
    ok &= length > 1e-3
    n = len(kps)
    if not ok.any():  # nothing reliable: fall back to the raw estimate
        ok = length > 1e-3
    if not ok.any():
        return np.zeros(n), np.ones(n)
    unit = tv / np.maximum(length[:, None], 1e-6)
    # running sums over reliable frames only (vectorised moving average)
    w = ok.astype(np.float64)
    vals = np.column_stack(
        [unit[:, 0] * w, unit[:, 1] * w, np.log(np.maximum(length, 1e-6)) * w, w]
    )
    csum = np.vstack([np.zeros((1, 4)), np.cumsum(vals, axis=0)])
    half = smooth // 2
    lo = np.clip(np.arange(n) - half, 0, n)
    hi = np.clip(np.arange(n) + half + 1, 0, n)
    win = csum[hi] - csum[lo]
    # frames with no reliable neighbour copy the nearest frame that has one
    has = win[:, 3] > 0
    if not has.all():
        idx = np.flatnonzero(has)
        near = idx[np.clip(np.searchsorted(idx, np.arange(n)), 0, len(idx) - 1)]
        win[~has] = win[near[~has]]
    ang = np.arctan2(win[:, 1], win[:, 0])  # circular mean of the torso direction
    scale = np.exp(win[:, 2] / win[:, 3])  # geometric mean: robust to a few bad lengths
    return ang, np.maximum(scale, 1e-3)


def normalise(kps: np.ndarray, anchor: str = "frame") -> np.ndarray:
    """Body-centric coordinates for a contiguous sequence (N, 17, 3) -> (N, 17, 3).

    anchor="frame": each frame centred on its own mid-hip (pure body posture, for cues).
    anchor="first": every frame centred on the first frame's mid-hip (keeps displacement,
    for the sequence models).
    """
    ang, scale = body_frame(kps)
    # rotate so the torso points to -y ("up" in image coordinates)
    theta = -np.pi / 2 - ang
    c, s = np.cos(theta), np.sin(theta)
    rot = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)  # (N, 2, 2)
    hip = _mid(kps, L_HIP, R_HIP)
    if anchor == "first":
        hip = np.repeat(hip[:1], len(kps), axis=0)
    xy = kps[:, :, :2] - hip[:, None, :]
    xy = np.einsum("nij,nkj->nki", rot, xy) / scale[:, None, None]
    out = np.empty_like(kps)
    out[:, :, :2] = xy
    out[:, :, 2] = kps[:, :, 2]
    return out


def window_starts(n: int, size: int, stride: int) -> np.ndarray:
    """Start indices of windows covering a sequence of length n (last window flush with the end)."""
    if n < size:
        return np.zeros(0, dtype=np.int64)
    starts = np.arange(0, n - size + 1, stride)
    if starts[-1] != n - size:
        starts = np.r_[starts, n - size]
    return starts


def flip_lr(kps: np.ndarray) -> np.ndarray:
    """Mirror a normalised pose left/right (data augmentation)."""
    out = kps[..., FLIP, :].copy()
    out[..., 0] *= -1
    return out
