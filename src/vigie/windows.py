"""Turn videos into fixed-length windows: pose tensors, cue features and an index back to frames."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cues import frame_features, window_features
from .data import Video
from .geometry import L_HIP, R_HIP, fill_gaps, normalise, window_starts

WIN = 24  # 1.6 s at 15 fps
STRIDE = 6


@dataclass
class WindowSet:
    pose: np.ndarray  # (W, WIN, 17, 3) float16, normalised, centred on the window's first mid-hip
    cues: np.ndarray  # (W, F*4) float32 aggregated concealment cues
    frame_cues: np.ndarray  # (W, WIN, F) float16 per-frame cues (for explanations)
    video: np.ndarray  # (W,) int32 index into the video list
    track: np.ndarray  # (W,) int32 index into video.tracks
    frame0: np.ndarray  # (W,) int32 first frame index of the window (in video frames)
    label: np.ndarray  # (W,) float32 share of theft frames in the window (nan if unlabelled)

    def __len__(self) -> int:
        return len(self.video)

    def subset(self, idx: np.ndarray) -> WindowSet:
        return WindowSet(*(getattr(self, f)[idx] for f in self.__dataclass_fields__))


def _empty() -> WindowSet:
    return WindowSet(
        np.zeros((0, WIN, 17, 3), np.float16), np.zeros((0, 56), np.float32),
        np.zeros((0, WIN, 14), np.float16), *(np.zeros(0, np.int32) for _ in range(3)),
        np.zeros(0, np.float32),
    )  # fmt: skip


def build_windows(
    videos: list[Video],
    stride: int = STRIDE,
    max_windows: int | None = None,
    seed: int = 0,
    min_len: int = WIN,
) -> WindowSet:
    """Window every track segment of every video.

    With ``max_windows`` the result is a uniform random sample (used for the large normal set).
    Tracks shorter than WIN are padded by repeating their edge frames when they are at least
    ``min_len`` long; the real-world test clips are short, so their tests use ``min_len=8``.
    """
    rng = np.random.default_rng(seed)

    def segments(t):
        for frames, kps in fill_gaps(t):
            if len(frames) < min_len:
                continue
            if len(frames) < WIN:  # pad short tracks at both ends
                pad = WIN - len(frames)
                a, b = pad // 2, pad - pad // 2
                kps = np.concatenate([np.repeat(kps[:1], a, 0), kps, np.repeat(kps[-1:], b, 0)])
                frames = np.arange(frames[0] - a, frames[-1] + b + 1, dtype=np.int32)
            yield frames, kps

    # first pass: count windows per track without keeping any pose data (RAM-friendly)
    counts = []  # (video, track, [windows per segment])
    for vi, v in enumerate(videos):
        for ti, t in enumerate(v.tracks):
            n = [len(window_starts(len(f), WIN, stride)) for f, _ in segments(t)]
            if sum(n):
                counts.append((vi, ti, n))
    total = sum(sum(c[2]) for c in counts)
    keep = None
    if max_windows is not None and total > max_windows:
        keep = np.zeros(total, bool)
        keep[rng.choice(total, max_windows, replace=False)] = True

    def plan():
        offset = 0
        for vi, ti, n in counts:
            if keep is not None and not keep[offset : offset + sum(n)].any():
                offset += sum(n)
                continue
            for frames, kps in segments(videos[vi].tracks[ti]):
                starts = window_starts(len(frames), WIN, stride)
                sel = (
                    np.ones(len(starts), bool)
                    if keep is None
                    else keep[offset : offset + len(starts)]
                )
                offset += len(starts)
                if sel.any():
                    yield vi, ti, frames, kps, starts[sel]

    out = {k: [] for k in ("pose", "cues", "frame_cues", "video", "track", "frame0", "label")}
    for vi, ti, frames, kps, starts in plan():
        body = normalise(kps, anchor="frame")
        track_xy = normalise(kps, anchor="first")
        ff = frame_features(body, track_xy)
        idx = starts[:, None] + np.arange(WIN)[None]
        seq = track_xy[idx].copy()  # (W, WIN, 17, 3)
        hip0 = (seq[:, 0, L_HIP, :2] + seq[:, 0, R_HIP, :2]) / 2
        seq[..., :2] -= hip0[:, None, None, :]
        out["pose"].append(seq.astype(np.float16))
        out["cues"].append(window_features(ff, starts, WIN))
        out["frame_cues"].append(ff[idx].astype(np.float16))
        out["video"].append(np.full(len(starts), vi, np.int32))
        out["track"].append(np.full(len(starts), ti, np.int32))
        out["frame0"].append(frames[starts].astype(np.int32))
        lab = videos[vi].labels
        if lab is None:
            out["label"].append(np.full(len(starts), np.nan, np.float32))
        else:
            fidx = np.clip(frames[idx], 0, len(lab) - 1)
            out["label"].append(lab[fidx].mean(axis=1).astype(np.float32))
    if not out["video"]:
        return _empty()
    return WindowSet(**{k: np.concatenate(v) for k, v in out.items()})


def window_frame_labels(videos: list[Video], ws: WindowSet) -> np.ndarray:
    """Frame labels inside each window, (W, WIN) uint8 (0 for unlabelled videos or padding)."""
    out = np.zeros((len(ws), WIN), np.uint8)
    for i, (vi, f0) in enumerate(zip(ws.video, ws.frame0)):
        lab = videos[vi].labels
        if lab is None:
            continue
        idx = f0 + np.arange(WIN)
        ok = (idx >= 0) & (idx < len(lab))
        out[i, ok] = lab[idx[ok]]
    return out


def save_windows(ws: WindowSet, path) -> None:
    np.savez(path, **{f: getattr(ws, f) for f in ws.__dataclass_fields__})


def load_windows(path) -> WindowSet:
    z = np.load(path, allow_pickle=False)
    return WindowSet(**{f: z[f] for f in WindowSet.__dataclass_fields__})
