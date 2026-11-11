"""Loading pose datasets in the STG-NF JSON layout (PoseLift, RetailS).

Each video is one JSON file::

    {"<person_id>": {"<frame_idx>": {"keypoints": [x0, y0, c0, ..., x16, y16, c16], ...}}}

with COCO-17 keypoints, plus an optional frame-level label array ``<video>.npy``
(0 = normal, 1 = shoplifting) in a sibling ``gt/test_frame_mask`` folder.

Parsed splits are cached as ``.npz`` (no pickle) so that later runs load in seconds.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

N_JOINTS = 17
COCO_NAMES = (
    "nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder", "l_elbow",
    "r_elbow", "l_wrist", "r_wrist", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle",
)  # fmt: skip


@dataclass
class Track:
    """One person's pose over time inside one video."""

    pid: str
    frames: np.ndarray  # (N,) int32, strictly increasing frame indices
    kps: np.ndarray  # (N, 17, 3) float32: x, y, confidence

    def __len__(self) -> int:
        return len(self.frames)


@dataclass
class Video:
    name: str  # e.g. "01_0222"
    camera: str
    tracks: list[Track] = field(default_factory=list)
    labels: np.ndarray | None = None  # (n_frames,) uint8, or None for unlabelled data
    n_frames: int = 0


def camera_of(name: str) -> str:
    return name.split("_")[0].lstrip("0") or "0"


def _clean(kps: np.ndarray) -> np.ndarray:
    """Replace NaN coordinates with the joint's nearest valid value; NaN joints get confidence 0."""
    bad = ~np.isfinite(kps).all(axis=2)  # (N, 17)
    kps = kps.copy()
    if bad.any():
        kps[bad, 2] = 0.0
        for j in np.where(bad.any(axis=0))[0]:
            ok = np.where(~bad[:, j])[0]
            if len(ok) == 0:
                kps[:, j, :2] = 0.0
                continue
            idx = np.arange(len(kps))
            nearest = ok[np.clip(np.searchsorted(ok, idx), 0, len(ok) - 1)]
            kps[bad[:, j], j, :2] = kps[nearest[bad[:, j]], j, :2]
    return kps


def load_pose_json(path: Path, min_len: int = 1) -> list[Track]:
    data = json.loads(Path(path).read_text())
    tracks = []
    for pid, frames in data.items():
        if len(frames) < min_len:
            continue
        idx = np.array(sorted(int(k) for k in frames), dtype=np.int32)
        kps = np.array(
            [frames[str(i)]["keypoints"] for i in idx], dtype=np.float32
        ).reshape(len(idx), N_JOINTS, 3)
        tracks.append(Track(pid=str(pid), frames=idx, kps=_clean(kps)))
    return tracks


def load_split(
    pose_dir: Path,
    label_dir: Path | None = None,
    min_len: int = 1,
    files: list[Path] | None = None,
) -> list[Video]:
    files = files if files is not None else sorted(Path(pose_dir).glob("*.json"))
    videos = []
    for f in files:
        name = f.stem
        tracks = load_pose_json(f, min_len=min_len)
        last = max((int(t.frames[-1]) for t in tracks), default=-1)
        labels = None
        if label_dir is not None:
            lp = Path(label_dir) / f"{name}.npy"
            if lp.exists():
                labels = np.load(lp, allow_pickle=False).astype(np.uint8).ravel()
        n_frames = max(last + 1, len(labels) if labels is not None else 0)
        videos.append(Video(name, camera_of(name), tracks, labels, n_frames))
    return videos


# ---------------------------------------------------------------- npz cache (pickle-free)


def save_cache(videos: list[Video], path: Path) -> None:
    meta = {"videos": []}
    frames, kps, owner = [], [], []
    labels, label_owner = [], []
    for vi, v in enumerate(videos):
        meta["videos"].append(
            {"name": v.name, "camera": v.camera, "n_frames": v.n_frames,
             "pids": [t.pid for t in v.tracks], "has_labels": v.labels is not None}
        )  # fmt: skip
        for ti, t in enumerate(v.tracks):
            frames.append(t.frames)
            kps.append(t.kps.astype(np.float16))
            owner.append(np.full((len(t), 2), (vi, ti), dtype=np.int32))
        if v.labels is not None:
            labels.append(v.labels)
            label_owner.append(np.full(len(v.labels), vi, dtype=np.int32))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        meta=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8),
        frames=np.concatenate(frames) if frames else np.zeros(0, np.int32),
        kps=np.concatenate(kps) if kps else np.zeros((0, N_JOINTS, 3), np.float16),
        owner=np.concatenate(owner) if owner else np.zeros((0, 2), np.int32),
        labels=np.concatenate(labels) if labels else np.zeros(0, np.uint8),
        label_owner=np.concatenate(label_owner) if label_owner else np.zeros(0, np.int32),
    )


def load_cache(path: Path) -> list[Video]:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(z["meta"].tobytes().decode())
    # keypoints stay float16 in memory (half the RAM); geometry casts each segment when used
    frames, kps, owner = z["frames"], z["kps"], z["owner"]
    labels, label_owner = z["labels"], z["label_owner"]
    videos = [
        Video(m["name"], m["camera"], [], None, m["n_frames"]) for m in meta["videos"]
    ]
    # split the flat arrays back into tracks: rows are contiguous per (video, track)
    if len(owner):
        change = np.flatnonzero(np.any(np.diff(owner, axis=0) != 0, axis=1)) + 1
        starts = np.r_[0, change]
        ends = np.r_[change, len(owner)]
        for s, e in zip(starts, ends):
            vi, ti = owner[s]
            videos[vi].tracks.append(Track(meta["videos"][vi]["pids"][ti], frames[s:e], kps[s:e]))
    if len(label_owner):
        change = np.flatnonzero(np.diff(label_owner) != 0) + 1
        for s, e in zip(np.r_[0, change], np.r_[change, len(label_owner)]):
            videos[label_owner[s]].labels = labels[s:e].copy()
    return videos


# ---------------------------------------------------------------- dataset registry

RAW = Path("data/raw")
CACHE = Path("data/cache")

SPLITS = {
    # name: (pose dir, label dir)
    "retails_train": (RAW / "retails/RetailS/RetailS_train/pose/train", None),
    "retails_staged": (
        RAW / "retails/RetailS/RetailS_test_staged/pose/test",
        RAW / "retails/RetailS/RetailS_test_staged/gt/test_frame_mask",
    ),
    "retails_real": (
        RAW / "retails/RetailS/RetailS_test_realworld/pose/test",
        RAW / "retails/RetailS/RetailS_test_realworld/gt/test_frame_mask",
    ),
    "poselift_train": (RAW / "poselift/PoseLift/Json_files/data/PoseLift/pose/train", None),
    "poselift_test": (
        RAW / "poselift/PoseLift/Json_files/data/PoseLift/pose/test",
        RAW / "poselift/PoseLift/Json_files/data/PoseLift/gt/test_frame_mask",
    ),
}


def retails_train_files(holdout_frac: float = 0.08, seed: int = 0) -> tuple[list[Path], list[Path]]:
    """Split RetailS-train videos by file into (fit, holdout).

    The holdout is long, unlabelled, normal-only footage used to measure false alarms per hour.
    The split is stratified by camera so every view appears in both parts.
    """
    pose_dir = SPLITS["retails_train"][0]
    by_cam: dict[str, list[Path]] = {}
    for f in sorted(pose_dir.glob("*.json")):
        by_cam.setdefault(camera_of(f.stem), []).append(f)
    rng = random.Random(seed)
    fit, hold = [], []
    for cam in sorted(by_cam):
        fs = by_cam[cam][:]
        rng.shuffle(fs)
        k = max(1, round(len(fs) * holdout_frac))
        hold += fs[:k]
        fit += fs[k:]
    return sorted(fit), sorted(hold)


def get_split(name: str, rebuild: bool = False, min_len: int = 1) -> list[Video]:
    """Load a named split, building its npz cache on first use."""
    cache = CACHE / f"{name}.npz"
    if cache.exists() and not rebuild:
        return load_cache(cache)
    if name in ("retails_fit", "retails_holdout"):
        fit, hold = retails_train_files()
        files = fit if name == "retails_fit" else hold
        videos = load_split(SPLITS["retails_train"][0], None, min_len=min_len, files=files)
    else:
        pose_dir, label_dir = SPLITS[name]
        videos = load_split(pose_dir, label_dir, min_len=min_len)
    save_cache(videos, cache)
    return videos
