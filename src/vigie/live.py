"""Live pipeline: video or webcam -> skeletons -> per-person risk -> privacy-view overlay + event log.

Pixels are used only to find skeletons. Everything downstream, including what is stored,
is pose data. The rendered "privacy view" pixelates the scene so nobody is recognisable.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .cues import frame_features, window_features
from .data import Track
from .geometry import L_HIP, R_HIP, fill_gaps, normalise
from .render import COLORS, draw_person, panel, privacy_background, sigmoid
from .scorer import Scorer
from .windows import WIN

FPS = 15
HISTORY = 70  # frames kept per person (4.7 s): window + "reach then conceal" look-back
STALE = 15  # forget a person unseen for 1 s



@dataclass
class PersonState:
    frames: deque = field(default_factory=lambda: deque(maxlen=HISTORY))
    kps: deque = field(default_factory=lambda: deque(maxlen=HISTORY))
    risk: float = 0.0  # smoothed fused score (logit scale)
    above: int = 0
    state: str = "normal"
    armed: bool = True
    reasons: list = field(default_factory=list)
    last_seen: int = 0
    alerts: int = 0


class LivePipeline:
    def __init__(self, scorer: Scorer, pose_model: str = "yolo11n-pose.pt", imgsz: int = 640,
                 update_every: int = 3, ema: float = 0.35, min_frames: int = 6, watch_margin: float = 1.0):  # fmt: skip
        from ultralytics import YOLO  # optional dependency, imported lazily

        self.yolo = YOLO(pose_model, task="pose")
        self.scorer = scorer
        self.imgsz = imgsz
        self.update_every = update_every
        self.ema = ema
        self.min_frames = min_frames  # consecutive updates above threshold (in pipeline frames)
        self.thr = scorer.threshold
        self.watch = scorer.threshold - watch_margin
        self.people: dict[int, PersonState] = {}
        self.events: list[dict] = []
        self.t = 0  # pipeline frame counter at 15 fps

    # ------------------------------------------------------------------ perception
    def _detect(self, frame: np.ndarray) -> list[tuple[int, np.ndarray]]:
        res = self.yolo.track(frame, persist=True, tracker="bytetrack.yaml", imgsz=self.imgsz,
                              conf=0.3, classes=[0], verbose=False)[0]  # fmt: skip
        if res.boxes is None or res.boxes.id is None or res.keypoints is None:
            return []
        ids = res.boxes.id.int().cpu().numpy()
        kps = res.keypoints.data.cpu().numpy().astype(np.float32)  # (N, 17, 3)
        return list(zip(ids.tolist(), kps))

    # ------------------------------------------------------------------ scoring
    def _update_scores(self) -> None:
        batch_pose, batch_cues, batch_ff, owners = [], [], [], []
        for pid, p in self.people.items():
            if self.t - p.last_seen > 2 or len(p.frames) < 8:
                continue
            tr = Track(str(pid), np.array(p.frames, np.int32), np.stack(p.kps))
            segs = fill_gaps(tr)
            frames, kps = segs[-1]
            if len(frames) < 8:
                continue
            if len(frames) < WIN:
                pad = WIN - len(frames)
                kps = np.concatenate([np.repeat(kps[:1], pad, 0), kps])
            body = normalise(kps, anchor="frame")
            track_xy = normalise(kps, anchor="first")
            ff = frame_features(body, track_xy)
            start = len(kps) - WIN
            seq = track_xy[start:].copy()
            hip0 = (seq[0, L_HIP, :2] + seq[0, R_HIP, :2]) / 2
            seq[..., :2] -= hip0
            batch_pose.append(seq)
            batch_cues.append(window_features(ff, np.array([start]), WIN)[0])
            batch_ff.append(ff[start:])
            owners.append(pid)
        if not owners:
            return
        fused = self.scorer.score(np.stack(batch_pose), np.stack(batch_cues))  # (B, WIN)
        for pid, f, ff in zip(owners, fused, batch_ff):
            p = self.people[pid]
            now = float(f[-6:].mean())
            p.risk = now if p.risk == 0.0 else self.ema * now + (1 - self.ema) * p.risk
            if p.armed:
                p.above = p.above + self.update_every if p.risk >= self.thr else 0
                if p.above >= self.min_frames:
                    p.state, p.armed, p.above = "alert", False, 0
                    p.alerts += 1
                    p.reasons = self.scorer.cue_ref.reasons(ff) or ["unusual movement"]
                    self.events.append({"t": round(self.t / FPS, 2), "person": int(pid),
                                        "risk": round(sigmoid(p.risk), 3), "reasons": p.reasons})  # fmt: skip
                else:
                    p.state = "watch" if p.risk >= self.watch else "normal"
            elif p.risk < self.thr - 1.5:  # calm again: re-arm
                p.armed, p.state = True, "normal"

    # ------------------------------------------------------------------ rendering
    def _render(self, frame: np.ndarray, privacy: bool, proc_fps: float) -> np.ndarray:
        h, w = frame.shape[:2]
        frame = privacy_background(frame) if privacy else frame.copy()
        active = [(pid, p) for pid, p in self.people.items() if self.t - p.last_seen <= 2 and p.kps]
        for pid, p in active:
            draw_person(frame, p.kps[-1], p.state, f"ID {pid}  {100 * sigmoid(p.risk):.0f}%")
        if any(p.state == "alert" for _, p in active):
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), COLORS["alert"], 8)
        people = [(f"ID {pid}", p.state, p.risk) for pid, p in sorted(active, key=lambda a: -a[1].risk)]
        n_alerts = sum(p.alerts for p in self.people.values())
        side = panel(h, "privacy view - skeletons only",
                     f"people {len(active)}   alerts {n_alerts}   {proc_fps:.0f} fps CPU",
                     people, self.thr,
                     [{**e, "person": f"ID {e['person']}"} for e in self.events])  # fmt: skip
        return np.hstack([frame, side])

    # ------------------------------------------------------------------ main loop
    def run(self, source, out_video: str | None = None, events_path: str | None = None,
            privacy: bool = True, show: bool = False, max_seconds: float | None = None) -> dict:  # fmt: skip
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {source}")
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = fps_in / FPS
        writer, next_t, i, last = None, 0.0, 0, None
        t_proc, n_proc = 0.0, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i + 1e-6 < next_t:  # keep only frames on the 15 fps grid
                i += 1
                continue
            next_t += step
            i += 1
            t0 = time.perf_counter()
            for pid, k in self._detect(frame):
                p = self.people.setdefault(pid, PersonState())
                p.frames.append(self.t)
                p.kps.append(k)
                p.last_seen = self.t
            if self.t % self.update_every == 0:
                self._update_scores()
            self.people = {k: v for k, v in self.people.items() if self.t - v.last_seen <= STALE}
            t_proc += time.perf_counter() - t0
            n_proc += 1
            last = self._render(frame, privacy, n_proc / max(t_proc, 1e-6))
            if out_video:
                if writer is None:
                    Path(out_video).parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), FPS,
                                             (last.shape[1], last.shape[0]))  # fmt: skip
                writer.write(last)
            if show:
                cv2.imshow("vigie", last)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            self.t += 1
            if max_seconds and self.t / FPS >= max_seconds:
                break
        cap.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()
        if events_path:
            Path(events_path).write_text("\n".join(json.dumps(e) for e in self.events))
        return {"frames": self.t, "proc_fps": n_proc / max(t_proc, 1e-6), "events": self.events}
