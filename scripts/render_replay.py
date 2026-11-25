"""Render real test-set thefts as a skeleton-only replay with Vigie's live risk and alerts.

The test data (PoseLift / RetailS real-world) contains no pixels, only skeletons, so this is
exactly what Vigie "sees". The labelled theft interval is shown in the corner for honesty.

Usage: python scripts/render_replay.py --clips 6 --out docs/media/replay.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from vigie.data import get_split
from vigie.eval import person_frame_scores
from vigie.geometry import torso_vectors
from vigie.render import FONT, PANEL_W, draw_person, panel, sigmoid
from vigie.scorer import Scorer
from vigie.windows import WIN, build_windows

FPS = 15
W, H = 960, 720


def states(scores: np.ndarray, thr: float, ema=0.35, min_frames=6, watch_margin=1.0):
    """Smoothed risk, state per frame and alert frames (same rules as the live pipeline)."""
    out_r, out_s, alerts = [], [], []
    r, above, armed = None, 0, True
    for i, x in enumerate(scores):
        r = x if r is None else ema * x + (1 - ema) * r
        if armed:
            above = above + 1 if r >= thr else 0
            if above >= min_frames:
                armed, above = False, 0
                alerts.append(i)
                st = "alert"
            else:
                st = "watch" if r >= thr - watch_margin else "normal"
        else:
            st = "alert"
            if r < thr - 1.5:
                armed, st = True, "normal"
        out_r.append(r)
        out_s.append(st)
    return np.array(out_r), out_s, alerts


def scene_transform(video):
    """Rotate the scene so people stand upright, then fit it into the canvas."""
    allk = np.concatenate([t.kps.astype(np.float32) for t in video.tracks])
    tv = torso_vectors(allk)
    ok = np.linalg.norm(tv, axis=1) > 1e-3
    u = tv[ok] / np.linalg.norm(tv[ok], axis=1, keepdims=True)
    ang = np.arctan2(u[:, 1].mean(), u[:, 0].mean())
    th = -np.pi / 2 - ang
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    pts = allk[..., :2][allk[..., 2] > 0.2] @ rot.T
    lo, hi = np.percentile(pts, 1, axis=0), np.percentile(pts, 99, axis=0)
    span = np.maximum(hi - lo, 1)
    s = min((W - 160) / span[0], (H - 160) / span[1])
    off = np.array([W, H]) / 2 - (lo + hi) / 2 * s

    def f(k):
        out = k.astype(np.float32).copy()
        out[:, :2] = k[:, :2].astype(np.float32) @ rot.T * s + off
        return out

    return f, max(2, int(round(3 * min(1.0, s))))


def render_clip(scorer: Scorer, video, ws, fused, vi: int, writer, clip_no: int, n_clips: int):
    tracks = {}
    for ti, t in enumerate(video.tracks):
        r = np.flatnonzero((ws.video == vi) & (ws.track == ti))
        if not len(r):
            continue
        sc = person_frame_scores(ws, fused, vi, ti)
        start = int((ws.frame0[r][:, None] + np.arange(WIN)[None]).min())
        risk, st, al = states(sc, scorer.threshold)
        tracks[ti] = (start, risk, st, al, dict(zip(t.frames.tolist(), range(len(t)))))
    tf, thick = scene_transform(video)
    events = []
    lab = video.labels
    for f in range(video.n_frames):
        img = np.full((H, W, 3), (34, 31, 28), np.uint8)
        for gx in range(0, W, 60):
            cv2.line(img, (gx, 0), (gx, H), (42, 39, 36), 1)
        for gy in range(0, H, 60):
            cv2.line(img, (0, gy), (W, gy), (42, 39, 36), 1)
        people, any_alert = [], False
        for ti, (start, risk, st, al, fmap) in tracks.items():
            i = f - start
            if i < 0 or i >= len(risk):
                continue
            label = f"ID {video.tracks[ti].pid}"
            if i in al:
                r = np.flatnonzero((ws.video == vi) & (ws.track == ti) & (ws.frame0 <= f) & (ws.frame0 + WIN > f))
                ff = ws.frame_cues[r[0]][: f - ws.frame0[r[0]] + 1].astype(np.float32) if len(r) else None
                reasons = (scorer.cue_ref.reasons(ff) if ff is not None and len(ff) else []) or ["unusual movement"]
                events.append({"t": f / FPS, "person": label, "reasons": reasons})
            people.append((label, st[i], risk[i]))
            any_alert |= st[i] == "alert"
            if f in fmap:
                k = tf(video.tracks[ti].kps[fmap[f]])
                draw_person(img, k, st[i], f"{label}  {100 * sigmoid(risk[i]):.0f}%", 0.2, thick)
        if any_alert:
            cv2.rectangle(img, (0, 0), (W - 1, H - 1), (60, 60, 240), 8)
        cv2.putText(img, f"REAL CCTV THEFT - clip {clip_no}/{n_clips} ({video.name})", (20, 34), FONT,
                    0.6, (200, 200, 200), 1, cv2.LINE_AA)  # fmt: skip
        if lab is not None and f < len(lab) and lab[f]:
            cv2.putText(img, "ground truth: theft", (20, 62), FONT, 0.55, (120, 120, 255), 1, cv2.LINE_AA)
        cv2.putText(img, f"{f / FPS:5.1f} s", (W - 110, H - 20), FONT, 0.55, (170, 170, 170), 1, cv2.LINE_AA)
        side = panel(H, "skeletons only - no pixels stored", f"people {len(people)}   alerts {len(events)}",
                     sorted(people, key=lambda p: -p[2]), scorer.threshold, events,
                     footer="data: PoseLift (WACV 2025), real store")  # fmt: skip
        writer.write(np.hstack([img, side]))
    return len(events)


def title_card(writer, lines: list[str], seconds: float = 2.5):
    img = np.full((H, W + PANEL_W, 3), (24, 22, 20), np.uint8)
    y = H // 2 - 30 * len(lines) // 2
    for i, line in enumerate(lines):
        s, t = (1.1, 2) if i == 0 else (0.65, 1)
        size = cv2.getTextSize(line, FONT, s, t)[0]
        cv2.putText(img, line, ((W + PANEL_W - size[0]) // 2, y), FONT, s, (235, 235, 235), t, cv2.LINE_AA)
        y += 48 if i == 0 else 34
    for _ in range(int(seconds * FPS)):
        writer.write(img)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="runs/heads/bundle")
    ap.add_argument("--clips", type=int, default=6)
    ap.add_argument("--out", default="docs/media/replay_raw.mp4")
    ap.add_argument("--names", nargs="*", default=None, help="specific clip names")
    args = ap.parse_args()
    scorer = Scorer.load(args.bundle)
    videos = get_split("retails_real")
    ws = build_windows(videos, min_len=8)
    fused = scorer.score(ws.pose, ws.cues)
    if args.names:
        pick = [i for i, v in enumerate(videos) if v.name in args.names]
    else:  # the longest clips: more context, chosen without looking at the scores
        pick = sorted(range(len(videos)), key=lambda i: -videos[i].n_frames)[: args.clips]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W + PANEL_W, H))
    title_card(writer, ["VIGIE", "privacy-first shoplifting detection",
                        "real thefts from a real store, seen as skeletons only"])  # fmt: skip
    for n, vi in enumerate(pick, 1):
        k = render_clip(scorer, videos[vi], ws, fused, vi, writer, n, len(pick))
        print(f"{videos[vi].name}: {videos[vi].n_frames} frames, {k} alerts, "
              f"theft frames {int(videos[vi].labels.sum())}")  # fmt: skip
    writer.release()
    print("written", args.out)


if __name__ == "__main__":
    main()
