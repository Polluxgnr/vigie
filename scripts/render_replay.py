"""Render real test-set clips as a skeleton-only replay with Vigie's live risk.

The test data (PoseLift / RetailS real-world) contains no pixels, only skeletons, so this is
exactly what Vigie "sees". Under the scene, a risk curve shows each person's fused score over
time; the labelled theft interval is shaded red, so the viewer can judge the model honestly.

Two operating points are drawn:
  * review (amber): 95th percentile of normal footage, for a human to glance at;
  * alert (red):    the shipped threshold, at most 1 false alert per camera-hour.

Clips are chosen by length among those containing a theft, never by score.

Usage: python scripts/render_replay.py --clips 5 --out runs/replay_raw.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from vigie.data import get_split
from vigie.eval import person_frame_scores
from vigie.geometry import torso_vectors
from vigie.render import COLORS, FONT, PANEL_W, draw_person, panel
from vigie.scorer import Scorer
from vigie.windows import WIN, build_windows

FPS = 15
W, H = 960, 720
SCENE_H = 520  # top part: skeletons; bottom part: risk timeline
TL_TOP, TL_BOT = SCENE_H + 30, H - 28
PERSON_COLS = [(214, 120, 42), (52, 104, 235), (122, 175, 27), (164, 123, 232)]  # BGR series


def run_states(scores: np.ndarray, review: float, alert: float, ema=0.35, min_frames=6):
    """Smoothed risk, per-frame state, and (frame, kind) flags, with the live pipeline's rules."""
    out_r, out_s, flags = [], [], []
    r, above_r, above_a, armed_r, armed_a = None, 0, 0, True, True
    for i, x in enumerate(scores):
        r = x if r is None else ema * x + (1 - ema) * r
        above_r = above_r + 1 if r >= review else 0
        above_a = above_a + 1 if r >= alert else 0
        if armed_a and above_a >= min_frames:
            flags.append((i, "alert"))
            armed_a = False
        if armed_r and above_r >= min_frames:
            flags.append((i, "review"))
            armed_r = False
        if r < review - 1.0:
            armed_r = armed_a = True
        st = "alert" if not armed_a else ("watch" if r >= review else "normal")
        out_r.append(r)
        out_s.append(st)
    return np.array(out_r), out_s, flags


def scene_transform(video):
    """Rotate the scene so people stand upright, then fit it into the scene area."""
    allk = np.concatenate([t.kps.astype(np.float32) for t in video.tracks])
    tv = torso_vectors(allk)
    ok = np.linalg.norm(tv, axis=1) > 1e-3
    u = tv[ok] / np.linalg.norm(tv[ok], axis=1, keepdims=True)
    th = -np.pi / 2 - np.arctan2(u[:, 1].mean(), u[:, 0].mean())
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    pts = allk[..., :2][allk[..., 2] > 0.2] @ rot.T
    lo, hi = np.percentile(pts, 0.5, axis=0), np.percentile(pts, 99.5, axis=0)
    span = np.maximum(hi - lo, 1)
    s = min((W - 200) / span[0], (SCENE_H - 120) / span[1])
    off = np.array([W / 2, SCENE_H / 2 + 25]) - (lo + hi) / 2 * s

    def f(k):
        out = k.astype(np.float32).copy()
        out[:, :2] = k[:, :2].astype(np.float32) @ rot.T * s + off
        return out

    return f, int(np.clip(round(2 + s * 2), 3, 5))


def draw_timeline(img, video, curves, f, review, alert):
    n = video.n_frames
    x0, x1 = 60, W - 30
    tx = lambda i: int(x0 + (x1 - x0) * i / max(n - 1, 1))
    lo, hi = -1.0, 7.0
    ty = lambda v: int(TL_BOT - (TL_BOT - TL_TOP) * (np.clip(v, lo, hi) - lo) / (hi - lo))
    cv2.rectangle(img, (x0, TL_TOP), (x1, TL_BOT), (30, 28, 26), -1)
    lab = video.labels
    if lab is not None:
        edges = np.flatnonzero(np.diff(np.r_[0, lab[:n], 0]))
        for a, b in zip(edges[::2], edges[1::2]):
            overlay = img.copy()
            cv2.rectangle(overlay, (tx(a), TL_TOP), (tx(b - 1), TL_BOT), (60, 60, 200), -1)
            cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
    for thr, col, name in [(review, COLORS["watch"], "review"), (alert, COLORS["alert"], "alert")]:
        y = ty(thr)
        for xs in range(x0, x1, 12):
            cv2.line(img, (xs, y), (min(xs + 6, x1), y), col, 1, cv2.LINE_AA)
        cv2.putText(img, name, (x1 - 52, y - 5), FONT, 0.4, col, 1, cv2.LINE_AA)
    for k, (start, risk, _, _) in enumerate(curves.values()):
        col = PERSON_COLS[k % len(PERSON_COLS)]
        upto = min(len(risk), f - start + 1)
        if upto < 2:
            continue
        pts = np.array([[tx(start + i), ty(risk[i])] for i in range(upto)], np.int32)
        cv2.polylines(img, [pts], False, col, 2, cv2.LINE_AA)
    cv2.line(img, (tx(f), TL_TOP), (tx(f), TL_BOT), (200, 200, 200), 1)
    cv2.putText(img, "risk over time   (red band: labelled theft)", (x0, TL_TOP - 10), FONT, 0.45,
                (170, 170, 170), 1, cv2.LINE_AA)  # fmt: skip


def render_clip(scorer, video, ws, fused, vi, writer, clip_no, n_clips, review):
    alert = scorer.threshold
    curves = {}
    for ti, t in enumerate(video.tracks):
        r = np.flatnonzero((ws.video == vi) & (ws.track == ti))
        if not len(r):
            continue
        sc = person_frame_scores(ws, fused, vi, ti)
        start = int(ws.frame0[r].min())
        risk, st, flags = run_states(sc, review, alert)
        curves[ti] = (start, risk, st, flags)
    tf, thick = scene_transform(video)
    events = []
    lab = video.labels
    for f in range(video.n_frames):
        img = np.full((H, W, 3), (34, 31, 28), np.uint8)
        for gx in range(0, W, 60):
            cv2.line(img, (gx, 0), (gx, SCENE_H), (42, 39, 36), 1)
        for gy in range(0, SCENE_H, 60):
            cv2.line(img, (0, gy), (W, gy), (42, 39, 36), 1)
        people, any_alert = [], False
        for k, (ti, (start, risk, st, flags)) in enumerate(curves.items()):
            i = f - start
            if i < 0 or i >= len(risk):
                continue
            label = f"ID {video.tracks[ti].pid}"
            for fi, kind in flags:
                if fi == i:
                    r = np.flatnonzero((ws.video == vi) & (ws.track == ti) & (ws.frame0 <= f)
                                       & (ws.frame0 + WIN > f))  # fmt: skip
                    ff = (
                        ws.frame_cues[r[0]][: f - ws.frame0[r[0]] + 1].astype(np.float32)
                        if len(r)
                        else None
                    )
                    reasons = (scorer.cue_ref.reasons(ff) if ff is not None else []) or [
                        "unusual movement"
                    ]
                    events.append({"t": f / FPS, "person": f"{label} {kind}", "reasons": reasons})
            people.append((label, st[i], risk[i]))
            any_alert |= st[i] == "alert"
            tr = video.tracks[ti]
            j = np.searchsorted(tr.frames, f)
            if j < len(tr) and tr.frames[j] == f:
                kk = tf(tr.kps[j])
                draw_person(img, kk, st[i], label, 0.2, thick)
                cv2.circle(img, (int(kk[0, 0]) - 2, int(kk[0, 1]) - 34), 5, PERSON_COLS[k % 4], -1)
        if any_alert:
            cv2.rectangle(img, (0, 0), (W - 1, SCENE_H), COLORS["alert"], 6)
        cv2.putText(img, f"REAL STORE CCTV, skeletons only - clip {clip_no}/{n_clips}", (20, 32), FONT,
                    0.6, (210, 210, 210), 1, cv2.LINE_AA)  # fmt: skip
        if lab is not None and f < len(lab) and lab[f]:
            cv2.putText(img, "ground truth: theft in progress", (20, 60), FONT, 0.55, (110, 110, 250), 1,
                        cv2.LINE_AA)  # fmt: skip
        cv2.putText(
            img, f"{f / FPS:4.1f} s", (W - 90, 32), FONT, 0.55, (170, 170, 170), 1, cv2.LINE_AA
        )
        draw_timeline(img, video, curves, f, review, alert)
        side = panel(H, "skeletons only - no pixels stored", f"people {len(people)}",
                     sorted(people, key=lambda p: -p[2]), alert, events,
                     footer="data: PoseLift test set (WACV 2025)")  # fmt: skip
        writer.write(np.hstack([img, side]))
    # hold the last frame for a second so the curve can be read
    for _ in range(FPS):
        writer.write(np.hstack([img, side]))
    return events


def title_card(writer, lines, seconds=3.0):
    img = np.full((H, W + PANEL_W, 3), (24, 22, 20), np.uint8)
    y = H // 2 - 24 * len(lines)
    for i, line in enumerate(lines):
        s, t = (1.4, 3) if i == 0 else (0.7, 1)
        size = cv2.getTextSize(line, FONT, s, t)[0]
        cv2.putText(
            img, line, ((W + PANEL_W - size[0]) // 2, y), FONT, s, (235, 235, 235), t, cv2.LINE_AA
        )
        y += 64 if i == 0 else 38
    for _ in range(int(seconds * FPS)):
        writer.write(img)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="runs/heads/bundle")
    ap.add_argument("--clips", type=int, default=5)
    ap.add_argument("--out", default="runs/replay_raw.mp4")
    ap.add_argument("--names", nargs="*", default=None)
    args = ap.parse_args()
    scorer = Scorer.load(args.bundle)
    review = scorer.meta.get("review_threshold", scorer.threshold - 1.5)
    videos = get_split("retails_real")
    ws = build_windows(videos, min_len=8)
    fused = scorer.score(ws.pose, ws.cues)
    if args.names:
        pick = [i for i, v in enumerate(videos) if v.name in args.names]
    else:
        with_theft = [
            i for i, v in enumerate(videos) if v.labels is not None and v.labels.max() > 0
        ]
        pick = sorted(with_theft, key=lambda i: -videos[i].n_frames)[: args.clips]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W + PANEL_W, H))
    title_card(writer, ["VIGIE", "privacy-first shoplifting detection",
                        "real thefts from a real store, seen as skeletons only"])  # fmt: skip
    for n, vi in enumerate(pick, 1):
        ev = render_clip(scorer, videos[vi], ws, fused, vi, writer, n, len(pick), review)
        print(f"{videos[vi].name}: {videos[vi].n_frames} frames, theft frames {int(videos[vi].labels.sum())}, "
              f"flags {[e['person'] for e in ev]}")  # fmt: skip
    writer.release()
    print("written", args.out)


if __name__ == "__main__":
    main()
