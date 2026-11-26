"""Drawing helpers shared by the live pipeline and the offline skeleton replays."""

from __future__ import annotations

import cv2
import numpy as np

SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13),
            (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]  # fmt: skip
COLORS = {"normal": (170, 200, 60), "watch": (0, 190, 255), "alert": (60, 60, 240)}  # BGR
PANEL_W = 380
FONT = cv2.FONT_HERSHEY_SIMPLEX


def sigmoid(x: float) -> float:
    return float(1 / (1 + np.exp(-x)))


def draw_person(img, k: np.ndarray, state: str, label: str, conf_min: float = 0.3, thick: int = 3):
    col = COLORS[state]
    for a, b in SKELETON:
        if k[a, 2] > conf_min and k[b, 2] > conf_min:
            cv2.line(
                img,
                (int(k[a, 0]), int(k[a, 1])),
                (int(k[b, 0]), int(k[b, 1])),
                col,
                thick,
                cv2.LINE_AA,
            )
    for j in range(17):
        if k[j, 2] > conf_min:
            cv2.circle(
                img,
                (int(k[j, 0]), int(k[j, 1])),
                max(2, thick + 1),
                (255, 255, 255),
                -1,
                cv2.LINE_AA,
            )
    vis = k[k[:, 2] > conf_min]
    if len(vis):
        x0, y0 = int(vis[:, 0].min()), int(vis[:, 1].min()) - 14
        cv2.putText(img, label, (x0, max(16, y0)), FONT, 0.6, col, 2, cv2.LINE_AA)


def privacy_background(frame: np.ndarray) -> np.ndarray:
    """Pixelate, blur and darken so that nobody is recognisable."""
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (max(1, w // 24), max(1, h // 24)), interpolation=cv2.INTER_AREA)
    out = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    out = cv2.GaussianBlur(out, (0, 0), 6)
    return (out * 0.45).astype(np.uint8)


def panel(h: int, title_sub: str, stats: str, people: list[tuple], threshold: float, events: list[dict],
          footer: str | None = None) -> np.ndarray:  # fmt: skip
    """people: [(label, state, risk_logit)], events: [{"t", "person", "reasons"}]."""
    pw = PANEL_W
    p = np.full((h, pw, 3), (24, 22, 20), np.uint8)

    def put(txt, y, s=0.55, c=(230, 230, 230), t=1):
        cv2.putText(p, txt, (18, y), FONT, s, c, t, cv2.LINE_AA)

    put("VIGIE", 40, 0.9, (255, 255, 255), 2)
    put(title_sub, 64, 0.45, (160, 160, 160))
    put(stats, 96, 0.5)
    y = 130
    thr_x = 18 + int((pw - 36) * sigmoid(threshold))
    for label, state, risk in people[:6]:
        col = COLORS[state]
        put(f"{label}  {state.upper()}", y, 0.5, col)
        y += 10
        cv2.rectangle(p, (18, y), (pw - 18, y + 10), (60, 60, 60), -1)
        cv2.rectangle(p, (18, y), (18 + int((pw - 36) * sigmoid(risk)), y + 10), col, -1)
        cv2.line(p, (thr_x, y - 3), (thr_x, y + 13), (255, 255, 255), 1)
        y += 30
    n_ev = min(5, len(events))
    y = max(y + 10, h - 40 - 42 * n_ev - 30 - (24 if footer else 0))
    put("EVENT LOG", y, 0.5, (200, 200, 200))
    y += 26
    for ev in events[-5:][::-1]:
        col = COLORS["watch"] if "review" in str(ev["person"]) else COLORS["alert"]
        put(f"{ev['t']:5.1f}s {ev['person']}", y, 0.42, col)
        put("  " + ", ".join(ev["reasons"])[:42], y + 18, 0.4, (200, 200, 200))
        y += 42
    if footer:
        put(footer, h - 18, 0.42, (150, 150, 150))
    return p
