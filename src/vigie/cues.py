"""Interpretable concealment cues computed from body-centric poses.

Every cue is in torso units (see ``geometry.normalise``), so it means the same thing whatever
the camera. They are the "reasons" an alert can cite, and the input of the supervised head.

Body landmarks in the normalised frame (y points down, the torso points to -y):
    mid-hip = (0, 0), mid-shoulder ~ (0, -1).
"""

from __future__ import annotations

import numpy as np

from .geometry import CONF_MIN

FPS = 15
NOSE, L_EAR, R_EAR = 0, 3, 4
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 11, 12, 13, 14, 15, 16

POCKET_DROP = 0.15  # front/side pockets sit a little below the hip joints
HISTORY = 3 * FPS  # "reach then conceal" looks back 3 seconds

FRAME_FEATURES = [
    "pocket_dist",  # closest wrist to a pocket point (either side)
    "waist_dist",  # closest wrist to the front waistband
    "torso_dist",  # closest wrist to the torso centre line (under a shirt / jacket)
    "reach",  # most extended arm, wrist-to-shoulder over arm length
    "hand_height",  # highest wrist above the hips
    "hand_hidden",  # a wrist unreliable while its elbow is visible
    "hand_speed",  # fastest wrist, torso lengths per second
    "pocket_dwell",  # seconds the hand has stayed in a pocket zone
    "reach_then_conceal",  # concealment now x a reach in the previous 3 s, same hand
    "head_yaw",  # signed head turn proxy (nose vs ears)
    "head_turn_rate",  # |d yaw / dt|, glancing around
    "crouch",  # leg extent below the hips (small = crouching)
    "shoulder_width",  # small = side-on or turned away from the camera
    "body_speed",  # displacement of the hips (needs anchor="first" input), torso/s
]


def _dist_to_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(float(ab @ ab), 1e-9), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[..., None] * ab), axis=-1)


def _dwell(mask: np.ndarray) -> np.ndarray:
    """Length of the current run of True values, per frame."""
    out = np.zeros(len(mask))
    run = 0
    for i, m in enumerate(mask):
        run = run + 1 if m else 0
        out[i] = run
    return out


def _past_max(x: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """max of x over the window [t-hi, t-lo] for every t (0 when empty)."""
    n = len(x)
    out = np.zeros(n)
    for t in range(n):
        a, b = max(0, t - hi), t - lo + 1
        if b > a:
            out[t] = x[a:b].max()
    return out


def frame_features(body: np.ndarray, track_xy: np.ndarray | None = None) -> np.ndarray:
    """Per-frame cues for a contiguous sequence.

    body:     (N, 17, 3) normalised with anchor="frame" (posture only).
    track_xy: (N, 17, 3) normalised with anchor="first" (keeps displacement), optional.
    Returns (N, len(FRAME_FEATURES)) float32.
    """
    xy, conf = body[:, :, :2], body[:, :, 2]
    n = len(body)
    pockets = np.array([[-0.2, POCKET_DROP], [0.2, POCKET_DROP]])
    # use the measured hip positions for the pockets when available
    hips = xy[:, [L_HIP, R_HIP]] + np.array([0.0, POCKET_DROP])
    pockets = np.where((conf[:, [L_HIP, R_HIP]] > CONF_MIN)[..., None], hips, pockets[None])

    per_hand = {}
    for side, (wr, el, sh) in {"L": (L_WR, L_EL, L_SH), "R": (R_WR, R_EL, R_SH)}.items():
        w = xy[:, wr]
        d_pocket = np.linalg.norm(w[:, None, :] - pockets, axis=-1).min(axis=1)
        d_waist = np.linalg.norm(w - np.array([0.0, 0.05]), axis=-1)
        d_torso = _dist_to_segment(w, np.array([0.0, 0.1]), np.array([0.0, -0.9]))
        arm = np.linalg.norm(xy[:, sh] - xy[:, el], axis=-1) + np.linalg.norm(
            xy[:, el] - w, axis=-1
        )
        arm = np.maximum(np.median(arm), 0.3)
        reach = np.linalg.norm(w - xy[:, sh], axis=-1) / arm
        hidden = ((conf[:, wr] < CONF_MIN) & (conf[:, el] >= CONF_MIN)).astype(float)
        speed = np.r_[0.0, np.linalg.norm(np.diff(w, axis=0), axis=-1)] * FPS
        conceal = np.exp(-np.minimum(d_pocket, d_torso) / 0.25)
        # reach must happen at least 0.2 s before the concealment
        rtc = conceal * _past_max(reach, 3, HISTORY)
        dwell = _dwell(d_pocket < 0.3) / FPS
        per_hand[side] = {"d_pocket": d_pocket, "d_waist": d_waist, "d_torso": d_torso, "reach": reach,
                              "height": -w[:, 1], "hidden": hidden, "speed": speed, "rtc": rtc, "dwell": dwell}  # fmt: skip

    L, R = per_hand["L"], per_hand["R"]
    ear_mid = (xy[:, L_EAR, 0] + xy[:, R_EAR, 0]) / 2
    ear_w = np.maximum(np.abs(xy[:, L_EAR, 0] - xy[:, R_EAR, 0]), 0.05)
    yaw = np.clip((xy[:, NOSE, 0] - ear_mid) / ear_w, -3, 3)
    yaw = np.where((conf[:, [NOSE, L_EAR, R_EAR]] > CONF_MIN).sum(1) >= 2, yaw, 0.0)
    turn = np.r_[0.0, np.abs(np.diff(yaw))] * FPS
    crouch = (xy[:, [L_ANK, R_ANK], 1].mean(1)).clip(0, 3)
    sh_w = np.linalg.norm(xy[:, L_SH] - xy[:, R_SH], axis=-1)
    if track_xy is not None:
        hip = (track_xy[:, L_HIP, :2] + track_xy[:, R_HIP, :2]) / 2
        body_speed = np.r_[0.0, np.linalg.norm(np.diff(hip, axis=0), axis=-1)] * FPS
    else:
        body_speed = np.zeros(n)

    feats = np.column_stack([
        np.minimum(L["d_pocket"], R["d_pocket"]),
        np.minimum(L["d_waist"], R["d_waist"]),
        np.minimum(L["d_torso"], R["d_torso"]),
        np.maximum(L["reach"], R["reach"]),
        np.maximum(L["height"], R["height"]),
        np.maximum(L["hidden"], R["hidden"]),
        np.maximum(L["speed"], R["speed"]),
        np.maximum(L["dwell"], R["dwell"]),
        np.maximum(L["rtc"], R["rtc"]),
        yaw,
        turn,
        crouch,
        sh_w,
        body_speed,
    ])  # fmt: skip
    return np.nan_to_num(feats.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


AGGS = ("mean", "min", "max", "std")


def window_features(ff: np.ndarray, starts: np.ndarray, size: int) -> np.ndarray:
    """Aggregate per-frame cues over windows -> (W, F * 4)."""
    if len(starts) == 0:
        return np.zeros((0, ff.shape[1] * len(AGGS)), dtype=np.float32)
    idx = starts[:, None] + np.arange(size)[None]
    w = ff[idx]  # (W, size, F)
    return np.concatenate([w.mean(1), w.min(1), w.max(1), w.std(1)], axis=1).astype(np.float32)


def window_feature_names() -> list[str]:
    return [f"{f}_{a}" for a in AGGS for f in FRAME_FEATURES]
