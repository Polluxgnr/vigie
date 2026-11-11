"""Turn concealment cues into short, human-readable reasons for an alert.

A cue becomes a reason only when it is rare in ordinary shopping footage: each cue value is
compared with its distribution on normal windows (percentile), in the direction that means
"suspicious" for that cue.
"""

from __future__ import annotations

import numpy as np

from .cues import FPS, FRAME_FEATURES

# cue -> (direction, template). direction +1: high is suspicious; -1: low is suspicious.
REASONS = {
    "reach_then_conceal": (+1, "hand went from shelf to body"),
    "pocket_dwell": (+1, "hand kept at pocket {v:.1f} s"),
    "pocket_dist": (-1, "hand at pocket"),
    "waist_dist": (-1, "hand at waistband"),
    "torso_dist": (-1, "hand under clothing"),
    "hand_hidden": (+1, "hand hidden by body"),
    "head_turn_rate": (+1, "looking around"),
    "crouch": (-1, "crouching"),
}


class CueReference:
    """Per-cue sorted values from normal footage, for percentile lookups."""

    def __init__(self, normal_frame_cues: np.ndarray, max_rows: int = 200_000, seed: int = 0):
        x = normal_frame_cues.reshape(-1, normal_frame_cues.shape[-1]).astype(np.float32)
        if len(x) > max_rows:
            x = x[np.random.default_rng(seed).choice(len(x), max_rows, replace=False)]
        self.sorted = np.sort(x, axis=0)

    def percentile(self, j: int, v: float) -> float:
        col = self.sorted[:, j]
        return float(np.searchsorted(col, v, side="right") / len(col))

    def reasons(self, frame_cues: np.ndarray, top: int = 2, rare: float = 0.97) -> list[str]:
        """Reasons for the most recent frames of one person, most unusual first.

        frame_cues: (T, F) per-frame cues; the last ~0.7 s are summarised.
        """
        recent = frame_cues[-10:]
        found = []
        for name, (sign, text) in REASONS.items():
            j = FRAME_FEATURES.index(name)
            v = float(recent[:, j].max() if sign > 0 else recent[:, j].min())
            p = self.percentile(j, v)
            rarity = p if sign > 0 else 1 - p
            if rarity >= rare:
                found.append((rarity, text.format(v=v if name != "pocket_dwell" else v)))
        found.sort(reverse=True)
        return [t for _, t in found[:top]]

    def to_arrays(self) -> dict[str, np.ndarray]:
        return {"cue_sorted": self.sorted.astype(np.float32)}

    @classmethod
    def from_sorted(cls, arr: np.ndarray) -> "CueReference":
        obj = cls.__new__(cls)
        obj.sorted = arr
        return obj


__all__ = ["CueReference", "REASONS", "FPS"]
