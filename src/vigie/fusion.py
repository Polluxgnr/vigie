"""Put scores from different models on one scale: "how unusual compared with normal footage".

Each raw score is mapped to its percentile among windows of ordinary shopping that no model was
trained on. A percentile of 0.999 means "only 1 normal window in 1000 scores this high", which
links every score directly to a false-alarm rate. Scores are then combined in logit space.
"""

from __future__ import annotations

import numpy as np


class NormalReference:
    def __init__(self, normal_scores: np.ndarray):
        self.ref = np.sort(np.asarray(normal_scores, dtype=np.float64).ravel())

    def percentile(self, s: np.ndarray) -> np.ndarray:
        n = len(self.ref)
        # +1 smoothing keeps percentiles strictly inside (0, 1)
        return (np.searchsorted(self.ref, s, side="right") + 0.5) / (n + 1)

    def logit(self, s: np.ndarray) -> np.ndarray:
        p = self.percentile(s)
        return np.log(p / (1 - p))


def fuse(logits: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    w = np.ones(len(logits)) if weights is None else np.asarray(weights, dtype=np.float64)
    return sum(wi * li for wi, li in zip(w, logits)) / w.sum()
