"""Supervised theft head: gradient-boosted trees on interpretable window features.

Inputs per window: the aggregated concealment cues plus summaries of the normality model's
per-frame error. Trained on staged thefts (positives), the calm parts of the same staged clips
(hard negatives) and ordinary shopping footage (easy negatives). Tested on real thefts only.

Trees are chosen over a neural head because the training set is small (a few thousand
positive windows), the features are already meaningful, and per-feature contributions can be
read back to explain an alert.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

from ..cues import window_feature_names

NORMALITY_FEATURES = ["norm_err_mean", "norm_err_max", "norm_err_last"]


def normality_features(per_frame: np.ndarray) -> np.ndarray:
    """Summaries of the per-frame normality error inside each window, (W, 3)."""
    return np.column_stack([per_frame.mean(1), per_frame.max(1), per_frame[:, -6:].mean(1)])


def feature_matrix(cues: np.ndarray, per_frame: np.ndarray | None) -> np.ndarray:
    if per_frame is None:
        return cues
    return np.concatenate([cues, normality_features(per_frame)], axis=1)


def feature_names(with_normality: bool = True) -> list[str]:
    return window_feature_names() + (NORMALITY_FEATURES if with_normality else [])


def make_head(seed: int = 0) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=1.0,
        class_weight="balanced",
        early_stopping=True,
        validation_fraction=0.15,
        random_state=seed,
    )


def importances(model, X: np.ndarray, y: np.ndarray, names: list[str], top: int = 12):
    r = permutation_importance(model, X, y, n_repeats=5, random_state=0, scoring="roc_auc")
    order = np.argsort(-r.importances_mean)[:top]
    return [(names[i], float(r.importances_mean[i])) for i in order]
