"""The full scoring stack in one object: three scorers, fusion, threshold, explanations.

Saved as a folder ("bundle"). The boosted-trees head is stored with joblib, which uses pickle:
only load bundles you produced yourself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import torch

from .explain import CueReference
from .fusion import NormalReference, fuse
from .models.classifier import TheftClassifier, predict
from .models.normality import PoseMaskedTransformer, load_model, save_model, score_frames
from .models.supervised import feature_matrix
from .windows import WIN


@dataclass
class Scorer:
    normality: PoseMaskedTransformer
    classifier: TheftClassifier
    gbm: object
    refs: dict[str, NormalReference]
    cue_ref: CueReference
    keys: list[str] = field(default_factory=lambda: ["normality", "gbm", "finetuned"])
    threshold: float = 0.0
    meta: dict = field(default_factory=dict)

    def components(self, pose: np.ndarray, cues: np.ndarray) -> dict[str, np.ndarray]:
        """Raw per-frame scores (W, WIN) of each model."""
        pf = score_frames(self.normality, pose)
        out = {"normality": pf}
        if "gbm" in self.keys:
            g = self.gbm.predict_proba(feature_matrix(cues, pf))[:, 1]
            out["gbm"] = np.repeat(g[:, None], WIN, 1)
        if "finetuned" in self.keys:
            cw, cf = predict(self.classifier, pose)
            out["finetuned"] = 0.5 * cf + 0.5 * cw[:, None]
        return out

    def fused(self, comps: dict[str, np.ndarray]) -> np.ndarray:
        return fuse([self.refs[k].logit(comps[k]) for k in self.keys])

    def score(self, pose: np.ndarray, cues: np.ndarray) -> np.ndarray:
        return self.fused(self.components(pose, cues))

    # ------------------------------------------------------------ persistence
    def save(self, folder) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        save_model(self.normality, folder / "normality.pt")
        torch.save(self.classifier.state_dict(), folder / "classifier.pt")
        joblib.dump(self.gbm, folder / "gbm.joblib")
        np.savez(folder / "references.npz", **{f"ref_{k}": r.ref for k, r in self.refs.items()},
                 **self.cue_ref.to_arrays())  # fmt: skip
        (folder / "scorer.json").write_text(
            json.dumps(
                {"keys": self.keys, "threshold": self.threshold, "meta": self.meta}, indent=2
            )
        )

    @classmethod
    def load(cls, folder) -> Scorer:
        folder = Path(folder)
        cfg = json.loads((folder / "scorer.json").read_text())
        norm = load_model(folder / "normality.pt")
        clf = TheftClassifier(norm)
        clf.load_state_dict(
            torch.load(folder / "classifier.pt", map_location="cpu", weights_only=True)
        )
        clf.eval()
        z = np.load(folder / "references.npz", allow_pickle=False)
        refs = {
            k[4:]: NormalReference.__new__(NormalReference) for k in z.files if k.startswith("ref_")
        }
        for k, r in refs.items():
            r.ref = z[f"ref_{k}"]
        return cls(norm, clf, joblib.load(folder / "gbm.joblib"), refs,
                   CueReference.from_sorted(z["cue_sorted"]), cfg["keys"], cfg["threshold"],
                   cfg.get("meta", {}))  # fmt: skip
