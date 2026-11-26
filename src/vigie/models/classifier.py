"""Theft classifier fine-tuned from the self-supervised normality transformer.

The normality model has read millions of frames of ordinary shopping and learned how bodies
move in this setting. Here its encoder is reused, a small head is added, and the whole thing is
fine-tuned on the few thousand staged-theft windows. This is the standard pretrain/fine-tune
recipe for small labelled sets, applied to skeletons.
"""

from __future__ import annotations

import copy
import os
import time

import numpy as np
import torch
from torch import nn

from .normality import PoseMaskedTransformer, _augment


class TheftClassifier(nn.Module):
    def __init__(self, backbone: PoseMaskedTransformer, dropout: float = 0.2):
        super().__init__()
        self.backbone = copy.deepcopy(backbone)
        d = backbone.cfg.d_model
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(dropout), nn.Linear(d, 1))
        self.frame_head = nn.Linear(d, 1)  # per-frame logits, for localisation

    def encode(self, pose: torch.Tensor) -> torch.Tensor:
        bb = self.backbone
        b, t = pose.shape[:2]
        x = pose.clone()
        x[..., :2] = x[..., :2] * (x[..., 2:3] > 0.2)
        h = bb.encoder(bb.inp(x.reshape(b, t, -1)) + bb.pos[:t])
        return bb.norm(h)  # (B, T, d)

    def forward(self, pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encode(pose)
        return self.head(h.mean(1)).squeeze(-1), self.frame_head(h).squeeze(-1)


def train_classifier(
    backbone: PoseMaskedTransformer,
    pose: np.ndarray,
    window_label: np.ndarray,
    frame_label: np.ndarray,
    epochs: int = 15,
    lr: float = 3e-4,
    batch: int = 128,
    seed: int = 0,
    log=print,
    checkpoint=None,
) -> TheftClassifier:
    """window_label (W,) in [0,1]; frame_label (W, T) in {0,1}.

    With ``checkpoint`` (a path), state is saved after every epoch and training resumes from it.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = TheftClassifier(backbone)
    params = [
        {"params": model.backbone.parameters(), "lr": lr / 3},  # gentler on pretrained weights
        {"params": list(model.head.parameters()) + list(model.frame_head.parameters()), "lr": lr},
    ]
    opt = torch.optim.AdamW(params, weight_decay=1e-3)
    pos_w = float((1 - window_label.mean()) / max(window_label.mean(), 1e-3))
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w))
    bce_f = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w))
    n = len(pose)
    start = 0
    if checkpoint is not None and os.path.exists(checkpoint):
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        rng.bit_generator.state = ck["rng"]
        start = ck["epoch"]
        log(f"classifier resumed after epoch {start}")
    for ep in range(start, epochs):
        model.train()
        t0, total = time.time(), 0.0
        order = rng.permutation(n)
        for i in range(0, n, batch):
            idx = np.sort(order[i : i + batch])
            p = torch.from_numpy(_augment(pose[idx].astype(np.float32), rng))
            yw = torch.from_numpy(window_label[idx].astype(np.float32))
            yf = torch.from_numpy(frame_label[idx].astype(np.float32))
            lw, lf = model(p)
            loss = bce(lw, yw) + 0.5 * bce_f(lf, yf)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(idx)
        log(f"classifier epoch {ep + 1}/{epochs} loss {total / n:.4f} ({time.time() - t0:.0f}s)")
        if checkpoint is not None:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "rng": rng.bit_generator.state, "epoch": ep + 1}, checkpoint)  # fmt: skip
    return model.eval()


@torch.no_grad()
def predict(
    model: TheftClassifier, pose: np.ndarray, batch: int = 1024
) -> tuple[np.ndarray, np.ndarray]:
    """Window probabilities (W,) and per-frame probabilities (W, T)."""
    model.eval()
    pw, pf = [], []
    for i in range(0, len(pose), batch):
        lw, lf = model(torch.from_numpy(pose[i : i + batch].astype(np.float32)))
        pw.append(torch.sigmoid(lw).numpy())
        pf.append(torch.sigmoid(lf).numpy())
    return np.concatenate(pw), np.concatenate(pf)
