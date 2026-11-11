"""Normality model: a small transformer that learns what ordinary shopping movement looks like.

Training (normal footage only): random blocks of frames are hidden and the model reconstructs
them from the visible frames; one of the blocks is always the end of the window, so the model
also learns to predict the next 0.4 s. At test time each block is hidden in turn, and the
reconstruction error of the hidden frames is the anomaly score. A frame gets a high score when
the person's movement there could not be predicted from what came just before and after.

Why this design: it needs no theft labels, it gives a score per frame (so an alert can point
to the moment), and it is about 0.3M parameters, so it trains and runs on a laptop CPU.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from ..geometry import CONF_MIN, flip_lr

BLOCK = 6  # frames hidden together (0.4 s)


@dataclass
class NormalityConfig:
    win: int = 24
    d_model: int = 96
    heads: int = 4
    layers: int = 3
    ff: int = 192
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch: int = 256
    epochs: int = 12
    seed: int = 0


class PoseMaskedTransformer(nn.Module):
    def __init__(self, cfg: NormalityConfig):
        super().__init__()
        self.cfg = cfg
        self.inp = nn.Linear(17 * 3, cfg.d_model)
        self.mask_token = nn.Parameter(torch.zeros(cfg.d_model))
        self.pos = nn.Parameter(torch.randn(cfg.win, cfg.d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            cfg.d_model, cfg.heads, cfg.ff, cfg.dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, cfg.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.out = nn.Linear(cfg.d_model, 17 * 2)

    def forward(self, pose: torch.Tensor, hide: torch.Tensor) -> torch.Tensor:
        """pose (B, T, 17, 3); hide (B, T) bool -> predicted xy (B, T, 17, 2)."""
        b, t = pose.shape[:2]
        x = pose.clone()
        x[..., :2] = x[..., :2] * (x[..., 2:3] > CONF_MIN)  # unreliable joints carry no position
        tok = self.inp(x.reshape(b, t, -1))
        tok = torch.where(hide[..., None], self.mask_token.expand_as(tok), tok)
        h = self.encoder(tok + self.pos[:t])
        return self.out(self.norm(h)).reshape(b, t, 17, 2)


def _random_hide(b: int, t: int, gen: torch.Generator) -> torch.Tensor:
    """Hide the last block plus one or two random blocks per sample."""
    hide = torch.zeros(b, t, dtype=torch.bool)
    hide[:, t - BLOCK :] = True
    n_blocks = t // BLOCK
    for i in range(b):
        k = int(torch.randint(1, 3, (1,), generator=gen))
        for blk in torch.randperm(n_blocks - 1, generator=gen)[:k].tolist():
            hide[i, blk * BLOCK : (blk + 1) * BLOCK] = True
    return hide


def _masked_error(pred: torch.Tensor, pose: torch.Tensor, hide: torch.Tensor) -> torch.Tensor:
    """Per-frame mean joint error over reliable joints, (B, T); 0 where not hidden."""
    w = (pose[..., 2] > CONF_MIN).float()  # (B, T, 17)
    err = torch.linalg.norm(pred - pose[..., :2], dim=-1)  # (B, T, 17)
    err = (err * w).sum(-1) / w.sum(-1).clamp(min=1.0)
    return err * hide


def _augment(pose: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pose = pose.copy()
    flip = rng.random(len(pose)) < 0.5
    pose[flip] = flip_lr(pose[flip])
    ang = rng.uniform(-0.25, 0.25, len(pose))  # about +-15 degrees
    sc = rng.uniform(0.9, 1.1, len(pose))
    c, s = np.cos(ang) * sc, np.sin(ang) * sc
    x, y = pose[..., 0].copy(), pose[..., 1].copy()
    pose[..., 0] = c[:, None, None] * x - s[:, None, None] * y
    pose[..., 1] = s[:, None, None] * x + c[:, None, None] * y
    drop = rng.random(pose.shape[:3]) < 0.05  # random missing joints
    pose[..., 2][drop] = 0.0
    return pose


def train_normality(
    train_pose: np.ndarray,
    val_pose: np.ndarray | None = None,
    cfg: NormalityConfig | None = None,
    log=print,
) -> PoseMaskedTransformer:
    cfg = cfg or NormalityConfig()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    model = PoseMaskedTransformer(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    n = len(train_pose)
    steps = cfg.epochs * math.ceil(n / cfg.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg.lr, total_steps=steps, pct_start=0.1)
    log(f"normality model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, "
        f"{n} windows, {steps} steps")  # fmt: skip
    for ep in range(cfg.epochs):
        model.train()
        t0, total = time.time(), 0.0
        order = rng.permutation(n)
        for i in range(0, n, cfg.batch):
            idx = np.sort(order[i : i + cfg.batch])
            pose = torch.from_numpy(_augment(train_pose[idx].astype(np.float32), rng))
            hide = _random_hide(len(pose), cfg.win, gen)
            err = _masked_error(model(pose, hide), pose, hide)
            loss = err.sum() / hide.sum()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        msg = f"epoch {ep + 1}/{cfg.epochs} train {total / n:.4f}"
        if val_pose is not None:
            msg += f" val {score_windows(model, val_pose).mean():.4f}"
        log(msg + f" ({time.time() - t0:.0f}s)")
    return model


@torch.no_grad()
def score_frames(model: PoseMaskedTransformer, pose: np.ndarray, batch: int = 1024) -> np.ndarray:
    """Per-frame anomaly scores (W, T): each block hidden in turn, its error recorded."""
    model.eval()
    t = model.cfg.win
    out = np.zeros((len(pose), t), np.float32)
    for i in range(0, len(pose), batch):
        p = torch.from_numpy(pose[i : i + batch].astype(np.float32))
        acc = torch.zeros(len(p), t)
        for blk in range(t // BLOCK):
            hide = torch.zeros(len(p), t, dtype=torch.bool)
            hide[:, blk * BLOCK : (blk + 1) * BLOCK] = True
            acc += _masked_error(model(p, hide), p, hide)
        out[i : i + batch] = acc.numpy()
    return out


def score_windows(model: PoseMaskedTransformer, pose: np.ndarray) -> np.ndarray:
    return score_frames(model, pose).mean(axis=1)


def save_model(model: PoseMaskedTransformer, path) -> None:
    torch.save({"cfg": asdict(model.cfg), "state": model.state_dict()}, path)


def load_model(path) -> PoseMaskedTransformer:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = PoseMaskedTransformer(NormalityConfig(**ckpt["cfg"]))
    model.load_state_dict(ckpt["state"])
    return model.eval()
