"""Train the normality transformer on normal-only footage and evaluate it on the test sets.

Usage:
    python scripts/train_normality.py --train retails_fit --windows 200000 --epochs 12
    python scripts/train_normality.py --train poselift_train   # apples-to-apples with PoseLift
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from vigie.data import get_split
from vigie.eval import bootstrap_ci, frame_scores, metrics
from vigie.models.normality import NormalityConfig, save_model, score_frames, train_normality
from vigie.windows import build_windows, save_windows

RUNS = Path("runs")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="retails_fit")
    ap.add_argument("--windows", type=int, default=200_000)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--tests", nargs="+", default=["retails_real", "retails_staged"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    tag = args.tag or f"normality_{args.train}"
    out = RUNS / tag
    out.mkdir(parents=True, exist_ok=True)

    t = time.time()
    train_videos = get_split(args.train)
    ws = build_windows(train_videos, max_windows=args.windows, seed=0)
    del train_videos
    print(f"train windows: {len(ws)} ({time.time() - t:.0f}s)", flush=True)
    rng = np.random.default_rng(1)
    val_idx = rng.choice(len(ws), min(5000, len(ws) // 20), replace=False)
    fit_idx = np.setdiff1d(np.arange(len(ws)), val_idx)

    model = train_normality(
        ws.pose[fit_idx], ws.pose[val_idx], NormalityConfig(epochs=args.epochs),
        log=lambda m: print(m, flush=True),
    )  # fmt: skip
    save_model(model, out / "model.pt")

    results = {}
    for name in args.tests:
        videos = get_split(name)
        tws = build_windows(videos, min_len=8)
        pf = score_frames(model, tws.pose)
        save_windows(tws, out / f"windows_{name}.npz")
        np.save(out / f"perframe_{name}.npy", pf)
        for sigma in (0, 5, 10):
            sc = frame_scores(videos, tws, pf.mean(1), per_frame=pf, sigma=sigma)
            m = metrics(videos, sc)
            ci = bootstrap_ci(videos, sc, n=500)
            results[f"{name}_sigma{sigma}"] = {**m.__dict__, "ci": ci}
            print(f"{name:15s} sigma={sigma:2d}  {m.row()}  "
                  f"CI95 AUC-ROC [{100 * ci['auc_roc'][0]:.1f}, {100 * ci['auc_roc'][1]:.1f}]",
                  flush=True)  # fmt: skip
    (out / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
