"""Figures for the README and the report, from the saved results (runs/heads).

Palette: categorical slot 1 (blue) for Vigie, neutral greys for published baselines,
light surface, text in text inks (never series colours).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from vigie.data import get_split  # noqa: E402
from vigie.eval import frame_scores  # noqa: E402
from vigie.windows import build_windows  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, BASE, ALERT = "#2a78d6", "#a3a29c", "#e34948"
OUT = Path("docs/figures")

PUBLISHED = {"GEPC": 60.61, "TSGAD": 63.35, "STG-NF": 67.46, "Shopformer": 69.15}


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.label.set_color(INK2)
    ax.xaxis.label.set_color(INK2)


def fig_benchmark(res: dict) -> None:
    rows = [(k, v, None, BASE) for k, v in PUBLISHED.items()]
    labels = {"normality": "Vigie: normality only", "gbm": "Vigie: cues + trees",
              "finetuned": "Vigie: fine-tuned net", "all three": "Vigie: fusion (shipped)"}  # fmt: skip
    for key, lab in labels.items():
        m = res[key]["sigma0"]
        rows.append((lab, 100 * m["auc_roc"], [100 * x for x in m["ci"]["auc_roc"]],
                     BLUE if key == "all three" else "#8db8ea"))  # fmt: skip
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    y = np.arange(len(rows))[::-1]
    for yi, (lab, v, ci, col) in zip(y, rows):
        ax.barh(yi, v - 50, left=50, height=0.6, color=col, edgecolor=SURFACE, linewidth=2)
        if ci:
            ax.plot(ci, [yi, yi], color=INK2, lw=1.2)
            ax.plot([ci[0]] * 2, [yi - 0.12, yi + 0.12], color=INK2, lw=1.2)
            ax.plot([ci[1]] * 2, [yi - 0.12, yi + 0.12], color=INK2, lw=1.2)
        ax.text(max(v, ci[1] if ci else v) + 0.6, yi, f"{v:.1f}", va="center", color=INK, fontsize=9)
    ax.set_yticks(y, [r[0] for r in rows], color=INK)
    ax.set_xlim(50, 100)
    ax.set_xlabel("Frame-level AUC-ROC on the PoseLift real-world test set (%)")
    ax.axvline(50, color=GRID, lw=1)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Grey: published baselines.  Blue: Vigie, with 95% bootstrap interval over videos.",
                 fontsize=9, color=INK2, loc="left")  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "benchmark.png", facecolor=SURFACE)
    plt.close(fig)


def fig_tradeoff(res: dict) -> None:
    curve = res["all three"].get("tradeoff") or []
    if not curve:
        return
    fa = [c["fa_per_hour"] for c in curve]
    rec = [100 * c["event_recall"] for c in curve]
    fig, ax = plt.subplots(figsize=(6, 3.6), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    ax.plot(fa, rec, color=BLUE, lw=2, marker="o", ms=5, markeredgecolor=SURFACE, markeredgewidth=1.5)
    op = res["all three"].get("operating_point")
    if op:
        ax.scatter([op["fa_per_hour"]], [100 * op["event_recall"]], s=90, facecolor="none",
                   edgecolor=INK, lw=1.5, zorder=5)  # fmt: skip
        ax.annotate("shipped threshold", (op["fa_per_hour"], 100 * op["event_recall"]),
                    xytext=(10, -14), textcoords="offset points", color=INK, fontsize=9)  # fmt: skip
    ax.set_xscale("symlog", linthresh=0.5)
    ax.set_xlabel("False alerts per camera-hour on unseen normal footage")
    ax.set_ylabel("Real theft events caught (%)")
    ax.set_ylim(0, 100)
    ax.grid(color=GRID, lw=0.6)
    fig.tight_layout()
    fig.savefig(OUT / "tradeoff.png", facecolor=SURFACE)
    plt.close(fig)


def fig_timelines(scores_path: Path, threshold: float, n: int = 4) -> None:
    videos = get_split("retails_real")
    ws = build_windows(videos, min_len=8)
    te = np.load(scores_path, allow_pickle=False)["all"]
    per_video = frame_scores(videos, ws, te.mean(1), per_frame=te, sigma=0)
    # longest clips that contain a theft, for readability
    order = sorted(range(len(videos)), key=lambda i: -videos[i].n_frames)
    pick = [i for i in order if videos[i].labels.max() > 0][:n]
    fig, axes = plt.subplots(len(pick), 1, figsize=(8, 1.5 * len(pick)), dpi=160, sharex=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, i in zip(np.atleast_1d(axes), pick):
        style(ax)
        v, s = videos[i], per_video[i]
        t = np.arange(len(s)) / 15
        lab = v.labels[: len(s)]
        ax.fill_between(t, 0, 1, where=lab > 0, transform=ax.get_xaxis_transform(),
                        color=ALERT, alpha=0.12, lw=0)  # fmt: skip
        ax.plot(t, s, color=BLUE, lw=2)
        ax.axhline(threshold, color=INK2, lw=1, ls=(0, (4, 3)))
        ax.set_ylabel(v.name, rotation=0, ha="right", va="center", fontsize=8)
        ax.set_yticks([])
    np.atleast_1d(axes)[-1].set_xlabel("seconds   (red band: labelled theft;  dashed: alert threshold)")
    fig.tight_layout()
    fig.savefig(OUT / "timelines.png", facecolor=SURFACE)
    plt.close(fig)


def fig_features(res: dict) -> None:
    feats = res.get("features") or []
    if not feats:
        return
    names = [f[0].replace("_", " ") for f in feats[:10]][::-1]
    vals = [f[1] for f in feats[:10]][::-1]
    fig, ax = plt.subplots(figsize=(6, 3.4), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    ax.barh(names, vals, color=BLUE, height=0.6, edgecolor=SURFACE, linewidth=2)
    ax.set_xlabel("Drop in AUC when the feature is shuffled (staged thefts)")
    ax.tick_params(axis="y", colors=INK)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "features.png", facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res = json.loads(Path("runs/heads/results_retails_real.json").read_text())
    fig_benchmark(res)
    fig_tradeoff(res)
    fig_features(res)
    thr = json.loads(Path("runs/heads/bundle/scorer.json").read_text())["threshold"]
    fig_timelines(Path("runs/heads/scores_retails_real.npz"), thr)
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
