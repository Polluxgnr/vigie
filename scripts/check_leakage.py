"""Detect identical pose sequences shared between dataset splits.

PoseLift and RetailS come from the same store and the same authors, so the same footage could
appear in several splits. Every window of 8 consecutive frames is fingerprinted (keypoints
rounded to the pixel); two splits sharing fingerprints share footage.

Usage: python scripts/check_leakage.py retails_fit retails_holdout retails_staged retails_real
"""

from __future__ import annotations

import hashlib
import sys
from itertools import combinations

import numpy as np

from vigie.data import get_split

K = 8


def fingerprints(name: str) -> tuple[set[bytes], dict[bytes, str]]:
    prints, where = set(), {}
    for v in get_split(name):
        for t in v.tracks:
            xy = np.round(t.kps[:, :, :2]).astype(np.int32)
            for s in range(0, len(xy) - K + 1, K):
                h = hashlib.blake2b(xy[s : s + K].tobytes(), digest_size=12).digest()
                prints.add(h)
                where.setdefault(h, v.name)
    return prints, where


def main() -> None:
    names = sys.argv[1:] or ["retails_fit", "retails_holdout", "retails_staged", "retails_real"]
    fp = {n: fingerprints(n) for n in names}
    for n in names:
        print(f"{n:16s} {len(fp[n][0]):9d} fingerprints")
    for a, b in combinations(names, 2):
        common = fp[a][0] & fp[b][0]
        vids = sorted({fp[b][1][h] for h in common})
        print(f"{a} x {b}: {len(common)} shared windows in {len(vids)} videos of {b} {vids[:8]}")


if __name__ == "__main__":
    main()
