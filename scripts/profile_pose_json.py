"""Quick profile of STG-NF style pose JSON files: {person: {frame: {keypoints: [51]}}}."""

import json
import random
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
files = sorted(root.glob("*.json"))
random.seed(0)
sample = random.sample(files, min(n, len(files)))
xs, ys, confs, lens, people, spans = [], [], [], [], [], []
for f in sample:
    d = json.loads(f.read_text())
    people.append(len(d))
    allframes = []
    for frames in d.values():
        fr = sorted(int(k) for k in frames)
        allframes += fr
        lens.append(len(fr))
        for k in list(frames)[:50]:
            kp = np.asarray(frames[k]["keypoints"], dtype=np.float32).reshape(17, 3)
            xs.append(kp[:, 0].max())
            ys.append(kp[:, 1].max())
            confs.append(kp[:, 2])
    spans.append((min(allframes), max(allframes)) if allframes else None)
print(f"{len(files)} files; sampled {len(sample)}")
print("people/file  median", np.median(people), "max", max(people))
print("track length median", np.median(lens), "p90", np.percentile(lens, 90), "max", max(lens))
print("max x", np.percentile(xs, 99), "max y", np.percentile(ys, 99))
c = np.concatenate(confs)
print("conf mean", c.mean().round(3), "share<0.3", (c < 0.3).mean().round(3))
print("frame spans", spans[:5])
