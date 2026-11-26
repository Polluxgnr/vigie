"""Download the PoseLift Google Drive folder one file at a time.

Google Drive rate-limits folder downloads that fire hundreds of requests in a
row ("Cannot retrieve the public link ... or have had many accesses"). This
script lists the folder once, then fetches each file with a pause and retries,
skipping files that are already on disk.

Usage:  python scripts/download_poselift.py [--out data/raw/poselift]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import gdown

FOLDER_URL = "https://drive.google.com/drive/folders/1aEkENZlVE4ZvF_BZXV1VJOwuiXQq6trn"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw/poselift")
    parser.add_argument("--pause", type=float, default=1.5, help="seconds between files")
    parser.add_argument("--retries", type=int, default=6)
    args = parser.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    files = gdown.download_folder(FOLDER_URL, output=str(out), skip_download=True, quiet=True)
    print(f"{len(files)} files listed")

    failed = []
    for i, f in enumerate(files, 1):
        target = Path(f.local_path)
        if target.exists() and target.stat().st_size > 0:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(args.retries):
            try:
                got = gdown.download(id=f.id, output=str(target), quiet=True)
                if got:
                    break
            except Exception as exc:  # noqa: BLE001  (gdown raises many error types)
                print(f"  retry {attempt + 1} for {f.path}: {type(exc).__name__}")
            time.sleep(args.pause * (2**attempt))
        else:
            failed.append(f.path)
        if i % 25 == 0:
            print(f"{i}/{len(files)} done, {len(failed)} failed so far", flush=True)
        time.sleep(args.pause)

    print(f"finished: {len(files) - len(failed)} ok, {len(failed)} failed")
    for p in failed:
        print("  FAILED", p)


if __name__ == "__main__":
    main()
