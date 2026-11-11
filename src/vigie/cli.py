"""Command line: ``vigie run VIDEO`` renders the privacy view and writes the event log."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(prog="vigie", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="analyse a video file or a webcam (use 0)")
    run.add_argument("source")
    run.add_argument("--bundle", default="runs/heads/bundle")
    run.add_argument("--pose-model", default="yolo11n-pose.pt",
                     help="Ultralytics pose weights, or an exported *_openvino_model folder")  # fmt: skip
    run.add_argument("--out", default=None, help="output video (default: runs/demo/<name>.mp4)")
    run.add_argument("--no-privacy", action="store_true", help="show raw pixels (debug only)")
    run.add_argument("--show", action="store_true")
    run.add_argument("--max-seconds", type=float, default=None)
    run.add_argument("--imgsz", type=int, default=640)
    exp = sub.add_parser("export", help="export YOLO11-pose to OpenVINO for faster CPU inference")
    exp.add_argument("--pose-model", default="yolo11n-pose.pt")
    exp.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    if args.cmd == "export":
        from ultralytics import YOLO

        print(YOLO(args.pose_model).export(format="openvino", imgsz=args.imgsz, half=False))
        return

    from .live import LivePipeline
    from .scorer import Scorer

    source = int(args.source) if args.source.isdigit() else args.source
    name = "webcam" if isinstance(source, int) else Path(source).stem
    out = args.out or f"runs/demo/{name}.mp4"
    pipe = LivePipeline(Scorer.load(args.bundle), pose_model=args.pose_model, imgsz=args.imgsz)
    res = pipe.run(source, out, str(Path(out).with_suffix(".events.jsonl")),
                   privacy=not args.no_privacy, show=args.show, max_seconds=args.max_seconds)  # fmt: skip
    print(json.dumps({"video": out, "frames": res["frames"], "fps": round(res["proc_fps"], 1),
                      "alerts": len(res["events"])}, indent=2))  # fmt: skip


if __name__ == "__main__":
    main()
