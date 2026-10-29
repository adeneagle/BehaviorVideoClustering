"""The part-map video that replaces the camera video downstream.

    python parts_video.py RUN_DIR [--size 256]

Writes RUN_DIR/parts.mp4: every frame's label map from run_video.py drawn
with render.solid() (part colours on black, sinks dropped) and area-resized to
--size, i.e. the frames embed_vjepa.py feeds the encoder.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from render import solid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()

    run = Path(args.run_dir)
    t = pd.read_csv(run / "frames.csv", usecols=["t"]).t
    fps = (len(t) - 1) / t.iloc[-1]
    w = cv2.VideoWriter(str(run / "parts.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (args.size, args.size))
    for f in sorted(Path(run, "labels").glob("labels_*.npz")):
        for lab in np.load(f)["labels"]:
            w.write(cv2.resize(solid(lab), (args.size, args.size), interpolation=cv2.INTER_AREA))
        print(f.stem, flush=True)
    w.release()
    print(f"-> {run / 'parts.mp4'}")


if __name__ == "__main__":
    main()
