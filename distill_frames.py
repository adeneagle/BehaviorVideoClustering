"""Turn sleap_teacher.py's predictions into a keypoint training set for the part network.

    python distill_frames.py RUN_DIR [--teacher NAME] [--out NAME]

RUN_DIR is the run_video.py output of the session the teacher ran on. Each predicted frame is
cut exactly as run_video.py cut it for the part network (768 px at frames.csv x0, y0), so the
student trains on the crops it will see at inference; the frames are read in one sequential pass
(cv2, luma, no seeking). Keypoints are stored in crop px with the teacher's per-node score
(NaN where it abstained, e.g. the snout of a rearing mouse).

Output: <out_root>/<NAME>/ (default distill_<session>): frames/*.jpg, index.csv in sleap_gt's
layout (+ <node>_score columns, source = teacher). train_seg.py --kp-distill uses it.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from common import DEFAULT_OUT_ROOT
from sleap_kp import NODES, read_slp

CROP = 768


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--out")
    ap.add_argument("--teacher", default="teacher", help="sleap_teacher.py --name (folder in <session>/sleap/)")
    ap.add_argument("--jpeg-q", type=int, default=95)
    args = ap.parse_args()

    run = Path(args.run_dir)
    session = run.parent.parent.name
    tdir = run.parent.parent / "sleap" / args.teacher
    parts = sorted(tdir.glob("part_*.slp"))
    if not parts:
        raise SystemExit(f"no teacher predictions in {tdir}")
    d = [read_slp(p, predicted=True) for p in parts]
    frame = np.concatenate([x["frame"] for x in d])
    xy = np.concatenate([x["xy"] for x in d])
    score = np.concatenate([x["score"] for x in d])
    order = np.argsort(frame)
    frame, xy, score = frame[order], xy[order], score[order]
    print(f"{len(frame)} teacher frames from {len(parts)} parts")

    fr = pd.read_csv(run / "frames.csv", usecols=["x0", "y0"])
    out = DEFAULT_OUT_ROOT / (args.out or f"distill_{session}" + ("" if args.teacher == "teacher" else f"_{args.teacher}"))
    (out / "frames").mkdir(parents=True, exist_ok=True)
    video = next(l.split(": ", 1)[1].strip() for l in open(run / "summary.txt") if l.startswith("video: "))
    cap = cv2.VideoCapture(video)
    want = set(frame.tolist())
    rows, f, last = [], 0, int(frame.max())
    pos = {int(v): i for i, v in enumerate(frame)}
    while f <= last:
        if f not in want:
            ok = cap.grab()
        else:
            ok, img = cap.read()
            if ok:
                i = pos[f]
                x0, y0 = int(fr.x0[f]), int(fr.y0[f])
                g = img[y0 : y0 + CROP, x0 : x0 + CROP, 0] if img.ndim == 3 else img[y0 : y0 + CROP, x0 : x0 + CROP]
                name = f"{session}_{f:07d}.jpg"
                cv2.imwrite(str(out / "frames" / name), g, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_q])
                r = dict(file=name, video=session, frame=f, x0=x0, y0=y0, crop=CROP)
                for k, n in enumerate(NODES):
                    r[f"{n}_x"], r[f"{n}_y"] = xy[i, k, 0] - x0, xy[i, k, 1] - y0
                    r[f"{n}_score"] = score[i, k] if not np.isnan(xy[i, k, 0]) else np.nan
                rows.append(dict(r, lr_suspect=False, source="teacher"))
        if not ok:
            raise SystemExit(f"video ended at frame {f}")
        f += 1
        if f % 100000 == 0:
            print(f"  frame {f}/{last}, {len(rows)} written", flush=True)
    pd.DataFrame(rows).to_csv(out / "index.csv", index=False)
    sc = pd.DataFrame(rows)[[f"{n}_score" for n in NODES]]
    print(f"{len(rows)} frames -> {out}; teacher score >= 0.5: "
          + ", ".join(f"{n} {np.mean(sc[f'{n}_score'] >= 0.5):.0%}" for n in NODES))


if __name__ == "__main__":
    main()
