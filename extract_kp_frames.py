"""Cache mouse-centred crops of the hand-labelled SLEAP frames.

    python extract_kp_frames.py [--slp LABELS.slp] [--crop 768]

These frames come from several sessions and animals and carry human-placed
keypoints, so they are the only ground truth that is independent of SAM.
Written to <out_root>/sleap_gt/: frames/<video>_<frame>.jpg and index.csv
(crop offset and keypoints in crop pixels; NaN = not visible).

index.csv also gets `lr_suspect`: frames whose hind AND fore paws are both on
the opposite sides of the body axis from the rest of the dataset, i.e. the
labeller most likely swapped left and right for the whole animal. They are
left out of training and scoring; worth fixing in the SLEAP project.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from common import DEFAULT_OUT_ROOT
from sleap_kp import NODES, SLEAP_DIR, SNOUT, TAIL_BASE, local_video, read_slp


def crop_origin(xy, crop, W, H):
    """Centre on the snout-tail midpoint (the body), else on the visible points."""
    ends = xy[[SNOUT, TAIL_BASE]]
    c = ends.mean(0) if not np.isnan(ends).any() else np.nanmean(xy, 0)
    x0 = int(np.clip(round(c[0] - crop / 2), 0, W - crop))
    y0 = int(np.clip(round(c[1] - crop / 2), 0, H - crop))
    return x0, y0


def lr_suspect(xy, min_sin=0.5):
    """xy [N,6,2] -> bool [N]: left/right mirrored relative to the dataset majority."""
    axis = xy[:, SNOUT] - xy[:, TAIL_BASE]

    def side(left, right):  # sine of the angle from the body axis to (left - right)
        d = xy[:, left] - xy[:, right]
        return (axis[:, 0] * d[:, 1] - axis[:, 1] * d[:, 0]) / (np.linalg.norm(axis, axis=1) * np.linalg.norm(d, axis=1))

    hind, fore = side(3, 4), side(1, 2)
    hind, fore = hind * np.sign(np.nanmedian(hind)), fore * np.sign(np.nanmedian(fore))
    with np.errstate(invalid="ignore"):
        return (hind < -min_sin) & (fore < 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slp", default=str(SLEAP_DIR / "subset_labels_new.slp"))
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--crop", type=int, default=768)
    args = ap.parse_args()

    d = read_slp(args.slp)
    out = Path(args.out_root) / "sleap_gt"
    (out / "frames").mkdir(parents=True, exist_ok=True)
    rows = []
    for v in np.unique(d["video"]):
        src = local_video(d["videos"][v])
        stem = Path(d["videos"][v]).stem
        cap = cv2.VideoCapture(str(src))
        W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        sel = np.flatnonzero(d["video"] == v)
        sel = sel[np.argsort(d["frame"][sel])]
        print(f"{stem}: {len(sel)} labelled frames", flush=True)
        for i in sel:
            fi, xy = int(d["frame"][i]), d["xy"][i]
            name = f"{stem}_{fi:06d}.jpg"
            x0, y0 = crop_origin(xy, args.crop, W, H)
            if not (out / "frames" / name).exists():
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ok, f = cap.read()
                if not ok:
                    print(f"  could not read frame {fi}")
                    continue
                cv2.imwrite(str(out / "frames" / name), f[y0 : y0 + args.crop, x0 : x0 + args.crop, 0],
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
            r = dict(file=name, video=stem, frame=fi, x0=x0, y0=y0, crop=args.crop)
            for n, (x, y) in zip(NODES, xy):
                r[f"{n}_x"], r[f"{n}_y"] = x - x0, y - y0
            rows.append(r)
    df = pd.DataFrame(rows)
    df["lr_suspect"] = lr_suspect(np.stack([df[[f"{n}_x", f"{n}_y"]].to_numpy() for n in NODES], 1))
    df.to_csv(out / "index.csv", index=False)
    print(f"wrote {len(df)} frames -> {out}")
    if df.lr_suspect.any():
        print(f"{df.lr_suspect.sum()} frames look left/right-mirrored in the SLEAP labels:")
        print(df[df.lr_suspect][["video", "frame"]].to_string(index=False))


if __name__ == "__main__":
    main()
