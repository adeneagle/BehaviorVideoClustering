"""Cache mouse-centred crops of frames with confident SLEAP PREDICTIONS.

    python extract_pred_frames.py PRED.slp [PRED.slp ...] [--n 1500] [--name sleap_pred]

The hand-labelled frames (extract_kp_frames.py) are few; whole-video
predictions from a good SLEAP model give many more frames with paw positions
and left/right, at the price of some wrong ones. A frame is used only if
    * both hind paws score >= --hi and the tail base >= 0.5
    * snout and each forepaw are either confident (>= --hi) or clearly
      absent (<= --lo); anything in between is ambiguous and the frame is
      skipped. Absent points are written as NaN ("not visible").
    * left/right agree with the rest of the data (hind paws on the usual
      sides of the tail->snout axis), when the snout is there to check
Frames are drawn evenly over the eight snout/forepaw visibility patterns
(so rearing and grooming postures are not swamped by walking) and spread
over the recording, at least --min-gap frames apart.

Output has the layout of sleap_gt/: <out_root>/<name>/frames/*.jpg and
index.csv. Then run prompt_from_keypoints.py --gt <that dir> --soft-absent,
and pass the dir to train_seg.py --kp-extra.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from common import DEFAULT_OUT_ROOT
from extract_kp_frames import crop_origin
from sleap_kp import NODES, SLEAP_DIR, SNOUT, TAIL_BASE, local_video, read_slp


def usable(d, hi, lo, hand_sign):
    """Rows of a prediction file that pass the rules above; returns (mask, xy with absent -> NaN, pattern)."""
    s, xy = d["score"], d["xy"].copy()
    present = s >= hi
    clear = present | (s <= lo) | np.isnan(xy[..., 0])
    ok = present[:, 3] & present[:, 4] & (s[:, TAIL_BASE] >= 0.5) & clear[:, :3].all(1)
    xy[~present] = np.nan
    xy[:, TAIL_BASE] = d["xy"][:, TAIL_BASE]
    axis = xy[:, SNOUT] - xy[:, TAIL_BASE]
    dh = xy[:, 3] - xy[:, 4]
    with np.errstate(invalid="ignore"):
        sin = (axis[:, 0] * dh[:, 1] - axis[:, 1] * dh[:, 0]) / (np.linalg.norm(axis, axis=1) * np.linalg.norm(dh, axis=1))
        ok &= ~(sin * hand_sign < 0.2)  # NaN (no snout) passes: nothing to check against
    pattern = present[:, 0] * 4 + present[:, 1] * 2 + present[:, 2]
    return ok, xy, pattern


def pick(frames, pattern, n, min_gap, rng):
    """Evenly over patterns, spread in time, >= min_gap apart."""
    chosen = []
    quota = n // 8
    for p in sorted(np.unique(pattern), key=lambda p: (pattern == p).sum()):  # rare patterns first
        cand = rng.permutation(np.flatnonzero(pattern == p))
        got = 0
        taken = np.array(sorted(frames[chosen])) if chosen else np.array([-10**9])
        for i in cand:
            j = np.searchsorted(taken, frames[i])
            near = min(abs(frames[i] - taken[max(j - 1, 0)]), abs(taken[min(j, len(taken) - 1)] - frames[i]))
            if near >= min_gap:
                chosen.append(i)
                taken = np.insert(taken, j, frames[i])
                got += 1
                if got == quota:
                    break
    return np.array(sorted(chosen, key=lambda i: frames[i]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("predictions", nargs="+", help=".slp prediction files (name or path; names are looked up in Sleap/predictions)")
    ap.add_argument("--n", type=int, default=1500, help="frames per file")
    ap.add_argument("--name", default="sleap_pred")
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--hi", type=float, default=0.7)
    ap.add_argument("--lo", type=float, default=0.2)
    ap.add_argument("--min-gap", type=int, default=60)
    ap.add_argument("--crop", type=int, default=768)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # the hand labels define which side "left" is on
    gt = read_slp(SLEAP_DIR / "subset_labels_new.slp")["xy"]
    ax, dh = gt[:, SNOUT] - gt[:, TAIL_BASE], gt[:, 3] - gt[:, 4]
    hand_sign = np.sign(np.nanmedian(ax[:, 0] * dh[:, 1] - ax[:, 1] * dh[:, 0]))

    rng = np.random.default_rng(args.seed)
    out = Path(args.out_root) / args.name
    (out / "frames").mkdir(parents=True, exist_ok=True)
    rows = []
    for pf in args.predictions:
        pf = Path(pf) if Path(pf).exists() else SLEAP_DIR / "predictions" / pf
        d = read_slp(pf, predicted=True)
        ok, xy, pattern = usable(d, args.hi, args.lo, hand_sign)
        for v in np.unique(d["video"]):
            m = np.flatnonzero(ok & (d["video"] == v))
            sel = m[pick(d["frame"][m], pattern[m], args.n, args.min_gap, rng)]
            vpath = d["videos"][v] if d["videos"][v].startswith("/") else str(pf.parent / d["videos"][v])
            src, stem = local_video(vpath), Path(vpath).stem
            cap = cv2.VideoCapture(str(src))
            W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"{pf.name} / {stem}: {ok.mean():.0%} of {len(ok)} predicted frames usable, taking {len(sel)}; "
                  f"per pattern {np.bincount(pattern[sel], minlength=8).tolist()}", flush=True)
            for n_done, i in enumerate(sel):
                fi = int(d["frame"][i])
                name = f"{stem}_{fi:07d}.jpg"
                x0, y0 = crop_origin(xy[i], args.crop, W, H)
                if not (out / "frames" / name).exists():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                    okr, f = cap.read()
                    if not okr:
                        continue
                    cv2.imwrite(str(out / "frames" / name), f[y0 : y0 + args.crop, x0 : x0 + args.crop, 0],
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
                r = dict(file=name, video=stem, frame=fi, x0=x0, y0=y0, crop=args.crop)
                for node, (x, y) in zip(NODES, xy[i]):
                    r[f"{node}_x"], r[f"{node}_y"] = x - x0, y - y0
                r.update(lr_suspect=False, source=pf.name, pattern=int(pattern[i]))
                rows.append(r)
                if n_done % 250 == 249:
                    print(f"  {n_done + 1}/{len(sel)}", flush=True)
    df = pd.DataFrame(rows).drop_duplicates("file")
    df.to_csv(out / "index.csv", index=False)
    print(f"wrote {len(df)} frames -> {out}")


if __name__ == "__main__":
    main()
