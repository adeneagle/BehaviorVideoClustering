"""Per-frame part centroids from a run_video.py output: the movement measurements that
score_behaviour.py predicts. They come from the part network's label maps only, so they are
independent of V-JEPA and of any clustering (not of segmentation errors).

    python part_tracks.py RUN_DIR [--workers 8]

writes RUN_DIR/part_tracks.npz:
  parts     names of the parts, in column order (body, forepaw_L, forepaw_R, hindpaw_L, hindpaw_R)
  xy        [n_frames, 5, 2] centroid in full-res image px (NaN where the part is absent)
  area      [n_frames, 5] px at the label maps' resolution
  heading   [n_frames] deg, image coords (0 = right, +90 = down): tail base -> body centroid,
            else hindpaw -> forepaw midpoints (axis.py), NaN where neither exists
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from axis import run_axes
from common import part_id

NAMES = ("body", "forepaw_L", "forepaw_R", "hindpaw_L", "hindpaw_R")
IDS = np.array([part_id(n) for n in NAMES])
CROP = 768  # run_video.py's crop side (full-res px)
BLOCK = 100


def _chunk(path):
    """Centroids (label-map px) and areas of every part in one labels_<first>.npz."""
    lab = np.load(path)["labels"]
    n, S = lab.shape[0], lab.shape[1]
    nid = 256  # any uint8 value (tail, implant and others are counted, then dropped)
    cnt, sx, sy = (np.zeros((n, nid)) for _ in range(3))
    xs = np.tile(np.arange(S, dtype=np.float64), S)
    ys = np.repeat(np.arange(S, dtype=np.float64), S)
    for a in range(0, n, BLOCK):
        blk = lab[a : a + BLOCK].reshape(-1, S * S).astype(np.int64)
        m = len(blk)
        keys = (blk + nid * np.arange(m)[:, None]).ravel()
        L = m * nid
        cnt[a : a + m] = np.bincount(keys, minlength=L).reshape(m, nid)
        sx[a : a + m] = np.bincount(keys, np.tile(xs, m), minlength=L).reshape(m, nid)
        sy[a : a + m] = np.bincount(keys, np.tile(ys, m), minlength=L).reshape(m, nid)
    c = cnt[:, IDS]
    with np.errstate(invalid="ignore", divide="ignore"):
        xy = np.stack([sx[:, IDS] / c, sy[:, IDS] / c], -1)
    return xy.astype(np.float32), c.astype(np.float32), S


def part_tracks(run_dir, workers=8):
    """Cached in RUN_DIR/part_tracks.npz (rebuilt when the label maps are newer)."""
    run = Path(run_dir)
    path = run / "part_tracks.npz"
    files = sorted(run.glob("labels/labels_*.npz"))
    if path.exists() and path.stat().st_mtime > max(f.stat().st_mtime for f in files):
        return dict(np.load(path))
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_chunk, files))
    xy = np.concatenate([r[0] for r in res])
    area = np.concatenate([r[1] for r in res])
    S = res[0][2]
    fr = pd.read_csv(run / "frames.csv", usecols=["x0", "y0"])
    assert len(fr) == len(xy), (len(fr), len(xy))
    # label-map px -> full-res image px (the map covers the 768 px crop at offset x0, y0)
    xy = (xy + 0.5) * (CROP / S) - 0.5 + fr[["x0", "y0"]].to_numpy()[:, None, :].astype(np.float32)
    ax = run_axes(run)
    heading = np.where(np.isnan(ax["tail_angle"]), ax["paw_angle"], ax["tail_angle"]).astype(np.float32)
    out = dict(parts=np.array(NAMES), xy=xy, area=area, heading=heading)
    np.savez(path, **out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    t = part_tracks(args.run_dir, args.workers)
    present = ~np.isnan(t["xy"][..., 0])
    print(f"{len(present)} frames; part present: " + ", ".join(f"{n} {p:.1%}" for n, p in zip(NAMES, present.mean(0)))
          + f"; heading defined {np.mean(~np.isnan(t['heading'])):.1%}")
