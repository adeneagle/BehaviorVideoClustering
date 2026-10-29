"""Body axis from a part label map: tail base -> body centroid.

The tail base is where the tail mask meets the body (see tail_base(); the
free tail swings and is useless for direction). Works on label maps of part
ids at any resolution.

The axis tilts towards the inside of a curled body (the centroid moves into
the bend), and gets short and noisy when the mouse rears and the body is
foreshortened; `length` (in px) is a usable confidence.

    python axis.py RUN_DIR [--workers 12]

writes RUN_DIR/axis.npz for a run_video.py output: per frame, tail_angle /
tail_len and paw_angle / paw_len (deg, px at the label maps' resolution;
NaN where undefined). embed_vjepa.py builds it when it is missing.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from common import part_id

BODY, TAIL = part_id("body"), part_id("tail")
FORE, HIND = (part_id("forepaw_L"), part_id("forepaw_R")), (part_id("hindpaw_L"), part_id("hindpaw_R"))


def centroid(mask):
    ys, xs = np.nonzero(mask)
    return np.array([xs.mean(), ys.mean()]) if len(xs) else None


def tail_base(lab, reach=None, max_gap=None, min_area=None):
    """(x, y) of the tail base, or None if no tail piece comes within max_gap
    of the body. The network often leaves a few px of background between the
    tail root and the body, so this takes the nearest tail piece of at least
    min_area px and the centroid of its pixels within `reach` of its closest
    approach to the body. Defaults scale with the map (768 px: reach 6,
    max_gap 24, min_area 300)."""
    S = lab.shape[0]
    reach, max_gap = reach or max(2, S // 128), max_gap or S // 32
    min_area = min_area or S * S // 2000
    body, tail = lab == BODY, lab == TAIL
    if not body.any() or not tail.any():
        return None
    dist = cv2.distanceTransform((~body).astype(np.uint8), cv2.DIST_L2, 5)
    n, comp, stats, _ = cv2.connectedComponentsWithStats(tail.astype(np.uint8))
    best = None
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] >= min_area:
            d = dist[comp == k].min()
            if d <= max_gap and (best is None or d < best[1]):
                best = (k, d)
    if best is None:
        return None
    k, d = best
    return centroid((comp == k) & (dist <= d + reach))


def tail_axis(lab):
    """(angle in degrees, length in px) of tail base -> body centroid; angle in
    image coordinates (0 = right, +90 = down). None without a tail base."""
    tb, c = tail_base(lab), centroid(lab == BODY)
    if tb is None or c is None:
        return None
    d = c - tb
    return float(np.degrees(np.arctan2(d[1], d[0]))), float(np.hypot(*d))


def paw_axis(lab):
    """Hindpaw midpoint -> forepaw midpoint, as embed_vjepa.py's first heading."""
    cs = [centroid(lab == i) for i in FORE + HIND]
    if any(c is None for c in cs):
        return None
    d = (cs[0] + cs[1]) / 2 - (cs[2] + cs[3]) / 2
    return float(np.degrees(np.arctan2(d[1], d[0]))), float(np.hypot(*d))


def angle_diff(a, b):
    """Absolute difference of two angles in degrees, in [0, 180]."""
    return np.abs((np.asarray(a) - np.asarray(b) + 180) % 360 - 180)


def _chunk_axes(path):
    rows = []
    for lab in np.load(path)["labels"]:
        t, p = tail_axis(lab), paw_axis(lab)
        rows.append((*(t or (np.nan, np.nan)), *(p or (np.nan, np.nan))))
    return np.array(rows, np.float32).reshape(-1, 4)


def run_axes(run_dir, workers=12):
    """Per-frame axes of a run_video.py output, cached in RUN_DIR/axis.npz."""
    path = Path(run_dir) / "axis.npz"
    files = sorted(Path(run_dir, "labels").glob("labels_*.npz"))
    if path.exists() and path.stat().st_mtime > max(f.stat().st_mtime for f in files):
        return dict(np.load(path))
    with ProcessPoolExecutor(workers) as ex:
        a = np.concatenate(list(ex.map(_chunk_axes, files)))
    out = dict(tail_angle=a[:, 0], tail_len=a[:, 1], paw_angle=a[:, 2], paw_len=a[:, 3])
    np.savez(path, **out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    a = run_axes(args.run_dir, args.workers)
    t, p = ~np.isnan(a["tail_angle"]), ~np.isnan(a["paw_angle"])
    both = t & p
    print(f"{len(t)} frames: tail axis {t.mean():.1%}, paw axis {p.mean():.1%}, either {(t | p).mean():.1%}; "
          f"tail vs paw median {np.median(angle_diff(a['tail_angle'][both], a['paw_angle'][both])):.1f} deg")
