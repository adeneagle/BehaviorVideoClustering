"""Pick representative behaviour clips and write mouse-centred crops for SAM3.

The mouse is ~300 px across in a 2248x2064 frame, and SAM3 resizes every
frame to 1008 px, so on the raw frame a paw ends up ~10-15 px wide. Each clip
is therefore cropped to a square window (default 768 px) that follows a
smoothed mouse centroid from background subtraction; the per-frame crop
offsets are saved so masks can be mapped back to full-frame coordinates (and
so global translation is not lost for later canonicalisation).

Usage:
    python extract_clips.py --video VIDEO.mkv --labels LABELS.csv \
        [--n-per-behavior 2] [--clip-sec 6] [--crop 768]
    python extract_clips.py ... --clips-csv my_clips.csv   # start_frame,n_frames[,clip_id]
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d, median_filter

from common import DEFAULT_OUT_ROOT, write_json

UNLABELED = "unlabeled"


def count_frames(cap):
    """CAP_PROP_FRAME_COUNT over-reports on these .mkv files; find the last decodable frame."""
    lo, hi = 0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while hi - lo > 1:
        mid = (lo + hi) // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        lo, hi = (mid, hi) if cap.read()[0] else (lo, mid)
    return lo + 1


def read_gray(cap, idx=None):
    if idx is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, f = cap.read()
    if not ok:
        raise RuntimeError(f"could not decode frame {idx}")
    return f[..., 0] if f.ndim == 3 else f  # the camera is monochrome (R==G==B)


def compute_background(cap, n_frames, n_samples=240):
    idx = np.linspace(0, n_frames - 1, n_samples).astype(int)
    stack = np.stack([read_gray(cap, i) for i in idx])
    return np.median(stack, axis=0).astype(np.uint8)


def parse_time(s):
    m, sec, ms = s.split(":")
    return int(m) * 60 + int(sec) + int(ms) / 1000.0


def load_bouts(labels_csv, fps):
    df = pd.read_csv(labels_csv)
    df["start_frame"] = (df["start"].map(parse_time) * fps).round().astype(int)
    df["end_frame"] = (df["end"].map(parse_time) * fps).round().astype(int)
    df["n"] = df["end_frame"] - df["start_frame"]
    return df


def per_frame_behaviour(bouts, n_frames):
    lab = np.full(n_frames, UNLABELED, dtype=object)
    for r in bouts.itertuples():
        lab[max(r.start_frame, 0) : min(r.end_frame, n_frames)] = r.behavior
    return lab


def select_clips(bouts, n_frames, clip_len, n_per_behavior, min_gap_s, fps):
    """Per behaviour, take the longest bouts, centred in a clip window, keeping
    clips from overlapping and same-behaviour clips >= min_gap_s apart so they
    sample different parts of the session."""
    chosen = []
    for beh, grp in bouts.groupby("behavior"):
        picked = 0
        for r in grp.sort_values("n", ascending=False).itertuples():
            centre = (r.start_frame + r.end_frame) // 2
            s = int(np.clip(centre - clip_len // 2, 0, n_frames - clip_len))
            e = s + clip_len
            if any(s < c["end"] and c["start"] < e for c in chosen):
                continue
            if any(
                c["target"] == beh and abs(c["start"] - s) < min_gap_s * fps
                for c in chosen
            ):
                continue
            chosen.append(
                dict(
                    start=s,
                    end=e,
                    target=beh,
                    bout_start=int(r.start_frame),
                    bout_end=int(r.end_frame),
                )
            )
            picked += 1
            if picked == n_per_behavior:
                break
    return sorted(chosen, key=lambda c: c["start"])


def locate_mouse(gray, bg, prev=None, diff_thr=12, scale=1.0):
    """Centroid of the mouse body (tail eroded away), or None.

    The fur is only slightly brighter than the dark floor, so the difference
    image breaks the mouse into fragments; a large closing merges them first.
    Among large blobs, the one nearest the previous centroid wins, which keeps
    the crop off wall reflections.

    `scale` is the size of gray/bg relative to the full-resolution frame
    (e.g. 0.25 for 4x-downscaled input); kernels and area limits follow it,
    and the centroid is returned in the input's own pixels.
    """
    k_open = max(3, int(round(5 * scale)) | 1)
    k_close = max(5, int(round(41 * scale)) | 1)
    m = (cv2.absdiff(gray, bg) > diff_thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k_open, k_open), np.uint8))
    m = cv2.morphologyEx(
        m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
    )
    n, lab, stats, cents = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return None, 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    big = np.flatnonzero(areas >= max(3000 * scale**2, 0.3 * areas.max())) + 1
    if prev is not None and len(big) > 1:
        k = big[np.argmin(np.linalg.norm(cents[big] - prev, axis=1))]
    else:
        k = big[np.argmax(stats[big, cv2.CC_STAT_AREA])]
    area = int(stats[k, cv2.CC_STAT_AREA])
    blob = (lab == k).astype(np.uint8)
    r = max(3, int(0.12 * np.sqrt(area)))  # thin parts (tail) vanish
    core = cv2.erode(blob, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2))
    ys, xs = np.nonzero(core if core.any() else blob)
    return np.array([xs.mean(), ys.mean()]), area


TRACK_MEDIAN, TRACK_SIGMA_S = 9, 0.15  # frames, seconds


def smooth_track(cents, fps):
    cents = np.asarray(cents, float)
    t = np.arange(len(cents))
    for d in range(2):
        ok = ~np.isnan(cents[:, d])
        if not ok.any():
            raise RuntimeError("mouse never found in clip")
        cents[:, d] = np.interp(t, t[ok], cents[ok, d])
    cents = median_filter(cents, size=(TRACK_MEDIAN, 1), mode="nearest")
    return gaussian_filter1d(cents, sigma=TRACK_SIGMA_S * fps, axis=0, mode="nearest")


def track_radius(fps):
    """Frames on either side of a frame that smooth_track() reads for it (scipy truncates the Gaussian at 4 sigma)."""
    return TRACK_MEDIAN // 2 + int(4.0 * TRACK_SIGMA_S * fps + 0.5)


def extract_clip(cap, bg, clip, crop, cdir, behaviours, fps, video, jpeg_q=95):
    frames, cents, areas, prev = [], [], [], None
    cap.set(cv2.CAP_PROP_POS_FRAMES, clip["start"])
    for _ in range(clip["end"] - clip["start"]):
        g = read_gray(cap)
        c, a = locate_mouse(g, bg, prev)
        frames.append(g)
        cents.append(c if c is not None else (np.nan, np.nan))
        areas.append(a)
        prev = c if c is not None else prev
    H, W = bg.shape
    track = smooth_track(cents, fps)
    offs = np.stack(
        [
            np.clip(np.round(track[:, 0] - crop / 2), 0, W - crop),
            np.clip(np.round(track[:, 1] - crop / 2), 0, H - crop),
        ],
        1,
    ).astype(int)
    fdir = Path(cdir) / "frames"
    fdir.mkdir(parents=True, exist_ok=True)
    for i, (g, (x0, y0)) in enumerate(zip(frames, offs)):
        cv2.imwrite(
            str(fdir / f"{i:05d}.jpg"),
            g[y0 : y0 + crop, x0 : x0 + crop],
            [cv2.IMWRITE_JPEG_QUALITY, jpeg_q],
        )
    beh = list(behaviours[clip["start"] : clip["end"]])
    meta = dict(
        video=str(video),
        fps=fps,
        start_frame=clip["start"],
        n_frames=len(frames),
        crop_size=crop,
        target_behavior=clip.get("target"),
        bout=[clip.get("bout_start"), clip.get("bout_end")],
        crop_offsets_xy=offs.tolist(),
        raw_centroids_xy=[[None if np.isnan(v) else float(v) for v in c] for c in cents],
        blob_area=areas,
        behavior_per_frame=beh,
        missed_localisation=int(np.isnan(np.asarray(cents, float)[:, 0]).sum()),
    )
    write_json(Path(cdir) / "meta.json", meta)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--n-per-behavior", type=int, default=2)
    ap.add_argument("--clip-sec", type=float, default=6.0)
    ap.add_argument("--min-gap-sec", type=float, default=60.0)
    ap.add_argument("--crop", type=int, default=768)
    ap.add_argument("--clips-csv", help="explicit clips: start_frame,n_frames[,clip_id]")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    video = Path(args.video)
    vdir = Path(args.out_root) / video.stem
    vdir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = count_frames(cap)
    print(f"{video.name}: {n_frames} frames @ {fps:.3f} fps")

    bg_path = vdir / "background.npy"
    if bg_path.exists():
        bg = np.load(bg_path)
    else:
        print("computing median background ...")
        bg = compute_background(cap, n_frames)
        np.save(bg_path, bg)

    bouts = load_bouts(args.labels, fps)
    behaviours = per_frame_behaviour(bouts, n_frames)
    clip_len = int(round(args.clip_sec * fps))
    if args.clips_csv:
        clips = []
        for row in csv.DictReader(open(args.clips_csv)):
            s = int(row["start_frame"])
            clips.append(dict(start=s, end=s + int(row["n_frames"]), clip_id=row.get("clip_id")))
    else:
        clips = select_clips(
            bouts, n_frames, clip_len, args.n_per_behavior, args.min_gap_sec, fps
        )

    rows = []
    for c in clips:
        tag = (c.get("target") or "manual").replace("/", "-").replace(" ", "-")
        cid = c.get("clip_id") or f"{c['start']:06d}_{tag}"
        cdir = vdir / "clips" / cid
        if (cdir / "meta.json").exists() and not args.overwrite:
            print(f"  {cid}: exists, skipping")
        else:
            m = extract_clip(cap, bg, c, args.crop, cdir, behaviours, fps, video)
            print(f"  {cid}: {m['n_frames']} frames, localisation misses {m['missed_localisation']}")
        beh = behaviours[c["start"] : c["end"]]
        vals, cnt = np.unique(beh, return_counts=True)
        mix = "; ".join(f"{v} {n / len(beh):.0%}" for v, n in sorted(zip(vals, cnt), key=lambda x: -x[1]))
        rows.append(
            dict(clip_id=cid, start_frame=c["start"], n_frames=c["end"] - c["start"],
                 start_sec=round(c["start"] / fps, 2), target=c.get("target"), behaviour_mix=mix)
        )
    pd.DataFrame(rows).to_csv(vdir / "manifest.csv", index=False)
    print(f"wrote {len(rows)} clips -> {vdir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
