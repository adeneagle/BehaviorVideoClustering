"""Score a part network against hand-placed SLEAP keypoints.

    python eval_keypoints.py RUN_DIR [--radius 12] [--tag NAME]

The SAM masks the network is trained on are not ground truth, and they all
come from one session. The SLEAP labels are human-placed and span several
sessions and animals (cache them first with extract_kp_frames.py), so this
is the test of (a) whether the right paw gets the right label and (b) whether
the network works on animals it has never seen.

For each visible paw keypoint, the predicted part within --radius px of it
(crop pixels) is one of:
    correct      the right paw
    wrong_side   the same limb pair, other side (an L/R swap)
    wrong_limb   a fore/hind confusion
    missed       no paw there (body or background)
For keypoints marked not visible, `phantom` is the share of frames where the
network predicts that paw anyway (> --min-area px).
snout / tail_base count as correct when body is within the radius.

For networks with a tail class, the predicted tail base (axis.py) is scored
against the tail_base keypoint (distance), and two body axes against the
human one, tail_base -> snout: tail base -> body centroid, and the paw axis
(hindpaw midpoint -> forepaw midpoint).

Output in <run>/keypoint_eval[_<tag>]/: per_keypoint.csv, summary.txt,
failures.jpg (worst frames), and prints the summary.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from axis import angle_diff, paw_axis, tail_axis, tail_base
from common import DEFAULT_OUT_ROOT, part_id
from render import enhance, overlay
from seg_model import load_checkpoint, to_input
from sleap_kp import NODES, PAW_NODES, SNOUT, TAIL_BASE

MIRROR = {"forepaw_L": "forepaw_R", "forepaw_R": "forepaw_L", "hindpaw_L": "hindpaw_R", "hindpaw_R": "hindpaw_L"}
KP_BGR = [(0, 255, 0), (60, 60, 255), (255, 120, 60), (60, 255, 255), (255, 60, 255), (255, 255, 255)]


@torch.no_grad()
def predict_images(model, paths, size, batch=32, device="cuda", flip_tta=False):
    """Label maps (part ids) at the images' own resolution."""
    cls_to_obj = None
    out = []
    for s in range(0, len(paths), batch):
        imgs = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in paths[s : s + batch]]
        H, W = imgs[0].shape
        g = np.stack([cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA) for im in imgs])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(to_input(torch.from_numpy(g).to(device))).float()
        if cls_to_obj is None:
            from seg_model import CLASS_OBJ_IDS
            cls_to_obj = torch.tensor(CLASS_OBJ_IDS[: logits.shape[1]], dtype=torch.uint8, device=device)
        cls = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False).argmax(1)
        out.append(cls_to_obj[cls].cpu().numpy())
    return np.concatenate(out)


def classify(lab, xy, radius, want, paw_ids):
    """What the network put at a keypoint. `want` is a part name."""
    H, W = lab.shape
    x, y = int(round(xy[0])), int(round(xy[1]))
    x0, x1, y0, y1 = max(x - radius, 0), min(x + radius + 1, W), max(y - radius, 0), min(y + radius + 1, H)
    if x0 >= x1 or y0 >= y1:
        return "outside_crop"
    yy, xx = np.ogrid[y0 - y : y1 - y, x0 - x : x1 - x]
    near = set(np.unique(lab[y0:y1, x0:x1][yy * yy + xx * xx <= radius * radius]).tolist())
    if paw_ids[want] in near:
        return "correct"
    if want == "body":
        return "missed"
    if paw_ids[MIRROR[want]] in near:
        return "wrong_side"
    if near & (set(paw_ids.values()) - {paw_ids["body"]}):
        return "wrong_limb"
    return "missed"


def score_keypoints(labs, idx, radius=12, min_area=150):
    """One row per keypoint: what the label maps (part ids, crop resolution) put there."""
    ids = {n: part_id(n) for n in list(MIRROR) + ["body"]}
    want = {SNOUT: "body", TAIL_BASE: "body", **PAW_NODES}
    rows = []
    for i, r in enumerate(idx.itertuples()):
        for k, node in enumerate(NODES):
            xy = (getattr(r, f"{node}_x"), getattr(r, f"{node}_y"))
            if np.isnan(xy[0]):
                res = None
                if k in PAW_NODES:
                    res = "phantom" if (labs[i] == ids[want[k]]).sum() > min_area else "absent_ok"
            else:
                res = classify(labs[i], xy, radius, want[k], ids)
            if res:
                rows.append(dict(file=r.file, video=r.video, frame=r.frame, node=node, part=want[k], result=res))
    return pd.DataFrame(rows)


def paw_rates(df):
    """Share of visible paw keypoints in each outcome, and of frames with every visible paw right."""
    paws = df[(df.part != "body") & ~df.result.isin(["phantom", "absent_ok", "outside_crop"])]
    out = {c: float((paws.result == c).mean()) for c in ("correct", "wrong_side", "wrong_limb", "missed")}
    out["all_paws_correct"] = float(paws.groupby("file").result.agg(lambda s: (s == "correct").all()).mean())
    return out


def score_axis(labs, idx):
    """One row per frame with a visible tail_base: tail base distance (crop px)
    and heading errors (deg) against tail_base -> snout."""
    rows = []
    for i, r in enumerate(idx.itertuples()):
        tb = np.array([r.tail_base_x, r.tail_base_y])
        if np.isnan(tb[0]):
            continue
        sn = np.array([r.snout_x, r.snout_y])
        true = np.degrees(np.arctan2(*(sn - tb)[::-1])) if not np.isnan(sn[0]) else np.nan
        p, ta, pa = tail_base(labs[i]), tail_axis(labs[i]), paw_axis(labs[i])
        rows.append(dict(file=r.file, video=r.video, tail_found=p is not None,
                         tail_dist=float(np.hypot(*(p - tb))) if p is not None else np.nan,
                         err_tail_axis=angle_diff(ta[0], true) if ta and not np.isnan(true) else np.nan,
                         err_paw_axis=angle_diff(pa[0], true) if pa and not np.isnan(true) else np.nan))
    return pd.DataFrame(rows)


def axis_rates(ax):
    """Summary of score_axis(): tail detection, distance and heading errors."""
    out = dict(tail_found=ax.tail_found.mean(), tail_dist_median=ax.tail_dist.median(),
               tail_within_12px=(ax.tail_dist <= 12).sum() / len(ax), tail_within_25px=(ax.tail_dist <= 25).sum() / len(ax))
    for k in ("tail_axis", "paw_axis"):
        e = ax[f"err_{k}"]
        out |= {f"{k}_defined": e.notna().mean(), f"{k}_err_median": e.median(),
                f"{k}_err_p90": e.quantile(0.9), f"{k}_err_over_45": (e > 45).sum() / e.notna().sum()}
    return out


def axis_lines(ax):
    a = axis_rates(ax)
    return [f"tail base ({len(ax)} frames with a visible tail_base keypoint): found {a['tail_found']:.1%}, "
            f"median distance {a['tail_dist_median']:.1f} px, within 12 px {a['tail_within_12px']:.1%}, "
            f"within 25 px {a['tail_within_25px']:.1%}",
            "heading error vs tail_base -> snout (deg):"] + [
            f"  {k:9s}: defined {a[f'{k}_defined']:.1%}, median {a[f'{k}_err_median']:.1f}, "
            f"90th pct {a[f'{k}_err_p90']:.1f}, > 45 deg {a[f'{k}_err_over_45']:.1%}" for k in ("tail_axis", "paw_axis")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="seg_runs/<run> directory")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--gt", default=str(DEFAULT_OUT_ROOT / "sleap_gt"))
    ap.add_argument("--radius", type=int, default=12, help="px in the 768 px crop (a paw is ~40 px across)")
    ap.add_argument("--min-area", type=int, default=150, help="crop px; smaller blobs are not a 'phantom' paw")
    ap.add_argument("--videos", nargs="*", help="only these videos (substring match)")
    ap.add_argument("--tag")
    args = ap.parse_args()

    run, gt = Path(args.run), Path(args.gt)
    idx = pd.read_csv(gt / "index.csv")
    n_suspect = int(idx.lr_suspect.sum())
    idx = idx[~idx.lr_suspect].reset_index(drop=True)  # mirrored SLEAP labels, see extract_kp_frames.py
    if args.videos:
        idx = idx[idx.video.map(lambda v: any(s in v for s in args.videos))].reset_index(drop=True)
    model, ck = load_checkpoint(run / args.ckpt)
    labs = predict_images(model, [gt / "frames" / f for f in idx.file], ck["cfg"]["size"])
    df = score_keypoints(labs, idx, args.radius, args.min_area)
    out = run / ("keypoint_eval" + (f"_{args.tag}" if args.tag else ""))
    out.mkdir(exist_ok=True)
    df.to_csv(out / "per_keypoint.csv", index=False)

    vis = df[~df.result.isin(["phantom", "absent_ok", "outside_crop"])]
    cols = ["correct", "wrong_side", "wrong_limb", "missed"]

    def table(by):
        t = pd.crosstab(vis[by], vis.result, normalize="index").reindex(columns=cols, fill_value=0) * 100
        t.insert(0, "n", vis.groupby(by).size())
        return t

    paws = vis[vis.part != "body"]
    per_frame = paws.groupby(["video", "file"]).result.agg(lambda s: (s == "correct").all())
    hid = df[df.result.isin(["phantom", "absent_ok"])]
    fmt = lambda v: f"{v:.1f}"
    lines = [
        f"model: {run.name} ({args.ckpt})   keypoints: {gt}",
        f"{len(idx)} hand-labelled frames, radius {args.radius} px "
        f"({n_suspect} frames with left/right-mirrored labels left out)", "",
        "% of visible keypoints, by keypoint:", table("node").to_string(float_format=fmt), "",
        "% of visible PAW keypoints, by video:",
        (pd.crosstab(paws.video, paws.result, normalize="index").reindex(columns=cols, fill_value=0) * 100)
        .assign(all_paws_correct=per_frame.groupby("video").mean() * 100)
        .assign(frames=per_frame.groupby("video").size()).to_string(float_format=fmt), "",
        f"all visible paws correct: {per_frame.mean():.1%} of frames",
        f"paw keypoints overall: " + "  ".join(f"{c} {(paws.result == c).mean():.1%}" for c in cols),
    ]
    if len(hid):
        ph = hid.groupby("node").result.agg(lambda s: (s == "phantom").mean() * 100)
        lines += ["", "paw marked not visible, but predicted anyway (% of such frames):",
                  ph.to_frame("phantom").assign(n=hid.groupby("node").size()).to_string(float_format=fmt)]
    if "tail" in ck["class_names"]:
        ax = score_axis(labs, idx)
        ax.to_csv(out / "per_frame_axis.csv", index=False)
        lines += ["", *axis_lines(ax), "", "by video (median tail base distance px / tail axis error deg / paw axis error deg):",
                  ax.groupby("video")[["tail_dist", "err_tail_axis", "err_paw_axis"]].median().to_string(float_format=fmt)]
    (out / "summary.txt").write_text("\n".join(lines))
    print("\n".join(lines))

    # worst frames: most wrong paw keypoints first, spread over videos
    bad = paws[paws.result != "correct"].groupby(["video", "file"]).size().rename("n_bad").reset_index()
    bad = bad.sort_values("n_bad", ascending=False).groupby("video").head(6)
    tiles = []
    for _, b in bad.head(36).iterrows():
        i = idx.index[idx.file == b.file][0]
        im = cv2.cvtColor(enhance(cv2.imread(str(gt / "frames" / b.file), cv2.IMREAD_GRAYSCALE)), cv2.COLOR_GRAY2BGR)
        ov = overlay(im, labs[i])
        for k, node in enumerate(NODES):
            x, y = idx.loc[i, f"{node}_x"], idx.loc[i, f"{node}_y"]
            if not np.isnan(x):
                cv2.circle(ov, (int(x), int(y)), 9, (0, 0, 0), -1)
                cv2.circle(ov, (int(x), int(y)), 7, KP_BGR[k], -1)
        ov = cv2.resize(ov, (384, 384), interpolation=cv2.INTER_AREA)
        cv2.putText(ov, f"{b.video[9:19]} {idx.loc[i, 'frame']}", (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        tiles.append(ov)
    if tiles:
        ncol = 6
        tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % ncol)
        cv2.imwrite(str(out / "failures.jpg"), np.vstack([np.hstack(tiles[i : i + ncol]) for i in range(0, len(tiles), ncol)]))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
