"""Paw masks for the hand-labelled SLEAP frames, with SAM3 clicked by the keypoints.

    python prompt_from_keypoints.py [--gt DIR] [--body-from SEG_RUN_DIR] [--soft-absent] [--redo]

Each cached frame (extract_kp_frames.py) is segmented on its own, exactly as
a person clicking in annotate.py without tracking: one click on each paw's
keypoint, with right-clicks on the other paws. Left/right therefore come
from the human keypoint labels, not from SAM or from a tracker.

Only the paws are taken from SAM. Clicking the body the same way was tried
and is not reliable on single frames (holes, missing bodies, and a tail sink
that floods the floor). Without --body-from, everything that is not a paw is
stored as "body or background, unknown which" (254) and train_seg.py scores
those pixels only for not being a paw. That alone lets the body class drift
on these sessions, so --body-from fills them from an existing network:
body where it is confident AND the blob touches a keypoint (stray blobs
become background), background where it is confident, 254 where it is not.
These body labels are the network's own opinion, not ground truth.

--soft-absent is for frames from SLEAP PREDICTIONS (extract_pred_frames.py),
where "not visible" only means the SLEAP model was unsure: wherever the
--body-from network sees a paw whose keypoint is absent, those pixels are
ignored instead of being taught as "not a paw". For hand labels, absent
means absent, so leave it off.

A SAM paw mask is kept only if it sits on its keypoint and has the size and
reach of a paw; otherwise (SAM took the tail, a wall reflection, the whole
body ...) a small disc on the keypoint is used and its surroundings are
marked "ignore".

Output in <out_root>/sleap_gt/:
    sam_paws.npz       raw SAM label maps (cached; --redo to recompute)
    kp_masks.npz       labels [N,H,W] uint8: part ids, 0 = background,
                       254 = body or background, 255 = ignore; files [N]
                       (rows as index.csv)
    paw_masks.csv      per frame and paw: area, kept or replaced by a disc
    kp_masks_sheet.jpg
"""

import argparse
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from common import DEFAULT_OUT_ROOT, part_id
from render import enhance, overlay
from sleap_kp import NODES, PAW_NODES

KP_BGR = [(0, 255, 0), (60, 60, 255), (255, 120, 60), (60, 255, 255), (255, 60, 255), (255, 255, 255)]
IGNORE, UNKNOWN = 255, 254
# limits in crop pixels (768 px crop; a spread hind paw is ~70 px long)
MIN_AREA, MAX_AREA, MAX_REACH, ON_KP = 120, 7000, 95, 6
DISC, IGNORE_RING = 7, 45


def keypoints(idx):
    return np.stack([idx[[f"{n}_x", f"{n}_y"]].to_numpy() for n in NODES], 1)  # [N,6,2]


def paw_prompts(kp, crop):
    """kp [6,2] crop px (NaN = not visible) -> {obj_id: (xy list, label list)}."""
    paws = {PAW_NODES[k]: kp[k] for k in PAW_NODES
            if not np.isnan(kp[k, 0]) and 0 <= kp[k, 0] < crop and 0 <= kp[k, 1] < crop}
    out = {}
    for name, p in paws.items():
        neg = [q for n, q in paws.items() if n != name]
        out[part_id(name)] = ([list(map(float, q)) for q in [p] + neg], [1] + [0] * len(neg))
    return out


def run_sam(gt, idx, chunk=200):
    from sam_session import PartSession, build_tracker

    tracker = build_tracker()
    crop = int(idx.crop.iloc[0])
    labels = np.zeros((len(idx), crop, crop), np.uint8)
    kps = keypoints(idx)
    order = [part_id(n) for n in ("hindpaw_L", "hindpaw_R", "forepaw_L", "forepaw_R")]
    for s in range(0, len(idx), chunk):
        with tempfile.TemporaryDirectory() as tmp:  # the tracker wants a folder of 00000.jpg ...
            for j, f in enumerate(idx.file.iloc[s : s + chunk]):
                os.symlink(gt / "frames" / f, Path(tmp) / f"{j:05d}.jpg")
            ses = PartSession(tracker, tmp, order)
            for j in range(ses.T):
                pr = paw_prompts(kps[s + j], ses.W)
                lab = None
                for o in order:
                    if o in pr:
                        lab = ses.set_points(j, o, *pr[o])
                if lab is not None:
                    labels[s + j] = lab
        print(f"  {min(s + chunk, len(idx))}/{len(idx)}", flush=True)
    return labels


def clean(raw, kps):
    """Raw SAM paw maps -> training targets (see module docstring) + a per-paw table."""
    N, H, W = raw.shape
    yy, xx = np.mgrid[:H, :W]
    out = np.zeros_like(raw)
    rows = []
    for i in range(N):
        fallback = []
        for k, name in PAW_NODES.items():
            x, y = kps[i, k]
            if np.isnan(x) or not (0 <= x < W and 0 <= y < H):
                continue
            m = raw[i] == part_id(name)
            d2 = (xx - x) ** 2 + (yy - y) ** 2
            area = int(m.sum())
            reach = float(np.sqrt(d2[m].max())) if area else 0.0
            near = float(np.sqrt(d2[m].min())) if area else np.inf
            ok = MIN_AREA <= area <= MAX_AREA and reach <= MAX_REACH and near <= ON_KP
            if ok:
                out[i][m] = part_id(name)
            else:
                fallback.append((name, d2))
            rows.append(dict(row=i, part=name, area=area, reach=round(reach, 1), kept=ok))
        for name, d2 in fallback:  # rings first, discs after, so a disc is never ignored
            out[i][(d2 <= IGNORE_RING**2) & (out[i] == 0)] = IGNORE
        for name, d2 in fallback:
            out[i][d2 <= DISC**2] = part_id(name)
    return out, pd.DataFrame(rows)


@torch.no_grad()
def fill_body(gt, idx, labels, run, sure=0.7, touch=15, batch=16, soft_absent=False):
    """Replace 'not a paw' (0) by body / background / UNKNOWN from a trained network."""
    from seg_model import load_checkpoint, to_input

    model, ck = load_checkpoint(Path(run) / "best.pt")
    size, body = ck["cfg"]["size"], part_id("body")
    kps = keypoints(idx)
    out = labels.copy()
    H, W = labels.shape[1:]
    for s in range(0, len(idx), batch):
        g = np.stack([cv2.resize(cv2.imread(str(gt / "frames" / f), cv2.IMREAD_GRAYSCALE), (size, size),
                                 interpolation=cv2.INTER_AREA) for f in idx.file.iloc[s : s + batch]])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(to_input(torch.from_numpy(g).cuda())).float()
        logits = torch.nn.functional.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        p_body = logits[:, :2].softmax(1)[:, 1].cpu().numpy()  # share of body among {background, body}
        seen = np.array(ck["class_obj_ids"], np.uint8)[logits.argmax(1).cpu().numpy()]  # the network's own parts
        for j in range(len(g)):
            i = s + j
            free = labels[i] == 0
            lab = out[i]
            lab[free & (p_body[j] > 1 - sure) & (p_body[j] < sure)] = UNKNOWN
            cand = ((p_body[j] >= sure) & free) | ((labels[i] != 0) & (labels[i] != IGNORE))
            n, comp = cv2.connectedComponents(cand.astype(np.uint8))
            keep = set()
            for x, y in kps[i]:
                if not np.isnan(x):
                    x0, y0 = int(max(x - touch, 0)), int(max(y - touch, 0))
                    keep |= set(np.unique(comp[y0 : int(y + touch) + 1, x0 : int(x + touch) + 1]).tolist())
            keep.discard(0)
            lab[free & (p_body[j] >= sure) & np.isin(comp, list(keep))] = body
            if soft_absent:
                for k, name in PAW_NODES.items():
                    if np.isnan(kps[i, k, 0]):
                        lab[free & (seen[j] == part_id(name))] = IGNORE
    return out


def sheet(gt, idx, labels, path, n=60, cols=10, tile=256):
    pick = np.linspace(0, len(idx) - 1, min(n, len(idx))).astype(int)
    tiles = []
    for i in pick:
        im = cv2.cvtColor(enhance(cv2.imread(str(gt / "frames" / idx.file.iloc[i]), cv2.IMREAD_GRAYSCALE)), cv2.COLOR_GRAY2BGR)
        lab = labels[i].copy()
        ig = lab >= UNKNOWN
        lab[ig] = 0
        ov = overlay(im, lab)
        ov[ig] = (0.6 * ov[ig] + 0.4 * np.array([0, 160, 0])).astype(np.uint8)
        ov = cv2.resize(ov, (tile, tile), interpolation=cv2.INTER_AREA)
        cv2.putText(ov, f"{idx.video.iloc[i][9:19]} {idx.frame.iloc[i]}", (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        tiles.append(ov)
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % cols)
    cv2.imwrite(str(path), np.vstack([np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", default=str(DEFAULT_OUT_ROOT / "sleap_gt"))
    ap.add_argument("--redo", action="store_true", help="re-run SAM even if sam_paws.npz exists")
    ap.add_argument("--body-from", help="seg_runs/<run>: fill the non-paw pixels from this network")
    ap.add_argument("--soft-absent", action="store_true",
                    help="keypoints come from predictions: a missing paw is 'unknown', not 'absent'")
    args = ap.parse_args()

    gt = Path(args.gt)
    idx = pd.read_csv(gt / "index.csv")
    raw_path = gt / "sam_paws.npz"
    if raw_path.exists() and not args.redo and list(np.load(raw_path)["files"]) == list(idx.file):
        raw = np.load(raw_path)["labels"]
    else:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            raw = run_sam(gt, idx)
        np.savez_compressed(raw_path, labels=raw, files=idx.file.to_numpy().astype(str))
    labels, tab = clean(raw, keypoints(idx))
    tab.insert(0, "file", idx.file.to_numpy()[tab.row])
    tab.insert(1, "video", idx.video.to_numpy()[tab.row])
    tab.drop(columns="row").to_csv(gt / "paw_masks.csv", index=False)
    if args.body_from:
        labels = fill_body(gt, idx, labels, args.body_from, soft_absent=args.soft_absent)
    else:
        if args.soft_absent:
            raise SystemExit("--soft-absent needs --body-from")
        labels[labels == 0] = UNKNOWN
    np.savez_compressed(gt / "kp_masks.npz", labels=labels, files=idx.file.to_numpy().astype(str),
                        body_from=str(args.body_from or ""))
    sheet(gt, idx, labels, gt / "kp_masks_sheet.jpg")
    print(f"{len(tab)} paw keypoints: SAM mask kept for {tab.kept.mean():.1%}, disc for the rest")
    print(tab.groupby("part").agg(n=("kept", "size"), kept=("kept", "mean"), median_area=("area", "median")).to_string(float_format=lambda v: f"{v:.2f}"))
    print(tab.groupby("video").kept.mean().to_string(float_format=lambda v: f"{v:.2f}"))
    print(f"wrote {gt / 'kp_masks.npz'}")


if __name__ == "__main__":
    main()
