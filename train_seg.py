"""Train the fast part-segmentation network on the SAM3 part masks.

    python train_seg.py                          # auto clip split, resnet34, 512 px
    python train_seg.py --val-clips 008253_itch 037309_rear
    python train_seg.py --val-clips none         # final model on every clip

Splits are by WHOLE CLIP: neighbouring frames are near-duplicates, so a
frame-level split would score the model on frames it has effectively seen.
Frames you flagged wrong in review mode are left out of training.

Augmentation: any rotation, scale, shift, brightness/gamma/noise/blur, and
horizontal flips WITH the left/right paw labels swapped (a mirrored mouse's
left paw is its right paw).

Two kinds of training frames are mixed (--kp-share sets the proportion):
  * the clips: every pixel labelled, by SAM tracking your clicks; one session
  * the hand-labelled SLEAP frames (extract_kp_frames.py, then
    prompt_from_keypoints.py): paws only, but with human-checked left/right
    and from several sessions and animals. Their body/background comes
    from an earlier network where it was confident; the rest is only
    scored for not being a paw.

A tail class is learnt from the clips' tail masks. Where a frame has no tail
label (a clip frame without a tail mask, or any SLEAP frame) background is
only "background or tail" (seg_model.PARTIAL).

Two validation scores, and the best checkpoint maximises their sum:
  * part IoU against SAM's masks on held-out clips (agreement with SAM, not
    ground truth)
  * share of hand-placed paw keypoints that get the right paw label, on
    held-out VIDEOS (--kp-val-videos), i.e. sessions the network never saw
For the final model pass --val-clips none --kp-val-videos none.

Keypoints (--kp-distill DIR ..., 2026-10-19): the network also learns heatmaps of the six SLEAP
nodes (seg_model.KP_NAMES), from the hand-placed keypoints of the SLEAP frames (a node marked not
visible = known absent) and from frames the lab's SLEAP model labelled (distill_frames.py; only
nodes it scored >= --kp-min-score; the rest are unknown, as are all clip frames). Loss: the
CenterNet focal loss on Gaussian targets (--kp-sigma, network px), weight --kp-weight. --init
starts from a trained part network (its segmentation weights; the keypoint head is new).
Validation adds keypoint errors (crop px = full-res px) on the held-out SLEAP videos.

Output: <video>/seg_runs/<time>_<encoder>/ with best.pt, last.pt, log.csv,
report.txt (per-class IoU, per-clip IoU, L/R swap rates, confusion matrix)
and val_examples.jpg (frame | SAM | network).
"""

import argparse
import math
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from common import DEFAULT_OUT_ROOT, frame_paths, load_masks, palette_lut, part_id, read_json, write_json
from seg_model import (
    BG_OR_TAIL,
    CLASS_NAMES,
    CLASS_OBJ_IDS,
    IGNORE,
    N_CLASSES,
    PARTIAL,
    KP_MIRROR,
    KP_NAMES,
    PartSegNet,
    decode_keypoints,
    mirror_class_lut,
    obj_to_class_lut,
    save_checkpoint,
    to_input,
)

OBJ2CLS = obj_to_class_lut()
MIRROR = mirror_class_lut()
CLS2OBJ = np.array(CLASS_OBJ_IDS + [0] * (256 - N_CLASSES), np.uint8)


# ---- data -------------------------------------------------------------------------
def review_flagged(cdir):
    qc = read_json(cdir / "qc.json", {}).get("reviewed", {})
    return {int(f) for f, bad in qc.items() if bad}


def swap_frames(vdir, min_tail_pct=50.0):
    """clip -> frames that evaluate.py puts on the minority L/R side, i.e.
    likely tracker identity swaps. Only trusted where the tail set the body's
    front (the paw-based fallback misreads grooming postures)."""
    path = vdir / "eval" / "swaps.csv"
    if not path.exists():
        return None
    import pandas as pd

    sw = pd.read_csv(path)
    if "front_from_tail_pct" not in sw.columns:
        return None
    out = {}
    for r in sw.itertuples():
        if r.front_from_tail_pct < min_tail_pct or not isinstance(r.swap_runs, str):
            continue
        for run in r.swap_runs.split(";"):
            a, b = map(int, run.split("-"))
            out.setdefault(r.clip, set()).update(range(a, b + 1))
    return out


def load_clip(cdir, size, drop=()):
    obj = load_masks(cdir)["labels"]
    labels = OBJ2CLS[obj]
    no_tail = ~(obj == part_id("tail")).any((1, 2))
    labels[no_tail[:, None, None] & (obj == 0)] = BG_OR_TAIL
    keep = [t for t in range(len(labels)) if t not in drop]
    paths = frame_paths(cdir)
    imgs = np.stack([cv2.resize(cv2.imread(str(paths[t]), cv2.IMREAD_GRAYSCALE), (size, size),
                                interpolation=cv2.INTER_AREA) for t in keep])
    labs = np.stack([cv2.resize(labels[t], (size, size), interpolation=cv2.INTER_NEAREST) for t in keep])
    return imgs, labs, keep


def load_kp_frames(gt, size):
    """Hand-labelled SLEAP frames with keypoint-clicked paw masks -> images,
    targets (classes, NOTPAW, IGNORE) and the keypoint table. They have no
    tail labels, so their background is BG_OR_TAIL. Frames whose SLEAP labels
    look left/right-mirrored are dropped."""
    import pandas as pd

    idx = pd.read_csv(gt / "index.csv")
    z = np.load(gt / "kp_masks.npz")
    if list(z["files"]) != list(idx.file):
        raise SystemExit(f"{gt}/kp_masks.npz does not match index.csv: re-run prompt_from_keypoints.py")
    ok = ~idx.lr_suspect.to_numpy()
    imgs = np.stack([cv2.resize(cv2.imread(str(gt / "frames" / f), cv2.IMREAD_GRAYSCALE), (size, size),
                                interpolation=cv2.INTER_AREA) for f in idx.file[ok]])
    labs = np.stack([cv2.resize(np.where(m == 0, BG_OR_TAIL, OBJ2CLS[m]), (size, size), interpolation=cv2.INTER_NEAREST)
                     for m in z["labels"][ok]])
    return imgs, labs, idx[ok].reset_index(drop=True)


def kp_table(idx, size, min_score=None):
    """index.csv rows -> [N, K, 3] float32: x, y in network px and a flag: 1 = there, 0 = known
    absent, -1 = unknown. Human labels (min_score None): NaN = not visible = absent. Teacher
    labels: only nodes scored >= min_score are used, the rest are unknown."""
    kp = np.full((len(idx), len(KP_NAMES), 3), -1.0, np.float32)
    sc = size / idx.crop.to_numpy(float)[:, None]
    for k, n in enumerate(KP_NAMES):
        x, y = idx[f"{n}_x"].to_numpy(float), idx[f"{n}_y"].to_numpy(float)
        vis = ~np.isnan(x)
        kp[:, k, 0], kp[:, k, 1] = (x + 0.5) * sc[:, 0] - 0.5, (y + 0.5) * sc[:, 0] - 0.5
        if min_score is None:
            kp[:, k, 2] = np.where(vis, 1, 0)
        else:
            kp[:, k, 2] = np.where(vis & (idx[f"{n}_score"].to_numpy(float) >= min_score), 1, -1)
    kp[..., :2] = np.nan_to_num(kp[..., :2])
    return kp


def load_distill_frames(d, size, min_score):
    """distill_frames.py output -> images, keypoint table, index (no segmentation targets).
    The images are resized once into d/frames_<size>.npy and then memory-mapped, so they stay out
    of RAM (17k frames are 4.5 GB; this machine has no swap, and running out of memory kills
    whatever the kernel picks, e.g. the desktop app)."""
    idx = pd.read_csv(d / "index.csv")
    cache = d / f"frames_{size}.npy"
    if not cache.exists() or cache.stat().st_mtime < (d / "index.csv").stat().st_mtime:
        tmp = d / f"frames_{size}.tmp.npy"
        mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(len(idx), size, size))
        for i, f in enumerate(idx.file):
            mm[i] = cv2.resize(cv2.imread(str(d / "frames" / f), cv2.IMREAD_GRAYSCALE), (size, size),
                               interpolation=cv2.INTER_AREA)
        mm.flush()
        del mm
        tmp.rename(cache)
    return np.load(cache, mmap_mode="r"), kp_table(idx, size, min_score), idx


class Stacked:
    """Several image arrays (in RAM or memory-mapped) indexed as one, without copying them."""

    def __init__(self, parts):
        self.parts = [p for p in parts if len(p)]
        self.ends = np.cumsum([len(p) for p in self.parts])

    def __len__(self):
        return int(self.ends[-1])

    def __getitem__(self, i):
        k = int(np.searchsorted(self.ends, i, side="right"))
        return self.parts[k][i - (self.ends[k - 1] if k else 0)]


class Frames(torch.utils.data.Dataset):
    """imgs [N,S,S]; labs [N,S,S] or None per frame (no segmentation target: all IGNORE);
    kps [N,K,3] (kp_table)."""

    def __init__(self, imgs, labs, kps, augment, flip=True):
        self.imgs, self.labs, self.kps, self.augment, self.flip = imgs, labs, kps, augment, flip

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img, lab, kp = np.asarray(self.imgs[i]), self.labs[i], self.kps[i].copy()
        if lab is None:
            lab = np.full(img.shape, IGNORE, np.uint8)
        if self.augment:
            img, lab, kp = self._augment(img, lab, kp, np.random.default_rng())
        return (torch.from_numpy(np.ascontiguousarray(img)), torch.from_numpy(np.ascontiguousarray(lab)),
                torch.from_numpy(kp))

    def _augment(self, img, lab, kp, rng):
        S = img.shape[0]
        M = cv2.getRotationMatrix2D((S / 2, S / 2), rng.uniform(0, 360), rng.uniform(0.85, 1.15))
        M[:, 2] += rng.uniform(-0.08, 0.08, 2) * S
        img = cv2.warpAffine(img, M, (S, S), flags=cv2.INTER_LINEAR, borderValue=0)
        lab = cv2.warpAffine(lab, M, (S, S), flags=cv2.INTER_NEAREST, borderValue=IGNORE)
        kp[:, :2] = kp[:, :2] @ M[:, :2].T + M[:, 2]
        if self.flip and rng.random() < 0.5:
            img, lab = img[:, ::-1], MIRROR[lab[:, ::-1]]
            kp[:, 0] = S - 1 - kp[:, 0]
            kp = kp[KP_MIRROR]
        out = (kp[:, 2] == 1) & ((kp[:, :2] < 0).any(1) | (kp[:, :2] > S - 1).any(1))
        kp[out, 2] = -1  # rotated out of the frame: unknown
        x = img.astype(np.float32) / 255
        x = np.power(np.clip(x * rng.uniform(0.6, 1.6), 0, 1), rng.uniform(0.7, 1.4))
        if rng.random() < 0.2:
            x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.5, 1.5))
        x = x + rng.normal(0, rng.uniform(0, 6) / 255, x.shape).astype(np.float32)
        return (np.clip(x, 0, 1) * 255).astype(np.uint8), lab, kp


def split_clips(clips, val_arg, vdir):
    names = [c.name for c in clips]
    if val_arg == ["none"]:
        return names, []
    if val_arg:
        missing = set(val_arg) - set(names)
        if missing:
            raise SystemExit(f"--val-clips not found among labelled clips: {sorted(missing)}")
        return [n for n in names if n not in val_arg], list(val_arg)
    # auto: hold out ~25% of clips, one per behaviour, preferring the 2nd clip of
    # each behaviour so the 1st stays in training
    n_val = max(1, round(0.25 * len(names))) if len(names) > 1 else 0
    by_beh = {}
    for n in names:
        by_beh.setdefault(read_json(vdir / "clips" / n / "meta.json").get("target_behavior"), []).append(n)
    val = []
    for group in sorted(by_beh.values(), key=len, reverse=True):
        if len(val) < n_val:
            val.append(group[-1])
    return [n for n in names if n not in val], val


# ---- loss / metrics -----------------------------------------------------------------
def seg_loss(logits, target):
    """Cross-entropy + Dice. A partial target (seg_model.PARTIAL, e.g. NOTPAW
    = background, body or tail) is known to be one of a set of classes:
    cross-entropy scores it on the summed probability of that set, and Dice
    leaves it out of those classes while still counting it as a negative for
    every other class."""
    logp = logits.float().log_softmax(1)
    valid, full = target != IGNORE, target < N_CLASSES
    ce = F.nll_loss(logp, torch.where(full, target, IGNORE).long(), ignore_index=IGNORE, reduction="sum")
    known = valid.unsqueeze(1).repeat(1, N_CLASSES, 1, 1)
    for code, allowed in PARTIAL.items():
        m = target == code
        ce = ce - torch.logsumexp(logp[:, allowed], 1)[m].sum()
        known[:, allowed] &= ~m.unsqueeze(1)
    ce = ce / valid.sum().clamp(min=1)
    p = logp.exp() * known
    t = F.one_hot(torch.where(full, target, 0).long(), N_CLASSES).permute(0, 3, 1, 2) * full.unsqueeze(1)
    inter = (p * t).sum((0, 2, 3))
    denom = p.sum((0, 2, 3)) + t.sum((0, 2, 3))
    dice = (1 - (2 * inter + 1) / (denom + 1))[1:].mean()  # foreground classes; background dominates otherwise
    return ce + dice


def kp_loss(logits, kp, sigma):
    """CenterNet focal loss. logits [B,K,H,W]; kp [B,K,3] (x, y, flag). A present node's target
    is a Gaussian (sigma, px) with 1 at its nearest pixel; an absent node's target is all 0; unknown
    nodes are left out. Normalised by the number of present nodes."""
    B, K, H, W = logits.shape
    flag = kp[..., 2]
    known = flag >= 0
    if not known.any():
        return logits.sum() * 0
    lg, pts, fl = logits[known].float(), kp[known][:, :2], flag[known]  # [M,H,W], [M,2], [M]
    ys = torch.arange(H, device=lg.device, dtype=torch.float32)[None, :, None]
    xs = torch.arange(W, device=lg.device, dtype=torch.float32)[None, None, :]
    cx, cy = pts[:, 0, None, None].round(), pts[:, 1, None, None].round()
    gt = torch.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma**2)) * (fl == 1)[:, None, None]
    pos = gt > 0.999
    logp, log1mp = F.logsigmoid(lg), F.logsigmoid(-lg)
    p = lg.sigmoid()
    pos_loss = (logp * (1 - p) ** 2)[pos].sum()
    neg_loss = (log1mp * p**2 * (1 - gt) ** 4)[~pos].sum()
    return -(pos_loss + neg_loss) / pos.sum().clamp(min=1)


@torch.no_grad()
def evaluate_kp_points(model, kp_val, device, size, batch=16):
    """Predicted vs hand-placed keypoints on the held-out SLEAP frames: per node the median error
    and PCK@10 / @20 (crop px = full-res px) over visible nodes, and how often an invisible node
    still gets a confident (> 0.5) prediction."""
    imgs, idx = kp_val
    model.eval()
    xy, sc = [], []
    for s in range(0, len(imgs), batch):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, k = model(to_input(torch.from_numpy(imgs[s : s + batch]).to(device)), keypoints=True)
        p, c = decode_keypoints(k)
        xy.append(p.cpu().numpy())
        sc.append(c.cpu().numpy())
    model.train()
    xy, sc = np.concatenate(xy), np.concatenate(sc)
    crop = idx.crop.to_numpy(float)[:, None]
    xy = (xy + 0.5) * (crop[..., None] / size) - 0.5
    out = {}
    for k, n in enumerate(KP_NAMES):
        g = idx[[f"{n}_x", f"{n}_y"]].to_numpy(float)
        vis = ~np.isnan(g[:, 0])
        e = np.linalg.norm(xy[vis, k] - g[vis], axis=1)
        e[sc[vis, k] < 0.1] = np.inf  # visible but not found
        out[n] = dict(median=float(np.median(e)), pck10=float(np.mean(e <= 10)), pck20=float(np.mean(e <= 20)),
                      false_present=float(np.mean(sc[~vis, k] > 0.5)) if (~vis).any() else float("nan"))
    out["mean_pck10"] = float(np.mean([v["pck10"] for v in out.values()]))
    return out


def confusion(pred, target):
    v = target < N_CLASSES  # partial targets are not scored
    idx = target[v].long() * N_CLASSES + pred[v].long()
    return torch.bincount(idx, minlength=N_CLASSES**2).view(N_CLASSES, N_CLASSES)


def scores(cm):
    cm = cm.double()
    iou = cm.diag() / (cm.sum(0) + cm.sum(1) - cm.diag()).clamp(min=1)
    swaps = {}
    for i, n in enumerate(CLASS_NAMES):
        if n.endswith("_L"):
            j = CLASS_NAMES.index(n[:-2] + "_R")
            swaps[n[:-2]] = float((cm[i, j] + cm[j, i]) / (cm[i].sum() + cm[j].sum()).clamp(min=1))
    present = cm.sum(1) > 0
    return dict(iou={n: float(iou[i]) if present[i] else float("nan") for i, n in enumerate(CLASS_NAMES)},
                mean_part_iou=float(iou[1:][present[1:]].mean()) if present[1:].any() else float("nan"),
                lr_swap=swaps)


@torch.no_grad()
def evaluate(model, val_sets, device, batch=8):
    model.eval()
    total = torch.zeros(N_CLASSES, N_CLASSES, dtype=torch.long, device=device)
    per_clip = {}
    for name, (imgs, labs) in val_sets.items():
        cm = torch.zeros_like(total)
        for s in range(0, len(imgs), batch):
            x = to_input(torch.from_numpy(imgs[s : s + batch]).to(device))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(x).argmax(1)
            cm += confusion(pred, torch.from_numpy(labs[s : s + batch]).to(device))
        per_clip[name] = scores(cm)
        total += cm
    model.train()
    return scores(total), per_clip, total.cpu().numpy()


@torch.no_grad()
def evaluate_keypoints(model, kp_val, device, batch=16, axis=False):
    """Paw keypoint outcomes (eval_keypoints.py) on the held-out SLEAP frames;
    with axis=True also the tail base / heading scores (score_axis)."""
    from eval_keypoints import paw_rates, score_axis, score_keypoints

    imgs, idx = kp_val
    crop = int(idx.crop.iloc[0])
    model.eval()
    labs = []
    for s in range(0, len(imgs), batch):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(to_input(torch.from_numpy(imgs[s : s + batch]).to(device))).argmax(1)
        labs += [cv2.resize(CLS2OBJ[m], (crop, crop), interpolation=cv2.INTER_NEAREST)
                 for m in pred.cpu().numpy().astype(np.uint8)]
    model.train()
    df = score_keypoints(labs, idx)
    if axis:
        return paw_rates(df), df, score_axis(labs, idx)
    return paw_rates(df), df


def example_sheet(model, val_sets, device, path, n=6, tile=256):
    lut = palette_lut()  # tail included
    cls_to_obj = np.array(CLASS_OBJ_IDS + [0] * (256 - N_CLASSES), np.uint8)
    rows = []
    items = [(nm, i) for nm, (imgs, _) in val_sets.items() for i in np.linspace(0, len(imgs) - 1, 3).astype(int)]
    model.eval()
    with torch.no_grad():
        for nm, i in items[:n]:
            img, lab = val_sets[nm][0][i], val_sets[nm][1][i]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(to_input(torch.from_numpy(img[None]).to(device))).argmax(1)[0].cpu().numpy()
            g = cv2.cvtColor(cv2.convertScaleAbs(img, alpha=2), cv2.COLOR_GRAY2BGR)
            sam = lut[cls_to_obj[np.where(lab >= N_CLASSES, 0, lab)]]
            net = lut[cls_to_obj[pred.astype(np.uint8)]]
            row = np.hstack([cv2.resize(a, (tile, tile), interpolation=cv2.INTER_NEAREST) for a in (g, sam, net)])
            cv2.putText(row, nm[:28], (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            rows.append(row)
    model.train()
    if rows:
        head = np.zeros((24, rows[0].shape[1], 3), np.uint8)
        for k, t in enumerate(("frame", "SAM masks (target)", "network")):
            cv2.putText(head, t, (k * tile + 6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imwrite(str(path), np.vstack([head] + rows))


# ---- main -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--video-stem")
    ap.add_argument("--val-clips", nargs="*", help="clip ids to hold out, or 'none'")
    ap.add_argument("--encoder", default="resnet34",
                    help="any timm features_only model, e.g. swin_tiny_patch4_window7_224, swin_small_patch4_window7_224")
    ap.add_argument("--size", type=int, default=512, help="train/infer resolution (crops are 768)")
    ap.add_argument("--iters", type=int, default=24000,
                    help="6000 is clearly undertrained (held-out right-paw rate 92%% vs 95%% at 24000)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--enc-lr-mult", type=float, default=1.0,
                    help="encoder learning rate relative to --lr (e.g. 0.3 for convnext/transformer encoders)")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--no-flip", action="store_true", help="disable mirrored (L/R-swapped) samples")
    ap.add_argument("--use-flagged", action="store_true", help="also train on frames flagged wrong in review")
    ap.add_argument("--keep-swap-frames", action="store_true",
                    help="keep frames evaluate.py marks as L/R swap candidates (dropped by default)")
    ap.add_argument("--exclude-clips", nargs="*", default=[], help="clip ids to leave out entirely")
    ap.add_argument("--kp-frames", help="hand-labelled SLEAP frames (default <out-root>/sleap_gt if prepared)")
    ap.add_argument("--no-kp", action="store_true", help="train on the clips only")
    ap.add_argument("--kp-extra", nargs="*", default=[],
                    help="more keypoint-frame folders (e.g. <out-root>/sleap_pred from SLEAP predictions); "
                         "training only, and never from a held-out video")
    ap.add_argument("--kp-val-videos", nargs="*", default=["2026-03-12", "2026-06-01"],
                    help="SLEAP videos held out entirely (substring match), or 'none'")
    ap.add_argument("--kp-share", type=float, default=0.5, help="share of training samples from hand-labelled SLEAP frames")
    ap.add_argument("--kp-extra-share", type=float, default=0.25,
                    help="share of training samples from --kp-extra frames (taken from the clips' share)")
    ap.add_argument("--kp-distill", nargs="*", default=[], help="distill_frames.py folders: teacher-labelled keypoint frames")
    ap.add_argument("--kp-distill-share", type=float, default=0.4,
                    help="share of training samples from --kp-distill frames (taken from the clips' share)")
    ap.add_argument("--kp-min-score", type=float, default=0.5, help="teacher nodes below this score are unknown")
    ap.add_argument("--kp-sigma", type=float, default=2.5, help="keypoint target Gaussian, network px")
    ap.add_argument("--kp-weight", type=float, default=1.0)
    ap.add_argument("--init", help="start from this part-network checkpoint (segmentation weights)")
    ap.add_argument("--keypoints", action="store_true",
                    help="learn keypoints from the hand-labelled frames even without --kp-distill")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    root = Path(args.out_root)
    stem = args.video_stem or sorted(p.name for p in root.iterdir() if (p / "clips").is_dir())[0]
    vdir = root / stem
    clips = sorted(c for c in (vdir / "clips").iterdir()
                   if (c / "masks.npz").exists() and c.name not in args.exclude_clips)
    if not clips:
        raise SystemExit("no clips with masks.npz yet")
    if args.exclude_clips:
        print(f"excluded: {', '.join(args.exclude_clips)}")
    swaps = None if args.keep_swap_frames else swap_frames(vdir)
    if not args.keep_swap_frames:
        csv = vdir / "eval" / "swaps.csv"
        newest_mask = max((c / "masks.npz").stat().st_mtime for c in clips)
        if swaps is None:
            print("!! no usable eval/swaps.csv: run evaluate.py first to drop L/R swap frames")
            swaps = {}
        elif csv.stat().st_mtime < newest_mask:
            raise SystemExit("eval/swaps.csv is older than some masks: re-run evaluate.py first "
                             "(or pass --keep-swap-frames)")
    train_names, val_names = split_clips(clips, args.val_clips, vdir)
    if not train_names:
        raise SystemExit("no training clips left after the split")
    print(f"train clips ({len(train_names)}): {', '.join(train_names)}")
    print(f"val clips   ({len(val_names)}): {', '.join(val_names) or '-'}")
    if len(clips) < 6:
        print(f"!! only {len(clips)} labelled clips: expect a model that memorises them and val "
              "scores that say little about new footage.")

    # Dropped frames are left out of validation too: a target with the wrong
    # L/R would mark a correct prediction as an error.
    def load(names):
        out, n_flag, n_swap = {}, 0, 0
        for n in names:
            cdir = vdir / "clips" / n
            flagged = set() if args.use_flagged else review_flagged(cdir)
            swapped = swaps.get(n, set()) if swaps else set()
            imgs, labs, _ = load_clip(cdir, args.size, flagged | swapped)
            out[n] = (imgs, labs)
            n_flag += len(flagged)
            n_swap += len(swapped - flagged)
        return out, n_flag, n_swap

    t0 = time.time()
    train_sets, tf, ts = load(train_names)
    val_sets, vf, vs = load(val_names)
    imgs = np.concatenate([v[0] for v in train_sets.values()])
    labs = np.concatenate([v[1] for v in train_sets.values()])
    print(f"loaded {len(imgs)} train frames, {sum(len(v[0]) for v in val_sets.values())} val frames "
          f"in {time.time() - t0:.0f}s; left out: {tf + vf} review-flagged, {ts + vs} L/R-swap frames")

    # hand-labelled SLEAP frames: paws only, several sessions
    n_kp = len(KP_NAMES) if (args.kp_distill or args.keypoints) else 0
    kps = np.full((len(imgs), len(KP_NAMES), 3), -1.0, np.float32)  # clips: keypoints unknown
    weights, kp_val, kp_train_videos, kp_val_videos = np.ones(len(imgs)), None, [], []
    gt = Path(args.kp_frames) if args.kp_frames else root / "sleap_gt"
    if not args.no_kp and (gt / "kp_masks.npz").exists():
        kimgs, klabs, kidx = load_kp_frames(gt, args.size)
        kkps = kp_table(kidx, args.size)
        held = kidx.video.map(lambda v: any(s in v for s in args.kp_val_videos)).to_numpy()
        n_hand = len(kidx)
        for extra in args.kp_extra:
            ei, el, ex = load_kp_frames(Path(extra), args.size)
            use = ~ex.video.map(lambda v: any(s in v for s in args.kp_val_videos)).to_numpy()
            print(f"extra keypoint frames from {extra}: {int(use.sum())}")
            kimgs, klabs = np.concatenate([kimgs, ei[use]]), np.concatenate([klabs, el[use]])
            kkps = np.concatenate([kkps, np.full((int(use.sum()), len(KP_NAMES), 3), -1.0, np.float32)])  # old model's points: unused
            kidx = pd.concat([kidx, ex[use]], ignore_index=True)
            held = np.r_[held, np.zeros(int(use.sum()), bool)]
        kp_train_videos, kp_val_videos = sorted(kidx.video[~held].unique()), sorted(kidx.video[held].unique())
        if held.any():
            kp_val = (kimgs[held], kidx[held].reset_index(drop=True))
        n_clip, n_hand_tr, n_extra = len(imgs), int((~held[:n_hand]).sum()), len(kidx) - n_hand
        imgs, labs = np.concatenate([imgs, kimgs[~held]]), np.concatenate([labs, klabs[~held]])
        kps = np.concatenate([kps, kkps[~held]])
        extra_share = args.kp_extra_share if n_extra else 0.0
        weights = np.r_[np.full(n_clip, (1 - args.kp_share - extra_share) / n_clip),
                        np.full(n_hand_tr, args.kp_share / n_hand_tr), np.full(n_extra, extra_share / max(n_extra, 1))]
        print(f"sample shares: clips {1 - args.kp_share - extra_share:.2f}, hand-labelled {args.kp_share:.2f}, "
              f"extra {extra_share:.2f}")
        print(f"SLEAP frames: {n_hand_tr} hand-labelled + {n_extra} extra train from {len(kp_train_videos)} videos, "
              f"{int(held.sum())} held out from {len(kp_val_videos)} videos ({', '.join(kp_val_videos) or '-'})")
    elif not args.no_kp:
        print(f"!! no {gt}/kp_masks.npz: training on the clips only")
    labs = list(labs)  # per frame: target array, or None (no segmentation target)
    if args.kp_distill:
        di, dk, n_d = [], [], 0
        for d in args.kp_distill:
            a, k, ix = load_distill_frames(Path(d), args.size, args.kp_min_score)
            if any(any(s in v for s in args.kp_val_videos) for v in ix.video.unique()):
                raise SystemExit(f"{d} comes from a held-out video")
            print(f"teacher-labelled frames from {d}: {len(a)} (nodes used: "
                  + ", ".join(f"{n} {np.mean(k[:, j, 2] == 1):.0%}" for j, n in enumerate(KP_NAMES)) + ")")
            di.append(a), dk.append(k)
        n_d = sum(len(a) for a in di)
        imgs, kps, labs = Stacked([imgs, *di]), np.concatenate([kps, *dk]), labs + [None] * n_d
        weights = np.r_[weights / weights.sum() * (1 - args.kp_distill_share), np.full(n_d, args.kp_distill_share / n_d)]
        print(f"teacher-labelled share {args.kp_distill_share:.2f} (the others scaled down to {1 - args.kp_distill_share:.2f})")

    run = vdir / "seg_runs" / f"{time.strftime('%Y%m%d_%H%M%S')}_{args.encoder}"
    run.mkdir(parents=True)
    cfg = dict(vars(args), train_clips=train_names, val_clips=val_names, video_stem=stem,
               kp_train_videos=kp_train_videos, kp_val_video_names=kp_val_videos)
    write_json(run / "config.json", cfg)

    device = "cuda"
    model = PartSegNet(args.encoder, img_size=args.size, n_kp=n_kp).to(device)
    if args.init:
        src = torch.load(args.init, map_location="cpu", weights_only=False)["state_dict"]
        own = model.state_dict()
        keep = {k: v for k, v in src.items() if k in own and own[k].shape == v.shape}
        model.load_state_dict(keep, strict=False)
        print(f"initialised {len(keep)}/{len(own)} tensors from {args.init}")
    enc = list(model.encoder.parameters())
    rest = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([dict(params=enc, lr=args.lr * args.enc_lr_mult), dict(params=rest, lr=args.lr)],
                            weight_decay=1e-4)
    warm = min(300, args.iters // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1, (i + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(i, args.iters) / args.iters)))
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=args.iters * args.batch, replacement=True)
    loader = torch.utils.data.DataLoader(Frames(imgs, labs, kps, True, not args.no_flip), batch_size=args.batch,
                                         sampler=sampler, num_workers=args.workers, pin_memory=True,
                                         drop_last=True, persistent_workers=args.workers > 0)

    log = open(run / "log.csv", "w")
    log.write("iter,loss,val_mean_part_iou," + ",".join(f"iou_{n}" for n in CLASS_NAMES[1:])
              + ",kp_correct,kp_wrong_side,kp_wrong_limb,kp_missed,kp_all_paws_correct"
              + "".join(f",pt_{n}_median,pt_{n}_pck10" for n in KP_NAMES[:n_kp]) + "\n")
    best, best_metrics, t0, run_loss = -1.0, None, time.time(), []
    for it, (x, y, kp) in enumerate(loader, 1):
        x, y = to_input(x.to(device, non_blocking=True)), y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x, keypoints=bool(n_kp))
        if n_kp:
            loss = seg_loss(out[0], y) + args.kp_weight * kp_loss(out[1], kp.to(device), args.kp_sigma)
        else:
            loss = seg_loss(out, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        run_loss.append(loss.item())
        if it % args.eval_every == 0 or it == args.iters:
            msg = f"it {it}/{args.iters}  loss {np.mean(run_loss):.3f}  {time.time() - t0:.0f}s"
            row = [it, np.mean(run_loss)]
            score, s, k = 0.0, None, None
            if val_sets:
                s, _, _ = evaluate(model, val_sets, device)
                row += [s["mean_part_iou"]] + [s["iou"][n] for n in CLASS_NAMES[1:]]
                msg += f"  val part IoU {s['mean_part_iou']:.3f}  " + " ".join(
                    f"{n.replace('paw', '')}:{s['iou'][n]:.2f}" for n in CLASS_NAMES[1:])
                score += s["mean_part_iou"]
            else:
                row += [float("nan")] * N_CLASSES
            if kp_val:
                k, _ = evaluate_keypoints(model, kp_val, device)
                row += list(k.values())
                msg += (f"  | keypoints: right paw {k['correct']:.1%}, wrong side {k['wrong_side']:.1%}, "
                        f"all paws {k['all_paws_correct']:.1%}")
                score += k["correct"]
                if n_kp:
                    pt = evaluate_kp_points(model, kp_val, device, args.size)
                    row += [v for n in KP_NAMES for v in (pt[n]["median"], pt[n]["pck10"])]
                    msg += f"  | points PCK@10 {pt['mean_pck10']:.1%} (snout {pt['snout']['pck10']:.1%}, median {pt['snout']['median']:.1f} px)"
                    score += pt["mean_pck10"]
                    k = dict(k, points=pt)
            if (val_sets or kp_val) and score > best:
                best, best_metrics = score, dict(clips=s, keypoints=k)
                save_checkpoint(run / "best.pt", model, cfg, best_metrics)
            print(msg, flush=True)
            log.write(",".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row) + "\n")
            log.flush()
            run_loss = []
    save_checkpoint(run / "last.pt", model, cfg)
    if not (val_sets or kp_val):
        save_checkpoint(run / "best.pt", model, cfg)

    # final report on the best checkpoint
    lines = [f"train: {train_names}", f"val:   {val_names}",
             f"SLEAP frames train: {kp_train_videos}", f"SLEAP frames held out: {kp_val_videos}", ""]
    state = torch.load(run / "best.pt", map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(state)
    if kp_val:
        from eval_keypoints import axis_lines

        k, df, ax = evaluate_keypoints(model, kp_val, device, axis=True)
        df.to_csv(run / "kp_val_per_keypoint.csv", index=False)
        lines += ["hand-placed paw keypoints on HELD-OUT videos (best ckpt):",
                  "  " + "  ".join(f"{c} {v:.1%}" for c, v in k.items()), "", *axis_lines(ax), ""]
        if n_kp:
            pt = evaluate_kp_points(model, kp_val, device, args.size)
            lines += ["predicted keypoints vs hand-placed, HELD-OUT videos (crop px = full-res px):"]
            lines += [f"  {n:16s} median {v['median']:6.1f}  PCK@10 {v['pck10']:.1%}  PCK@20 {v['pck20']:.1%}  "
                      f"confident when not visible {v['false_present']:.1%}" for n, v in pt.items() if n != "mean_pck10"]
            lines += [""]
    if val_sets:
        s, per_clip, cm = evaluate(model, val_sets, device)
        lines += ["NOTE: targets are SAM masks, so these measure agreement with SAM on unseen clips.", "",
                  f"mean part IoU (best ckpt): {s['mean_part_iou']:.3f}",
                  "per class IoU: " + "  ".join(f"{n} {v:.3f}" for n, v in s["iou"].items()),
                  "L/R swap rate (share of paw pixels given the wrong side): "
                  + "  ".join(f"{k} {v:.3%}" for k, v in s["lr_swap"].items()), "", "per clip:"]
        for n, cs in per_clip.items():
            lines.append(f"  {n}: part IoU {cs['mean_part_iou']:.3f}  "
                         + " ".join(f"{k}:{v:.2f}" for k, v in cs["iou"].items() if k != "background"))
        w = max(len(n) for n in CLASS_NAMES)
        lines += ["", "confusion (rows = SAM, cols = network, pixels):",
                  " " * (w + 1) + " ".join(f"{n[:9]:>9}" for n in CLASS_NAMES)]
        lines += [f"{n:<{w}} " + " ".join(f"{v:>9d}" for v in cm[i]) for i, n in enumerate(CLASS_NAMES)]
        example_sheet(model, val_sets, device, run / "val_examples.jpg")
    # speed: frames/s at the training resolution, batch 1 and 16
    model.eval()
    sp = {}
    for b in (1, 16):
        x = torch.zeros(b, 3, args.size, args.size, device=device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(5):
                model(x)
            torch.cuda.synchronize()
            t = time.time()
            for _ in range(20):
                model(x)
            torch.cuda.synchronize()
        sp[b] = 20 * b / (time.time() - t)
    lines += ["", f"inference speed @ {args.size}px: {sp[1]:.0f} fps (batch 1), {sp[16]:.0f} fps (batch 16)"]
    (run / "report.txt").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nrun saved to {run}")


if __name__ == "__main__":
    main()
