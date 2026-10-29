"""Does temporal smoothing improve the paw labels on hand-labelled frames?

    python eval_temporal.py RUN_DIR [--videos SUBSTR ...]

For every hand-labelled SLEAP frame in the run's held-out videos, reads the
neighbouring video frames (same crop), predicts, smooths the softmax over time
(temporal.py) and scores the centre frame against the keypoints, as in
eval_keypoints.py. Window 0 is the single-frame baseline.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from common import DEFAULT_OUT_ROOT
from eval_keypoints import paw_rates, score_keypoints
from seg_model import CLASS_OBJ_IDS, load_checkpoint, to_input
from sleap_kp import SLEAP_DIR, local_video, read_slp
from temporal import temporal_smooth

SETTINGS = [("none", None, 0)] + [(f"uniform ±{k}", None, k) for k in (1, 2, 3, 5)] + \
           [(f"gauss σ={s}", s, int(round(3 * s))) for s in (1, 1.5, 2, 3)]


@torch.no_grad()
def window_probs(model, frames, size, device="cuda"):
    g = np.stack([cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA) for f in frames])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model(to_input(torch.from_numpy(g).to(device))).float().softmax(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--gt", default=str(DEFAULT_OUT_ROOT / "sleap_gt"))
    ap.add_argument("--videos", nargs="*", help="default: the run's held-out videos")
    ap.add_argument("--radius", type=int, default=12)
    ap.add_argument("--slp", default=str(SLEAP_DIR / "subset_labels_new.slp"))
    args = ap.parse_args()

    run, gt = Path(args.run), Path(args.gt)
    model, ck = load_checkpoint(run / args.ckpt)
    size = ck["cfg"]["size"]
    vids = args.videos or ck["cfg"]["kp_val_video_names"]
    idx = pd.read_csv(gt / "index.csv")
    idx = idx[~idx.lr_suspect & idx.video.isin(vids) if not args.videos else
              ~idx.lr_suspect & idx.video.map(lambda v: any(s in v for s in vids))].reset_index(drop=True)
    d = read_slp(args.slp)
    src = {Path(p).stem: local_video(p) for p in d["videos"]}
    R = max(r for _, _, r in SETTINGS)
    labs = {s[0]: [] for s in SETTINGS}
    caps = {}
    for n, r in enumerate(idx.itertuples()):
        cap = caps.setdefault(r.video, cv2.VideoCapture(str(src[r.video])))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        first = max(r.frame - R, 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        frames = []
        for _ in range(first, min(r.frame + R + 1, total)):
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f[r.y0 : r.y0 + r.crop, r.x0 : r.x0 + r.crop, 0])
        c = r.frame - first
        prob = window_probs(model, frames, size)
        for name, sigma, rad in SETTINGS:
            lo, hi = max(c - rad, 0), min(c + rad + 1, len(frames))
            p = temporal_smooth(prob[lo:hi], sigma=sigma, radius=rad)[c - lo] if rad else prob[c]
            p = F.interpolate(p[None], size=(r.crop, r.crop), mode="bilinear", align_corners=False)[0]
            lut = torch.tensor(CLASS_OBJ_IDS[: p.shape[0]], dtype=torch.uint8, device=p.device)
            labs[name].append(lut[p.argmax(0)].cpu().numpy())
        if n % 100 == 0:
            print(f"{n}/{len(idx)}", flush=True)

    rows = []
    for name, _, _ in SETTINGS:
        df = score_keypoints(np.stack(labs[name]), idx, args.radius)
        rows.append(dict(setting=name, **paw_rates(df)))
    out = pd.DataFrame(rows).set_index("setting")
    txt = f"{run.name}: {len(idx)} frames from {sorted(idx.video.unique())}\n" + \
          (out * 100).to_string(float_format=lambda v: f"{v:.1f}")
    (run / "temporal_eval.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
