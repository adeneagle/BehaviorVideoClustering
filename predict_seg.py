"""Run a trained part-segmentation network on clips, in SAM's mask format.

    python predict_seg.py RUN_DIR [--clip ID ...] [--no-videos]

Writes masks_<tag>.npz, overlay_<tag>.mp4, parts_<tag>.mp4 and
contact_sheet_<tag>.jpg next to SAM's outputs (tag = run dir name), then
    python evaluate.py --masks masks_<tag>.npz
scores the network with the same stability / L/R checks as SAM. Any clip
with frames works, labelled or not.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from common import PARTS, frame_paths, read_json, save_masks
from render import contact_sheet, write_videos
from seg_model import load_checkpoint, to_input


@torch.no_grad()
def predict_clip(model, cdir, size, batch=16, device="cuda"):
    paths = frame_paths(cdir)
    H, W = cv2.imread(str(paths[0]), cv2.IMREAD_GRAYSCALE).shape
    cls_to_obj = torch.tensor([0] + [p.obj_id for p in PARTS if p.render], dtype=torch.uint8, device=device)
    obj_ids = [p.obj_id for p in PARTS]
    labels = np.zeros((len(paths), H, W), np.uint8)
    scores = np.full((len(paths), len(obj_ids)), np.nan, np.float32)
    for s in range(0, len(paths), batch):
        g = np.stack([cv2.resize(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE), (size, size),
                                 interpolation=cv2.INTER_AREA) for p in paths[s : s + batch]])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(to_input(torch.from_numpy(g).to(device)))
        prob = F.interpolate(logits.float(), size=(H, W), mode="bilinear", align_corners=False).softmax(1)
        conf, cls = prob.max(1)
        lab = cls_to_obj[cls]
        labels[s : s + len(g)] = lab.cpu().numpy()
        for c in range(1, len(cls_to_obj)):
            k = obj_ids.index(int(cls_to_obj[c]))
            m = cls == c
            n = m.flatten(1).sum(1)
            mean_conf = (conf * m).flatten(1).sum(1) / n.clamp(min=1)
            scores[s : s + len(g), k] = torch.where(n > 0, mean_conf, torch.nan).cpu().numpy()
    return labels, scores, obj_ids


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="seg_runs/<run> directory (uses best.pt)")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--clip", nargs="*", help="clip ids (default: all clips)")
    ap.add_argument("--no-videos", action="store_true")
    args = ap.parse_args()

    run = Path(args.run)
    model, ck = load_checkpoint(run / args.ckpt)
    size, tag = ck["cfg"]["size"], run.name
    vdir = run.parent.parent
    clips = sorted(c for c in (vdir / "clips").iterdir() if (c / "meta.json").exists())
    if args.clip:
        clips = [c for c in clips if c.name in args.clip]
    held_out = set(ck["cfg"].get("val_clips", []))
    for c in clips:
        t0 = time.time()
        labels, scores, obj_ids = predict_clip(model, c, size)
        dt = time.time() - t0
        K = len(obj_ids)
        save_masks(c, labels, scores, np.zeros((len(labels), K, K), np.int32), obj_ids,
                   name=f"masks_{tag}.npz")
        if not args.no_videos:
            write_videos(c, labels, read_json(c / "meta.json")["fps"], suffix=f"_{tag}")
        contact_sheet(c, labels, path=c / f"contact_sheet_{tag}.jpg")
        role = "val" if c.name in held_out else "train" if c.name in ck["cfg"]["train_clips"] else "unlabelled"
        print(f"{c.name} [{role}]: {len(labels)} frames in {dt:.1f}s ({len(labels) / dt:.0f} fps incl. I/O)")
    print(f"\nscore with:  python evaluate.py --masks masks_{tag}.npz")


if __name__ == "__main__":
    main()
