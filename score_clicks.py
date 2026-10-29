"""Score the SLEAP teacher and a keypoint part network against hand clicks (sleap_click_project.py).

    python score_clicks.py CLICKS.slp TEACHER.slp CKPT RUN_DIR [--groups FRAMES.json] [--nodes snout,tail_base]

CLICKS.slp: the labelled project (user instances); TEACHER.slp: the SLEAP model's predictions on the
same frames; CKPT: the part network; RUN_DIR: the session's run_video.py output (crop offsets).
Per node and frame group (FRAMES.json, e.g. even / disagree / other): for nodes you placed, the
error of each model (full-res px; a model that is not confident counts as a miss), and for nodes
you marked not visible, how often each model is confident anyway. Confident = teacher score >=
--teacher-thr (SLEAP's 0.5), student score >= --student-thr (its heatmap peaks run lower; ~0.2).
Only --nodes are scored (default snout and tail base): nodes left where SLEAP's GUI dropped them
would otherwise count as clicks.

Output: <CLICKS dir>/<CLICKS stem>_scores.csv (per frame and node) and _summary.txt
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from eval_distill import CROP, read_frames
from seg_model import KP_NAMES, decode_keypoints, load_checkpoint, to_input
from sleap_kp import NODES, read_slp


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clicks")
    ap.add_argument("teacher")
    ap.add_argument("ckpt")
    ap.add_argument("run_dir")
    ap.add_argument("--groups")
    ap.add_argument("--nodes", default="snout,tail_base")
    ap.add_argument("--teacher-thr", type=float, default=0.5)
    ap.add_argument("--student-thr", type=float, default=0.2)
    args = ap.parse_args()

    nodes = args.nodes.split(",")
    h = read_slp(args.clicks)
    if not len(h["frame"]):
        raise SystemExit(f"no hand-placed instances in {args.clicks} yet")
    t = read_slp(args.teacher, predicted=True)
    trow = {int(f): i for i, f in enumerate(t["frame"])}
    group = {}
    if args.groups:
        for g, fs in json.load(open(args.groups)).items():
            group.update({int(f): g for f in fs})

    run = Path(args.run_dir)
    fr = pd.read_csv(run / "frames.csv", usecols=["x0", "y0"])
    video = next(l.split(": ", 1)[1].strip() for l in open(run / "summary.txt") if l.startswith("video: "))
    model, ck = load_checkpoint(args.ckpt)
    size = ck["cfg"]["size"]
    frames = h["frame"].tolist()
    imgs = read_frames(video, frames)
    crops = np.stack([cv2.resize(imgs[f][fr.y0[f] : fr.y0[f] + CROP, fr.x0[f] : fr.x0[f] + CROP], (size, size),
                                 interpolation=cv2.INTER_AREA) for f in frames])
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        _, k = model(to_input(torch.from_numpy(crops).cuda()), keypoints=True)
    sxy, ssc = decode_keypoints(k)
    sxy = (sxy.float().cpu().numpy() + 0.5) * (CROP / size) - 0.5 + np.stack([fr.x0[frames], fr.y0[frames]], 1)[:, None]
    ssc = ssc.float().cpu().numpy()

    rows = []
    for i, f in enumerate(frames):
        for n in nodes:
            a, b = NODES.index(n), KP_NAMES.index(n)
            gx, gy = h["xy"][i, a]
            r = dict(frame=f, group=group.get(f, "-"), node=n, visible=not np.isnan(gx))
            j = trow.get(f)
            tx, ty = (t["xy"][j, a] if j is not None else (np.nan, np.nan))
            r["teacher_conf"] = j is not None and not np.isnan(tx) and t["score"][j, a] >= args.teacher_thr
            r["student_conf"] = bool(ssc[i, b] >= args.student_thr)
            r["teacher_err"] = float(np.hypot(tx - gx, ty - gy)) if r["visible"] and r["teacher_conf"] else np.nan
            r["student_err"] = float(np.hypot(sxy[i, b, 0] - gx, sxy[i, b, 1] - gy)) if r["visible"] and r["student_conf"] else np.nan
            rows.append(r)
    df = pd.DataFrame(rows)

    def summ(d):
        v, nv = d[d.visible], d[~d.visible]
        out = {"clicked": len(v), "not_visible": len(nv)}
        for m in ("teacher", "student"):
            e = v[f"{m}_err"].fillna(np.inf)  # not confident on a visible point = miss
            out[f"{m}_median"] = float(np.median(e)) if len(e) else np.nan
            out[f"{m}_within10"] = float((e <= 10).mean()) if len(e) else np.nan
            out[f"{m}_within20"] = float((e <= 20).mean()) if len(e) else np.nan
            out[f"{m}_missed"] = float((~v[f"{m}_conf"]).mean()) if len(v) else np.nan
            out[f"{m}_false"] = float(nv[f"{m}_conf"].mean()) if len(nv) else np.nan
        return pd.Series(out)

    stem = Path(args.clicks).with_suffix("")
    df.to_csv(f"{stem}_scores.csv", index=False)
    tab = df.groupby(["node", "group"]).apply(summ)
    tab = pd.concat([tab, df.groupby("node").apply(summ).assign(group="ALL").set_index("group", append=True)]).sort_index()
    txt = (f"{len(frames)} hand-clicked frames; teacher {Path(args.teacher).name} (conf >= {args.teacher_thr}), "
           f"student {args.ckpt} (conf >= {args.student_thr}); errors in full-res px, a miss counts as inf\n\n"
           + tab.to_string(float_format=lambda x: f"{x:.2f}") + "\n")
    Path(f"{stem}_summary.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
