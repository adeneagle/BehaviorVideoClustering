"""Compare a keypoint part network (the student) with the SLEAP teacher on a held-out session.

    python eval_distill.py CKPT RUN_DIR [--slp A.slp B.slp ...] [--tag NAME]

RUN_DIR is that session's run_video.py output; --slp the teacher's full-pipeline predictions on
some of its frames (default: every .slp in <session>/sleap/ except hand-click projects; not the teacher/ training parts).
The student sees exactly run_video.py's crop of each frame. Per node: distance student - teacher
(full-res px) where the teacher is confident (score >= 0.5), and how often the two disagree on
presence (one confident, the other not). Reported for all frames and per .slp file, so bouts
picked for a behaviour (grooming) can be read on their own.

Output: <CKPT dir>/distill_eval[_<tag>]/ per_frame.csv, summary.txt, examples.jpg
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from seg_model import KP_NAMES, decode_keypoints, load_checkpoint, to_input
from sleap_kp import read_slp

CROP = 768


def read_frames(video, frames):
    """Gray frames by index: seek to the start of every run of consecutive frames, then read on."""
    cap = cv2.VideoCapture(str(video))
    out, f_next = {}, None
    for f in sorted(frames):
        if f != f_next:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            raise SystemExit(f"could not read frame {f}")
        out[f] = img[..., 0] if img.ndim == 3 else img
        f_next = f + 1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt")
    ap.add_argument("run_dir")
    ap.add_argument("--slp", nargs="*")
    ap.add_argument("--min-score", type=float, default=0.5)
    ap.add_argument("--tag")
    args = ap.parse_args()

    run = Path(args.run_dir)
    # default: the teacher's prediction files, not hand-click projects (sleap_click_project.py)
    slps = ([Path(p) for p in args.slp] if args.slp else
            sorted(p for p in (run.parent.parent / "sleap").glob("*.slp") if "clicks" not in p.stem or p.stem.endswith("_teacher")))
    model, ck = load_checkpoint(args.ckpt)
    assert len(ck.get("kp_names", [])) == len(KP_NAMES), "checkpoint has no keypoint head"
    size = ck["cfg"]["size"]
    fr = pd.read_csv(run / "frames.csv", usecols=["x0", "y0"])
    video = next(l.split(": ", 1)[1].strip() for l in open(run / "summary.txt") if l.startswith("video: "))

    rows, sheet = [], []
    for slp in slps:
        t = read_slp(slp, predicted=True)
        imgs = read_frames(video, t["frame"].tolist())
        for a in range(0, len(t["frame"]), 32):
            fs = t["frame"][a : a + 32]
            crops = np.stack([cv2.resize(imgs[f][fr.y0[f] : fr.y0[f] + CROP, fr.x0[f] : fr.x0[f] + CROP], (size, size),
                                         interpolation=cv2.INTER_AREA) for f in fs])
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                _, k = model(to_input(torch.from_numpy(crops).cuda()), keypoints=True)
            xy, sc = decode_keypoints(k)
            xy = (xy.cpu().numpy() + 0.5) * (CROP / size) - 0.5
            sc = sc.cpu().numpy()
            for j, f in enumerate(fs):
                i = a + j
                xy_f = xy[j] + [fr.x0[f], fr.y0[f]]
                r = dict(slp=slp.stem, frame=int(f))
                for n_i, n in enumerate(KP_NAMES):
                    tx, ty = t["xy"][i, n_i]
                    ts = t["score"][i, n_i] if not np.isnan(tx) else 0.0
                    r[f"{n}_teacher_score"], r[f"{n}_student_score"] = ts, float(sc[j, n_i])
                    r[f"{n}_err"] = float(np.hypot(xy_f[n_i, 0] - tx, xy_f[n_i, 1] - ty)) if not np.isnan(tx) else np.nan
                rows.append(r)
                if j == 0 and a % 128 == 0 and len(sheet) < 24:
                    g = cv2.cvtColor(cv2.convertScaleAbs(cv2.resize(crops[j], (384, 384)), alpha=2), cv2.COLOR_GRAY2BGR)
                    s = 384 / CROP
                    for n_i in range(len(KP_NAMES)):
                        tx, ty = t["xy"][i, n_i]
                        if not np.isnan(tx):
                            cv2.circle(g, (int((tx - fr.x0[f]) * s), int((ty - fr.y0[f]) * s)), 7, (0, 255, 0), 2)
                        if sc[j, n_i] >= 0.5:
                            cv2.drawMarker(g, (int((xy_f[n_i, 0] - fr.x0[f]) * s), int((xy_f[n_i, 1] - fr.y0[f]) * s)),
                                           (0, 0, 255), cv2.MARKER_TILTED_CROSS, 10, 2)
                    cv2.putText(g, f"{slp.stem} {f}", (4, 16), 0, 0.45, (0, 255, 255), 1)
                    sheet.append(g)
    df = pd.DataFrame(rows)

    def summarise(d):
        out = {}
        for n in KP_NAMES:
            conf_t = d[f"{n}_teacher_score"] >= args.min_score
            conf_s = d[f"{n}_student_score"] >= args.min_score
            e = d.loc[conf_t, f"{n}_err"]
            out[n] = dict(n=int(conf_t.sum()), median=e.median(), within10=(e <= 10).mean(), within20=(e <= 20).mean(),
                          teacher_only=float((conf_t & ~conf_s).mean()), student_only=float((conf_s & ~conf_t).mean()))
        return pd.DataFrame(out).T

    out = Path(args.ckpt).parent / ("distill_eval" + (f"_{args.tag}" if args.tag else ""))
    out.mkdir(exist_ok=True)
    df.to_csv(out / "per_frame.csv", index=False)
    lines = [f"student {args.ckpt}", f"vs teacher on {run.parent.parent.name}: {len(df)} frames from "
             + ", ".join(s.stem for s in slps),
             f"distances in full-res px where the teacher scores >= {args.min_score}; teacher_only / student_only = "
             "share of frames where only one of the two is confident", "", "ALL", summarise(df).to_string(float_format=lambda v: f"{v:.3f}")]
    for s, d in df.groupby("slp"):
        lines += ["", s, summarise(d).to_string(float_format=lambda v: f"{v:.3f}")]
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if sheet:
        rows_img = [np.hstack(sheet[i : i + 6]) for i in range(0, len(sheet) - len(sheet) % 6, 6)]
        cv2.imwrite(str(out / "examples.jpg"), np.vstack(rows_img))


if __name__ == "__main__":
    main()
