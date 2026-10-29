"""Cluster-labelled video of a clustered run.

    python label_video.py RUN_DIR [--emb vjepa_window] [--start-min 0] [--end-min END]
                          [--context-s 60] [--out FILE] [--labels FILE.npz --key NAME[,NAME...]]

Every frame of RUN_DIR/preview.mp4 (crop | part overlay, every 2nd frame) gets
a third panel with the part map as the encoder saw it (rotated by its window's
heading if --emb was canonicalised), the cluster of the window centred
nearest to it, and that window's body-frame forward speed and turn rate
(image-clockwise > 0). Below runs an ethogram strip of --context-s seconds
centred on the current frame. Cluster colours match overview.png; grey is
HDBSCAN noise; dark is no window (the mouse was lost, so the window was dropped).

--labels: any partition file with 'start' and one label array per key over embed_vjepa.py windows
(tw_finch.py's labels.npz, score_behaviour.py's labels.npz) instead of clusters/windows.csv. --key
picks the partitions: the first labels the panel, each gets its own ethogram row.

Output (H.264, via ffmpeg): RUN_DIR/<emb>/clusters/labeled[_<start>-<end>min].mp4, or with
--labels next to that file: labeled_<first key>[_<start>-<end>min].mp4
"""

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

from embed_vjepa import LabelChunks
from render import solid

PANEL, ROW, AXIS = 384, 40, 20
NO_WINDOW = np.array([20, 20, 20], np.uint8)


def cluster_colours(labels):
    cmap = matplotlib.colormaps["tab20"]
    rgb = np.array([cmap(k % 20)[:3] if k >= 0 else (0.55, 0.55, 0.55) for k in labels])
    return (rgb[:, ::-1] * 255).astype(np.uint8)  # BGR


def load_windows(run, emb, labels, keys):
    """Windows (centre, heading, fwd_speed, turn_rate) with one label column per key, and the
    output folder. Default: clusters/windows.csv (column 'cluster')."""
    if not labels:
        return pd.read_csv(run / emb / "clusters" / "windows.csv"), ["cluster"], run / emb / "clusters"
    z, f = np.load(labels), np.load(run / emb / "features.npz")
    row = {int(s): i for i, s in enumerate(f["start"])}
    idx = np.array([row[int(s)] for s in z["start"]])
    keys = keys or [k for k in z.files if k != "start"][:1]
    w = pd.DataFrame({k: f[k][idx] for k in ("start", "centre", "heading", "fwd_speed", "turn_rate")})
    for k in keys:
        w[k] = z[k]
    return w, keys, Path(labels).parent


def nearest_window(centres, frames, hop):
    """Index of the window centred nearest to each frame; -1 where none is within half a hop."""
    k = np.clip(np.searchsorted(centres, frames), 1, len(centres) - 1)
    k -= np.abs(frames - centres[k - 1]) <= np.abs(centres[k] - frames)
    return np.where(np.abs(centres[k] - frames) <= hop / 2, k, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--emb", default="vjepa_window")
    ap.add_argument("--start-min", type=float, default=0)
    ap.add_argument("--end-min", type=float)
    ap.add_argument("--context-s", type=float, default=60)
    ap.add_argument("--out")
    ap.add_argument("--labels", help="partition file (.npz with 'start' + label arrays) instead of clusters/windows.csv")
    ap.add_argument("--key", default="", help="--labels: partitions to show, comma separated (default: the first)")
    args = ap.parse_args()

    run = Path(args.run_dir)
    cfg = json.load(open(run / args.emb / "config.json"))
    w, keys, cdir = load_windows(run, args.emb, args.labels, [k for k in args.key.split(",") if k])
    fps, hop = cfg["fps"], cfg["hop"]
    canon = cfg.get("canon", "none") != "none"
    cols = {k: cluster_colours(w[k].to_numpy()) for k in keys}
    centres = w.centre.to_numpy()
    STRIP = ROW * len(keys) + AXIS

    cap = cv2.VideoCapture(str(run / "preview.mp4"))
    n_prev = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    p0 = int(args.start_min * 60 * fps / 2)
    p1 = min(n_prev, int(args.end_min * 60 * fps / 2)) if args.end_min else n_prev
    cap.set(cv2.CAP_PROP_POS_FRAMES, p0)
    tag = f"_{args.start_min:g}-{args.end_min:g}min" if args.start_min or args.end_min else ""
    out = Path(args.out) if args.out else cdir / (f"labeled_{keys[0]}{tag}.mp4" if args.labels else f"labeled{tag}.mp4")
    W, H = 3 * PANEL, PANEL + STRIP
    enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                            "-s", f"{W}x{H}", "-r", f"{fps / 2:.4f}", "-i", "-", "-c:v", "libx264",
                            "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)

    chunks = LabelChunks(run)
    half = int(args.context_s / 2 * fps / hop)  # hops either side in the strip
    offs = np.arange(-half, half + 1) * hop
    for p in range(p0, p1):
        ok, prev = cap.read()
        if not ok:
            break
        f = 2 * p
        k = int(nearest_window(centres, np.array([f]), hop)[0])

        lab = cv2.resize(solid(chunks.get([f])[0]), (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)
        if k < 0:
            cv2.rectangle(lab, (0, 0), (PANEL - 1, 54), (60, 60, 60), -1)
            cv2.putText(lab, "no window (mouse lost)", (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
        else:
            r = w.iloc[k]
            if canon:  # same rotation as embed_vjepa.py: the window's heading -> up
                M = cv2.getRotationMatrix2D((PANEL / 2, PANEL / 2), 90 + r.heading, 1.0)
                lab = cv2.warpAffine(lab, M, (PANEL, PANEL), flags=cv2.INTER_NEAREST)
            v = int(r[keys[0]])
            name = (f"{keys[0]}: {v}" if args.labels else f"cluster {v}") if v >= 0 else "noise"
            cv2.rectangle(lab, (0, 0), (PANEL - 1, 54), tuple(int(c) for c in cols[keys[0]][k]), -1)
            cv2.putText(lab, name, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7 if args.labels else 0.8, (0, 0, 0), 2)
            cv2.putText(lab, f"fwd {r.fwd_speed:.0f} px/s  turn {r.turn_rate:+.0f} deg/s", (8, 47),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        cv2.putText(lab, f"{f / fps / 60:.2f} min" + ("  (as encoded, rotated)" if canon and k >= 0 else ""),
                    (8, PANEL - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # ethogram strip: one block per hop around the current frame, one row per partition
        near = nearest_window(centres, f + offs, hop)
        rows = []
        for key in keys:
            blocks = np.where((near >= 0)[:, None], cols[key][np.maximum(near, 0)], NO_WINDOW)
            row = cv2.resize(blocks[None], (W, ROW), interpolation=cv2.INTER_NEAREST)
            if len(keys) > 1:
                cv2.putText(row, key, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            rows.append(row)
        strip = np.vstack(rows + [np.full((AXIS, W, 3), 20, np.uint8)])
        cv2.line(strip, (W // 2, 0), (W // 2, STRIP - AXIS - 1), (255, 255, 255), 2)
        cv2.putText(strip, f"-{args.context_s / 2:g} s", (4, STRIP - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(strip, f"+{args.context_s / 2:g} s", (W - 44, STRIP - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        frame = np.vstack([np.hstack([prev, lab]), strip])
        enc.stdin.write(frame.tobytes())
        if (p - p0) % 9000 == 0:
            print(f"  {f / fps / 60:.1f} min", flush=True)
    enc.stdin.close()
    enc.wait()
    print(f"-> {out}")


if __name__ == "__main__":
    main()
