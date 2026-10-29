"""Cluster-labelled video of a clustered run.

    python label_video.py RUN_DIR [--emb vjepa_window] [--start-min 0] [--end-min END]
                          [--context-s 60] [--out FILE]

Every frame of RUN_DIR/preview.mp4 (crop | part overlay, every 2nd frame) gets
a third panel with the part map as the encoder saw it (rotated by its window's
heading if --emb was canonicalised), the cluster of the window centred
nearest to it, and that window's body-frame forward speed and turn rate
(image-clockwise > 0). Below runs an ethogram strip of --context-s seconds
centred on the current frame. Cluster colours match overview.png; grey is
HDBSCAN noise.

Output (H.264, via ffmpeg): RUN_DIR/<emb>/clusters/labeled[_<start>-<end>min].mp4
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

PANEL, STRIP = 384, 60


def cluster_colours(labels):
    cmap = matplotlib.colormaps["tab20"]
    rgb = np.array([cmap(k % 20)[:3] if k >= 0 else (0.55, 0.55, 0.55) for k in labels])
    return (rgb[:, ::-1] * 255).astype(np.uint8)  # BGR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--emb", default="vjepa_window")
    ap.add_argument("--start-min", type=float, default=0)
    ap.add_argument("--end-min", type=float)
    ap.add_argument("--context-s", type=float, default=60)
    ap.add_argument("--out")
    args = ap.parse_args()

    run = Path(args.run_dir)
    cdir = run / args.emb / "clusters"
    cfg = json.load(open(run / args.emb / "config.json"))
    w = pd.read_csv(cdir / "windows.csv")
    fps, hop = cfg["fps"], cfg["hop"]
    canon = cfg.get("canon", "none") != "none"
    col = cluster_colours(w.cluster.to_numpy())
    centre0 = w.centre.iloc[0]

    cap = cv2.VideoCapture(str(run / "preview.mp4"))
    n_prev = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    p0 = int(args.start_min * 60 * fps / 2)
    p1 = min(n_prev, int(args.end_min * 60 * fps / 2)) if args.end_min else n_prev
    cap.set(cv2.CAP_PROP_POS_FRAMES, p0)
    tag = f"_{args.start_min:g}-{args.end_min:g}min" if args.start_min or args.end_min else ""
    out = Path(args.out) if args.out else cdir / f"labeled{tag}.mp4"
    W, H = 3 * PANEL, PANEL + STRIP
    enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                            "-s", f"{W}x{H}", "-r", f"{fps / 2:.4f}", "-i", "-", "-c:v", "libx264",
                            "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)

    chunks = LabelChunks(run)
    half = int(args.context_s / 2 * fps / hop)  # windows either side in the strip
    for p in range(p0, p1):
        ok, prev = cap.read()
        if not ok:
            break
        f = 2 * p
        k = int(np.clip(round((f - centre0) / hop), 0, len(w) - 1))
        r = w.iloc[k]

        lab = cv2.resize(solid(chunks.get([f])[0]), (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)
        if canon:  # same rotation as embed_vjepa.py: the window's heading -> up
            M = cv2.getRotationMatrix2D((PANEL / 2, PANEL / 2), 90 + r.heading, 1.0)
            lab = cv2.warpAffine(lab, M, (PANEL, PANEL), flags=cv2.INTER_NEAREST)
        c = tuple(int(v) for v in col[k])
        name = f"cluster {r.cluster}" if r.cluster >= 0 else "noise"
        cv2.rectangle(lab, (0, 0), (PANEL - 1, 54), c, -1)
        cv2.putText(lab, name, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(lab, f"fwd {r.fwd_speed:.0f} px/s  turn {r.turn_rate:+.0f} deg/s", (8, 47),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        cv2.putText(lab, f"{f / fps / 60:.2f} min" + ("  (as encoded, rotated)" if canon else ""),
                    (8, PANEL - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # ethogram strip: one block per window, current window under the centre line
        idx = np.arange(k - half, k + half + 1)
        valid = (idx >= 0) & (idx < len(w))
        blocks = np.where(valid[:, None], col[np.clip(idx, 0, len(w) - 1)], 20)
        strip = cv2.resize(blocks[None], (W, STRIP - 20), interpolation=cv2.INTER_NEAREST)
        strip = np.vstack([strip, np.full((20, W, 3), 20, np.uint8)])
        cv2.line(strip, (W // 2, 0), (W // 2, STRIP - 21), (255, 255, 255), 2)
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
