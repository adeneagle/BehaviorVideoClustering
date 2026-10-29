"""Write run_video.py's keypoints as a SLEAP predictions file, to review them in the SLEAP GUI.

    ~/.local/share/uv/tools/sleap/bin/python kp_to_slp.py RUN_DIR [--min-score 0.2] [--start-sec S]
                                                         [--end-sec E] [--suggest 200] [--out FILE]

Runs in the SLEAP tool env. RUN_DIR is a run_video.py output with keypoints (frames.csv kp_* columns);
the video is the one in its summary.txt. Every frame in range becomes one predicted instance with the
lab project's skeleton (same six nodes); a node scored below --min-score is left out (shown as
missing), and the scores are kept as point scores. --suggest adds that many random frames with the
mouse in view as Labeling Suggestions, to step through an unbiased sample.

Open with `sleap-label RUN_DIR/predictions.slp`. Ctrl+Shift+arrows (or the seekbar) move through
the video; the right panel lists the suggestions. Correcting a prediction there (double-click it)
makes it a user label, so the same file can become training frames.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import sleap_io as sio

PROJECT = "/mnt/howlin/mnt/data/samba/yttri/Aden/Sleap/subset_labels_new.slp"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--min-score", type=float, default=0.2)
    ap.add_argument("--start-sec", type=float, default=0)
    ap.add_argument("--end-sec", type=float)
    ap.add_argument("--suggest", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    args = ap.parse_args()

    run = Path(args.run_dir)
    video_path = next(l.split(": ", 1)[1].strip() for l in open(run / "summary.txt") if l.startswith("video: "))
    skeleton = sio.load_slp(PROJECT).skeletons[0]
    nodes = [n.name for n in skeleton.nodes]
    cols = [f"kp_{n}_{c}" for n in nodes for c in ("x", "y", "score")]
    df = pd.read_csv(run / "frames.csv", usecols=["frame", "t", "mouse_found"] + cols)
    df = df[(df.t >= args.start_sec) & ((df.t < args.end_sec) if args.end_sec else True)]
    xy = np.stack([df[[f"kp_{n}_x", f"kp_{n}_y"]].to_numpy(np.float32) for n in nodes], 1)  # frames, nodes, 2
    sc = np.stack([df[f"kp_{n}_score"].to_numpy(np.float32) for n in nodes], 1)
    xy[sc < args.min_score] = np.nan

    video = sio.Video.from_filename(video_path)
    arr = np.concatenate([xy, sc[..., None]], -1)[:, None]  # frames, 1 instance, nodes, (x, y, score)
    labels = sio.Labels.from_numpy(arr, videos=[video], skeletons=skeleton, first_frame=int(df.frame.iloc[0]),
                                   return_confidence=True)
    labels.provenance["source"] = f"run_video.py keypoints, {run}"
    seen = df.frame.to_numpy()[df.mouse_found.to_numpy()]
    if args.suggest and len(seen):
        pick = np.sort(np.random.default_rng(args.seed).choice(seen, size=min(args.suggest, len(seen)), replace=False))
        labels.suggestions = [sio.SuggestionFrame(video=video, frame_idx=int(f)) for f in pick]
    out = Path(args.out) if args.out else run / "predictions.slp"
    labels.save(str(out))
    print(f"{len(labels)} frames ({df.frame.iloc[0]}-{df.frame.iloc[-1]}), {len(labels.suggestions)} suggestions -> {out}")


if __name__ == "__main__":
    main()
