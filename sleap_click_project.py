"""Build a SLEAP labelling project for hand-clicking chosen frames of one video (blind: no predictions).

    ~/.local/share/uv/tools/sleap/bin/python sleap_click_project.py FRAMES.json VIDEO OUT.slp [--seed 0]

Runs in the SLEAP tool env. FRAMES.json maps group names to frame lists (e.g. made for grooming
frames); every frame becomes a SLEAP suggestion, in shuffled order so the labeller cannot tell the
groups apart, and the skeleton is the lab project's (Sleap/subset_labels_new.slp). The file holds
no instances, so nothing anchors the clicks to either model.

Label in the GUI: `sleap-label OUT.slp`, Labeling Suggestions (right panel) -> go to a suggested
frame, add an instance (Ctrl+I / double-click), drag the nodes, right-click a node to mark it not
visible, Ctrl+S. Then score_clicks.py compares the models with your clicks.
"""

import argparse
import json

import numpy as np
import sleap_io as sio

PROJECT = "/mnt/howlin/mnt/data/samba/yttri/Aden/Sleap/subset_labels_new.slp"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames_json")
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    groups = json.load(open(args.frames_json))
    frames = sorted({int(f) for v in groups.values() for f in v})
    order = np.random.default_rng(args.seed).permutation(len(frames))
    skeleton = sio.load_slp(PROJECT).skeletons[0]
    video = sio.Video.from_filename(args.video)
    labels = sio.Labels(videos=[video], skeletons=[skeleton],
                        suggestions=[sio.SuggestionFrame(video=video, frame_idx=frames[i]) for i in order])
    labels.save(args.out)
    print(f"{len(frames)} suggested frames ({', '.join(f'{k} {len(v)}' for k, v in groups.items())}), "
          f"{video.shape} -> {args.out}")


if __name__ == "__main__":
    main()
