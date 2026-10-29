"""Read SLEAP keypoints (.slp is HDF5) without needing sleap installed.

The SLEAP project lives on the lab server; its files store video paths as
seen from there (/mnt/data/samba/yttri/Aden/...). local_video() maps them to
the mount on this machine and follows the server-side symlinks by hand.

Node order: snout, left/right front paw, left/right hind paw, tail_base.
Left/right are the animal's own, as for the part names in common.py.
"""

import json
import os
from pathlib import Path

import numpy as np

SERVER_ROOT = "/mnt/data/samba/yttri/Aden"
MOUNT_ROOT = "/mnt/howlin/mnt/data/samba/yttri/Aden"
SLEAP_DIR = Path(MOUNT_ROOT) / "Sleap"

NODES = ("snout", "left_front_paw", "right_front_paw", "left_hind_paw", "right_hind_paw", "tail_base")
SNOUT, TAIL_BASE = 0, 5
# keypoint index -> part name
PAW_NODES = {1: "forepaw_L", 2: "forepaw_R", 3: "hindpaw_L", 4: "hindpaw_R"}


def local_video(path):
    """Server path (possibly a server-side symlink) -> readable local path."""
    p = str(path)
    for _ in range(5):
        if p.startswith(SERVER_ROOT):
            p = MOUNT_ROOT + p[len(SERVER_ROOT):]
        if not os.path.islink(p):
            break
        t = os.readlink(p)
        p = t if os.path.isabs(t) else os.path.normpath(os.path.join(os.path.dirname(p), t))
    return Path(p)


def read_slp(path, predicted=False):
    """Keypoints of the human-labelled (or predicted) instances in a .slp file.

    Returns dict: videos [n_videos] server paths, video [N] index into videos,
    frame [N], xy [N,6,2] full-frame px (NaN where not visible), score [N,6]
    (predictions only; 1 for human labels), instance_score [N].
    One instance per frame is assumed (single mouse); if a frame has several,
    the highest-scoring one is kept.
    """
    import h5py

    with h5py.File(path, "r") as f:
        md = json.loads(f["metadata"].attrs["json"])
        nodes = [n["name"] for n in md["nodes"]]
        assert tuple(nodes) == NODES, nodes
        videos = [json.loads(v)["filename"] for v in f["videos_json"][:]]
        frames, inst = f["frames"][:], f["instances"][:]
        pts = f["pred_points" if predicted else "points"][:]
    inst = inst[inst["instance_type"] == (1 if predicted else 0)]
    order = np.argsort(frames["frame_id"])
    k = order[np.searchsorted(frames["frame_id"][order], inst["frame_id"])]
    vid, fidx = frames["video"][k].astype(int), frames["frame_idx"][k].astype(int)
    n = len(NODES)
    assert np.all(inst["point_id_end"] - inst["point_id_start"] == n)
    idx = inst["point_id_start"][:, None].astype(np.int64) + np.arange(n)
    xy = np.stack([pts["x"][idx], pts["y"][idx]], -1)
    xy[~pts["visible"][idx]] = np.nan
    score = pts["score"][idx] if predicted else np.ones(idx.shape)
    iscore = inst["score"].astype(float)
    # one instance per (video, frame): keep the best
    key = vid.astype(np.int64) * 10**9 + fidx
    o = np.lexsort((-iscore, key))
    keep = o[np.r_[True, np.diff(key[o]) != 0]]
    return dict(videos=videos, video=vid[keep], frame=fidx[keep], xy=xy[keep],
                score=score[keep], instance_score=iscore[keep])


def per_frame(d, n_frames, video=0):
    """Dense [n_frames,6,2] and [n_frames,6] arrays (NaN / 0 where no instance)."""
    m = (d["video"] == video) & (d["frame"] < n_frames)
    xy = np.full((n_frames, len(NODES), 2), np.nan)
    sc = np.zeros((n_frames, len(NODES)))
    xy[d["frame"][m]], sc[d["frame"][m]] = d["xy"][m], d["score"][m]
    return xy, sc
