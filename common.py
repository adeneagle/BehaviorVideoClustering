"""Shared paths, part definitions and I/O for the anatomical part-mask pilot.

Layout of one video's output directory (``<out_root>/<video_stem>/``)::

    background.npy            median background (full-res, uint8 gray)
    manifest.csv              one row per extracted clip
    clips/<clip_id>/
        frames/00000.jpg ...  mouse-centred crops fed to SAM3
        meta.json             source video, frame range, per-frame crop offsets,
                              per-frame behaviour labels
        prompts.json          click rounds (written by annotate.py or by an
                              automatic prompter, e.g. from SLEAP keypoints)
        masks.npz             labels [T,H,W] uint8 (0 = background, k = part k),
                              obj_scores [T,K], raw_overlap [T,K,K]
        qc.json               frames a human flagged as wrong, per part
        overlay.mp4 / parts.mp4 / contact_sheet.jpg
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_OUT_ROOT = Path("/home/aeagle/Aden/Data/Behavior/PartSeg")


@dataclass(frozen=True)
class Part:
    obj_id: int  # SAM3 tracker object id and label-map value
    name: str
    bgr: tuple  # overlay colour (OpenCV BGR)
    key: str  # hotkey in annotate.py
    render: bool = True  # drawn in the solid-colour "parts" video


# Left/right are the ANIMAL's left/right, not image left/right. In an
# underside view the image is mirrored relative to a top view, so check
# against the animal's anatomy, not the screen.
#
# head and trunk are one "body" region: from below there is no visible
# boundary between them, so a split would jitter and read as movement. A
# separate snout part was tried and dropped (tracked poorly); head direction
# should come from keypoints instead. Ids are kept stable (1 is retired) so
# saved clicks never change meaning.
#
# tail and implant are optional "sink" objects: the tracker keeps object masks
# non-overlapping, so giving distractors their own object stops them being
# absorbed into body/paw masks if right-clicks alone do not hold them off.
# They are left out of the solid-colour render.
PARTS = (
    Part(2, "body", (200, 200, 200), "2"),
    Part(3, "forepaw_L", (60, 60, 255), "3"),
    Part(4, "forepaw_R", (255, 120, 60), "4"),
    Part(5, "hindpaw_L", (60, 255, 255), "5"),
    Part(6, "hindpaw_R", (255, 60, 255), "6"),
    Part(7, "tail", (90, 140, 90), "7", render=False),
    Part(8, "implant", (40, 40, 160), "8", render=False),
)
ANATOMICAL = tuple(p for p in PARTS if p.render)
PART_BY_ID = {p.obj_id: p for p in PARTS}
# Left/right pairs, used to detect identity swaps.
LR_PAIRS = (("forepaw_L", "forepaw_R"), ("hindpaw_L", "hindpaw_R"))


def part_id(name):
    return next(p.obj_id for p in PARTS if p.name == name)


def palette_lut(render_only=False):
    """256x3 BGR lookup table mapping label value -> colour."""
    lut = np.zeros((256, 3), np.uint8)
    for p in PARTS:
        if p.render or not render_only:
            lut[p.obj_id] = p.bgr
    return lut


def clip_dir(out_root, video_stem, clip_id):
    return Path(out_root) / video_stem / "clips" / clip_id


def frame_paths(cdir):
    fdir = Path(cdir) / "frames"
    return sorted(fdir.glob("*.jpg"), key=lambda p: int(p.stem))


def read_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def write_json(path, obj):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def save_masks(cdir, labels, obj_scores, raw_overlap, obj_ids, name="masks.npz"):
    np.savez_compressed(
        Path(cdir) / name,
        labels=labels,
        obj_scores=obj_scores,
        raw_overlap=raw_overlap,
        obj_ids=np.asarray(obj_ids),
    )


def load_masks(cdir, name="masks.npz"):
    z = np.load(Path(cdir) / name)
    return {k: z[k] for k in z.files}
