"""Overlay / solid-colour renders of part label maps."""

from pathlib import Path

import cv2
import numpy as np

from common import PARTS, frame_paths, palette_lut

_LUT = palette_lut()
_LUT_RENDER = palette_lut(render_only=True)


def enhance(img, gain=2.0):
    """Display-only brightening; the footage is dark."""
    return cv2.convertScaleAbs(img, alpha=gain)


def overlay(img, labels, alpha=0.5, contours=True):
    col = _LUT[labels]
    fg = labels[..., None] > 0
    out = np.where(fg, (alpha * col + (1 - alpha) * img).astype(np.uint8), img)
    if contours:
        for p in PARTS:
            m = (labels == p.obj_id).astype(np.uint8)
            if m.any():
                cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(out, cs, -1, p.bgr, 1)
    return out


def solid(labels):
    """The representation the clustering stage will see: fixed colours, no
    pixels from the camera, background and sink parts black."""
    return _LUT_RENDER[labels]


def write_videos(cdir, labels, fps, gain=2.0, suffix=""):
    cdir = Path(cdir)
    paths = frame_paths(cdir)
    T, H, W = labels.shape
    four = cv2.VideoWriter_fourcc(*"mp4v")
    ov = cv2.VideoWriter(str(cdir / f"overlay{suffix}.mp4"), four, fps, (2 * W, H))
    pv = cv2.VideoWriter(str(cdir / f"parts{suffix}.mp4"), four, fps, (W, H))
    for t in range(T):
        img = enhance(cv2.imread(str(paths[t])), gain)
        o = overlay(img, labels[t])
        cv2.putText(o, str(t), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        ov.write(np.hstack([img, o]))
        pv.write(solid(labels[t]))
    ov.release()
    pv.release()


def contact_sheet(cdir, labels, n=12, cols=6, tile=256, gain=2.0, path=None):
    cdir = Path(cdir)
    paths = frame_paths(cdir)
    T = len(labels)
    idx = np.linspace(0, T - 1, n).astype(int)
    tiles = []
    for t in idx:
        o = overlay(enhance(cv2.imread(str(paths[t])), gain), labels[t])
        o = cv2.resize(o, (tile, tile), interpolation=cv2.INTER_AREA)
        cv2.putText(o, str(t), (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        tiles.append(o)
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % cols)
    sheet = np.vstack([np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)])
    cv2.imwrite(str(path or cdir / "contact_sheet.jpg"), sheet)
    return sheet
