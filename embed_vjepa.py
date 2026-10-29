"""V-JEPA 2 embeddings of the part-map video, or of the raw video cropped and canonicalised the same way.

    python embed_vjepa.py RUN_DIR [--canon window|none] [--model facebook/vjepa2-vitl-fpc64-256]
                          [--frames 16] [--sample-fps 15] [--hop-sec 0.5] [--sheet]
                          [--input parts|raw|masked[,...]] [--layers 12,16,20] [--out-name NAME]
                          [--backbone vjepa2|vjepa2_1-l] [--centre raw|crop]

--backbone vjepa2_1-l: V-JEPA 2.1 ViT-L/16 at 384 px (see Encoder); --layers from 6,12,18,24.

--input (default parts):
  parts   the label maps as solid part colours (below)
  raw     the camera frames (grayscale x2 gain, as RGB) cropped around the animal: the same
          crop centre, scale and per-window rotation as the part maps, but cut 1088 px wide so
          the rotation leaves no empty corners. Read from the video in one sequential pass.
  masked  raw, with everything outside the (dilated) predicted part mask set to black
Several inputs share one decoding pass and are written to <out-name>_<input>/.

RUN_DIR is a run_video.py output (<video>/full_video/<model>/). Every frame's
label map is drawn as solid part colours on black (render.solid()), so the
encoder sees only the segmentation, never the fur, lighting or arena. The map
covers the 768 px mouse-centred crop; it is area-resized 512 -> 256 px (the
encoder's input) without zooming, because a stretched mouse reaches the crop
edge.

Windows: `--frames` frames sampled at `--sample-fps` (16 at 15 fps = 1.07 s),
one window every `--hop-sec`. Each window's feature is the mean of the
encoder's last-layer tokens (1024-d for ViT-L).

Heading (axis.py, per frame): tail base -> body centroid, or the paw axis
where there is no tail base. A window's heading is their circular mean,
tail frames weighted by axis length (short = rearing/foreshortened), paw
frames at half weight; a window with no heading at all gets one
interpolated from its neighbours.

--canon window (default) rotates all frames of a window by that ONE angle so
the mouse faces up: heading is removed between windows, while turning within
a window stays visible. Translation is still removed per frame by the crop,
so the window's body-frame motion is saved as numbers instead (fwd / lat
speed, turn rate). --canon none keeps the image orientation.

--centre: what each frame is centred on. `raw` (default since 2026-10-18) is
causal: see below. `crop` (the old behaviour, for reproducing earlier runs) is
run_video.py's crop, which follows a CENTRED smoothing of the track that reads
~0.7 s ahead: the mouse's offset from the crop centre is raw(t) - smoothed(t),
which shows where it is about to go (on 10-08 it adds ~0.05 R^2 to the next
0.13-1 s of displacement beyond the past displacement). `raw` shifts every
frame so the localizer's own centre for that frame (frames.csv cx, cy; no
smoothing) sits in the middle: causal, as score_behaviour.py needs.

Output in RUN_DIR/vjepa/ (--canon none) or RUN_DIR/vjepa_window/ (or --out-name):
  features.npz  feats [N,D] float16 (last layer; --layers adds feats_L<l>, the token mean of block l);
                start / centre / stop frame per window;
                heading [N] (deg, image coords: 0 = right, +90 = down) and
                heading_R [N] (weighted consistency in 0..1) and
                heading_src [N] (share of the weight from the tail axis;
                -1 = interpolated); fwd_speed / lat_speed [N] (body-centre
                px/s along / across the heading, full-res px; lat > 0 =
                image-clockwise of the heading) and turn_rate [N] (deg/s,
                > 0 = image-clockwise; mirrored relative to a top view)
  config.json
  canon_examples.jpg (--sheet): first and last frame of some windows, as fed
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from axis import run_axes
from common import palette_lut
from nvdec_reader import gray_batches, pick_decoder

VJEPA21_REPO = Path("/home/aeagle/Aden/Code/third_party/vjepa2")  # facebookresearch/vjepa2 @ 204698b (2026-03-23)
VJEPA21_WEIGHTS = Path("/home/aeagle/Aden/Data/Behavior/PartSeg/models/vjepa2_1/vjepa2_1_vitl_encoder_only.pt")
RAW_CUT = 1088  # crop side (full-res px) for the raw input: 768 * sqrt(2), so rotation keeps the corners filled
RAW_GAIN = 2.0  # the footage is dark (render.enhance uses the same gain for display)

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class LabelChunks:
    """Random access to frames of run_video.py's labels_<first>.npz chunks."""

    def __init__(self, run_dir):
        self.files = sorted(Path(run_dir, "labels").glob("labels_*.npz"))
        self.firsts = np.array([int(f.stem.split("_")[1]) for f in self.files])
        self.cache = {}

    def get(self, idx):
        idx = np.asarray(idx)
        chunk = np.searchsorted(self.firsts, idx, side="right") - 1
        out = None
        for c in np.unique(chunk):
            if c not in self.cache:
                if len(self.cache) >= 2:
                    self.cache.pop(min(self.cache))
                self.cache[c] = np.load(self.files[c])["labels"]
            sel = chunk == c
            lab = self.cache[c][idx[sel] - self.firsts[c]]
            if out is None:
                out = np.empty((len(idx),) + lab.shape[1:], np.uint8)
            out[sel] = lab
        return out


def palette(device):
    """render.solid()'s colours as RGB in 0..1: parts only, sinks and background black."""
    return torch.from_numpy(palette_lut(render_only=True)[:, ::-1].copy()).float().div(255).to(device)


def frame_headings(ax):
    """Per-frame heading (deg) and weight from axis.py's output."""
    tail, paw = ax["tail_angle"], ax["paw_angle"]
    w_tail = np.clip(ax["tail_len"] / np.nanmedian(ax["tail_len"]), 0, 1)
    use_tail = ~np.isnan(tail)
    ang = np.where(use_tail, tail, paw)
    w = np.where(use_tail, w_tail, np.where(np.isnan(paw), 0, 0.5))
    return np.nan_to_num(ang), w, use_tail


def window_headings(ang, w, use_tail, idx, min_weight=1.0):
    """idx [N,T] frames -> heading (deg), consistency R, tail share of the weight
    (-1 where the window had less than min_weight and was interpolated)."""
    a, ww = np.radians(ang[idx]), w[idx]
    c, s = (ww * np.cos(a)).sum(1), (ww * np.sin(a)).sum(1)
    tot = ww.sum(1)
    R = np.hypot(c, s) / np.maximum(tot, 1e-9)
    ok = tot >= min_weight
    src = np.where(ok, (ww * use_tail[idx]).sum(1) / np.maximum(tot, 1e-9), -1)
    if not ok.all():  # interpolate the unit vector between neighbouring windows
        n = np.arange(len(idx))
        u = np.c_[c, s] / np.maximum(np.hypot(c, s), 1e-9)[:, None]
        c = np.interp(n, n[ok], u[ok, 0])
        s = np.interp(n, n[ok], u[ok, 1])
    return np.degrees(np.arctan2(s, c)), R, src


def body_motion(xy, ang, idx, head, fps, edge=4):
    """Window body-frame speeds (px/s) from the track, turn rate (deg/s) from the
    per-frame headings: circular mean of the last `edge` sampled frames minus
    the first `edge`."""
    dt = (idx[:, -1] - idx[:, 0]) / fps
    v = (xy[idx[:, -1]] - xy[idx[:, 0]]) / dt[:, None]
    h = np.radians(head)
    fwd = v[:, 0] * np.cos(h) + v[:, 1] * np.sin(h)
    lat = -v[:, 0] * np.sin(h) + v[:, 1] * np.cos(h)
    a = np.radians(ang[idx])
    first = np.arctan2(np.sin(a[:, :edge]).sum(1), np.cos(a[:, :edge]).sum(1))
    last = np.arctan2(np.sin(a[:, -edge:]).sum(1), np.cos(a[:, -edge:]).sum(1))
    turn = (np.degrees(last - first) + 180) % 360 - 180
    return fwd, lat, turn / ((idx[:, -edge:].mean(1) - idx[:, :edge].mean(1)) / fps)


def rotation_grid(head_deg, n_rep, size, device, scale=1.0, shift=None):
    """Sampling grid that turns a window's heading to 'up' (image -90 deg). scale < 1 samples only
    the central part of the input (the raw input is cut wider than the output). shift [B*n_rep, 2]:
    per-frame input point (normalised -1..1 coords) that lands in the output centre."""
    phi = torch.deg2rad(torch.as_tensor(head_deg, dtype=torch.float32, device=device) + 90)
    c, s = torch.cos(phi) * scale, torch.sin(phi) * scale
    z = torch.zeros_like(c)
    theta = torch.stack([torch.stack([c, -s, z], 1), torch.stack([s, c, z], 1)], 1)  # output -> input
    theta = theta.repeat_interleave(n_rep, 0)
    if shift is not None:
        theta[:, :, 2] = torch.as_tensor(shift, dtype=torch.float32, device=device)
    return F.affine_grid(theta, (len(theta), 3, size, size), align_corners=False)


class Encoder:
    """Uniform call for the two backbones: video [B,T,3,S,S] (ImageNet-normalised) ->
    (token mean of the last layer [B,D], {block l: token mean [B,D]}).

    vjepa2      Hugging Face facebook/vjepa2-* (any of its 24 blocks via --layers)
    vjepa2_1-l  V-JEPA 2.1 ViT-L/16 (384 px, distilled from ViT-G; Meta's checkpoint, the
                repo's own model code). Its blocks 6/12/18/24 have their own trained output
                norms (the deeply supervised layers); --layers must be a subset of those."""

    def __init__(self, backbone, hf_model, layers, device):
        self.backbone, self.layers = backbone, layers
        if backbone == "vjepa2":
            from transformers import VJEPA2Model

            self.m = VJEPA2Model.from_pretrained(hf_model, torch_dtype=torch.float16, attn_implementation="sdpa").to(device).eval()
        else:
            import sys

            sys.path.insert(0, str(VJEPA21_REPO))
            from app.vjepa_2_1.models import vision_transformer as vit

            assert set(layers) <= {6, 12, 18, 24}, "V-JEPA 2.1 ViT-L exposes blocks 6, 12, 18, 24"
            self.m = vit.vit_large(patch_size=16, img_size=(384, 384), num_frames=64, tubelet_size=2, use_sdpa=True,
                                   use_SiLU=False, wide_SiLU=True, uniform_power=False, use_rope=True,
                                   img_temporal_dim_size=1, interpolate_rope=True, out_layers=[5, 11, 17, 23])
            self.m.load_state_dict(torch.load(VJEPA21_WEIGHTS, map_location="cpu")["ema_encoder"], strict=True)
            self.m = self.m.to(device).eval()

    def __call__(self, video):
        if self.backbone == "vjepa2":
            o = self.m(pixel_values_videos=video, skip_predictor=True, output_hidden_states=bool(self.layers))
            # hidden_states[0] is the patch embedding, [l] the output of block l
            return o.last_hidden_state.float().mean(1), {L: o.hidden_states[L].float().mean(1) for L in self.layers}
        with torch.autocast("cuda", dtype=torch.float16):  # its RoPE runs in fp32, so autocast, not .half()
            outs = self.m(video.permute(0, 2, 1, 3, 4).float())  # [B,3,T,S,S]
        by_block = {L: o.float().mean(1) for L, o in zip((6, 12, 18, 24), outs)}
        return by_block[24], {L: by_block[L] for L in self.layers}


class RawCrops:
    """Frames needed by the windows, cut RAW_CUT px wide around each frame's crop centre and
    area-resized by the part maps' scale (768 -> 512), decoded in one sequential pass."""

    def __init__(self, video, frames_df, needed, lab_size, decoder, device, batch=64):
        self.cx = frames_df.x0.to_numpy() + 384  # centre of run_video.py's 768 px crop
        self.cy = frames_df.y0.to_numpy() + 384
        self.needed = np.zeros(len(frames_df) + 1, bool)
        self.needed[np.asarray(needed)] = True
        self.out = int(round(RAW_CUT * lab_size / 768))
        self.dev, self.store = device, {}
        self.it = gray_batches(video, batch, 0, int(np.max(needed)) + 1, decoder)
        self.decoded = 0

    def _cut(self, g, i):
        H, W = g.shape
        h = RAW_CUT // 2
        x0, y0 = int(round(self.cx[i])) - h, int(round(self.cy[i])) - h
        a = g[max(y0, 0) : min(y0 + RAW_CUT, H), max(x0, 0) : min(x0 + RAW_CUT, W)].float()
        a = F.pad(a, (max(-x0, 0), max(x0 + RAW_CUT - W, 0), max(-y0, 0), max(y0 + RAW_CUT - H, 0)))
        return F.interpolate(a[None, None], size=(self.out, self.out), mode="area")[0, 0].to(torch.uint8)

    def get(self, idx):
        """[len(idx), out, out] uint8 on the device; frames older than min(idx) are dropped."""
        while self.decoded <= idx.max():
            i0, g = next(self.it)
            for j in np.flatnonzero(self.needed[i0 : i0 + len(g)]):
                self.store[i0 + j] = self._cut(g[j], i0 + j)
            self.decoded = i0 + len(g)
        lo = idx.min()
        for f in [f for f in self.store if f < lo]:
            del self.store[f]
        return torch.stack([self.store[i] for i in idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--backbone", choices=["vjepa2", "vjepa2_1-l"], default="vjepa2")
    ap.add_argument("--model", default="facebook/vjepa2-vitl-fpc64-256", help="Hugging Face model (vjepa2 backbone)")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--sample-fps", type=float, default=15)
    ap.add_argument("--hop-sec", type=float, default=0.5)
    ap.add_argument("--size", type=int, help="encoder input px (default 256 for vjepa2, 384 for vjepa2_1-l)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--canon", choices=["window", "none"], default="window")
    ap.add_argument("--centre", choices=["crop", "raw"], default="raw",
                    help="centre each frame on the (look-ahead smoothed) crop, or on its own raw localizer centre")
    ap.add_argument("--layers", default="", help="also save the token mean of these encoder layers (1-24), e.g. 8,16")
    ap.add_argument("--out-name", help="output folder in RUN_DIR (default vjepa / vjepa_window)")
    ap.add_argument("--sheet", action="store_true", help="write canon_examples.jpg")
    ap.add_argument("--input", default="parts", help="parts, raw, masked, or several comma separated")
    ap.add_argument("--video", help="raw/masked input: the video (default: the one in RUN_DIR/summary.txt)")
    ap.add_argument("--decoder", choices=["auto", "nvdec", "cpu"], default="cpu")
    ap.add_argument("--mask-dilate", type=int, default=7, help="masked input: dilate the part mask by this many px (512 px scale)")
    ap.add_argument("--max-windows", type=int, help="only the first N windows (for a quick check)")
    args = ap.parse_args()

    args.size = args.size or (384 if args.backbone == "vjepa2_1-l" else 256)
    run = Path(args.run_dir)
    inputs = args.input.split(",")
    base = args.out_name or ("vjepa" if args.canon == "none" else f"vjepa_{args.canon}")
    outs = {k: run / (base if inputs == ["parts"] else f"{base}_{k}") for k in inputs}
    layers = [int(v) for v in args.layers.split(",") if v]
    for o in outs.values():
        o.mkdir(exist_ok=True)
    frames = pd.read_csv(run / "frames.csv", usecols=["frame", "t", "x0", "y0", "cx", "cy"])
    n_frames = len(frames)
    fps = (n_frames - 1) / frames.t.iloc[-1]
    step = max(1, round(fps / args.sample_fps))
    span = (args.frames - 1) * step + 1
    hop = max(1, round(args.hop_sec * fps))
    starts = np.arange(0, n_frames - span + 1, hop)[: args.max_windows]
    print(f"{n_frames} frames at {fps:.2f} fps; windows of {args.frames} frames every {step} "
          f"({span / fps:.2f} s), hop {hop} ({hop / fps:.2f} s): {len(starts)} windows", flush=True)

    offs = np.arange(args.frames) * step
    widx = starts[:, None] + offs[None]
    ang, w, use_tail = frame_headings(run_axes(run))
    head, head_R, head_src = window_headings(ang, w, use_tail, widx)
    fwd, lat, turn = body_motion(frames[["cx", "cy"]].to_numpy(), ang, widx, head, fps)
    # raw localizer centre relative to the 768 px crop centre, full-res px (0 where the mouse was not found)
    off = np.nan_to_num(frames[["cx", "cy"]].to_numpy() - (frames[["x0", "y0"]].to_numpy() + 384))
    if args.centre == "crop":
        off[:] = 0
    else:
        print(f"re-centring on the raw track: offset from the crop centre median {np.median(np.hypot(*off.T)):.0f} px", flush=True)
    print(f"window headings: {np.mean(head_src > 0.5):.1%} mostly tail axis, {np.mean((head_src >= 0) & (head_src <= 0.5)):.1%} "
          f"mostly paw axis, {np.mean(head_src < 0):.1%} interpolated; median consistency R {np.median(head_R):.2f}", flush=True)

    dev = torch.device("cuda")
    encode = Encoder(args.backbone, args.model, layers, dev)
    pal, mean, std = palette(dev), MEAN.to(dev), STD.to(dev)
    chunks = LabelChunks(run)
    lab_size = chunks.get(np.array([0])).shape[-1]
    raw = None
    if {"raw", "masked"} & set(inputs):
        video = args.video or next(l.split(": ", 1)[1].strip() for l in open(run / "summary.txt") if l.startswith("video: "))
        raw = RawCrops(video, frames, np.unique(widx), lab_size, pick_decoder(args.decoder, Path(video)), dev)
        pad = (raw.out - lab_size) // 2

    feats = {k: [] for k in inputs}
    extra = {k: {L: [] for L in layers} for k in inputs}
    sheet = {k: [] for k in inputs}
    show = set(np.linspace(0, len(starts) - 1, 12).astype(int) // args.batch * args.batch) if args.sheet else set()
    t0 = time.time()
    for b in range(0, len(starts), args.batch):
        idx = widx[b : b + args.batch].ravel()
        lab = torch.from_numpy(chunks.get(idx)).to(dev, non_blocking=True)
        g = raw.get(idx) if raw is not None else None
        with torch.no_grad():
            for k in inputs:
                if k == "parts":
                    img, scale = pal[lab.long()].permute(0, 3, 1, 2), 1.0  # [B*T,3,512,512] in 0..1
                else:
                    x = (g.float() * (RAW_GAIN / 255)).clamp_(0, 1)[:, None]  # [B*T,1,724,724]
                    if k == "masked":
                        m = (lab > 0).float()[:, None]
                        if args.mask_dilate:
                            m = F.max_pool2d(m, 2 * args.mask_dilate + 1, 1, args.mask_dilate)
                        x = x * F.pad(m, (pad, raw.out - lab_size - pad) * 2)
                    img, scale = x, lab_size / raw.out  # one channel until after the resize
                if args.canon == "window" or args.centre == "raw":
                    # the parts input spans the 768 px crop, the raw one RAW_CUT px, both centred on the crop
                    shift = off[idx] / (384.0 if k == "parts" else RAW_CUT / 2)
                    h = head[b : b + args.batch] if args.canon == "window" else np.full(len(idx) // args.frames, -90.0)
                    grid = rotation_grid(h, args.frames, lab_size, dev, scale, shift)
                    img = F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
                elif scale != 1.0:
                    img = img[..., pad : pad + lab_size, pad : pad + lab_size]
                img = F.interpolate(img, size=(args.size, args.size), mode="area").expand(-1, 3, -1, -1)
                if b in show:
                    sheet[k].append(np.hstack([img[j].permute(1, 2, 0).cpu().numpy() for j in (0, args.frames - 1)]))
                img = ((img - mean) / std).half().view(-1, args.frames, 3, args.size, args.size)
                last, by_layer = encode(img)
                feats[k].append(last.cpu().numpy().astype(np.float16))
                for L in layers:
                    extra[k][L].append(by_layer[L].cpu().numpy().astype(np.float16))
        if (b // args.batch) % 50 == 0:
            done = b + len(idx) // args.frames
            print(f"  {done}/{len(starts)}  {done / (time.time() - t0):.1f} windows/s", flush=True)

    for k, out in outs.items():
        np.savez(out / "features.npz", feats=np.concatenate(feats[k]),
                 **{f"feats_L{L}": np.concatenate(v) for L, v in extra[k].items()},
                 start=starts, centre=starts + span // 2,
                 stop=starts + span, heading=head, heading_R=head_R, heading_src=head_src,
                 fwd_speed=fwd, lat_speed=lat, turn_rate=turn, fps=fps)
        json.dump(dict(vars(args), input=k, fps=fps, step=step, span=span, hop=hop, n_windows=len(starts)),
                  open(out / "config.json", "w"), indent=2)
        if sheet[k]:
            import cv2

            rows = [np.hstack(sheet[k][i : i + 3]) for i in range(0, len(sheet[k]) - len(sheet[k]) % 3, 3)]
            cv2.imwrite(str(out / "canon_examples.jpg"), (np.vstack(rows)[..., ::-1] * 255).astype(np.uint8))
        print(f"done: {len(starts)} windows in {time.time() - t0:.0f}s -> {out / 'features.npz'}")


if __name__ == "__main__":
    main()
