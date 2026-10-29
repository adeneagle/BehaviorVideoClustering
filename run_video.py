"""Part masks for a whole video: one pass, frames on the GPU, TensorRT.

    python run_video.py VIDEO.mkv [--labels LABELS.csv] [--model RUN_DIR|CKPT]
                        [--backend trt|torch] [--precision fp16|fp32]
                        [--decoder auto|nvdec|cpu] [--out-dir DIR]
                        [--start-sec S] [--end-sec E] [--no-preview]

For every batch of decoded frames (nvdec_reader.py; they stay on the GPU):
  1. downscale the frame and find the body centre with the SAM2-distilled
     localizer (localizer.py). It follows the body, not the implant or tether,
     and ignores a hand in the arena.
  2. smooth the track with the filter that cut the training clips. It looks
     ~0.7 s ahead, so the part network runs that far behind the decoder
     (StreamingTrack) instead of in a second pass over the video.
  3. crop a 768 px window around the track (as for the training clips),
     resize to the network's input size, predict parts.

Both networks run as TensorRT engines (trt_engines.py; built and cached on
first use, ~4 min). --backend torch is the plain PyTorch path, ~3x slower.
--decoder auto times NVDEC and cv2 on the first frames and takes the faster.

Output in <out root>/<video>/full_video/<model>/ (or --out-dir):
  labels/labels_<first frame>.npz  labels [n,S,S] uint8 part ids (as masks.npz),
                                   1000 frames per file; read with load_parts()
  frames.csv     per frame: behaviour label, crop offset (full-res px), the
                 localizer's body centre and presence score, area and mean
                 confidence of each part, whether the body touches the crop edge;
                 with a keypoint network (train_seg.py --kp-distill) also
                 kp_<node>_x / _y (full-res image px) and kp_<node>_score
                 (heatmap peak, 0-1; runs lower than SLEAP's: ~0.15-0.3 matches
                 SLEAP's 0.5 presence rate) for the six SLEAP nodes
  summary.txt    part presence by behaviour, edge contact, frames with no body
  preview.mp4    crop | overlay at 384 px, every 2nd frame (30 fps)
"""

import argparse
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

import localizer
import seg_model
from common import DEFAULT_OUT_ROOT, PART_BY_ID
from extract_clips import count_frames, load_bouts, per_frame_behaviour, smooth_track, track_radius
from nvdec_reader import area_resize, area_weights, gray_batches, pick_decoder
from render import enhance, overlay

CHUNK = 1000
MIN_PRESENCE = 0.5  # localizer presence below this: no mouse in the frame


def load_parts(out_dir, start=0, stop=None):
    """Label maps for frames [start, stop) of a run_video.py output."""
    files = sorted(Path(out_dir, "labels").glob("labels_*.npz"))
    firsts = [int(f.stem.split("_")[1]) for f in files]
    stop = stop if stop is not None else firsts[-1] + len(np.load(files[-1])["labels"])
    out = []
    for f, a in zip(files, firsts):
        if a + CHUNK <= start or a >= stop:
            continue
        lab = np.load(f)["labels"]
        out.append(lab[max(start - a, 0) : stop - a])
    return np.concatenate(out)


class StreamingTrack:
    """extract_clips.smooth_track() for a track that is still growing.

    A frame's crop offset is final once the mouse has been seen `radius` frames
    past it; the offsets are then the same as smoothing the finished track.
    Frames where the localizer sees no mouse are interpolated between the
    frames on either side that do. After `max_gap_s` of them in a row the
    track stops waiting and holds the last seen position, which bounds how
    many full frames are kept on the GPU.
    """

    def __init__(self, n, fps, crop, W, H, origin=0, max_gap_s=2.0):
        self.fps, self.crop, self.W, self.H, self.origin = fps, crop, W, H, origin
        self.radius = track_radius(fps)
        self.max_gap = int(round(max_gap_s * fps))
        self.raw = np.full((n, 2), np.nan)  # localizer centre, full-res px
        self.presence = np.zeros(n, np.float32)
        self.pos = np.empty((n, 2))  # raw with the unseen frames filled in; final for the first `known`
        self.n = self.known = 0
        self.last = None  # (frame, position) the mouse was last seen at
        self.done = False

    def _hold(self, upto):
        p = self.last[1] if self.last else np.array([self.W / 2, self.H / 2])
        self.pos[self.known : upto] = p
        self.known, self.last = upto, (upto - 1, p)

    def add(self, xy, presence):
        a, b = self.n, self.n + len(xy)
        self.raw[a:b], self.presence[a:b], self.pos[a:b] = xy, presence, xy
        self.n = b
        for k in range(a, b):
            if presence[k - a] >= MIN_PRESENCE:
                if self.known < k:  # unseen frames were waiting for this one
                    j, p = self.last or (k, xy[k - a])
                    t = np.arange(self.known, k)
                    for d in range(2):
                        self.pos[t, d] = np.interp(t, [j, k], [p[d], xy[k - a, d]])
                self.known, self.last = k + 1, (k, xy[k - a].copy())
            elif k + 1 - self.known > self.max_gap:
                self._hold(k + 1)

    def finish(self):
        if self.known < self.n:
            self._hold(self.n)
        self.done = True

    def final(self):
        """Frames before this one have their final offsets."""
        return self.origin + (self.known if self.done else max(self.known - self.radius, 0))

    def offsets(self, a, b):
        """Crop offsets (x0, y0) for frames [a, b), which must be final."""
        a, b = a - self.origin, b - self.origin
        lo, hi = max(a - self.radius, 0), min(b + self.radius, self.known)
        sm = smooth_track(self.pos[lo:hi].copy(), self.fps)[a - lo : b - lo]
        return np.stack([np.clip(np.round(sm[:, 0] - self.crop / 2), 0, self.W - self.crop),
                         np.clip(np.round(sm[:, 1] - self.crop / 2), 0, self.H - self.crop)], 1).astype(np.int32)


def resolve_model(path):
    """A seg_runs/<run> directory or a checkpoint file -> (checkpoint, name for the output folder)."""
    p = Path(path)
    if p.is_dir():
        return p / "best.pt", p.name
    return p, (p.parent.name if p.parent.parent.name == "seg_runs" else p.stem)


def load_models(backend, loc_path, ckpt, precision, max_batch):
    """-> (loc_fn, seg_fn, meta).
    loc_fn(uint8 [B,352,384] small frames) -> (xy in small-frame px [B,2], presence logit [B])
    seg_fn(uint8 [B,S,S] gray crops)       -> class logits [B,C,S,S], or (class logits, keypoint
                                              logits [B,K,S,S]) for a network with a keypoint head"""
    if backend == "trt":
        import trt_engines

        loc = trt_engines.localizer_engine(loc_path, precision=precision, max_batch=max_batch)
        seg = trt_engines.segmenter_engine(ckpt, precision=precision, max_batch=max_batch)
        print(f"TensorRT {precision}: {Path(loc.path).name}, {Path(seg.path).name}", flush=True)
        if len(seg.outputs) > 1:  # name the outputs: the engine's output order is its own
            order = [seg.outputs.index(n) for n in ("logits", "kp_logits")]

            def seg_fn(g):
                o = seg(g)
                return o[order[0]], o[order[1]]

            return loc, seg_fn, seg.meta
        return loc, seg, seg.meta
    loc = localizer.load_localizer(loc_path)
    net, ck = seg_model.load_checkpoint(ckpt)

    def loc_fn(x):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            xy, _, pl = loc(x)
        return xy, pl

    def seg_fn(g):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return net(seg_model.to_input(g), keypoints=bool(net.n_kp))

    return loc_fn, seg_fn, dict(size=ck["cfg"]["size"], class_obj_ids=ck["class_obj_ids"],
                                kp_names=list(ck.get("kp_names", [])))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--model", default=str(seg_model.WEIGHTS), help="seg_runs/<run> directory or a checkpoint file")
    ap.add_argument("--loc", default=str(localizer.WEIGHTS), help="localizer weights")
    ap.add_argument("--labels", help="behaviour label CSV (for per-behaviour summaries)")
    ap.add_argument("--out-dir", help="default: <out root>/<video stem>/full_video/<model name>")
    ap.add_argument("--crop", type=int, default=768, help="must match the training clips")
    ap.add_argument("--backend", choices=["trt", "torch"], default="trt",
                    help="trt = cached TensorRT engines; torch = PyTorch with bf16 autocast")
    ap.add_argument("--precision", choices=["fp16", "fp32"], default="fp16", help="TensorRT engine precision")
    ap.add_argument("--decoder", choices=["auto", "nvdec", "cpu"], default="cpu",
                    help="auto = the faster of NVDEC and cv2 on this video's first frames. Not the default: the NVDEC probe "
                         "takes time proportional to the file (~20 min of idle GPU on a 90 GB network video), "
                         "and cv2 won on every Rig2 video anyway")
    ap.add_argument("--start-sec", type=float, default=0)
    ap.add_argument("--end-sec", type=float)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--no-preview", action="store_true")
    args = ap.parse_args()

    video = Path(args.video)
    ckpt, name = resolve_model(args.model)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = count_frames(cap)
    cap.release()
    crop = args.crop
    out = Path(args.out_dir) if args.out_dir else DEFAULT_OUT_ROOT / video.stem / "full_video" / name
    (out / "labels").mkdir(parents=True, exist_ok=True)

    loc_fn, seg_fn, meta = load_models(args.backend, args.loc, ckpt, args.precision, args.batch)
    size = meta["size"]
    assert all(o in PART_BY_ID for o in meta["class_obj_ids"][1:]), "model's classes are not this code's parts"
    parts = [PART_BY_ID[o] for o in meta["class_obj_ids"][1:]]  # class c = parts[c - 1]
    decoder = pick_decoder(args.decoder, video)

    start = int(round(args.start_sec * fps))
    stop = min(n_frames, int(round(args.end_sec * fps))) if args.end_sec else n_frames
    start -= start % CHUNK  # chunk files are aligned to multiples of CHUNK
    beh = (per_frame_behaviour(load_bouts(args.labels, fps), n_frames) if args.labels
           else np.full(n_frames, "unlabeled", dtype=object))
    # the track filter reads `radius` frames either side, so locate that far beyond the range
    radius = track_radius(fps)
    first, last = max(start - radius, 0), min(stop + radius, n_frames)
    track = StreamingTrack(last - first, fps, crop, W, H, origin=first)

    scale = np.array([W / localizer.W, H / localizer.H])
    wy, wx = area_weights(H, localizer.H), area_weights(W, localizer.W)
    wc = area_weights(crop, size)
    kp_names = list(meta.get("kp_names", []))
    cls_to_obj = torch.tensor(list(meta["class_obj_ids"]), dtype=torch.uint8, device="cuda")
    n_cls = len(cls_to_obj)

    # CPU work per batch (csv rows, label chunks, preview) runs in order on one worker thread
    cpu, saver = ThreadPoolExecutor(1), ThreadPoolExecutor(4)
    jobs, saves = deque(), deque()
    preview = None
    if not args.no_preview:
        preview = cv2.VideoWriter(str(out / "preview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps / 2, (768, 384))
    rows, chunk = [], []

    def flush_chunk():
        if chunk:
            saves.append(saver.submit(np.savez_compressed, out / "labels" / f"labels_{rows[-len(chunk)]['frame']:06d}.npz",
                                      labels=np.stack(chunk)))
            chunk.clear()
            while len(saves) > 6:  # compression is behind: wait rather than pile up label maps
                saves.popleft().result()

    def write_batch(i0, off, imgs, labels, area, mconf, edge, kxy, ksc):
        for j in range(len(labels)):
            i = i0 + j
            k = i - first
            r = dict(frame=i, t=round(i / fps, 4), behaviour=beh[i], x0=int(off[j, 0]), y0=int(off[j, 1]),
                     cx=round(float(track.raw[k, 0]), 1), cy=round(float(track.raw[k, 1]), 1),
                     mouse_found=bool(track.presence[k] >= MIN_PRESENCE), presence=round(float(track.presence[k]), 3),
                     body_at_edge=bool(edge[j]))
            for c, p in enumerate(parts, start=1):
                r[f"area_{p.name}"] = int(area[j, c])
                r[f"conf_{p.name}"] = round(float(mconf[j, c]), 3) if area[j, c] else np.nan
            for q, n in enumerate(kp_names):  # full-res image px; score = heatmap peak (not SLEAP's scale)
                r[f"kp_{n}_x"], r[f"kp_{n}_y"] = round(float(kxy[j, q, 0]), 1), round(float(kxy[j, q, 1]), 1)
                r[f"kp_{n}_score"] = round(float(ksc[j, q]), 3)
            rows.append(r)
            chunk.append(labels[j])
            if len(chunk) == CHUNK:
                flush_chunk()
            if preview is not None and i % 2 == 0:
                im = cv2.cvtColor(enhance(imgs[j]), cv2.COLOR_GRAY2BGR)
                im = cv2.resize(im, (384, 384), interpolation=cv2.INTER_AREA)
                lb = cv2.resize(labels[j], (384, 384), interpolation=cv2.INTER_NEAREST)
                ov = overlay(im, lb)
                cv2.putText(ov, f"{i} {beh[i]}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                preview.write(np.hstack([im, ov]))

    held = deque()  # (first frame, [b,H,W] gray frames on the GPU) not yet through the part network
    nxt = start

    def segment(upto):
        nonlocal nxt
        while nxt < upto:
            n = min(args.batch, upto - nxt)
            off = track.offsets(nxt, nxt + n)
            crops = torch.empty((n, crop, crop), dtype=torch.uint8, device="cuda")
            for j in range(n):
                while held[0][0] + len(held[0][1]) <= nxt + j:
                    held.popleft()
                i0, g = held[0]
                crops[j] = g[nxt + j - i0, off[j, 1] : off[j, 1] + crop, off[j, 0] : off[j, 0] + crop]
            imgs = area_resize(crops, wc, wc)
            out = seg_fn(imgs)
            kxy = ksc = None
            if kp_names:
                out, kl = out
                kxy, ksc = seg_model.decode_keypoints(kl)
                # network px -> crop px -> full-res image px
                kxy = ((kxy + 0.5) * (crop / size) - 0.5 + torch.as_tensor(off, device="cuda")[:, None, :]).cpu().numpy()
                ksc = ksc.cpu().numpy()
            conf, cls = out.float().softmax(1).max(1)
            part = (cls + n_cls * torch.arange(n, device="cuda")[:, None, None]).flatten()  # (frame, class) bin
            area = torch.bincount(part, minlength=n * n_cls).view(n, n_cls)
            mconf = torch.bincount(part, weights=conf.flatten(), minlength=n * n_cls).view(n, n_cls) / area.clamp(min=1)
            body = cls == 1
            edge = body[:, 0].any(1) | body[:, -1].any(1) | body[:, :, 0].any(1) | body[:, :, -1].any(1)
            jobs.append(cpu.submit(write_batch, nxt, off, imgs.cpu().numpy(), cls_to_obj[cls].cpu().numpy(),
                                   area.cpu().numpy(), mconf.cpu().numpy(), edge.cpu().numpy(), kxy, ksc))
            while len(jobs) > 8:  # the CPU side is behind: wait for it
                jobs.popleft().result()
            if (nxt - start) // 5000 != (nxt + n - start) // 5000:
                print(f"  {nxt + n}/{stop}  {(nxt + n - start) / (time.time() - t0):.0f} fps", flush=True)
            nxt += n

    print(f"{video.name}: {W}x{H}, {n_frames} frames, {fps:.2f} fps; "
          f"frames {start}-{stop - 1} ({(stop - start) / fps / 60:.1f} min) ...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        for i0, gray in gray_batches(video, args.batch, first, last, decoder):
            xy, pl = loc_fn(area_resize(gray, wy, wx))
            track.add(xy.float().cpu().numpy() * scale, torch.sigmoid(pl.float()).cpu().numpy())
            held.append((i0, gray))
            segment(min(track.final(), stop))
        track.finish()
        if first + track.n < last:
            print(f"WARNING: the decoder stopped at frame {first + track.n}, expected {last}")
        stop = min(stop, first + track.n)
        segment(stop)
    while jobs:
        jobs.popleft().result()
    flush_chunk()
    while saves:
        saves.popleft().result()
    if preview is not None:
        preview.release()
    dt = time.time() - t0

    df = pd.DataFrame(rows)
    df.to_csv(out / "frames.csv", index=False)
    names = [p.name for p in parts]
    pres = df.groupby("behaviour")[[f"area_{n}" for n in names]].agg(lambda a: (a > 0).mean())
    pres.columns = names
    pres.insert(0, "frames", df.groupby("behaviour").size())
    lines = [
        f"video: {video}", f"model: {name} ({ckpt})",
        f"backend: {args.backend}{' ' + args.precision if args.backend == 'trt' else ' bf16'}, decoder: {decoder}",
        f"frames {start}-{stop - 1}: {len(df)} in {dt:.0f}s ({len(df) / dt:.0f} fps)", "",
        f"no mouse seen by the localizer (track interpolated): {(~df.mouse_found).mean():.2%}",
        f"no body predicted: {(df.area_body == 0).mean():.2%}",
        f"body touching the crop edge (mouse may be cut off): {df.body_at_edge.mean():.2%}", "",
        "part presence (share of frames with a non-empty mask) by behaviour:",
        pres.to_string(float_format=lambda v: f"{v:.2f}"), "",
        "median part area (px at network resolution) when present:",
        pd.Series({n: df.loc[df[f'area_{n}'] > 0, f'area_{n}'].median() for n in names}).to_string(),
    ]
    (out / "summary.txt").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
