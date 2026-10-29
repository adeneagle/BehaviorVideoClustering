"""Gray video frames as uint8 CUDA tensors, decoded by NVDEC or by cv2.

Adapted from the lab pipeline (Aden/code/video_processing/nvdec_reader.py).
Everything in this project was trained on `cv2.VideoCapture(...).read()[..., 0]`:
the blue channel of swscale's yuv420p -> BGR conversion. The Basler videos are
monochrome h264 (chroma is a constant 128), so that channel is a pure function
of luma: B = CV2_GRAY_LUT[Y], swscale's limited -> full range stretch. Both
decoders therefore deliver the luma plane and the table is applied on the GPU:

  nvdec  NVDEC -> NV12 luma (already on the GPU) -> table
  cpu    cv2 with CAP_PROP_CONVERT_RGB off (it then returns the luma plane and
         skips the BGR conversion, ~2x faster than a normal read) -> upload -> table

Frames stay on the GPU; area_resize() reproduces cv2's INTER_AREA there (the
same pixels as cv2.resize for the sizes used here).

    python nvdec_reader.py validate VIDEO   # every frame of both decoders vs a plain cv2 read
    python nvdec_reader.py bench VIDEO      # decode speed of both
"""

import argparse
import queue
import threading
import time

import cv2
import numpy as np
import torch

# swscale yuv420p (tv range) -> bgr24, any channel, at U=V=128. derive_cv2_lut() regenerates it.
CV2_GRAY_LUT = np.array(
    [0] * 17 + [
     1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33,
     34, 36, 37, 38, 39, 40, 41, 43, 44, 45, 46, 47, 48, 50, 51, 52, 53, 54, 55, 57, 58, 59, 60, 61, 62, 64, 65,
     66, 67, 68, 69, 71, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 83, 85, 86, 87, 88, 89, 90, 91, 93, 94, 95, 96,
     97, 98, 100, 101, 102, 103, 104, 105, 107, 108, 109, 110, 111, 112, 114, 115, 116, 117, 118, 119, 121, 122,
     123, 124, 125, 126, 128, 129, 130, 131, 132, 133, 135, 136, 137, 138, 139, 140, 142, 143, 144, 145, 146,
     147, 149, 150, 151, 152, 153, 154, 156, 157, 158, 159, 160, 161, 163, 164, 165, 166, 167, 168, 170, 171,
     172, 173, 174, 175, 176, 178, 179, 180, 181, 182, 183, 185, 186, 187, 188, 189, 190, 192, 193, 194, 195,
     196, 197, 199, 200, 201, 202, 203, 204, 206, 207, 208, 209, 210, 211, 213, 214, 215, 216, 217, 218, 220,
     221, 222, 223, 224, 225, 227, 228, 229, 230, 231, 232, 234, 235, 236, 237, 238, 239, 241, 242, 243, 244,
     245, 246, 248, 249, 250, 251, 252, 253] + [255] * 21, dtype=np.uint8)
assert CV2_GRAY_LUT.shape == (256,)


class NotMonochrome(RuntimeError):
    """The video has real chroma, so cv2's channel is not a function of luma."""


def derive_cv2_lut(workdir="/tmp"):
    """Regenerate CV2_GRAY_LUT: losslessly encode a 0..255 luma ramp (chroma 128), read it with cv2."""
    import os
    import subprocess

    w, h = 256, 64
    frame = np.concatenate([np.tile(np.arange(256, dtype=np.uint8), h), np.full(w * h // 2, 128, np.uint8)])
    path = os.path.join(workdir, "luma_ramp.mkv")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{w}x{h}",
                    "-i", "-", "-frames:v", "1", "-c:v", "libx264", "-qp", "0", "-color_range", "tv", path],
                   input=frame.tobytes(), check=True)
    ok, f = cv2.VideoCapture(path).read()
    return f[h // 2, :, 0].copy()


def area_weights(n_in, n_out, device="cuda"):
    """[n_out, n_in] weights of cv2's INTER_AREA along one axis (its computeResizeAreaTab)."""
    scale = n_in / n_out
    w = np.zeros((n_out, n_in), np.float32)
    for d in range(n_out):
        f1 = d * scale
        f2 = f1 + scale
        cell = min(scale, n_in - f1)
        s2 = min(int(np.floor(f2)), n_in - 1)
        s1 = min(int(np.ceil(f1)), s2)
        if s1 - f1 > 1e-3:
            w[d, s1 - 1] = (s1 - f1) / cell
        w[d, s1:s2] = 1.0 / cell
        if f2 - s2 > 1e-3:
            w[d, s2] = min(f2 - s2, 1.0, cell) / cell
    return torch.from_numpy(w).to(device)


def area_resize(x_u8, wy, wx):
    """[B,H,W] uint8 -> [B,h,w] uint8, as cv2.resize(..., interpolation=cv2.INTER_AREA)."""
    return (wy @ x_u8.float() @ wx.T).round_().clamp_(0, 255).byte()


def _apply_lut(lut, luma, out):
    for i in range(len(luma)):  # per frame: the int32 index of a whole batch would be > 1 GB
        torch.index_select(lut, 0, luma[i].reshape(-1).int(), out=out[i].view(-1))


def _produce_nvdec(video, batch, start, stop, put):
    import PyNvVideoCodec as nvc

    dec = nvc.ThreadedDecoder(str(video), buffer_size=2 * batch, gpu_id=0,
                              output_color_type=nvc.OutputColorType.NATIVE)
    meta = dec.get_stream_metadata()
    lut = torch.from_numpy(CV2_GRAY_LUT).cuda()
    stream = torch.cuda.Stream()
    n = 0
    try:
        with torch.cuda.stream(stream):
            while stop is None or n < stop:
                want = batch if n >= start else min(batch, start - n)  # frames before start are dropped
                want = want if stop is None else min(want, stop - n)
                frames = dec.get_batch_frames(want)
                if not frames:
                    break
                if n >= start:
                    out = torch.empty((len(frames), meta.height, meta.width), dtype=torch.uint8, device="cuda")
                    for i, fr in enumerate(frames):
                        if fr.format != nvc.Pixel_Format.NV12:
                            raise NotMonochrome(f"decoder output is {fr.format}, expected NV12")
                        y_plane, uv_plane = fr.cuda()
                        if i == 0 and not bool((torch.as_tensor(uv_plane, device="cuda") == 128).all()):
                            raise NotMonochrome(f"frame {n}: chroma is not a constant 128 (colour video?)")
                        _apply_lut(lut, torch.as_tensor(y_plane, device="cuda")[None, ..., 0], out[i : i + 1])
                    stream.synchronize()  # the decoder reuses its surfaces on the next call
                    if not put((n, out)):
                        break
                n += len(frames)
                if len(frames) < want:
                    break
    finally:
        dec.end()


def luma_read_ok(video, n=4):
    """Does cv2 with CAP_PROP_CONVERT_RGB off return the luma plane, and does the table turn it
    into a normal read's channel? (False for colour videos and for builds that ignore the flag.)"""
    ref, raw = cv2.VideoCapture(str(video)), cv2.VideoCapture(str(video))
    if not raw.set(cv2.CAP_PROP_CONVERT_RGB, 0):
        return False
    for _ in range(n):
        (ok, f), (ok2, y) = ref.read(), raw.read()
        if not ok:
            break
        if not ok2 or y.dtype != np.uint8 or y.shape != f.shape[:2] or not np.array_equal(CV2_GRAY_LUT[y], f[..., 0]):
            return False
    return True


def _produce_cv2(video, batch, start, stop, put):
    luma = luma_read_ok(video)
    cap = cv2.VideoCapture(str(video))
    if luma:
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    host = torch.empty((batch, h, w), dtype=torch.uint8).pin_memory()
    host_np = host.numpy()
    lut = torch.from_numpy(CV2_GRAY_LUT).cuda()
    n = start
    while stop is None or n < stop:
        want = batch if stop is None else min(batch, stop - n)
        b = 0
        while b < want:
            # luma frames are decoded straight into the pinned buffer (saves a 4.6 MB copy each)
            ok, f = cap.read(host_np[b]) if luma else cap.read()
            if not ok:
                break
            if not np.shares_memory(f, host_np):
                host_np[b] = f if luma else f[..., 0]
            b += 1
        if not b:
            break
        g = host[:b].cuda()
        if luma:
            out = torch.empty_like(g)
            _apply_lut(lut, g, out)
            g = out
        if not put((n, g)):
            break
        n += b
        if b < want:
            break


def gray_batches(video, batch, start=0, stop=None, decoder="nvdec", depth=2):
    """Yield (index of the first frame, [b,H,W] uint8 CUDA tensor) for frames [start, stop).

    Decoding runs on a background thread, `depth` batches ahead. Ends early if
    the video has fewer than `stop` frames. cv2 seeks to `start`; NVDEC's seek
    is not frame-accurate in .mkv, so it decodes from frame 0 and drops frames."""
    q, cancel = queue.Queue(maxsize=depth), threading.Event()

    def put(item):
        while not cancel.is_set():
            try:
                q.put(item, timeout=0.2)
                return True
            except queue.Full:
                pass
        return False

    def produce():
        try:
            (_produce_nvdec if decoder == "nvdec" else _produce_cv2)(video, batch, start, stop, put)
            put(None)
        except BaseException as e:  # surface decode errors in the consumer
            put(e)

    threading.Thread(target=produce, daemon=True).start()
    try:
        while (item := q.get()) is not None:
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        cancel.set()


def read_plain(cap, n):
    """n frames the way the training data was read."""
    out = []
    for _ in range(n):
        ok, f = cap.read()
        if not ok:
            break
        out.append(f[..., 0])
    return out


def check_matches_cv2(video, decoder="nvdec", n=16):
    """The first n frames against a plain cv2 read -> (identical, message)."""
    ref = read_plain(cv2.VideoCapture(str(video)), n)
    try:
        got = [f for _, g in gray_batches(video, len(ref), 0, len(ref), decoder) for f in g.cpu().numpy()]
    except NotMonochrome as e:
        return False, str(e)
    if len(got) != len(ref):
        return False, f"{decoder} returned {len(got)} of the first {len(ref)} frames"
    bad = [i for i, (a, b) in enumerate(zip(got, ref)) if not np.array_equal(a, b)]
    return not bad, (f"first {len(ref)} frames identical to cv2" if not bad else f"frames {bad} differ from cv2")


def decode_fps(video, decoder, n=640, batch=64):
    t0 = time.time()
    got = sum(len(g) for _, g in gray_batches(video, batch, 0, n, decoder))
    torch.cuda.synchronize()
    return got / (time.time() - t0)


def pick_decoder(decoder, video):
    """'auto' -> whichever of nvdec / cpu decodes this video faster on this machine; nvdec only
    counts if it gives the same pixels as cv2. (One NVDEC engine is slower than a many-core CPU
    for these 4.6 MP frames on an RTX 3090; newer cards have more of them.)"""
    if decoder == "cpu":
        return "cpu"
    try:
        ok, msg = check_matches_cv2(video, "nvdec")
    except Exception as e:  # no NVDEC / PyNvVideoCodec / unsupported codec
        ok, msg = False, f"{type(e).__name__}: {e}"
    if not ok:
        if decoder == "nvdec":
            raise RuntimeError(f"--decoder nvdec unusable for {video}: {msg}")
        print(f"decoder: cv2 (NVDEC unusable: {msg})", flush=True)
        return "cpu"
    if decoder == "nvdec":
        print(f"decoder: NVDEC ({msg})", flush=True)
        return "nvdec"
    fps = {d: decode_fps(video, d) for d in ("nvdec", "cpu")}
    best = max(fps, key=fps.get)
    print(f"decoder: {best} (NVDEC {fps['nvdec']:.0f} fps, cv2 {fps['cpu']:.0f} fps on the first frames; "
          f"NVDEC {msg})", flush=True)
    return best


def cmd_validate(args):
    """Every frame of each decoder against a plain cv2 read, and the frame counts."""
    bad = False
    for decoder in args.decoders:
        cap = cv2.VideoCapture(args.video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
        t0, n, differ = time.time(), 0, 0
        for i0, g in gray_batches(args.video, 64, args.start, args.max_frames and args.start + args.max_frames,
                                  decoder):
            ref = read_plain(cap, len(g))
            if len(ref) < len(g):
                raise SystemExit(f"{decoder}: cv2 ran out at frame {i0 + len(ref)} but {decoder} kept going")
            differ += int((g != torch.from_numpy(np.stack(ref)).cuda()).flatten(1).any(1).sum())
            n += len(g)
            if n % 6400 < 64:
                print(f"  {decoder}: {n} frames, {n / (time.time() - t0):.0f} fps", flush=True)
        extra = 0 if args.max_frames else len(read_plain(cap, 10**9))
        print(f"{decoder}: compared {n} frames from {args.start}: {differ} differ; "
              f"cv2 had {extra} frames beyond the last")
        bad |= bool(differ or extra)
    if bad:
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("validate", "bench", "lut"):
        p = sub.add_parser(name)
        if name != "lut":
            p.add_argument("video")
            p.add_argument("--decoders", nargs="+", default=["nvdec", "cpu"], choices=["nvdec", "cpu"])
            p.add_argument("--start", type=int, default=0)
            p.add_argument("--max-frames", type=int, default=None if name == "validate" else 6400)
    args = ap.parse_args()
    if args.cmd == "lut":
        lut = derive_cv2_lut()
        print("matches CV2_GRAY_LUT:", np.array_equal(lut, CV2_GRAY_LUT))
        print(lut.tolist())
    elif args.cmd == "validate":
        cmd_validate(args)
    else:
        for decoder in args.decoders:
            print(f"{decoder}: {decode_fps(args.video, decoder, args.max_frames):.0f} fps "
                  f"(decode + table, frames left on the GPU)")


if __name__ == "__main__":
    main()
