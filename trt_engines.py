"""TensorRT engines for the body localizer and the part-segmentation network.

Adapted from the lab pipeline (Aden/code/video_processing/trt_engines.py).
Each model is exported to ONNX (dynamic batch) and compiled to a serialized
engine. Engines are drop-in replacements for the torch calls in run_video.py:

    loc = localizer_engine()    # (B,352,384) uint8 small frame -> xy_small (B,2) f32, pres_logit (B,) f32
    seg = segmenter_engine()    # (B,S,S) uint8 gray crop       -> class logits (B,C,S,S) f32
                                #   (+ keypoint logits (B,K,S,S) f32 for a network with a keypoint head)

Both take the uint8 image; the normalisation is inside the engine.

An engine is only valid for one GPU architecture + TensorRT version + weights
file, so the cache filename encodes all three (weights by content hash): a
retrained model, another GPU or a TensorRT upgrade triggers a rebuild (~1 min
each) instead of loading a stale engine. The cache is `trt_engines/` next to
the weights file.

    python trt_engines.py build                      # build / reuse the engines for the default weights
    python trt_engines.py validate VIDEO [--n 512]   # TensorRT and bf16 torch vs fp32 torch on real frames
    python trt_engines.py bench                      # throughput, torch vs TensorRT

Needs (pip): tensorrt-cu12 onnx onnxscript
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import localizer
import seg_model

WORKSPACE_GB = 4
_LOGGER = None


def _logger():
    """One process-wide TRT logger (TRT warns and ignores any second one)."""
    global _LOGGER
    if _LOGGER is None:
        import tensorrt as trt

        _LOGGER = trt.Logger(trt.Logger.WARNING)
    return _LOGGER


# ── runtime ─────────────────────────────────────────────────────────────────

class TRTModule:
    """Executes a serialized engine on torch CUDA tensors.

    Runs on its own stream (TRT adds syncs when enqueued on the legacy default stream),
    fenced both ways against torch's current stream, so callers see ordinary stream
    semantics: outputs are fresh torch tensors and `.cpu()` afterwards syncs as usual.
    """

    def __init__(self, engine_path):
        import tensorrt as trt

        engine_path = str(engine_path)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(_logger()).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"could not deserialize {engine_path} (TRT version / GPU mismatch?)")
        self.context = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        dtypes = {trt.float32: torch.float32, trt.float16: torch.float16, trt.uint8: torch.uint8,
                  trt.int32: torch.int32, trt.int64: torch.int64, trt.bool: torch.bool}
        self.dtypes = {n: dtypes[self.engine.get_tensor_dtype(n)] for n in names}
        self.path = engine_path
        self.stream = torch.cuda.Stream()
        meta = engine_path[: -len(".engine")] + ".json"  # written at build: source weights, input size, ...
        self.meta = json.load(open(meta)) if os.path.exists(meta) else {}

    def __call__(self, *xs):
        if len(xs) != len(self.inputs):
            raise ValueError(f"{os.path.basename(self.path)} takes {len(self.inputs)} inputs, got {len(xs)}")
        for name, x in zip(self.inputs, xs):
            if x.dtype != self.dtypes[name] or not x.is_cuda:
                raise TypeError(f"input {name!r}: engine wants cuda {self.dtypes[name]}, got {x.device} {x.dtype}")
            x = x.contiguous()
            self.context.set_input_shape(name, tuple(x.shape))
            self.context.set_tensor_address(name, x.data_ptr())
        outs = []
        for name in self.outputs:
            out = torch.empty(tuple(self.context.get_tensor_shape(name)), dtype=self.dtypes[name], device="cuda")
            self.context.set_tensor_address(name, out.data_ptr())
            outs.append(out)
        # Inputs/outputs live on the caller's stream; the caller's stream waits for the engine
        # before returning, so freeing/reusing these tensors afterwards is ordered correctly.
        cur = torch.cuda.current_stream()
        self.stream.wait_stream(cur)
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError(f"TensorRT execution failed for {self.path}")
        cur.wait_stream(self.stream)
        return outs[0] if len(outs) == 1 else tuple(outs)


# ── export wrappers ─────────────────────────────────────────────────────────

class LocalizerExport(nn.Module):
    """Localizer2 minus the heatmap output (only xy + presence are used)."""

    def __init__(self, loc):
        super().__init__()
        self.loc = loc

    def forward(self, x_u8):
        xy, _, pres = self.loc(x_u8)
        return xy, pres


class SegmenterExport(nn.Module):
    """PartSegNet on the uint8 gray crop: seg_model.to_input() without its in-place ops."""

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer("mean", seg_model.IMG_MEAN.clone())
        self.register_buffer("std", seg_model.IMG_STD.clone())

    def forward(self, gray_u8):
        x = (gray_u8.float() / 255.0).unsqueeze(1).expand(-1, 3, -1, -1)
        return self.net((x - self.mean) / self.std, keypoints=bool(self.net.n_kp))


# ── build / cache ───────────────────────────────────────────────────────────

def _digest(path, n=12):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()[:n]


def engine_path(name, weights_path, precision, max_batch):
    import tensorrt as trt

    major, minor = torch.cuda.get_device_capability()
    tag = f"{name}_{_digest(weights_path)}_{precision}_b{max_batch}_sm{major}{minor}_trt{trt.__version__}"
    return str(Path(weights_path).parent / "trt_engines" / (tag + ".engine"))


def export_onnx(module, example, onnx_path, input_name, output_names, max_batch):
    batch = torch.export.Dim("batch", min=1, max=max_batch)
    with torch.no_grad():
        torch.onnx.export(module, (example,), onnx_path, input_names=[input_name],
                          output_names=output_names, dynamic_shapes=({0: batch},),
                          dynamo=True, external_data=False)


def build_engine(onnx_path, out_path, input_shape, precision, max_batch):
    """Compile ONNX -> serialized engine. input_shape excludes the (dynamic) batch dim."""
    import tensorrt as trt

    builder = trt.Builder(_logger())
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, _logger())
    if not parser.parse_from_file(onnx_path):
        errs = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT could not parse {onnx_path}: {errs}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_GB << 30)
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision != "fp32":
        raise ValueError(f"precision must be fp16 or fp32, got {precision!r}")
    profile = builder.create_optimization_profile()
    name = network.get_input(0).name
    profile.set_shape(name, (1, *input_shape), (max_batch, *input_shape), (max_batch, *input_shape))
    config.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT engine build failed for {onnx_path}")
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(serialized)
    os.replace(tmp, out_path)  # atomic: a killed build never leaves a truncated engine in the cache


def _get_engine(name, weights_path, precision, max_batch, rebuild, make_export):
    """make_export() -> (module, example_input, input_name, output_names, input_shape, meta)."""
    if not torch.cuda.is_available():
        raise RuntimeError("the TensorRT backend needs a CUDA GPU (use --backend torch)")
    path = engine_path(name, weights_path, precision, max_batch)
    if rebuild or not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        module, example, in_name, out_names, in_shape, meta = make_export()
        onnx_path = path[: -len(".engine")] + ".onnx"
        t0 = time.time()
        print(f"[trt] exporting {name} -> {os.path.basename(onnx_path)}", flush=True)
        export_onnx(module, example, onnx_path, in_name, out_names, max_batch)
        print(f"[trt] building {precision} engine (max batch {max_batch}), one-time, ~1 min", flush=True)
        build_engine(onnx_path, path, in_shape, precision, max_batch)
        os.remove(onnx_path)
        json.dump(dict(meta, weights=os.path.abspath(weights_path), precision=precision, max_batch=max_batch,
                       gpu=torch.cuda.get_device_name(0)),
                  open(path[: -len(".engine")] + ".json", "w"), indent=1)
        print(f"[trt] built {os.path.basename(path)} in {time.time() - t0:.0f}s", flush=True)
    return TRTModule(path)


def localizer_engine(loc_path=None, *, precision="fp16", max_batch=64, rebuild=False):
    loc_path = str(loc_path or localizer.WEIGHTS)

    def make():
        loc = localizer.load_localizer(loc_path)
        example = torch.randint(0, 256, (2, localizer.H, localizer.W), dtype=torch.uint8, device="cuda")
        return (LocalizerExport(loc).eval(), example, "frames_u8", ["xy_small", "pres_logit"],
                (localizer.H, localizer.W), dict(model="Localizer2"))

    return _get_engine("loc2", loc_path, precision, max_batch, rebuild, make)


def segmenter_engine(ckpt_path=None, *, precision="fp16", max_batch=64, rebuild=False):
    ckpt_path = str(ckpt_path or seg_model.WEIGHTS)

    def make():
        net, ck = seg_model.load_checkpoint(ckpt_path)
        size = ck["cfg"]["size"]
        example = torch.randint(0, 256, (2, size, size), dtype=torch.uint8, device="cuda")
        outs = ["logits", "kp_logits"] if net.n_kp else ["logits"]
        return (SegmenterExport(net).cuda().eval(), example, "crops_u8", outs, (size, size),
                dict(model="PartSegNet", encoder=ck["cfg"]["encoder"], size=size,
                     class_names=list(ck["class_names"]), class_obj_ids=list(ck["class_obj_ids"]),
                     kp_names=list(ck.get("kp_names", []))))

    return _get_engine("partseg", ckpt_path, precision, max_batch, rebuild, make)


# ── CLI: build / validate / bench ───────────────────────────────────────────

def _real_inputs(video, n, crop, size, loc):
    """n frames from 8 places in the video -> (small frames, part-network crops around the body, native scale)."""
    import cv2

    from nvdec_reader import area_resize, area_weights, gray_batches

    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) * 0.98)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    wy, wx = area_weights(H, localizer.H), area_weights(W, localizer.W)
    wc = area_weights(crop, size)
    smalls, crops = [], []
    per = max(1, n // 8)
    for s in np.linspace(0, total - per, 8).astype(int):
        for _, g in gray_batches(video, per, int(s), int(s) + per, "cpu"):
            small = area_resize(g, wy, wx)
            with torch.no_grad():
                xy = loc(small)[0].float().cpu().numpy() * [W / localizer.W, H / localizer.H]
            x0 = np.clip(np.round(xy[:, 0] - crop / 2), 0, W - crop).astype(int)
            y0 = np.clip(np.round(xy[:, 1] - crop / 2), 0, H - crop).astype(int)
            c = torch.stack([g[i, y0[i] : y0[i] + crop, x0[i] : x0[i] + crop] for i in range(len(g))])
            smalls.append(small)
            crops.append(area_resize(c, wc, wc))
    return torch.cat(smalls), torch.cat(crops), np.array([W / localizer.W, H / localizer.H])


def _batched(fn, x, bs):
    with torch.no_grad():
        outs = [fn(x[i : i + bs]) for i in range(0, len(x), bs)]
    if isinstance(outs[0], tuple):
        return [torch.cat([o[k].float() for o in outs]) for k in range(len(outs[0]))]
    return torch.cat([o.float() for o in outs])


def cmd_validate(args):
    bs = args.max_batch
    loc = localizer.load_localizer(args.loc)
    net, ck = seg_model.load_checkpoint(args.model)
    loc_ref = lambda x: (lambda xy, _, pl: (xy, pl))(*loc(x))
    seg_ref = lambda g: net(seg_model.to_input(g))

    def bf16(fn):
        def run(x):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return fn(x)
        return run

    small, crops, scale = _real_inputs(args.video, args.n, args.crop, ck["cfg"]["size"], loc_ref)
    report = {"frames": len(small), "precision": args.precision, "localizer": {}, "segmenter": {}}

    xy0, pl0 = [t.cpu().numpy() for t in _batched(loc_ref, small, bs)]
    cls0 = _batched(lambda g: seg_ref(g).argmax(1).byte(), crops, bs)
    names = ck["class_names"]
    paths = [("torch bf16 autocast (the path used until now)", bf16(loc_ref), bf16(seg_ref)),
             (f"tensorrt {args.precision}",
              localizer_engine(args.loc, precision=args.precision, max_batch=bs),
              segmenter_engine(args.model, precision=args.precision, max_batch=bs))]
    for tag, loc_fn, seg_fn in paths:
        xy, pl = [t.cpu().numpy() for t in _batched(loc_fn, small, bs)]
        err = np.linalg.norm((xy - xy0) * scale, axis=1)
        p, p0 = 1 / (1 + np.exp(-pl)), 1 / (1 + np.exp(-pl0))
        report["localizer"][tag] = dict(
            xy_err_native_px_median=round(float(np.median(err)), 3),
            xy_err_native_px_max=round(float(err.max()), 3),
            presence_absdiff_max=round(float(np.abs(p - p0).max()), 5))
        cls = _batched(lambda g: seg_fn(g).argmax(1).byte(), crops, bs)
        iou = {}
        for c, name in enumerate(names):
            a, b = cls == c, cls0 == c
            iou[name] = round(float((a & b).sum() / (a | b).sum().clamp(min=1)), 5)
        report["segmenter"][tag] = dict(
            pixels_differing=float((cls != cls0).float().mean()),
            frames_with_over_50_px_differing=float(((cls != cls0).flatten(1).sum(1) > 50).float().mean()),
            iou_vs_fp32=iou)
    print(json.dumps(report, indent=1))


def _bench(fn, n=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


def cmd_bench(args):
    bs = args.max_batch
    loc = localizer.load_localizer(args.loc)
    net, ck = seg_model.load_checkpoint(args.model)
    size = ck["cfg"]["size"]
    small = torch.randint(0, 256, (bs, localizer.H, localizer.W), dtype=torch.uint8, device="cuda")
    crops = torch.randint(0, 256, (bs, size, size), dtype=torch.uint8, device="cuda")
    rows = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        rows.append(("localizer torch bf16", _bench(lambda: loc(small))))
        rows.append(("segmenter torch bf16", _bench(lambda: net(seg_model.to_input(crops)))))
    loc_e = localizer_engine(args.loc, precision=args.precision, max_batch=bs)
    seg_e = segmenter_engine(args.model, precision=args.precision, max_batch=bs)
    rows.append((f"localizer tensorrt {args.precision}", _bench(lambda: loc_e(small))))
    rows.append((f"segmenter tensorrt {args.precision}", _bench(lambda: seg_e(crops))))
    for name, t in rows:
        print(f"{name:28s} batch {bs}: {t * 1e3:7.2f} ms  {bs / t:7.0f} frames/s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("build", "validate", "bench"):
        p = sub.add_parser(name)
        p.add_argument("--loc", default=str(localizer.WEIGHTS))
        p.add_argument("--model", default=str(seg_model.WEIGHTS), help="part-network checkpoint")
        p.add_argument("--max-batch", type=int, default=64)
        p.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
        if name == "build":
            p.add_argument("--rebuild", action="store_true")
        if name == "validate":
            p.add_argument("video")
            p.add_argument("--n", type=int, default=512)
            p.add_argument("--crop", type=int, default=768)
    args = ap.parse_args()
    if args.cmd == "build":
        for e in (localizer_engine(args.loc, precision=args.precision, max_batch=args.max_batch, rebuild=args.rebuild),
                  segmenter_engine(args.model, precision=args.precision, max_batch=args.max_batch, rebuild=args.rebuild)):
            print(f"[trt] ready: {e.path}  inputs={e.inputs} outputs={e.outputs}")
    elif args.cmd == "validate":
        cmd_validate(args)
    else:
        cmd_bench(args)


if __name__ == "__main__":
    main()
