"""Run the lab's SLEAP model (the distillation teacher) on a sample of a session's frames.

    ~/.local/share/uv/tools/sleap/bin/python sleap_teacher.py RUN_DIR [--chunks 900] [--chunk-len 20]
                                            [--frames-json FILE --name NAME]

Runs in the SLEAP tool env (not `sam3`). Picks --chunks runs of --chunk-len consecutive frames,
spread at random over the frames where run_video.py found the mouse (consecutive frames keep the
video reader sequential; seeks in these .mkv were checked frame-accurate against the part masks).
The full top-down pipeline (ConvNeXt-T centroid + Swin-S instance, the 1405-label models) runs in
parts of --part frames, each saved as it finishes, so an interrupted run resumes.

--frames-json takes a chosen frame list instead ({"frames": [...]}, e.g. mined grooming frames).

Output: RUN_DIR/../../sleap/<NAME, default teacher>/part_<k>.slp and frames.json (the frame list).
distill_frames.py turns them into a training set.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = Path("/mnt/howlin/mnt/data/samba/yttri/Aden/Sleap/models/hpo/retrain")
TEACHER = [str(MODELS / "centroid_convnext_tiny_1405lbl"), str(MODELS / "swint_small_imagenet_1405lbl")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--chunks", type=int, default=900)
    ap.add_argument("--chunk-len", type=int, default=20)
    ap.add_argument("--part", type=int, default=1000, help="frames per saved .slp")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frames-json", help="predict these frames ({'frames': [...]}) instead of a random sample")
    ap.add_argument("--name", default="teacher", help="output folder in <session>/sleap/")
    args = ap.parse_args()

    run = Path(args.run_dir)
    out = run.parent.parent / "sleap" / args.name
    out.mkdir(parents=True, exist_ok=True)
    video = next(l.split(": ", 1)[1].strip() for l in open(run / "summary.txt") if l.startswith("video: "))
    fl = out / "frames.json"
    if fl.exists():
        frames = json.load(open(fl))["frames"]
    elif args.frames_json:
        frames = sorted(int(f) for f in json.load(open(args.frames_json))["frames"])
        json.dump(dict(video=video, frames=frames, source=args.frames_json), open(fl, "w"))
    else:
        found = pd.read_csv(run / "frames.csv", usecols=["mouse_found"]).mouse_found.to_numpy()
        L = args.chunk_len
        cs = np.r_[0, np.cumsum(~found)]
        ok = np.flatnonzero(cs[L:] - cs[:-L] == 0)  # chunk starts with the mouse in all L frames
        rng = np.random.default_rng(args.seed)
        starts = np.sort(rng.choice(ok, size=min(args.chunks, len(ok)), replace=False))
        starts = starts[np.r_[True, np.diff(starts) >= L]]
        frames = sorted({int(s + i) for s in starts for i in range(L)})
        json.dump(dict(video=video, frames=frames, chunk_len=L, seed=args.seed), open(fl, "w"))
    print(f"{len(frames)} frames from {video}", flush=True)

    from sleap_nn.legacy_predict import run_inference

    for k, a in enumerate(range(0, len(frames), args.part)):
        dst = out / f"part_{k:03d}.slp"
        if dst.exists():
            continue
        run_inference(data_path=video, model_paths=TEACHER, frames=frames[a : a + args.part], max_instances=1,
                      batch_size=1, peak_threshold=0.2, output_path=str(dst), device="cuda")
        print(f"part {k}: frames {a}-{min(a + args.part, len(frames))} done", flush=True)


if __name__ == "__main__":
    main()
