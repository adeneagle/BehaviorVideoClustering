"""Headless: replay each clip's prompts.json, propagate, save masks + videos.

Lets you click points for every clip in annotate.py (no need to wait for
propagation there), then track everything in one batch:

    python propagate.py                 # every clip with prompts but no masks
    python propagate.py --all           # redo every clip with prompts
    python propagate.py --clip 012510_locomotion

prompts.json is the only input, so automatically generated prompts (e.g.
from SLEAP keypoints) go through exactly the same path.
"""

import argparse
import time
from pathlib import Path

from common import DEFAULT_OUT_ROOT, PARTS, read_json, save_masks, write_json
from render import contact_sheet, write_videos
from sam_session import PartSession, build_tracker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--video-stem")
    ap.add_argument("--clip", action="append", help="clip id (repeatable)")
    ap.add_argument("--all", action="store_true", help="re-run clips that already have masks")
    ap.add_argument("--no-videos", action="store_true")
    args = ap.parse_args()

    root = Path(args.out_root)
    stem = args.video_stem or sorted(p.name for p in root.iterdir() if (p / "clips").is_dir())[0]
    clips = sorted((root / stem / "clips").iterdir())
    if args.clip:
        clips = [c for c in clips if c.name in args.clip]
    todo = [c for c in clips if (c / "prompts.json").exists()
            and (args.all or args.clip or not (c / "masks.npz").exists())]
    skipped = [c.name for c in clips if not (c / "prompts.json").exists()]
    if skipped:
        print(f"no prompts yet (skipped): {', '.join(skipped)}")
    if not todo:
        print("nothing to do")
        return

    tracker = build_tracker()
    obj_ids = [p.obj_id for p in PARTS]
    for c in todo:
        t0 = time.time()
        prompts = read_json(c / "prompts.json")
        events = prompts["events"]
        if not any(e["op"] == "points" for e in events):
            print(f"{c.name}: prompts.json has no clicks, skipping")
            continue
        s = PartSession(tracker, c / "frames", obj_ids)
        s.replay(events)
        if s.result is None or s.stale:  # clicks after the last propagate (or none yet)
            s.propagate()
            prompts.update(events=s.events,
                           n_propagations=sum(e["op"] == "propagate" for e in s.events),
                           prompted_frames=s.prompted_frames())
            write_json(c / "prompts.json", prompts)
        r = s.result
        save_masks(c, r["labels"], r["obj_scores"], r["raw_overlap"], obj_ids)
        if not args.no_videos:
            write_videos(c, r["labels"], read_json(c / "meta.json")["fps"])
        contact_sheet(c, r["labels"])
        del s
        print(f"{c.name}: done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
