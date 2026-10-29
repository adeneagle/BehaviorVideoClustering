"""Click-prompt, propagate, correct and review part masks, clip by clip.

    python annotate.py [--video-stem STEM] [--clip CLIP_ID]

Workflow per clip
  1. Pick a frame where all parts are visible (a/d, arrows, slider).
  2. For each part: press its number, left-click on it (+), right-click (-)
     to cut away what it should not include. The preview updates per click.
     2 body = head + trunk as one region (right-click the tail and implant
     off it). There is no part 1 (the snout part was retired).
     Tail (7) and implant (8) are optional: prompt them only if they keep
     leaking into body/paw masks despite right-clicks.
     A part hidden on this frame can be prompted on another frame instead.
  3. p  propagate through the whole clip (forward + backward).
  4. Scrub; where a part is wrong, add clicks on that frame, then p again.
     Each correction makes that frame an extra conditioning frame; keep
     count of how many you needed, that number is itself a result.
  5. r  review mode: steps through every --review-step frames. Press the
     number of each part that is WRONG (identity swap, grossly wrong shape,
     missing while visible, or on something else), then Enter for next.
  6. s  save masks + videos (also done on clip switch / quit).

Keys
  2-8 select part (review mode: toggle "wrong")   left/right click  +/- point
  a d / <- ->  frame -1/+1     w x / up down  frame +10/-10    space  play
  [ ]  prev/next prompted frame   u  undo click   c  clear part on frame
  C C  clear ALL labels on this clip (files backed up, not deleted)
  p  propagate    v  view: overlay/solid/raw    + -  display gain
  r  review mode  Enter  next review frame  b  previous review frame
  n / N  next / previous clip    s  save    h  help    q / Esc  save & quit
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from common import (
    DEFAULT_OUT_ROOT,
    PART_BY_ID,
    PARTS,
    frame_paths,
    load_masks,
    read_json,
    save_masks,
    write_json,
)
from render import contact_sheet, enhance, overlay, solid, write_videos

WIN = "mouse parts"
PANEL_W = 340
KEY_LEFT = {81, 65361, 2424832, ord("a")}
KEY_RIGHT = {83, 65363, 2555904, ord("d")}
KEY_UP = {82, 65362, 2490368, ord("w")}
KEY_DOWN = {84, 65364, 2621440, ord("x")}
KEY_ENTER = {13, 10, 65293, 65421}


class Annotator:
    def __init__(self, vdir, clip_ids, tracker_factory, review_step, display):
        self.vdir = Path(vdir)
        self.clip_ids = clip_ids
        self.tracker_factory = tracker_factory
        self.tracker = None
        self.review_step = review_step
        self.display = display
        self.ci = 0
        self.part = PARTS[0].obj_id
        self.view = 0
        self.gain = 2.0
        self.playing = False
        self.show_help = True
        self.confirm_clear = False
        self.status = ""
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.setMouseCallback(WIN, self.on_mouse)

    # ---- clip lifecycle ----------------------------------------------------
    @property
    def cdir(self):
        return self.vdir / "clips" / self.clip_ids[self.ci]

    def open_clip(self, ci):
        self.ci = ci
        cdir = self.cdir
        self.meta = read_json(cdir / "meta.json")
        self.paths = frame_paths(cdir)
        self.T = len(self.paths)
        self.frames = {}
        self.t = 0
        self.session = None  # created lazily: GPU load + replay only when needed
        self.labels = None  # propagated label stack [T,H,W]
        self.preview = {}  # frame -> label map from clicks after last propagation
        self.saved_events = read_json(cdir / "prompts.json", {}).get("events", [])
        self.clicks = {}  # (frame, obj) -> (xy, labels), mirrors the saved/live session
        self._clicks_from_events(self.saved_events)
        self.dirty = False
        if (cdir / "masks.npz").exists():
            self.labels = load_masks(cdir)["labels"]
        qc = read_json(cdir / "qc.json", {})
        self.reviewed = {int(k): set(v) for k, v in qc.get("reviewed", {}).items()}
        self.review = False
        self.review_frames = list(range(0, self.T, self.review_step))
        self.ri = 0
        pf = sorted({f for f, _ in self.clicks})
        if pf:
            self.t = pf[0]
        cv2.destroyWindow(WIN)
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.setMouseCallback(WIN, self.on_mouse)
        cv2.resizeWindow(WIN, self.display + PANEL_W, self.display)
        cv2.createTrackbar("frame", WIN, self.t, self.T - 1, self.on_trackbar)
        self.status = f"opened {self.clip_ids[ci]}" + (
            f" ({len(self.saved_events)} saved events)" if self.saved_events else ""
        )

    def _clicks_from_events(self, events):
        self.clicks = {}
        for ev in events:
            if ev["op"] == "points":
                self.clicks[(ev["frame"], ev["obj_id"])] = (ev["xy"], ev["labels"])
            elif ev["op"] == "clear":
                self.clicks.pop((ev["frame"], ev["obj_id"]), None)
            elif ev["op"] == "reset":
                self.clicks = {}

    def ensure_session(self):
        if self.session is not None:
            return
        from sam_session import PartSession, build_tracker

        if self.tracker is None:
            self.set_status("loading SAM3 (first time only) ...")
            self.tracker = build_tracker()
        self.set_status("loading frames ...")
        self.session = PartSession(self.tracker, self.cdir / "frames", [p.obj_id for p in PARTS])
        if self.saved_events:
            n_prop = sum(e["op"] == "propagate" for e in self.saved_events)
            self.set_status(f"replaying {len(self.saved_events)} saved events ({n_prop} propagations) ...")
            self.session.replay(self.saved_events, progress=self.on_progress)
            if self.session.result is not None and not self.session.stale:
                self.labels = self.session.result["labels"]

    def save(self, videos=True):
        cdir = self.cdir
        if self.session is not None and self.session.events:
            write_json(cdir / "prompts.json", {
                "clip_id": self.clip_ids[self.ci],
                "parts": {p.obj_id: p.name for p in PARTS},
                "n_propagations": sum(e["op"] == "propagate" for e in self.session.events),
                "prompted_frames": self.session.prompted_frames(),
                "events": self.session.events,
            })
            self.saved_events = list(self.session.events)
        if self.reviewed:
            write_json(cdir / "qc.json", {
                "review_step": self.review_step,
                "reviewed": {str(k): sorted(v) for k, v in sorted(self.reviewed.items())},
            })
        if self.session is not None and self.session.result is not None and not self.session.stale and self.dirty:
            r = self.session.result
            save_masks(cdir, r["labels"], r["obj_scores"], r["raw_overlap"], [p.obj_id for p in PARTS])
            if videos:
                self.set_status("writing overlay / parts videos ...")
                write_videos(cdir, r["labels"], self.meta["fps"])
                contact_sheet(cdir, r["labels"])
            self.dirty = False
            self.status = f"saved masks + videos -> {cdir.name}"
        elif self.session is not None and self.session.stale:
            self.status = "saved prompts; masks NOT saved (clicks since last propagate - press p)"
        else:
            self.status = "saved"

    # ---- events -------------------------------------------------------------
    def on_trackbar(self, v):
        self.t = int(v)

    def on_mouse(self, ev, x, y, flags, _):
        if ev not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN) or self.review:
            return
        H = cv2.imread(str(self.paths[0])).shape[0]
        s = self.display / H
        x, y = x / s, y / s
        if x >= H:  # click on the side panel
            return
        neg = ev == cv2.EVENT_RBUTTONDOWN or (flags & cv2.EVENT_FLAG_SHIFTKEY)
        xy, lab = self.clicks.get((self.t, self.part), ([], []))
        self.set_clicks(xy + [[x, y]], lab + [0 if neg else 1])

    def set_clicks(self, xy, lab):
        self.ensure_session()
        n_resets = sum(e["op"] == "reset" for e in self.session.events)
        out = self.session.set_points(self.t, self.part, xy, lab)
        if xy:
            self.clicks[(self.t, self.part)] = (xy, lab)
        else:
            self.clicks.pop((self.t, self.part), None)
        if out is not None:
            self.preview[self.t] = out
        if sum(e["op"] == "reset" for e in self.session.events) > n_resets:
            self.labels, self.preview = None, {self.t: out}
            self.status = "new part after tracking: session rebuilt, propagate again"

    def propagate(self):
        self.ensure_session()
        if not self.session.prompted_frames():
            self.status = "no clicks yet"
            return
        self.preview = {}
        self.labels = np.zeros((self.T,) + self.frame(0).shape[:2], np.uint8)
        r = self.session.propagate(progress=self.on_progress)
        self.labels = r["labels"]
        self.dirty = True
        n_prop = sum(e["op"] == "propagate" for e in self.session.events)
        self.status = f"propagated in {r['seconds']:.0f}s (round {n_prop}); {len(self.session.prompted_frames())} prompted frames"
        self.save(videos=False)

    def on_progress(self, f, lab):
        if self.labels is not None and self.labels.shape[0] == self.T:
            self.labels[f] = lab
        self.t = f
        if f % 4 == 0:
            self.status = f"propagating ... frame {f}"
            self.draw()
            cv2.waitKey(1)

    def set_status(self, s):
        self.status = s
        self.draw()
        cv2.waitKey(1)

    # ---- drawing --------------------------------------------------------------
    def frame(self, t):
        if t not in self.frames:
            if len(self.frames) > 64:
                self.frames.pop(next(iter(self.frames)))
            self.frames[t] = cv2.imread(str(self.paths[t]))
        return self.frames[t]

    def current_labels(self):
        if self.t in self.preview:
            return self.preview[self.t]
        if self.labels is not None:
            return self.labels[self.t]
        return None

    def draw(self):
        img = enhance(self.frame(self.t), self.gain)
        lab = self.current_labels()
        if lab is not None:
            img = (overlay(img, lab), solid(lab), img)[self.view]
        s = self.display / img.shape[0]
        img = cv2.resize(img, (self.display, self.display), interpolation=cv2.INTER_LINEAR)
        for (f, o), (xy, ls) in self.clicks.items():
            if f != self.t:
                continue
            col = PART_BY_ID[o].bgr if o in PART_BY_ID else (128, 128, 128)  # retired part
            for (x, y), l in zip(xy, ls):
                c = (int(x * s), int(y * s))
                cv2.circle(img, c, 7, col, -1)
                cv2.circle(img, c, 8, (0, 255, 0) if l else (0, 0, 255), 2)
                if not l:
                    cv2.line(img, (c[0] - 4, c[1]), (c[0] + 4, c[1]), (0, 0, 0), 2)
        stale = self.session is not None and self.session.stale and bool(self.preview)
        if self.clicks and (self.labels is None or stale):
            msg = ("NOT TRACKED - clicks only segment their own frame. Press p to track the clip (~1.5 min)"
                   if self.labels is None else
                   "Clicks changed since tracking - press p to re-track the clip")
            cv2.rectangle(img, (0, self.display - 34), (self.display, self.display), (0, 0, 0), -1)
            cv2.putText(img, msg, (10, self.display - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 200, 255), 1, cv2.LINE_AA)
        if self.review and self.t in self.reviewed:
            bad = self.reviewed[self.t]
            cv2.rectangle(img, (0, 0), (self.display - 1, self.display - 1),
                          (0, 0, 255) if bad else (0, 200, 0), 6)
        canvas = np.zeros((self.display, self.display + PANEL_W, 3), np.uint8)
        canvas[:, : self.display] = img
        self.draw_panel(canvas[:, self.display :])
        cv2.imshow(WIN, canvas)

    def draw_panel(self, p):
        y = [22]

        def put(txt, col=(220, 220, 220), sc=0.5):
            cv2.putText(p, txt, (10, y[0]), cv2.FONT_HERSHEY_SIMPLEX, sc, col, 1, cv2.LINE_AA)
            y[0] += int(30 * sc + 6)

        beh = self.meta["behavior_per_frame"][self.t]
        put(f"clip {self.ci + 1}/{len(self.clip_ids)}: {self.clip_ids[self.ci]}", sc=0.45)
        put(f"frame {self.t}/{self.T - 1}   t={self.t / self.meta['fps']:.2f}s", sc=0.55)
        put(f"label: {beh}", (120, 220, 255), 0.55)
        pf = sorted({f for f, _ in self.clicks})
        put(f"prompted frames: {pf[:8]}{' ...' if len(pf) > 8 else ''}", sc=0.42)
        state = ("not tracked" if self.labels is None else
                 "STALE - press p" if self.session is not None and self.session.stale and self.preview else "tracked")
        put(f"masks: {state}", (0, 200, 255) if "STALE" in state else (180, 180, 180))
        put(f"view: {('overlay', 'solid parts', 'raw')[self.view]}   gain {self.gain:.1f}", sc=0.45)
        y[0] += 4
        if self.review:
            bad = self.reviewed.get(self.t, set())
            put(f"REVIEW {self.ri + 1}/{len(self.review_frames)}  (n done: {len(self.reviewed)})", (0, 255, 255))
            put("press part numbers that are WRONG,", sc=0.42)
            put("Enter = next, b = back, r = exit", sc=0.42)
        else:
            bad = set()
        for q in PARTS:
            here = (self.t, q.obj_id) in self.clicks
            mark = ">" if q.obj_id == self.part and not self.review else " "
            col = (0, 0, 255) if q.obj_id in bad else q.bgr
            put(f"{mark}{q.key} {q.name}{'  *' if here else ''}{'  WRONG' if q.obj_id in bad else ''}", col, 0.55)
        y[0] += 4
        if self.show_help:
            for line in __doc__.split("Keys")[1].strip().splitlines():
                for chunk in (line[i : i + 46] for i in range(0, len(line), 46)):
                    put(chunk.strip(), (150, 150, 150), 0.36)
        y[0] = self.display - 12
        put(self.status[:52], (0, 255, 0), 0.42)

    # ---- main loop --------------------------------------------------------------
    def goto(self, t):
        self.t = int(np.clip(t, 0, self.T - 1))
        cv2.setTrackbarPos("frame", WIN, self.t)

    def handle_key(self, k):
        ch = chr(k & 0xFF) if k != -1 and (k & 0xFF) < 128 else ""
        if ch != "C":
            self.confirm_clear = False
        if self.review:
            if ch.isdigit() and int(ch) in PART_BY_ID:
                s = self.reviewed.setdefault(self.t, set())
                s ^= {int(ch)}
                return True
            if k in KEY_ENTER or ch == "n":
                self.reviewed.setdefault(self.t, set())
                self.ri = min(self.ri + 1, len(self.review_frames) - 1)
                self.goto(self.review_frames[self.ri])
                if len(self.reviewed) == len(self.review_frames):
                    self.status = "all review frames done - s to save"
                return True
            if ch == "b":
                self.ri = max(self.ri - 1, 0)
                self.goto(self.review_frames[self.ri])
                return True
        if ch.isdigit() and int(ch) in PART_BY_ID:
            self.part = int(ch)
        elif k in KEY_LEFT:
            self.goto(self.t - 1)
        elif k in KEY_RIGHT:
            self.goto(self.t + 1)
        elif k in KEY_UP:
            self.goto(self.t + 10)
        elif k in KEY_DOWN:
            self.goto(self.t - 10)
        elif ch == " ":
            self.playing = not self.playing
        elif ch in "[]":
            pf = sorted({f for f, _ in self.clicks})
            nxt = [f for f in pf if (f > self.t if ch == "]" else f < self.t)]
            if nxt:
                self.goto(nxt[0] if ch == "]" else nxt[-1])
        elif ch == "u":
            xy, lab = self.clicks.get((self.t, self.part), ([], []))
            if xy:
                self.set_clicks(xy[:-1], lab[:-1])
        elif ch == "c":
            if (self.t, self.part) in self.clicks:
                self.set_clicks([], [])
        elif ch == "C":
            if self.confirm_clear:
                self.clear_clip()
            else:
                self.confirm_clear = True
                self.status = "press C again to clear ALL labels on this clip"
                return True
        elif ch == "p":
            self.propagate()
        elif ch == "v":
            self.view = (self.view + 1) % 3
        elif ch in "+=":
            self.gain = min(self.gain + 0.25, 5)
        elif ch == "-":
            self.gain = max(self.gain - 0.25, 0.5)
        elif ch == "r":
            if self.labels is None:
                self.status = "propagate before reviewing"
            else:
                self.review = not self.review
                if self.review:
                    todo = [i for i, f in enumerate(self.review_frames) if f not in self.reviewed]
                    self.ri = todo[0] if todo else 0
                    self.goto(self.review_frames[self.ri])
        elif ch == "s":
            self.save()
        elif ch == "h":
            self.show_help = not self.show_help
        elif ch in ("n", "N"):
            self.save()
            self.open_clip((self.ci + (1 if ch == "n" else -1)) % len(self.clip_ids))
        elif ch == "q" or k == 27:
            self.save()
            return False
        return True

    def clear_clip(self):
        """Drop every click, mask and review flag on this clip. Saved files are
        moved to <video>/_cleared_labels_<time>/, not deleted."""
        import shutil
        import time

        backup = self.vdir / f"_cleared_labels_{time.strftime('%Y%m%d_%H%M%S')}" / "clips" / self.cdir.name
        moved = []
        for name in ("prompts.json", "qc.json", "masks.npz", "overlay.mp4", "parts.mp4", "contact_sheet.jpg"):
            f = self.cdir / name
            if f.exists():
                backup.mkdir(parents=True, exist_ok=True)
                shutil.move(str(f), str(backup / name))
                moved.append(name)
        if self.session is not None:
            self.session.reset()
            self.session.events.clear()
        self.saved_events, self.clicks, self.preview = [], {}, {}
        self.labels, self.reviewed, self.review, self.dirty = None, {}, False, False
        self.status = f"cleared this clip ({len(moved)} files backed up)"

    def run(self, start=0):
        self.open_clip(start)
        while True:
            self.draw()
            k = cv2.waitKeyEx(15)
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                self.save()
                break
            if self.playing:
                self.goto((self.t + 1) % self.T)
            if k != -1 and not self.handle_key(k):
                break
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--video-stem", help="defaults to the only/first video under out-root")
    ap.add_argument("--clip", help="start at this clip id")
    ap.add_argument("--review-step", type=int, default=30, help="review every N-th frame")
    ap.add_argument("--display", type=int, default=900, help="displayed frame size (px)")
    args = ap.parse_args()

    root = Path(args.out_root)
    stems = sorted(p.name for p in root.iterdir() if (p / "clips").is_dir())
    stem = args.video_stem or stems[0]
    vdir = root / stem
    clip_ids = sorted(p.name for p in (vdir / "clips").iterdir() if (p / "meta.json").exists())
    if not clip_ids:
        sys.exit(f"no clips in {vdir}; run extract_clips.py first")
    start = clip_ids.index(args.clip) if args.clip else 0
    Annotator(vdir, clip_ids, None, args.review_step, args.display).run(start)


if __name__ == "__main__":
    main()
