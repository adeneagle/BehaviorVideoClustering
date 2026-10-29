"""SAM3 tracker session over one clip, with a replayable event log.

Every interaction is recorded as an event so that a session can be re-run
exactly, headless (propagate.py), or produced by a program instead of a
person (e.g. clicks derived from SLEAP keypoints). Events:

    {"op": "points", "frame": f, "obj_id": o, "xy": [[x, y], ...], "labels": [1, 0, ...]}
        the full click list for (frame, obj) so far; SAM refines iteratively, so
        each click is its own event rather than only the final list
    {"op": "clear", "frame": f, "obj_id": o}
    {"op": "propagate"}   forward from the earliest prompted frame, then backward
    {"op": "reset"}       drop all prompts and objects (needed to add a part
                          after tracking has started)

Click coordinates are crop pixels (the frames SAM3 sees).
"""

import time

import numpy as np
import torch


def build_tracker():
    from sam3.model_builder import build_sam3_video_model

    model = build_sam3_video_model()
    tracker = model.tracker
    tracker.backbone = model.detector.backbone
    del model  # the detector head is not used for click tracking
    return tracker


class PartSession:
    def __init__(self, tracker, frames_dir, obj_ids):
        self.tracker = tracker
        self.obj_ids = list(obj_ids)  # canonical part order for outputs
        st = tracker.init_state(video_path=str(frames_dir))
        self.images = st["images"]
        self.H, self.W, self.T = st["video_height"], st["video_width"], st["num_frames"]
        self.state = st
        self.events = []
        self.clicks = {}  # (frame, obj_id) -> (xy list, label list)
        self.result = None  # dict from propagate()
        self.stale = True

    # ---- state ------------------------------------------------------------
    def _fresh_state(self):
        # Re-use the decoded frames; only the prompt/memory bookkeeping is reset.
        self.tracker.clear_all_points_in_video(self.state)

    @property
    def tracking_started(self):
        return self.state["tracking_has_started"]

    def prompted_frames(self):
        return sorted({f for (f, _o), (xy, _l) in self.clicks.items() if xy})

    # ---- events -----------------------------------------------------------
    def apply(self, ev, progress=None):
        op = ev["op"]
        if op == "points":
            return self._points(ev["frame"], ev["obj_id"], ev["xy"], ev["labels"], log=ev)
        if op == "clear":
            return self._clear(ev["frame"], ev["obj_id"], log=ev)
        if op == "propagate":
            return self.propagate(progress=progress)
        if op == "reset":
            return self.reset()
        raise ValueError(f"unknown op {op}")

    def replay(self, events, progress=None):
        for ev in events:
            self.apply(ev, progress=progress)

    def reset(self):
        self._fresh_state()
        self.clicks.clear()
        self.events.append({"op": "reset"})
        self.result, self.stale = None, True

    def set_points(self, frame, obj_id, xy, labels):
        """Set the click list for (frame, obj). If obj is new after tracking
        started, rebuild the session from the current clicks first."""
        if self.tracking_started and obj_id not in self.state["obj_id_to_idx"]:
            keep = [(k, v) for k, v in self.clicks.items() if v[0]]
            self.reset()
            for (f, o), (pxy, plab) in keep:
                self._points(f, o, pxy, plab)
        if not xy:
            return self._clear(frame, obj_id)
        return self._points(frame, obj_id, xy, labels)

    def _points(self, frame, obj_id, xy, labels, log=None):
        rel = torch.tensor([[x / self.W, y / self.H] for x, y in xy], dtype=torch.float32)
        _, ids, _, vmasks = self.tracker.add_new_points_or_box(
            inference_state=self.state,
            frame_idx=int(frame),
            obj_id=int(obj_id),
            points=rel,
            labels=torch.tensor(labels, dtype=torch.int32),
            clear_old_points=True,
        )
        self.clicks[(frame, obj_id)] = ([list(map(float, p)) for p in xy], list(map(int, labels)))
        self.events.append(
            log or {"op": "points", "frame": int(frame), "obj_id": int(obj_id),
                    "xy": [list(map(float, p)) for p in xy], "labels": list(map(int, labels))}
        )
        self.stale = True
        return self._to_labels(ids, vmasks)

    def _clear(self, frame, obj_id, log=None):
        self.clicks.pop((frame, obj_id), None)
        self.events.append(log or {"op": "clear", "frame": int(frame), "obj_id": int(obj_id)})
        self.stale = True
        if obj_id not in self.state["obj_id_to_idx"]:
            return None
        out = self.tracker.clear_all_points_in_frame(self.state, int(frame), int(obj_id))
        if out is None:
            return None
        _, ids, _, vmasks = out
        return self._to_labels(ids, vmasks)

    # ---- outputs ----------------------------------------------------------
    def _to_labels(self, ids, vmasks):
        """(K,1,H,W) logits, already made non-overlapping by the tracker -> label map."""
        v = vmasks[:, 0]
        lab = torch.zeros(v.shape[-2:], dtype=torch.uint8, device=v.device)
        best = v.argmax(0)
        fg = (v > 0).any(0)
        id_lut = torch.tensor(list(ids), dtype=torch.uint8, device=v.device)
        lab[fg] = id_lut[best[fg]]
        return lab.cpu().numpy()

    def propagate(self, progress=None):
        """Track all parts over the whole clip: forward from the earliest
        prompted frame, then backward from it. Returns a dict of arrays."""
        tr, st = self.tracker, self.state
        tr.propagate_in_video_preflight(st)
        start = min(st["output_dict"]["cond_frame_outputs"])
        K = len(self.obj_ids)
        labels = np.zeros((self.T, self.H, self.W), np.uint8)
        scores = np.full((self.T, K), np.nan, np.float32)
        overlap = np.zeros((self.T, K, K), np.int32)  # raw (pre-argmax) low-res pixel counts
        col = None
        t0 = time.time()
        passes = [(start, False)] + ([(start, True)] if start > 0 else [])
        for s, rev in passes:
            for f, ids, low, vmasks, osc in tr.propagate_in_video(
                st, start_frame_idx=s, max_frame_num_to_track=self.T,
                reverse=rev, tqdm_disable=True,
            ):
                if col is None:
                    col = [self.obj_ids.index(i) if i in self.obj_ids else None for i in ids]
                labels[f] = self._to_labels(ids, vmasks)
                b = (low[:, 0] > 0).flatten(1).float()
                inter = (b @ b.T).round().int().cpu().numpy()
                sc = osc.float().flatten().cpu().numpy()
                for a, ca in enumerate(col):
                    if ca is None:
                        continue
                    scores[f, ca] = sc[a]
                    for b_, cb in enumerate(col):
                        if cb is not None:
                            overlap[f, ca, cb] = inter[a, b_]
                if progress:
                    progress(f, labels[f])
        self.events.append({"op": "propagate"})
        self.result = dict(labels=labels, obj_scores=scores, raw_overlap=overlap,
                           seconds=time.time() - t0)
        self.stale = False
        return self.result
