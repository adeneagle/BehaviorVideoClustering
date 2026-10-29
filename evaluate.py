"""Score part masks for identity, stability and accuracy across all clips.

There is no ground truth, so this combines automatic proxies with the human
review flags from annotate.py (r mode):

  per clip x part (part_metrics.csv)
    presence        fraction of frames with a non-empty mask
    flicker         short (<=3 frame) dropouts per 100 frames: mask vanishes and returns
    tiou_median     median IoU between consecutive frames (temporal stability)
    area_cv         area coefficient of variation; area_jumps = % of frame pairs
                    whose area changes by >1.5x (sudden grow/shrink = leak or loss)
    fragmented      % frames where the mask is >1 connected piece
    jitter_px       RMS of centroid minus its 2-frame Gaussian smoothing: high-
                    frequency boundary noise, not real movement
    contested       median share of the part's raw (pre-argmax) mask that another
                    part also claimed; high = two objects fighting over pixels
    human_error     % of reviewed frames flagged wrong (the accuracy number)
  identity (swaps.csv)
    for each L/R paw pair, which side of the body axis the L paw is on
    relative to the R paw. A clip's `side` is what most of its frames say;
    runs of frames on the other side are swap candidates. The axis is the
    body's long axis, pointed away from the tail base where the tail is
    tracked (front_from = tail), else forepaws-vs-hindpaws midpoints (paws;
    unreliable when the mouse sits up to groom), else kept pointing the same
    way as its neighbouring frames. `side` should agree across clips (same
    camera, same convention): a clip that disagrees with the majority
    probably had L/R clicked the other way round.
  per behaviour (behaviour_metrics.csv): the same stability numbers split by
    the manual behaviour label of each frame.
  effort: prompted frames, clicks and propagation rounds per clip.

    python evaluate.py            -> <video>/eval/
"""

import argparse
import warnings
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from common import DEFAULT_OUT_ROOT, LR_PAIRS, PARTS, load_masks, part_id, read_json


def part_series(labels, obj_id):
    """Per-frame area, centroid, component count, and IoU with the previous frame."""
    T = len(labels)
    area = np.zeros(T)
    cen = np.full((T, 2), np.nan)
    ncomp = np.zeros(T, int)
    tiou = np.full(T, np.nan)
    prev = None
    for t in range(T):
        m = labels[t] == obj_id
        a = m.sum()
        area[t] = a
        if a:
            ys, xs = np.nonzero(m)
            cen[t] = xs.mean(), ys.mean()
            n, _, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
            ncomp[t] = int((st[1:, cv2.CC_STAT_AREA] >= 20).sum())
        if prev is not None and (a or prev.any()):
            tiou[t] = (m & prev).sum() / (m | prev).sum()
        prev = m
    return area, cen, ncomp, tiou


def short_dropouts(present, max_len=3):
    n, t, T = 0, 0, len(present)
    while t < T:
        if not present[t]:
            s = t
            while t < T and not present[t]:
                t += 1
            if s > 0 and t < T and t - s <= max_len:
                n += 1
        t += 1
    return n


def jitter(cen, present):
    ok = present & ~np.isnan(cen[:, 0])
    if ok.sum() < 10:
        return np.nan
    c = cen[ok]
    res = c - gaussian_filter1d(c, sigma=2, axis=0, mode="nearest")
    return float(np.sqrt((res**2).sum(1).mean()))


def stability(area, ncomp, tiou, cen, sel):
    present = area > 0
    a, p = area[sel], present[sel]
    both = p[1:] & p[:-1]
    ratio = np.abs(np.log((a[1:] + 1) / (a[:-1] + 1)))[both]
    return dict(
        n_frames=int(sel.sum()),
        presence=p.mean(),
        flicker_per100=100 * short_dropouts(p) / max(len(p), 1),
        tiou_median=np.nanmedian(tiou[sel]) if np.isfinite(tiou[sel]).any() else np.nan,
        area_median=np.median(a[p]) if p.any() else 0,
        area_cv=a[p].std() / a[p].mean() if p.sum() > 1 else np.nan,
        area_jumps_pct=100 * (ratio > np.log(1.5)).mean() if len(ratio) else np.nan,
        fragmented_pct=100 * (ncomp[sel][p] > 1).mean() if p.any() else np.nan,
        jitter_px=jitter(cen[sel], p),
    )


def long_axis(mask):
    """Unit vector along the mask's principal axis (sign arbitrary), or NaN."""
    m = cv2.moments(mask.astype(np.uint8), binaryImage=True)
    if m["m00"] == 0:
        return np.array([np.nan, np.nan])
    th = 0.5 * np.arctan2(2 * m["mu11"], m["mu20"] - m["mu02"])
    return np.array([np.cos(th), np.sin(th)])


def body_axis(labels, body_id, front_hint, min_hint_px=15.0):
    """Per-frame unit vector along the body's long axis, pointing forward.

    The long axis comes from the body mask; its sign comes from `front_hint`
    (forepaw midpoint minus hindpaw midpoint, which an L/R swap cannot
    change) where that is clearly non-zero. Elsewhere the sign follows the
    neighbouring frame: forward from each hinted frame, then backward for
    frames before the first. Returns (axis [T,2], hinted [T] bool).
    """
    T = len(labels)
    pca = np.stack([long_axis(labels[t] == body_id) for t in range(T)])
    proj = np.einsum("td,td->t", np.nan_to_num(pca), np.nan_to_num(front_hint))
    hinted = ~np.isnan(pca[:, 0]) & ~np.isnan(front_hint[:, 0]) & (np.abs(proj) > min_hint_px)
    axis = np.full((T, 2), np.nan)
    axis[hinted] = pca[hinted] * np.sign(proj[hinted])[:, None]

    def fill(order):
        prev = None
        for t in order:
            if hinted[t] or (prev is None and not np.isnan(axis[t, 0])):
                prev = axis[t]
            elif prev is not None and not np.isnan(pca[t, 0]):
                axis[t] = pca[t] if pca[t] @ prev >= 0 else -pca[t]
                prev = axis[t]

    fill(range(T))
    fill(range(T - 1, -1, -1))  # frames before the first hinted one
    if np.isnan(axis[:, 0]).all() and not np.isnan(pca[:, 0]).all():
        # no hint anywhere: sign is arbitrary but kept consistent in time,
        # which is all the within-clip swap check needs
        first = np.flatnonzero(~np.isnan(pca[:, 0]))[0]
        axis[first] = pca[first]
        fill(range(T))
        fill(range(T - 1, -1, -1))
    return axis, hinted


def tail_base(labels, body_id, tail_id, ring_px=6):
    """Per-frame point where the tail meets the body (tail pixels within
    ring_px of the body mask), or NaN. Marks the rear in any posture."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring_px + 1,) * 2)
    out = np.full((len(labels), 2), np.nan)
    for t, lab in enumerate(labels):
        tail = lab == tail_id
        if not tail.any():
            continue
        near = tail & cv2.dilate((lab == body_id).astype(np.uint8), k).astype(bool)
        if near.any():
            ys, xs = np.nonzero(near)
            out[t] = xs.mean(), ys.mean()
    return out


def pair_midpoint(series, names):
    """Mean centroid of whichever of the named parts are present, per frame."""
    present = [series[n][1] for n in names if n in series]
    if not present:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN frames -> NaN
        return np.nanmean(np.stack(present), axis=0)


def lateral(cen_paw, cen_body, axis):
    """Signed distance of the paw from the body's front-back axis (px)."""
    d = cen_paw - cen_body
    return axis[:, 0] * d[:, 1] - axis[:, 1] * d[:, 0]


def runs(mask):
    out, t = [], 0
    while t < len(mask):
        if mask[t]:
            s = t
            while t < len(mask) and mask[t]:
                t += 1
            out.append((s, t - 1))
        t += 1
    return out


def evaluate_clip(cdir, eval_dir, margin_px, min_swap_frames, masks_name="masks.npz"):
    meta = read_json(cdir / "meta.json")
    prompts = read_json(cdir / "prompts.json", {})
    # review flags judge SAM's masks, so they only apply to masks.npz
    qc = read_json(cdir / "qc.json", {}).get("reviewed", {}) if masks_name == "masks.npz" else {}
    m = load_masks(cdir, masks_name)
    labels, ids = m["labels"], list(m["obj_ids"])
    T = len(labels)
    beh = np.array(meta["behavior_per_frame"][:T])
    events = prompts.get("events", [])
    prompted = sorted({e["frame"] for e in events if e["op"] == "points"})

    series, rows, brows = {}, [], []
    for p in PARTS:
        area, cen, ncomp, tiou = part_series(labels, p.obj_id)
        if not (area > 0).any():
            continue
        series[p.name] = (area, cen)
        k = ids.index(p.obj_id)
        ov = m["raw_overlap"][:, k, :]
        own = ov[:, k].astype(float)
        others = np.delete(ov, k, axis=1).max(1) if ov.shape[1] > 1 else np.zeros(T)
        contested = np.where(own > 0, others / np.maximum(own, 1), np.nan)
        rev = [int(f) for f in qc]
        bad = [f for f in rev if p.obj_id in qc[str(f)]]
        row = dict(clip=cdir.name, part=p.name, target=meta.get("target_behavior"),
                   **stability(area, ncomp, tiou, cen, np.ones(T, bool)),
                   contested_median=np.nanmedian(contested),
                   score_median=np.nanmedian(m["obj_scores"][:, k]),
                   n_reviewed=len(rev),
                   human_error_pct=100 * len(bad) / len(rev) if rev else np.nan)
        rows.append(row)
        for b in np.unique(beh):
            sel = beh == b
            if sel.sum() < 10:
                continue
            rb = [f for f in rev if f < T and beh[f] == b]
            bb = [f for f in rb if p.obj_id in qc[str(f)]]
            brows.append(dict(clip=cdir.name, part=p.name, behaviour=b,
                              **stability(area, ncomp, tiou, cen, sel),
                              n_reviewed=len(rb), n_wrong=len(bb)))

    swaps = []
    fore = pair_midpoint(series, ("forepaw_L", "forepaw_R"))
    hind = pair_midpoint(series, ("hindpaw_L", "hindpaw_R"))
    if "body" in series:
        cb = series["body"][1]
        hint = fore - hind if fore is not None and hind is not None else np.full((T, 2), np.nan)
        from_tail = np.zeros(T, bool)
        if "tail" in series:  # body centre minus tail base points forward in any posture
            tb = tail_base(labels, part_id("body"), part_id("tail"))
            from_tail = ~np.isnan(tb[:, 0])
            hint[from_tail] = (cb - tb)[from_tail]
        axis, hinted = body_axis(labels, part_id("body"), hint)
        for l_name, r_name in LR_PAIRS:
            if l_name not in series or r_name not in series:
                continue
            d = lateral(series[l_name][1], cb, axis) - lateral(series[r_name][1], cb, axis)
            ok = ~np.isnan(d)
            clear = ok & (np.abs(d) > margin_px)
            if not clear.any():
                continue
            # The clip's side is the one most frames agree on. A single
            # reference frame is fragile: it can catch the paws mid-crossing.
            sign = 1.0 if (d[clear] > 0).mean() >= 0.5 else -1.0
            flipped = clear & (np.sign(d) == -sign)
            long_runs = [r for r in runs(flipped) if r[1] - r[0] + 1 >= min_swap_frames]
            swaps.append(dict(clip=cdir.name, pair=f"{l_name}/{r_name}",
                              side=int(sign), side_agreement_pct=100 * (1 - flipped[clear].mean()),
                              flipped_pct=100 * flipped[ok].mean(),
                              front_from_tail_pct=100 * (hinted & from_tail).mean(),
                              front_from_paws_pct=100 * (hinted & ~from_tail).mean(),
                              swap_runs=";".join(f"{a}-{b}" for a, b in long_runs),
                              n_swap_runs=len(long_runs)))
            series[f"_lr_{l_name}"] = d * sign

    effort = dict(clip=cdir.name, target=meta.get("target_behavior"),
                  prompted_frames=len(prompted),
                  click_events=sum(e["op"] in ("points", "clear") for e in events),
                  propagation_rounds=sum(e["op"] == "propagate" for e in events),
                  reviewed_frames=len(qc),
                  frames_with_any_error=sum(bool(v) for v in qc.values()))
    plot_clip(cdir.name, series, beh, qc, eval_dir / f"{cdir.name}.png", prompted)
    return rows, brows, swaps, effort


def plot_clip(name, series, beh, qc, path, prompted):
    parts = [p for p in PARTS if p.name in series]
    lr = [k for k in series if k.startswith("_lr_")]
    fig, axes = plt.subplots(2 + len(lr), 1, figsize=(12, 2.2 * (2 + len(lr))), sharex=True)
    T = len(beh)
    cats = list(dict.fromkeys(beh))
    cmap = plt.get_cmap("tab10")
    ax = axes[0]
    for i, c in enumerate(cats):
        ax.fill_between(np.arange(T), 0, 1, where=beh == c, color=cmap(i % 10), step="mid", label=c)
    for f in prompted:
        ax.axvline(f, color="k", lw=1)
    for f, bad in qc.items():
        ax.plot(int(f), 0.5, "x" if bad else "o", color="red" if bad else "green", ms=6)
    ax.set_yticks([])
    ax.set_title(f"{name}: behaviour label, prompted frames (|), review (o ok / x wrong)", fontsize=9)
    ax.legend(fontsize=7, ncol=len(cats), loc="upper right")
    ax = axes[1]
    for p in parts:
        ax.plot(series[p.name][0], color=np.array(p.bgr[::-1]) / 255, label=p.name, lw=1)
    ax.set_yscale("symlog")
    ax.set_ylabel("area px")
    ax.legend(fontsize=7, ncol=4)
    for ax, k in zip(axes[2:], lr):
        ax.plot(series[k], lw=1)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel("L-R lateral\n(+ = clip majority)", fontsize=8)
        ax.set_title(k[4:].replace("_L", "") + " pair: negative = sides swapped", fontsize=8)
    axes[-1].set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--video-stem")
    ap.add_argument("--swap-margin-px", type=float, default=10.0)
    ap.add_argument("--min-swap-frames", type=int, default=6, help="~0.1 s at 60 fps")
    ap.add_argument("--masks", default="masks.npz",
                    help="mask file per clip, e.g. masks_<run>.npz from predict_seg.py")
    args = ap.parse_args()
    root = Path(args.out_root)
    stem = args.video_stem or sorted(p.name for p in root.iterdir() if (p / "clips").is_dir())[0]
    vdir = root / stem
    tag = Path(args.masks).stem.removeprefix("masks").lstrip("_")
    eval_dir = vdir / (f"eval_{tag}" if tag else "eval")
    eval_dir.mkdir(exist_ok=True)

    R, B, S, E = [], [], [], []
    for c in sorted((vdir / "clips").iterdir()):
        if not (c / args.masks).exists():
            continue
        r, b, s, e = evaluate_clip(c, eval_dir, args.swap_margin_px, args.min_swap_frames, args.masks)
        R += r; B += b; S += s; E.append(e)
    if not R:
        print("no clips with masks yet")
        return
    pm, bm, sw, ef = map(pd.DataFrame, (R, B, S, E))
    pm.to_csv(eval_dir / "part_metrics.csv", index=False)
    bm.to_csv(eval_dir / "behaviour_metrics.csv", index=False)
    sw.to_csv(eval_dir / "swaps.csv", index=False)
    ef.to_csv(eval_dir / "effort.csv", index=False)

    cols = ["presence", "flicker_per100", "tiou_median", "area_jumps_pct", "fragmented_pct",
            "jitter_px", "contested_median", "human_error_pct"]
    pd.set_option("display.width", 200)
    pd.set_option("display.precision", 2)
    lines = [f"{len(ef)} clips evaluated\n", "== per part (median over clips) ==",
             pm.groupby("part", sort=False)[cols].median().to_string(), ""]
    beh_tab = bm.groupby(["behaviour", "part"]).agg(
        tiou=("tiou_median", "median"), presence=("presence", "median"),
        reviewed=("n_reviewed", "sum"), wrong=("n_wrong", "sum"))
    beh_tab["error_pct"] = 100 * beh_tab.wrong / beh_tab.reviewed.replace(0, np.nan)
    lines += ["== by behaviour ==", beh_tab.to_string(), ""]
    if len(sw):
        maj = sw.groupby("pair")["side"].agg(lambda s: s.mode().iloc[0])
        sw["side_disagrees_with_majority"] = sw.apply(lambda r: r["side"] != maj[r["pair"]], axis=1)
        sw.to_csv(eval_dir / "swaps.csv", index=False)
        lines += ["== L/R identity ==", sw.drop(columns=["swap_runs"]).to_string(index=False), ""]
        if sw.side_disagrees_with_majority.any():
            lines.append("!! clips whose prompted L/R side disagrees with the majority (check prompts): "
                         + ", ".join(sw.loc[sw["side_disagrees_with_majority"], "clip"].unique()))
    lines += ["== effort ==", ef.to_string(index=False)]
    txt = "\n".join(lines)
    (eval_dir / "summary.txt").write_text(txt)
    print(txt)
    print(f"\nwrote {eval_dir}")


if __name__ == "__main__":
    main()
