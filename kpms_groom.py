"""keypoint-MoSeq syllables from run_video.py's keypoints, and the syllables inside one labelled behaviour
(default face/head/paw groom): are there distinct sub-behaviours?

    ~/miniconda3/envs/kpms/bin/python kpms_groom.py RUN_DIR [--behaviour "face/head/paw groom"]
                [--fps-out 30] [--target-dur-s 0.4] [--kappas-ar 1e5,1e6,1e7] [--full-iters 300]

Runs in the `kpms` conda env (keypoint-moseq 0.8 + JAX on the GPU), not `sam3`. RUN_DIR is a run_video.py
output made with a keypoint network (frames.csv has kp_<node>_x / _y / _score for the six SLEAP nodes)
and --labels (frames.csv 'behaviour').

Model. keypoint-MoSeq (Weinreb et al., Nat Methods 2024) on all six nodes, subsampled to --fps-out
(keypoint-MoSeq's defaults assume ~30 fps). Heading axis: snout (anterior) -> tail base (posterior).
Low-score keypoints are not dropped: keypoint-MoSeq weights each point by its confidence. Following
the keypoint-MoSeq docs: PCA to the dims explaining 90% of the aligned pose variance; an AR-only fit
(50 iters) for each --kappas-ar, keeping the kappa whose median syllable duration is closest to
--target-dur-s; then the full model from there with kappa / 10 (--full-iters).

Grooming analysis (frames labelled --behaviour; syllables assigned per frame):
  share of the behaviour's frames per syllable, enrichment = P(syllable | behaviour) / P(syllable),
  instances (syllable runs that overlap the behaviour), and a body-frame kinematic profile per
  syllable from the keypoints in those frames (aligned on tail base -> snout, body-length units):
  forepaw distances to the snout, forepaw height along the body axis, forepaw speed, hind paw spread,
  body length (curled vs stretched), snout confidence.
  "groom syllables" = >= 3% of the behaviour's frames and enrichment >= 2.

Output in RUN_DIR/kpms/ (the keypoint-MoSeq project, model checkpoints, results.h5) and
RUN_DIR/kpms/groom/:
  summary.txt          fit, kappa choice, syllable durations, groom syllable table
  groom_syllables.csv  the table
  behaviour_by_syllable.csv  frames per (syllable, behaviour)
  ethogram.png         the longest bouts of the behaviour: syllables vs the label
  clips/syllable_<s>.mp4  up to 8 instances of each groom syllable inside the behaviour, from the raw
                       video, 400 px around the body, keypoints drawn, 0.5 s before onset to 1 s after
"""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

KP = ("snout", "left_front_paw", "right_front_paw", "left_hind_paw", "right_hind_paw", "tail_base")
SKELETON = [["tail_base", "snout"], ["snout", "left_front_paw"], ["snout", "right_front_paw"],
            ["tail_base", "left_hind_paw"], ["tail_base", "right_hind_paw"]]


def load_keypoints(run):
    f = pd.read_csv(run / "frames.csv")
    xy = np.stack([f[[f"kp_{n}_x", f"kp_{n}_y"]].to_numpy() for n in KP], 1).astype(np.float64)  # [T, 6, 2]
    conf = np.stack([f[f"kp_{n}_score"].to_numpy() for n in KP], 1).astype(np.float64)
    return f, xy, conf


def run_lengths(z):
    edges = np.flatnonzero(np.diff(z)) + 1
    starts = np.r_[0, edges]
    return starts, np.diff(np.r_[starts, len(z)])


def median_duration(model):
    """Median syllable duration (frames) over the model's current state sequences."""
    zs = np.asarray(model["states"]["z"])
    mask = None
    lens = []
    for z in zs:
        _, l = run_lengths(z)
        lens.append(l[1:-1] if len(l) > 2 else l)  # drop runs cut by the segment ends
    return float(np.median(np.concatenate(lens)))


def body_frame(xy):
    """Keypoints in the body frame: origin tail base, x along tail base -> snout, units of body length."""
    tb, sn = xy[:, 5], xy[:, 0]
    ax = sn - tb
    L = np.linalg.norm(ax, axis=1)
    L = np.where(L > 1, L, np.nan)
    u = ax / L[:, None]
    rel = xy - tb[:, None]
    fwd = (rel * u[:, None]).sum(-1)
    lat = rel[..., 0] * -u[:, None, 1] + rel[..., 1] * u[:, None, 0]
    return np.stack([fwd, lat], -1) / np.nanmedian(L), L / np.nanmedian(L)


def profile(xy, conf, fps):
    """Per-frame interpretable measures (body-length units)."""
    b, blen = body_frame(xy)
    d = lambda i, j: np.linalg.norm(b[:, i] - b[:, j], axis=1)
    sp = np.r_[np.nan, np.linalg.norm(np.diff(xy[:, 1:3], axis=0), axis=-1).mean(1) * fps]  # forepaw speed, px/s
    return pd.DataFrame(dict(
        LF_snout=d(1, 0), RF_snout=d(2, 0), forepaw_gap=d(1, 2),
        forepaws_ahead=(b[:, 1, 0] + b[:, 2, 0]) / 2 - b[:, 0, 0],  # < 0: forepaws behind the snout (over the head)
        forepaw_asym=d(1, 0) - d(2, 0),  # > 0: left paw further from the snout than the right
        hind_spread=d(3, 4), body_len=blen, forepaw_speed_px_s=sp, snout_conf=conf[:, 0]))


def clip_montage(video, frames, cx, cy, xy, starts, path, fps, half=200, before=0.5, after=1.0, tile=(4, 2)):
    """Instances side by side: 2*half px crops around the body centre at the onset, keypoints drawn."""
    cap = cv2.VideoCapture(str(video))
    nb, na = int(before * fps), int(after * fps)
    clips = []
    for s in starts[: tile[0] * tile[1]]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(s - nb, 0))
        x0, y0 = int(cx[s]) - half, int(cy[s]) - half
        seq = []
        for t in range(max(s - nb, 0), min(s + na, len(cx))):
            ok, fr = cap.read()
            if not ok:
                break
            fr = cv2.copyMakeBorder(fr, half, half, half, half, cv2.BORDER_CONSTANT)
            for k in range(len(KP)):
                p = (int(xy[t, k, 0]) + half, int(xy[t, k, 1]) + half)
                cv2.circle(fr, p, 5, [(0, 0, 255), (0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 128, 0), (255, 255, 255)][k], -1)
            crop = fr[y0 + half : y0 + 3 * half, x0 + half : x0 + 3 * half].copy()
            if t >= s:
                cv2.rectangle(crop, (0, 0), (2 * half - 1, 2 * half - 1), (0, 200, 255), 3)  # syllable on
            cv2.putText(crop, f"{t / fps:.1f}s", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            seq.append(crop)
        clips.append(seq)
    if not clips:
        return
    n = max(len(c) for c in clips)
    blank = np.zeros((2 * half, 2 * half, 3), np.uint8)
    clips = [c + [c[-1]] * (n - len(c)) for c in clips] + [[blank] * n] * (tile[0] * tile[1] - len(clips))
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps / 2, (tile[0] * 2 * half, tile[1] * 2 * half))
    for t in range(n):
        rows = [np.hstack([clips[r * tile[0] + c][t] for c in range(tile[0])]) for r in range(tile[1])]
        w.write(np.vstack(rows))
    w.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--behaviour", default="face/head/paw groom")
    ap.add_argument("--fps-out", type=float, default=30)
    ap.add_argument("--target-dur-s", type=float, default=0.4, help="median syllable duration to aim for (AR-only stage)")
    ap.add_argument("--kappas-ar", default="1e5,1e6,1e7")
    ap.add_argument("--full-iters", type=int, default=300)
    ap.add_argument("--min-share", type=float, default=0.03)
    ap.add_argument("--min-enrich", type=float, default=2.0)
    ap.add_argument("--video", help="default: the one in RUN_DIR/summary.txt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import jax
    import keypoint_moseq as kpms

    run = Path(args.run_dir)
    proj = run / "kpms"
    out = proj / "groom"
    (out / "clips").mkdir(parents=True, exist_ok=True)
    f, xy_full, conf_full = load_keypoints(run)
    fps_in = (len(f) - 1) / f.t.iloc[-1]
    step = max(1, round(fps_in / args.fps_out))
    fps = fps_in / step
    sub = np.arange(0, len(f), step)
    name = run.parent.parent.name
    coords, confs = {name: xy_full[sub]}, {name: conf_full[sub]}
    print(f"{len(f)} frames at {fps_in:.1f} fps -> every {step}th: {len(sub)} at {fps:.1f} fps", flush=True)

    kpms.setup_project(str(proj), overwrite=True)
    kpms.update_config(str(proj), bodyparts=list(KP), use_bodyparts=list(KP), skeleton=SKELETON,
                       anterior_bodyparts=["snout"], posterior_bodyparts=["tail_base"], fps=fps,
                       conf_threshold=0.15)  # our heatmap scores run lower than SLEAP's (README)
    config = lambda: kpms.load_config(str(proj))
    data, metadata = kpms.format_data(coords, confs, **config())
    pca = kpms.fit_pca(**data, **config())
    kpms.save_pca(pca, str(proj))
    cs = np.cumsum(pca.explained_variance_ratio_)
    latent_dim = int(np.searchsorted(cs, 0.9) + 1)
    kpms.update_config(str(proj), latent_dim=latent_dim)
    print(f"PCA: {latent_dim} dims explain {cs[latent_dim - 1]:.1%} of the aligned pose variance", flush=True)

    # ---- kappa for the AR-only stage: aim at the target median duration
    target = args.target_dur_s * fps
    trials = []
    for kappa in [float(v) for v in args.kappas_ar.split(",")]:
        m = kpms.init_model(data, pca=pca, **config(), seed=jax.random.PRNGKey(args.seed))
        m = kpms.update_hypparams(m, kappa=kappa)
        m, mname = kpms.fit_model(m, data, metadata, str(proj), f"ar_kappa{kappa:g}", ar_only=True, num_iters=50,
                                  generate_progress_plots=False)
        trials.append((kappa, median_duration(m), mname))
        print(f"  AR-only kappa {kappa:g}: median syllable {trials[-1][1] / fps:.2f} s", flush=True)
    kappa_ar, dur_ar, ar_name = min(trials, key=lambda r: abs(np.log(r[1] / target)))

    # ---- full model from the chosen AR-only fit, kappa / 10 (the docs' usual step)
    m, data, metadata, it = kpms.load_checkpoint(str(proj), ar_name, iteration=50)
    kappa_full = kappa_ar / 10
    m = kpms.update_hypparams(m, kappa=kappa_full)
    m = kpms.fit_model(m, data, metadata, str(proj), ar_name, ar_only=False, start_iter=it,
                       num_iters=it + args.full_iters, generate_progress_plots=False)[0]
    kpms.reindex_syllables_in_checkpoint(str(proj), ar_name)
    m, data, metadata, it = kpms.load_checkpoint(str(proj), ar_name)
    res = kpms.extract_results(m, metadata, str(proj), ar_name, overwrite=True)
    syl_sub = np.asarray(res[name]["syllable"])
    syl = np.repeat(syl_sub, step)[: len(f)]  # back to every frame
    if len(syl) < len(f):
        syl = np.r_[syl, np.full(len(f) - len(syl), syl[-1])]
    _, lens = run_lengths(syl_sub)
    print(f"full model: kappa {kappa_full:g}, median syllable {np.median(lens) / fps:.2f} s, "
          f"{len(np.unique(syl_sub))} syllables used", flush=True)

    # ---- syllables inside the behaviour
    beh = f.behaviour.to_numpy()
    inb = beh == args.behaviour
    ct = pd.crosstab(syl, beh)
    ct.to_csv(out / "behaviour_by_syllable.csv")
    p_s = np.bincount(syl, minlength=syl.max() + 1) / len(syl)
    p_s_b = np.bincount(syl[inb], minlength=syl.max() + 1) / inb.sum()
    starts, lens_f = run_lengths(syl)
    prof = profile(xy_full, conf_full, fps_in)
    rows = []
    for s in np.flatnonzero(p_s_b > 0):
        runs = [(a, l) for a, l in zip(starts, lens_f) if syl[a] == s and inb[a : a + l].any()]
        sel = inb & (syl == s)
        top = pd.crosstab(syl[syl == s], beh[syl == s], normalize="index").iloc[0].sort_values(ascending=False)
        rows.append(dict(syllable=s, share_of_behaviour=p_s_b[s], enrichment=p_s_b[s] / p_s[s], instances=len(runs),
                         median_dur_s=np.median([l for _, l in runs]) / fps_in if runs else np.nan,
                         share_of_syllable_in_behaviour=sel.sum() / (syl == s).sum(),
                         other_labels=", ".join(f"{k} {v:.0%}" for k, v in top.items() if k != args.behaviour and v >= 0.1),
                         **prof[sel].median().round(2).to_dict()))
    tab = pd.DataFrame(rows).sort_values("share_of_behaviour", ascending=False).set_index("syllable")
    tab.to_csv(out / "groom_syllables.csv")
    groom = tab[(tab.share_of_behaviour >= args.min_share) & (tab.enrichment >= args.min_enrich)]
    base = prof[inb].median().round(2)

    lines = [f"run: {run}", f"behaviour: {args.behaviour}: {inb.sum()} frames ({inb.sum() / fps_in:.1f} s), "
             f"{len(run_lengths(inb.astype(int))[0][inb[run_lengths(inb.astype(int))[0]]])} bouts",
             f"keypoint-MoSeq {kpms.__version__}: {len(sub)} frames at {fps:.1f} fps, PCA {latent_dim} dims "
             f"({cs[latent_dim - 1]:.1%}); AR-only kappas " + ", ".join(f"{k:g} -> {d / fps:.2f} s" for k, d, _ in trials)
             + f"; chose {kappa_ar:g}; full model kappa {kappa_full:g}, {args.full_iters} iters: median syllable "
             f"{np.median(lens) / fps:.2f} s, {len(np.unique(syl_sub))} syllables used",
             f"syllables seen in the behaviour: {len(tab)}; covering 80% of its frames: "
             f"{int(np.searchsorted(np.cumsum(tab.share_of_behaviour.to_numpy()), 0.8) + 1)}",
             f"groom syllables (>= {args.min_share:.0%} of the behaviour, enrichment >= {args.min_enrich:g}): {list(groom.index)}",
             "", "profile columns: body-length units in the body frame (tail base -> snout); forepaws_ahead < 0 = "
             "forepaws behind the snout; forepaw_asym > 0 = left paw further from the snout; forepaw speed px/s",
             f"all {args.behaviour} frames, median: " + ", ".join(f"{k} {v}" for k, v in base.items()), "",
             tab.to_string(float_format=lambda v: f"{v:.2f}")]
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    json.dump(dict(vars(args), kappa_ar=kappa_ar, kappa_full=kappa_full, latent_dim=latent_dim, model=ar_name,
                   fps=fps, step=step), open(out / "config.json", "w"), indent=2)

    # ---- ethogram of the longest bouts
    b_starts, b_lens = run_lengths(inb.astype(int))
    bouts = sorted([(a, l) for a, l in zip(b_starts, b_lens) if inb[a]], key=lambda r: -r[1])[:6]
    cmap = plt.get_cmap("tab20")
    fig, axes = plt.subplots(len(bouts), 1, figsize=(14, 1.1 * len(bouts) + 0.6), squeeze=False)
    for ax, (a, l) in zip(axes[:, 0], bouts):
        lo, hi = max(a - int(2 * fps_in), 0), min(a + l + int(2 * fps_in), len(f))
        t = np.arange(lo, hi) / fps_in
        ax.scatter(t, np.zeros(hi - lo), c=[cmap(s % 20) for s in syl[lo:hi]], marker="|", s=300)
        ax.scatter(t[inb[lo:hi]], np.full(inb[lo:hi].sum(), -1), c="k", marker="|", s=80)
        for s0, l0 in zip(*run_lengths(syl[lo:hi])):
            if l0 >= 0.25 * fps_in:
                ax.text(t[s0], 0.45, str(syl[lo + s0]), fontsize=7)
        ax.set_yticks([0, -1], ["syllable", "label"])
        ax.set_ylim(-1.6, 1)
    axes[-1, 0].set_xlabel("time (s)")
    axes[0, 0].set_title(f"{name}: keypoint-MoSeq syllables in the {len(bouts)} longest '{args.behaviour}' bouts")
    fig.tight_layout()
    fig.savefig(out / "ethogram.png", dpi=110)

    # ---- clips of the groom syllables, instances inside the behaviour (longest first)
    video = args.video or next(l.split(": ", 1)[1].strip() for l in open(run / "summary.txt") if l.startswith("video: "))
    cx, cy = f.cx.to_numpy(), f.cy.to_numpy()
    for s in groom.index:
        runs = sorted([(a, l) for a, l in zip(starts, lens_f) if syl[a] == s and inb[a : a + l].any()], key=lambda r: -r[1])
        clip_montage(video, f, cx, cy, xy_full, [a for a, _ in runs], out / "clips" / f"syllable_{s:02d}.mp4", fps_in)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
