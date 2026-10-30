"""Sub-types within one labelled behaviour (default face/head/paw groom) from embed_vjepa.py features:
cluster only the windows inside the behaviour and test whether the clusters are distinct, recurring
and stable, rather than one continuum or one cluster per bout.

    python groom_vjepa.py RUN_DIR [--emb vjepa21_fine_raw] [--feat feats] [--behaviour "face/head/paw groom"]
                          [--min-frac 0.75] [--ks 2,3,4,5,6,8] [--clips-k K] [--max-per-bout N]

RUN_DIR: a run_video.py output made with --labels (frames.csv 'behaviour'); keypoint columns (a keypoint
network) are used for the kinematic profile if present. Use short, dense windows for short bouts
(e.g. embed_vjepa.py --sample-fps 30 --hop-sec 0.1: 0.52 s windows every 0.1 s).

Windows: at least --min-frac of their frames carry the label and the mouse is found in all of them.
Each window belongs to the label bout that contains its centre. --max-per-bout N keeps at most N
evenly spaced windows per bout, so a few long bouts cannot form clusters of their own. Features: standardised, PCA fitted
on these windows (dims for 90% of the variance, at most 20).

Per k (k-means, 20 starts):
  silhouette       how separated the clusters are (cosine on the PCA features)
  null_sil         the same for k-means on a Gaussian with the same covariance and n (mean of 20):
                   one blob, no sub-types. excess = silhouette - null_sil (z in SDs of the null)
  stability_ari    refit on bootstrap resamples of BOUTS (20); ARI with the full fit on the
                   windows both contain (mean)
  bout_ami         AMI of cluster vs bout: near 1 = each cluster is one bout's windows
  recurring        clusters seen in >= 3 bouts with >= 10% of their windows outside their top bout
Per cluster at --clips-k (default: the k with the largest excess z): share, bouts, top-bout share,
kinematic profile (kpms_groom.profile: body-frame keypoint measures, medians over the windows'
frames), and clips/cluster_<c>.mp4 (up to 8 windows from different bouts, raw video, keypoints).

Output in RUN_DIR/<emb>/groom/: summary.txt, windows.csv, ethogram.png, clips/
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from kpms_groom import KP, clip_montage, profile


def run_starts(x):
    """Start index of every run of equal values in x, and run ids per element."""
    starts = np.r_[0, np.flatnonzero(np.diff(x)) + 1]
    return starts, np.cumsum(np.r_[True, np.diff(x) != 0]) - 1


def kmeans(Z, k, seed):
    return KMeans(k, n_init=20, random_state=seed).fit(Z)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--emb", default="vjepa21_fine_raw")
    ap.add_argument("--feat", default="feats")
    ap.add_argument("--behaviour", default="face/head/paw groom")
    ap.add_argument("--min-frac", type=float, default=0.75)
    ap.add_argument("--ks", default="2,3,4,5,6,8")
    ap.add_argument("--clips-k", type=int)
    ap.add_argument("--max-per-bout", type=int, help="at most this many evenly spaced windows per bout")
    ap.add_argument("--video", help="default: the one in RUN_DIR/summary.txt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run = Path(args.run_dir)
    out = run / args.emb / "groom" / (args.feat + (f"_max{args.max_per_bout}" if args.max_per_bout else ""))
    (out / "clips").mkdir(parents=True, exist_ok=True)
    z = np.load(run / args.emb / "features.npz")
    f = pd.read_csv(run / "frames.csv")
    fps = float(z["fps"])
    inb = (f.behaviour.to_numpy() == args.behaviour).astype(int)
    found = f.mouse_found.to_numpy()
    c_in, c_lost = np.r_[0, np.cumsum(inb)], np.r_[0, np.cumsum(~found)]
    start, stop, centre = z["start"], z["stop"], z["centre"]
    frac = (c_in[stop] - c_in[start]) / (stop - start)
    keep = (frac >= args.min_frac) & (c_lost[stop] - c_lost[start] == 0) & (inb[centre] == 1)
    _, run_id = run_starts(inb)
    if args.max_per_bout:
        for b in np.unique(run_id[centre[keep]]):
            idx = np.flatnonzero(keep & (run_id[centre] == b))
            if len(idx) > args.max_per_bout:
                drop = np.setdiff1d(idx, idx[np.linspace(0, len(idx) - 1, args.max_per_bout).round().astype(int)])
                keep[drop] = False
    bout = run_id[centre[keep]]
    bout = np.unique(bout, return_inverse=True)[1]
    n_bouts_all = int(inb[run_starts(inb)[0]].sum())
    print(f"{keep.sum()} windows in {bout.max() + 1} of {n_bouts_all} '{args.behaviour}' bouts", flush=True)

    X = StandardScaler().fit_transform(z[args.feat][keep].astype(np.float32))
    pca = PCA(min(20, len(X) - 1), random_state=args.seed).fit(X)
    d = int(min(20, np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.9) + 1))
    Z = pca.transform(X)[:, :d]
    rng = np.random.default_rng(args.seed)
    cov = np.cov(Z.T)
    bouts_u = np.unique(bout)

    rows, fits = [], {}
    for k in [int(v) for v in args.ks.split(",")]:
        km = kmeans(Z, k, args.seed)
        lab = km.labels_
        fits[k] = lab
        sil = silhouette_score(Z, lab, metric="cosine")
        null = [silhouette_score(G, kmeans(G, k, s).labels_, metric="cosine")
                for s, G in enumerate(rng.multivariate_normal(np.zeros(d), cov, size=(20, len(Z))))]
        aris = []
        for s in range(20):
            pick = rng.choice(bouts_u, len(bouts_u), replace=True)
            idx = np.concatenate([np.flatnonzero(bout == b) for b in pick])
            lab_b = kmeans(Z[idx], k, s).predict(Z)
            aris.append(adjusted_rand_score(lab, lab_b))
        rec = 0
        for c in range(k):
            bc = bout[lab == c]
            top = np.bincount(bc).max() / len(bc)
            rec += (len(np.unique(bc)) >= 3) and (1 - top >= 0.1)
        rows.append(dict(k=k, silhouette=sil, null_sil=np.mean(null), excess=sil - np.mean(null),
                         excess_z=(sil - np.mean(null)) / (np.std(null) + 1e-12), stability_ari=np.mean(aris),
                         bout_ami=adjusted_mutual_info_score(bout, lab), recurring=f"{rec}/{k}"))
        print(f"  k={k}: silhouette {sil:.3f} (null {np.mean(null):.3f}), stability {np.mean(aris):.2f}, "
              f"bout AMI {rows[-1]['bout_ami']:.2f}, recurring {rec}/{k}", flush=True)
    tab = pd.DataFrame(rows).set_index("k")
    kc = args.clips_k or int(tab.excess_z.idxmax())
    lab = fits[kc]

    # ---- per cluster at kc: bouts and kinematics
    has_kp = all(f"kp_{n}_x" in f for n in KP)
    if has_kp:
        xy = np.stack([f[[f"kp_{n}_x", f"kp_{n}_y"]].to_numpy() for n in KP], 1).astype(np.float64)
        conf = np.stack([f[f"kp_{n}_score"].to_numpy() for n in KP], 1)
        prof = profile(xy, conf, fps)
        wprof = pd.DataFrame([prof.iloc[a:b].median() for a, b in zip(start[keep], stop[keep])])
    crow = []
    for c in range(kc):
        m = lab == c
        bc = bout[m]
        crow.append(dict(cluster=c, windows=m.sum(), share=m.mean(), bouts=len(np.unique(bc)),
                         top_bout_share=np.bincount(bc).max() / m.sum(),
                         **(wprof[m].median().round(2).to_dict() if has_kp else {})))
    ctab = pd.DataFrame(crow).set_index("cluster")
    fmt = lambda v: f"{v:.3f}" if abs(v) < 100 else f"{v:.0f}"
    lines = [f"run: {run}", f"embedding: {args.emb} [{args.feat}], window {np.median(stop - start) / fps:.2f} s, "
             f"hop {np.median(np.diff(start)) / fps:.2f} s",
             f"behaviour: {args.behaviour}; windows with >= {args.min_frac:.0%} of frames labelled: {keep.sum()} "
             f"in {bout.max() + 1} of {n_bouts_all} bouts" + (f" (at most {args.max_per_bout} per bout)" if args.max_per_bout else ""),
             f"features: standardised, PCA {d} dims "
             f"({np.cumsum(pca.explained_variance_ratio_)[d - 1]:.1%} of the variance of these windows)", "",
             tab.to_string(float_format=fmt), "",
             f"clusters at k={kc} (largest excess z unless --clips-k); profile = keypoint measures (kpms_groom.profile), "
             "body-length units; forepaws_ahead < 0 = forepaws behind the snout", ctab.to_string(float_format=fmt)]
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    pd.DataFrame(dict(start=start[keep], centre=centre[keep], t=centre[keep] / fps, bout=bout,
                      **{f"k{k}": v for k, v in fits.items()})).to_csv(out / "windows.csv", index=False)

    # ---- ethogram: the longest bouts, cluster per window centre
    cmap = plt.get_cmap("tab10")
    order = np.argsort(-np.bincount(bout))[:8]
    fig, axes = plt.subplots(len(order), 1, figsize=(12, 0.7 * len(order) + 0.8), squeeze=False)
    for ax, b in zip(axes[:, 0], order):
        m = bout == b
        ax.scatter(centre[keep][m] / fps, np.zeros(m.sum()), c=[cmap(v) for v in lab[m]], marker="|", s=300)
        ax.set_yticks([])
        ax.set_ylabel(f"bout {b}", rotation=0, ha="right", fontsize=8)
    axes[-1, 0].set_xlabel("time (s)")
    axes[0, 0].set_title(f"{args.emb} [{args.feat}]: k={kc} clusters in the longest '{args.behaviour}' bouts")
    fig.tight_layout()
    fig.savefig(out / "ethogram.png", dpi=110)

    # ---- clips: per cluster, the most central window from up to 8 different bouts
    if has_kp:
        video = args.video or next(l.split(": ", 1)[1].strip() for l in open(run / "summary.txt") if l.startswith("video: "))
        km = kmeans(Z, kc, args.seed)
        dist = np.linalg.norm(Z - km.cluster_centers_[km.labels_], axis=1)
        for c in range(kc):
            idx = np.flatnonzero(km.labels_ == c)
            idx = idx[np.argsort(dist[idx])]
            seen, pick = set(), []
            for i in idx:
                if bout[i] not in seen:
                    seen.add(bout[i])
                    pick.append(start[keep][i])
                if len(pick) == 8:
                    break
            clip_montage(video, f, f.cx.to_numpy(), f.cy.to_numpy(), xy, np.array(pick), out / "clips" / f"cluster_{c}.mp4",
                         fps, before=0.25, after=0.75)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
