"""TW-FINCH (Sarfraz et al., CVPR 2021) on embed_vjepa.py window features: temporally weighted
first-neighbour clustering into action segments, and those segments grouped into recurring types.

    python tw_finch.py RUN_DIR [--emb vjepa21_rc] [--feat feats] [--pca 50] [--ks 5,10,20,40,80]

TW-FINCH. Every window gets a time stamp t in (0, 1] (its centre frame / the session's last frame)
and distance d(i, j) = cosine distance of the features x |t_i - t_j|. Each window is linked to its
first neighbour (the j with the smallest d); the connected components of that graph are the first
partition. Each cluster is then replaced by the mean of its windows' features and time stamps, and
the same step repeats until one cluster is left or nothing merges: a hierarchy of partitions. An
exact number of clusters K comes from the last level with more than K clusters by merging the
closest pair (same distance) one at a time. This follows the authors' code
(github.com/ssarfraz/FINCH-Clustering, TW-FINCH/python/twfinch.py), with the distances computed in
GPU chunks instead of an N x N matrix. The paper uses it on single videos of a few minutes, where
each action happens once or a few times, and scores it against labels with Hungarian matching.

On a 4 h session the time weight makes every cluster a stretch of time, so the direct K-cluster
partitions (tw_k<K>) are mostly long blocks of the session, not recurring behaviours. This script
therefore also uses TW-FINCH as a segmenter: the runs of equal labels at one hierarchy level are
segments, and k-means on the segments' mean features (weighted by length) turns them into K
recurring types (twseg_l<level>_k<K>). Control (shufseg_l<level>_k<K>): the same segment lengths in
a random order, then the same k-means. It has the same amount of averaging over neighbouring
windows, so it isolates whether TW-FINCH's boundaries are better than arbitrary ones.

Leak warning for score_behaviour.py: a segment's label depends on all its windows, so a window's
label sees up to a segment's length into the future. Its future-movement scores are inflated
against causal partitions (km_emb, hdb); compare twseg with shufseg, not with those. A leak-free
check here: segment homogeneity, the share of variance of window kinematics and posture
(part_tracks.py, independent of V-JEPA) explained by segment identity, TW-FINCH vs shuffled
lengths at the same segment count.

Windows where the localizer lost the mouse in any frame are left out (as cluster_embeddings.py);
their absence widens the time gap, and a gap of more than one hop also ends a segment.

Output in RUN_DIR/<emb>/twfinch/ (or --out-name):
  labels.npz     start + one label array per partition over the kept windows (score_behaviour.py --labels)
  segments.npz   per level: segment id of every kept window
  summary.txt    hierarchy (clusters, segments, segment lengths), homogeneity, partitions
  overview.png   ethograms: tw_k* over the session; types vs controls over a 10 min excerpt
  config.json
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

CHUNK = 4096  # rows per GPU block of the distance matrix


# ---------------------------------------------------------------- TW-FINCH
def _unit(M):
    return M / M.norm(dim=1, keepdim=True).clamp_min(1e-12)


def tw_dist(U, t, rows):
    """Distances from rows of U (unit vectors) to all of U: cosine distance x |time difference|."""
    return (1 - U[rows] @ U.T).clamp_min_(0) * (t[rows, None] - t[None]).abs()


def first_neighbours(M, t):
    """Index of every row's first neighbour under the time-weighted cosine distance."""
    U, n = _unit(M), len(M)
    nn = torch.empty(n, dtype=torch.long, device=M.device)
    for a in range(0, n, CHUNK):
        rows = torch.arange(a, min(a + CHUNK, n), device=M.device)
        d = tw_dist(U, t, rows)
        d[torch.arange(len(rows)), rows] = float("inf")
        nn[rows] = d.argmin(1)
    return nn.cpu().numpy()


def cluster_means(X, t, lab, n_clust):
    """Mean feature and time stamp of every cluster (means of the windows, as FINCH)."""
    lab_t = torch.as_tensor(lab, device=X.device)
    cnt = torch.zeros(n_clust, device=X.device).index_add_(0, lab_t, torch.ones(len(lab), device=X.device))
    M = torch.zeros(n_clust, X.shape[1], device=X.device).index_add_(0, lab_t, X) / cnt[:, None]
    tm = torch.zeros(n_clust, device=X.device, dtype=t.dtype).index_add_(0, lab_t, t) / cnt.to(t.dtype)
    return M, tm, cnt


def in_time_order(lab):
    """Relabel 0..K-1 in order of first appearance."""
    _, first = np.unique(lab, return_index=True)
    order = np.argsort(np.argsort(first))
    return order[np.unique(lab, return_inverse=True)[1]]


def tw_finch(X, t):
    """Hierarchy of partitions: list of label arrays (finest first), as twfinch.FINCH(tw_finch=True)."""
    levels, lab, n_prev = [], np.arange(len(X)), len(X)
    M, tm = X, t
    while True:
        nn = first_neighbours(M, tm)
        n = len(nn)
        g = coo_matrix((np.ones(n), (np.arange(n), nn)), shape=(n, n))
        n_clust, u = connected_components(g, directed=True, connection="weak")
        if n_clust == 1 or n_clust >= n_prev:  # the reference drops the single-cluster level
            break
        lab = in_time_order(u[lab])
        levels.append(lab)
        n_prev = n_clust
        M, tm, _ = cluster_means(X, t, lab, n_clust)
    return levels


def merge_to(X, t, lab, k):
    """From a partition with more than k clusters, merge the closest pair (TW distance between
    cluster means) until k are left (twfinch.req_numclust)."""
    n_clust = int(lab.max()) + 1
    M, tm, cnt = cluster_means(X, t, lab, n_clust)
    U = _unit(M)
    D = torch.cat([tw_dist(U, tm, torch.arange(a, min(a + CHUNK, n_clust), device=X.device))
                   for a in range(0, n_clust, CHUNK)])
    D.fill_diagonal_(float("inf"))
    owner = np.arange(n_clust)
    dead = torch.zeros(n_clust, dtype=torch.bool, device=X.device)
    for _ in range(n_clust - k):
        a, b = sorted(divmod(int(D.argmin()), n_clust))
        M[a] = (cnt[a] * M[a] + cnt[b] * M[b]) / (cnt[a] + cnt[b])
        tm[a] = (cnt[a] * tm[a] + cnt[b] * tm[b]) / (cnt[a] + cnt[b])
        cnt[a] += cnt[b]
        dead[b] = True
        U[a] = _unit(M[a : a + 1])[0]
        row = tw_dist(U, tm, torch.tensor([a], device=X.device))[0]
        row[dead] = float("inf")
        row[a] = float("inf")
        D[a] = D[:, a] = row
        D[b] = D[:, b] = float("inf")
        owner[owner == b] = a
    return in_time_order(owner[lab])


# ---------------------------------------------------------------- segments and types
def segments_of(lab, gap):
    """Segment id per window: runs of equal labels, also cut where gap (time jump) is True."""
    cut = np.r_[False, (np.diff(lab) != 0) | gap[1:]]
    return np.cumsum(cut)


def shuffled_segments(seg, gap, rng):
    """Same segment lengths in random order, cut at the same gaps (lengths shuffled within each
    gap-free stretch so no segment spans a gap)."""
    out, nxt = np.empty_like(seg), 0
    stretch = np.cumsum(np.r_[False, gap[1:]])
    for s in np.unique(stretch):
        idx = np.flatnonzero(stretch == s)
        lens = np.bincount(seg[idx] - seg[idx].min())
        lens = rng.permutation(lens[lens > 0])
        out[idx] = nxt + np.repeat(np.arange(len(lens)), lens)
        nxt += len(lens)
    return out


def segment_types(Z, seg, k, seed):
    """k-means (length-weighted) on segment mean features -> type label per window."""
    n_seg = seg.max() + 1
    cnt = np.bincount(seg, minlength=n_seg)
    means = np.zeros((n_seg, Z.shape[1]))
    np.add.at(means, seg, Z)
    means /= cnt[:, None]
    km = KMeans(k, n_init=4, random_state=seed).fit(means, sample_weight=cnt)
    return in_time_order(km.labels_[seg])


def homogeneity(K, seg):
    """Share of the variance of the (standardised) columns of K explained by segment identity."""
    n_seg = seg.max() + 1
    cnt = np.bincount(seg, minlength=n_seg)
    sums = np.zeros((n_seg, K.shape[1]))
    np.add.at(sums, seg, K)
    within = ((K - (sums / cnt[:, None])[seg]) ** 2).sum()
    return 1 - within / ((K - K.mean(0)) ** 2).sum()


def window_kinematics(run, z, keep):
    """Per kept window: body-frame speeds and turn rate (features.npz) + posture at the centre frame
    (part_tracks.py), rank-standardised. Independent of V-JEPA."""
    from score_behaviour import frame_measures

    post = frame_measures(run)["post"][z["centre"][keep]]
    K = np.column_stack([z[k][keep] for k in ("fwd_speed", "lat_speed", "turn_rate")] + [post])
    K = np.nan_to_num(K)
    r = K.argsort(0).argsort(0) / (len(K) - 1)  # ranks in 0..1: robust to the heavy-tailed speeds
    return StandardScaler().fit_transform(r)


def bout_lengths(lab):
    return np.diff(np.r_[0, np.flatnonzero(np.diff(lab)) + 1, len(lab)])


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--emb", default="vjepa21_rc", help="embed_vjepa.py output folder in RUN_DIR")
    ap.add_argument("--feat", default="feats", help="feature key in features.npz (feats = last layer)")
    ap.add_argument("--pca", type=int, default=50, help="standardise + PCA to this many dims first (0 = raw features)")
    ap.add_argument("--ks", default="5,10,20,40,80", help="numbers of clusters / segment types")
    ap.add_argument("--max-seg-s", type=float, default=60.0, help="use hierarchy levels with mean segment length up to this as segmenters")
    ap.add_argument("--shuffles", type=int, default=5, help="shuffled-length controls for the homogeneity score")
    ap.add_argument("--out-name", default="twfinch")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    run = Path(args.run_dir)
    out = run / args.emb / args.out_name
    out.mkdir(parents=True, exist_ok=True)
    ks = [int(k) for k in args.ks.split(",")]
    rng = np.random.default_rng(args.seed)

    z = np.load(run / args.emb / "features.npz")
    fps = float(z["fps"])
    found = pd.read_csv(run / "frames.csv", usecols=["mouse_found"]).mouse_found.to_numpy()
    cs = np.r_[0, np.cumsum(~found)]
    keep = cs[z["stop"]] - cs[z["start"]] == 0
    start, centre = z["start"][keep], z["centre"][keep]
    hop = int(np.median(np.diff(z["start"])))
    hop_s = hop / fps
    gap = np.r_[False, np.diff(start) > hop]
    feats = z[args.feat][keep].astype(np.float32)
    if args.pca:
        pca = PCA(args.pca, random_state=args.seed).fit(StandardScaler().fit_transform(feats))
        Z = pca.transform(StandardScaler().fit_transform(feats))
        feat_desc = f"standardised, PCA {args.pca} ({pca.explained_variance_ratio_.sum():.1%} variance)"
    else:
        Z, feat_desc = feats, "raw"
    print(f"{keep.sum()}/{len(keep)} windows kept ({hop_s:.2f} s hop, {gap.sum()} gaps); features {feat_desc}", flush=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.as_tensor(Z, dtype=torch.float32, device=dev)
    t = torch.as_tensor(centre / (len(found) - 1), dtype=torch.float64, device=dev)  # fp64: 1/N steps over hours
    levels = tw_finch(X, t)
    print(f"TW-FINCH hierarchy in {time.time() - t0:.0f}s: " + " -> ".join(str(l.max() + 1) for l in levels), flush=True)

    parts, lines = {}, []
    # ---- direct TW-FINCH at K clusters
    counts = np.array([l.max() + 1 for l in levels])
    for k in ks:
        above = np.flatnonzero(counts >= k)
        if not len(above):
            print(f"  K={k}: finest level has only {counts[0]} clusters; skipped")
            continue
        base = levels[above[-1]]
        parts[f"tw_k{k}"] = base if counts[above[-1]] == k else merge_to(X, t, base, k)

    # ---- hierarchy table + segmenters
    Kin = window_kinematics(run, z, keep)
    seg_levels, rows = {}, []
    for i, lab in enumerate(levels):
        seg = segments_of(lab, gap)
        lens = np.bincount(seg) * hop_s
        h_tw = homogeneity(Kin, seg)
        h_sh = [homogeneity(Kin, shuffled_segments(seg, gap, rng)) for _ in range(args.shuffles)]
        rows.append(dict(level=i, clusters=lab.max() + 1, segments=seg.max() + 1, seg_median_s=np.median(lens),
                         seg_mean_s=lens.mean(), seg_p90_s=np.quantile(lens, 0.9),
                         segs_per_cluster=(seg.max() + 1) / (lab.max() + 1),
                         homog_tw=h_tw, homog_shuffled=np.mean(h_sh), homog_gain=h_tw - np.mean(h_sh),
                         homog_gain_z=(h_tw - np.mean(h_sh)) / (np.std(h_sh) + 1e-12)))
        if lens.mean() <= args.max_seg_s and seg.max() + 1 > max(ks):
            seg_levels[i] = seg
    hier = pd.DataFrame(rows).set_index("level")
    print(hier.to_string(float_format=lambda v: f"{v:.3f}"), flush=True)

    for i, seg in seg_levels.items():
        sh = shuffled_segments(seg, gap, np.random.default_rng(args.seed + 1000 + i))
        for k in ks:
            parts[f"twseg_l{i}_k{k}"] = segment_types(Z, seg, k, args.seed)
            parts[f"shufseg_l{i}_k{k}"] = segment_types(Z, sh, k, args.seed)
    for k in ks:  # per-window k-means on the same features, for the ethogram and summary
        parts[f"win_km_k{k}"] = in_time_order(KMeans(k, n_init=4, random_state=args.seed).fit(Z).labels_)
    print(f"{len(parts)} partitions ({time.time() - t0:.0f}s)", flush=True)

    # ---- summary
    prow = []
    for name, lab in parts.items():
        bl = bout_lengths(lab) * hop_s
        p = np.bincount(lab) / len(lab)
        p = p[p > 0]
        seg = segments_of(lab, gap)
        prow.append(dict(partition=name, k=len(p), H_bits=-(p * np.log2(p)).sum(), bout_median_s=np.median(bl),
                         bout_mean_s=bl.mean(), bouts_per_cluster=len(bl) / len(p), homog=homogeneity(Kin, seg)))
    psum = pd.DataFrame(prow).set_index("partition")
    psum["homog_of_types"] = [homogeneity(Kin, lab) for lab in parts.values()]
    fmt = lambda v: f"{v:.3f}" if abs(v) < 100 else f"{v:.0f}"
    lines = [f"run: {run}", f"embedding: {args.emb} [{args.feat}], {feat_desc}",
             f"windows: {keep.sum()} of {len(keep)} kept, hop {hop_s:.2f} s, {gap.sum()} gaps",
             f"TW-FINCH levels (clusters): {' -> '.join(map(str, counts))}", "",
             "hierarchy: segments = runs of equal labels; homog = share of kinematic + posture variance explained "
             "by segment identity (part_tracks, not V-JEPA); shuffled = same lengths in random order "
             f"(mean of {args.shuffles}); gain_z in SDs of the shuffles",
             hier.to_string(float_format=fmt), "",
             f"segmenter levels used for twseg / shufseg: {sorted(seg_levels)}",
             "partitions: homog = variance explained by the partition's bouts; homog_of_types = by its labels",
             psum.to_string(float_format=fmt)]
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-1:]))

    np.savez_compressed(out / "labels.npz", start=start, **parts)
    np.savez_compressed(out / "segments.npz", start=start, **{f"level{i}": segments_of(l, gap) for i, l in enumerate(levels)})
    json.dump(dict(vars(args), run_dir=str(run), n_windows=int(keep.sum()), hop_s=hop_s, levels=counts.tolist(),
                   seg_levels=sorted(seg_levels)), open(out / "config.json", "w"), indent=2)
    plot(parts, start / fps, hop_s, seg_levels, ks, out / "overview.png", f"{run.parent.parent.name} / {args.emb}")
    print(f"done in {time.time() - t0:.0f}s -> {out}")


def plot(parts, tsec, hop_s, seg_levels, ks, path, title):
    """Top: direct TW-FINCH partitions over the session. Bottom: 10 min excerpt, types vs controls."""
    cmap = plt.get_cmap("tab20")
    direct = [n for n in parts if n.startswith("tw_k")]
    k_mid = 20 if 20 in ks else ks[len(ks) // 2]
    excerpt = [f"win_km_k{k_mid}"] + [f"{p}_l{i}_k{k_mid}" for i in sorted(seg_levels) for p in ("twseg", "shufseg")]
    excerpt = [n for n in excerpt if n in parts]
    fig, axes = plt.subplots(2, 1, figsize=(16, 2 + 0.45 * (len(direct) + len(excerpt))),
                             gridspec_kw=dict(height_ratios=[max(len(direct), 1), max(len(excerpt), 1)]))

    def strip(ax, names, sel, x):
        for r, n in enumerate(names):
            lab = parts[n][sel]
            ax.scatter(x, np.full(len(x), r), c=[cmap(v % 20) for v in lab], marker="|", s=120, lw=1)
        ax.set_yticks(range(len(names)), names)
        ax.set_ylim(len(names) - 0.5, -0.5)

    every = slice(None)
    strip(axes[0], direct, every, tsec / 60)
    axes[0].set_xlabel("time (min)")
    axes[0].set_title(f"{title}: TW-FINCH at K clusters over the session (colour = cluster)")
    mid = tsec[len(tsec) // 2]
    sel = (tsec >= mid) & (tsec < mid + 600)
    strip(axes[1], excerpt, sel, tsec[sel] / 60)
    axes[1].set_xlabel("time (min)")
    axes[1].set_title(f"10 min excerpt: {k_mid} types from per-window k-means, TW-FINCH segments (twseg) "
                      "and shuffled-length segments (shufseg) per level")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


if __name__ == "__main__":
    main()
