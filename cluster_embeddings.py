"""Cluster embed_vjepa.py's window features: PCA -> UMAP -> HDBSCAN.

    python cluster_embeddings.py RUN_DIR [--emb vjepa_window] [--pca 50] [--umap-dim 10]
                                 [--min-cluster-frac 0.005] [--examples 12]

--emb is the embed_vjepa.py output folder in RUN_DIR (vjepa = no
canonicalisation, vjepa_window = rotated per window).

Windows where the localizer lost the mouse in any frame are left out.

Output in RUN_DIR/<emb>/clusters/:
  windows.csv      per window: start/centre frame, time, cluster (-1 = noise),
                   membership probability, 2-d UMAP, heading, speed and (if
                   saved) body-frame forward / lateral speed and turn rate
  summary.txt      per cluster: share of time, bout length, speed, forward
                   speed and turn rate, body area, heading concentration
                   (R near 1 = the cluster is one direction the mouse faces
                   in the image, not a behaviour) and how well a linear
                   read-out recovers heading from the features
  overview.png     UMAP by cluster and by heading, ethogram
  contact_sheet.jpg  middle frame of example windows, one row per cluster
  examples/cluster_<k>.mp4  example windows from preview.mp4 (crop | overlay)
"""

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler


def bouts(labels):
    """Run lengths of consecutive equal labels: (label, length) pairs."""
    edges = np.flatnonzero(np.diff(labels)) + 1
    starts = np.r_[0, edges]
    return labels[starts], np.diff(np.r_[starts, len(labels)])


def heading_r2(X, heading):
    """Cross-validated R^2 of (cos, sin) heading from the features (blocked folds)."""
    ok = ~np.isnan(heading)
    y = np.c_[np.cos(np.deg2rad(heading[ok])), np.sin(np.deg2rad(heading[ok]))]
    pred = cross_val_predict(RidgeCV(alphas=np.logspace(-2, 4, 13)), X[ok], y, cv=KFold(5))
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean(0)) ** 2).sum()


def read_preview(cap, first, n):
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    out = []
    for _ in range(n):
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--emb", default="vjepa_window", help="embed_vjepa.py output folder in RUN_DIR")
    ap.add_argument("--pca", type=int, default=50)
    ap.add_argument("--umap-dim", type=int, default=10)
    ap.add_argument("--neighbors", type=int, default=30)
    ap.add_argument("--min-cluster-frac", type=float, default=0.005, help="HDBSCAN min cluster size, share of windows")
    ap.add_argument("--select", choices=["leaf", "eom"], default="leaf",
                    help="HDBSCAN cluster selection: leaf = finest stable clusters; eom (excess of mass) merges "
                         "a continuous cloud into one cluster, which happens once heading no longer splits it")
    ap.add_argument("--examples", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import umap

    run = Path(args.run_dir)
    out = run / args.emb / "clusters"
    (out / "examples").mkdir(parents=True, exist_ok=True)
    z = np.load(run / args.emb / "features.npz")
    motion = [k for k in ("fwd_speed", "lat_speed", "turn_rate") if k in z]
    fr = pd.read_csv(run / "frames.csv", usecols=["frame", "t", "behaviour", "cx", "cy", "mouse_found", "area_body"])
    fps = float(z["fps"])
    start, centre, stop = z["start"], z["centre"], z["stop"]

    found = fr.mouse_found.to_numpy()
    cs = np.r_[0, np.cumsum(~found)]
    keep = cs[stop] - cs[start] == 0
    # body-centre speed over the window, full-res px/s
    xy = fr[["cx", "cy"]].to_numpy()
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    cstep = np.r_[0, np.cumsum(step)]
    speed = (cstep[stop - 1] - cstep[start]) / ((stop - 1 - start) / fps)

    X = StandardScaler().fit_transform(z["feats"][keep].astype(np.float32))
    pca = PCA(args.pca, random_state=args.seed).fit(X)
    Xp = pca.transform(X)
    print(f"{keep.sum()}/{len(keep)} windows kept; PCA {args.pca}: {pca.explained_variance_ratio_.sum():.1%} variance", flush=True)
    emb = umap.UMAP(n_components=args.umap_dim, n_neighbors=args.neighbors, min_dist=0.0, random_state=args.seed).fit_transform(Xp)
    emb2 = umap.UMAP(n_components=2, n_neighbors=args.neighbors, min_dist=0.1, random_state=args.seed).fit_transform(Xp)
    hdb = HDBSCAN(min_cluster_size=max(20, int(args.min_cluster_frac * len(Xp))), min_samples=10,
                  cluster_selection_method=args.select).fit(emb)
    lab, prob = hdb.labels_, hdb.probabilities_
    ks = sorted(set(lab) - {-1})

    head = z["heading"][keep]
    df = pd.DataFrame(dict(start=start[keep], centre=centre[keep], t=centre[keep] / fps, cluster=lab, prob=prob.round(3),
                           umap_x=emb2[:, 0], umap_y=emb2[:, 1], heading=head.round(1),
                           speed=speed[keep].round(1), **{k: z[k][keep].round(1) for k in motion},
                           behaviour=fr.behaviour.to_numpy()[centre[keep]]))
    df.to_csv(out / "windows.csv", index=False)

    # ---- summary
    r2_all = heading_r2(Xp, head)
    hop_s = np.median(np.diff(start)) / fps
    rows = []
    for k in ks + [-1]:
        m = lab == k
        a = np.deg2rad(head[m & ~np.isnan(head)])
        R = np.hypot(np.cos(a).mean(), np.sin(a).mean()) if len(a) else np.nan
        bl, bn = bouts(lab)
        rows.append(dict(cluster=k, windows=m.sum(), share=m.mean(), median_bout_s=np.median(bn[bl == k]) * hop_s,
                         speed_px_s=np.median(speed[keep][m]),
                         **({"fwd_px_s": np.median(df.fwd_speed[m]), "turn_deg_s": np.median(df.turn_rate[m]),
                             "abs_turn_deg_s": np.median(np.abs(df.turn_rate[m]))} if motion else {}),
                         heading_R=R,
                         heading_mean=np.rad2deg(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) if len(a) else np.nan,
                         body_area=np.median(fr.area_body.to_numpy()[centre[keep][m]])))
    summ = pd.DataFrame(rows).set_index("cluster")
    a = np.deg2rad(head[~np.isnan(head)])
    lines = [f"run: {run}", f"windows: {len(keep)} ({(~keep).sum()} dropped: mouse lost), hop {hop_s:.2f} s",
             f"PCA {args.pca} dims: {pca.explained_variance_ratio_.sum():.1%} variance; UMAP {args.umap_dim}-d; "
             f"HDBSCAN ({args.select}) min size {hdb.min_cluster_size}: {len(ks)} clusters, {np.mean(lab == -1):.1%} noise",
             f"heading from features (ridge on PCA, 5-fold blocked CV): R^2 = {r2_all:.2f}",
             f"heading concentration over all windows: R = {np.hypot(np.cos(a).mean(), np.sin(a).mean()):.2f}", "",
             summ.to_string(float_format=lambda v: f"{v:.3f}" if abs(v) < 10 else f"{v:.0f}")]
    if df.behaviour.nunique() > 1:
        lines += ["", "behaviour labels by cluster (share of the cluster's windows):",
                  pd.crosstab(df.cluster, df.behaviour, normalize="index").to_string(float_format=lambda v: f"{v:.2f}")]
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ---- figures
    cmap = plt.get_cmap("tab20")
    col = np.array([cmap(k % 20) if k >= 0 else (0.85, 0.85, 0.85, 1) for k in lab])
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(2, 2, 1)
    ax.scatter(emb2[:, 0], emb2[:, 1], c=col, s=2, lw=0)
    for k in ks:
        c = np.median(emb2[lab == k], 0)
        ax.text(*c, str(k), fontsize=9, weight="bold", ha="center")
    ax.set_title("UMAP, HDBSCAN clusters (grey = noise)")
    ax = fig.add_subplot(2, 2, 2)
    sc = ax.scatter(emb2[:, 0], emb2[:, 1], c=head, cmap="hsv", vmin=-180, vmax=180, s=2, lw=0)
    fig.colorbar(sc, ax=ax, label="heading (deg)")
    ax.set_title(f"UMAP coloured by heading in the image (linear R^2 {r2_all:.2f})")
    ax = fig.add_subplot(2, 1, 2)
    ax.scatter(df.t / 60, lab, c=col, s=3, marker="|")
    ax.set_xlabel("time (min)")
    ax.set_ylabel("cluster")
    ax.set_yticks(ks + [-1])
    ax.set_title("ethogram")
    fig.tight_layout()
    fig.savefig(out / "overview.png", dpi=110)

    # ---- examples: windows spread over the session, confident members first
    prev = run / "preview.mp4"
    if not prev.exists():
        return
    cap = cv2.VideoCapture(str(prev))
    pstep = 2  # run_video.py writes every 2nd frame
    rng = np.random.default_rng(args.seed)
    sheet_rows = []
    for k in ks:
        idx = np.flatnonzero((lab == k) & (prob >= np.quantile(prob[lab == k], 0.5)))
        idx = np.sort(rng.choice(idx, min(args.examples, len(idx)), replace=False))
        clips = []
        for i in idx:
            a, b = df.start.iloc[i] // pstep, -(-(df.start.iloc[i] + (stop - start)[0]) // pstep)
            fr_ = read_preview(cap, a, b - a)
            fr_ = [cv2.resize(f, (384, 192), interpolation=cv2.INTER_AREA) for f in fr_]
            for f in fr_:
                cv2.putText(f, f"{df.t.iloc[i] / 60:.1f} min", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            clips.append(fr_)
        n = max(len(c) for c in clips)
        clips = [c + [c[-1]] * (n - len(c)) for c in clips] + [[np.zeros((192, 384, 3), np.uint8)] * n] * (-len(clips) % 3)
        w = cv2.VideoWriter(str(out / "examples" / f"cluster_{k:02d}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps / pstep, (3 * 384, 192 * len(clips) // 3))
        for _ in range(2):
            for t in range(n):
                w.write(np.vstack([np.hstack([c[t] for c in clips[r : r + 3]]) for r in range(0, len(clips), 3)]))
        w.release()
        mid = [cv2.resize(c[n // 2][:, :192], (128, 128)) for c in clips[: min(8, len(idx))]]
        mid += [np.zeros((128, 128, 3), np.uint8)] * (8 - len(mid))
        row = np.hstack([np.full((128, 40, 3), 255, np.uint8)] + mid)
        cv2.putText(row, str(k), (4, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        sheet_rows.append(row)
    cv2.imwrite(str(out / "contact_sheet.jpg"), np.vstack(sheet_rows))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
