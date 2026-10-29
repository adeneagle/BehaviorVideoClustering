"""Score clustering hyperparameters by how much the spikes say about the clusters.

    python sweep_neural.py RUN_DIR --segment SEG_DIR [--emb vjepa_window] [--quality rf_good]

Re-runs cluster_embeddings.py's PCA 50 -> UMAP 10-d (same seed) once per n_neighbors, then labels
every kept window with a set of partitions:
  hdb_<sel>_f<frac>_n<nn>  HDBSCAN on the UMAP (noise windows given the label of their 15 nearest
                           clustered neighbours, so every partition covers the same windows)
  km_vjepa_k<k>            k-means on the PCA-50 V-JEPA features
  km_kin_k<k>              k-means on the body-frame kinematics only (matched-k control: is the
                           embedding worth more to the neurons than speed / turn / part-area bins?)
  shuffled_k<k>            km_vjepa labels with bouts circularly shifted against time
                           (keeps k, occupancy and bout structure; destroys the behaviour link)
For each: blocked-CV XGBoost decoding of the labels from spikes (neural_decode.decode_cv, 5 folds) and the reverse,
cross-validated R^2 of the units' sqrt counts from cluster means ("encoding R^2").

Output: RUN_DIR/<emb>/neural/sweep.csv, sweep.png
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

import neural_decode as nd


def encoding_r2(X, y, folds, buffer):
    """Blocked-CV R^2 of each unit's (standardised) activity predicted by its training-fold cluster mean."""
    num = den = 0.0
    for tr, te in nd.blocked_folds(len(y), folds, buffer):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Z_tr, Z_te = (X[tr] - mu) / sd, (X[te] - mu) / sd
        k = y.max() + 1
        S = np.zeros((k, X.shape[1]))
        np.add.at(S, y[tr], Z_tr)
        cnt = np.bincount(y[tr], minlength=k)[:, None]
        M = np.where(cnt > 0, S / np.maximum(cnt, 1), 0)
        num += ((Z_te - M[y[te]]) ** 2).sum()
        den += (Z_te ** 2).sum()
    return 1 - num / den


def relabel(lab):
    return np.unique(lab, return_inverse=True)[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--segment", required=True)
    ap.add_argument("--emb", default="vjepa_window")
    ap.add_argument("--quality", default="rf_good")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run = Path(args.run_dir)
    out = run / args.emb / "neural"
    out.mkdir(exist_ok=True)
    z = np.load(run / args.emb / "features.npz")
    cfg = json.load(open(run / args.emb / "config.json"))
    fr = pd.read_csv(run / "frames.csv", usecols=["mouse_found"])
    found = fr.mouse_found.to_numpy()
    cs = np.r_[0, np.cumsum(~found)]
    keep = cs[z["stop"]] - cs[z["start"]] == 0  # as cluster_embeddings.py
    start, stop = z["start"][keep], z["stop"][keep]

    sess = nd.load_session(args.segment, args.quality)
    hop_s = cfg["hop"] * sess["period"]
    buffer = int(round(10 / hop_s))
    X = nd.sqrt_counts(nd.window_counts(sess, start, stop))
    K, _ = nd.kinematics(run, args.emb, start, stop, None)

    Xf = StandardScaler().fit_transform(z["feats"][keep].astype(np.float32))
    Xp = PCA(50, random_state=args.seed).fit_transform(Xf)
    parts = {}
    for nn in (15, 30, 100):
        emb = umap.UMAP(n_components=10, n_neighbors=nn, min_dist=0.0, random_state=args.seed).fit_transform(Xp)
        grid = [("leaf", f) for f in (0.0025, 0.005, 0.01, 0.02)] + [("eom", 0.005)] if nn == 30 else [("leaf", 0.005)]
        for sel, frac in grid:
            lab = HDBSCAN(min_cluster_size=max(20, int(frac * len(Xp))), min_samples=10,
                          cluster_selection_method=sel).fit(emb).labels_
            noise = np.mean(lab < 0)
            if (lab >= 0).sum() and noise > 0:
                knn = KNeighborsClassifier(15).fit(emb[lab >= 0], lab[lab >= 0])
                lab = lab.copy()
                lab[lab < 0] = knn.predict(emb[lab < 0])
            parts[f"hdb_{sel}_f{frac}_n{nn}"] = (relabel(lab), noise, "hdbscan")
        print(f"umap n_neighbors={nn} done", flush=True)
    Ks = StandardScaler().fit_transform(K)
    rng = np.random.default_rng(args.seed)
    for k in (5, 10, 20, 40):
        lv = KMeans(k, n_init=4, random_state=args.seed).fit_predict(Xp)
        parts[f"km_vjepa_k{k}"] = (lv, 0.0, "kmeans_vjepa")
        parts[f"km_kin_k{k}"] = (KMeans(k, n_init=4, random_state=args.seed).fit_predict(Ks), 0.0, "kmeans_kin")
        parts[f"shuffled_k{k}"] = (np.roll(lv, rng.integers(len(lv) // 4, 3 * len(lv) // 4)), 0.0, "shuffled")

    rows = []
    for name, (lab, noise, fam) in parts.items():
        k = lab.max() + 1
        P, prior, used = nd.decode_cv(X, lab, k, args.folds, buffer)
        s = nd.scores(P, prior, lab)
        r = dict(partition=name, family=fam, k=k, hdb_noise=noise, **s, enc_r2=encoding_r2(X, lab, args.folds, buffer))
        rows.append(r)
        print({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in r.items()}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "sweep.csv", index=False)

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for fam, mk in [("hdbscan", "o"), ("kmeans_vjepa", "s"), ("kmeans_kin", "^"), ("shuffled", "x")]:
        d = df[df.family == fam].sort_values("k")
        for a, col in zip(ax, ["bits", "frac_H", "enc_r2"]):
            a.plot(d.k, d[col], mk + ("-" if fam != "hdbscan" else ""), label=fam)
    for a, col in zip(ax, ["decoded bits / window", "bits / H(labels)", "encoding R^2 (cluster means)"]):
        a.set_xscale("log")
        a.set_xlabel("number of clusters k")
        a.set_ylabel(col)
        a.legend()
    fig.tight_layout()
    fig.savefig(out / "sweep.png", dpi=110)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
