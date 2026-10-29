"""Tune the clustering against the spikes, scoring what the clusters add BEYOND kinematics.

    python tune_neural.py RUN_DIR --segment SEG_DIR [--emb vjepa_window] [--feat feats] [--grid default]
                          [--configs FILE.json] [--tag NAME]

sweep_neural.py showed that a plain neural score (decoded bits, encoding R^2) mostly rewards more
clusters and speed/turn bins: k-means on kinematics alone beat the V-JEPA clusters at every k. So
here the score is the encoding gain over kinematics:

  dR2 = R^2(spikes ~ kin + clusters) - R^2(spikes ~ kin)
  (--detrend-min N first removes each unit's N-minute moving average: slow drift;
   --kin-pos adds arena position (quadratic, plus an 8 x 8 grid) and image heading to kin, for
   raw-video inputs that can see the walls)

pooled over units (sqrt counts per ~1 s window, z-scored on the training folds), ridge, blocked
5-fold CV with 10 s buffers. kin = body-frame speed / forward / lateral / turn and part areas with
all pairwise products (77 columns). A partition that is only kinematic bins adds little; one that
splits behaviour the way the neurons do adds more. Every config is run over a min-cluster-size
grid, giving a dR2-vs-k curve, and is compared with the kinematics k-means curve at the same k
(excess = dR2 - kin-kmeans dR2 interpolated in log k).

A config sets: features (V-JEPA layer key, + standardised kinematics with weight kin_w), PCA dims,
UMAP dims / neighbours / min_dist, HDBSCAN selection / min_samples, min_cluster_frac. Noise
windows get the label of their 15 nearest clustered neighbours (every partition covers the same
windows).

Output in RUN_DIR/<emb>/neural/tune_<tag>/: results.csv (one row per config x size), labels.npz
(every partition), curves.png
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

DEFAULT = dict(feat="feats", kin_w=0.0, pca=50, umap_dim=10, nn=30, min_dist=0.0, select="leaf", min_samples=10)
FRACS = (0.0025, 0.005, 0.01, 0.02)


def grid(name):
    """One-at-a-time variations around DEFAULT (cluster_embeddings.py's settings)."""
    var = {"default": [{}],
           "pca": [dict(pca=20), dict(pca=100)],
           "umap": [dict(umap_dim=5), dict(umap_dim=20), dict(nn=15), dict(nn=100), dict(min_dist=0.1)],
           "hdb": [dict(min_samples=3), dict(min_samples=30), dict(select="eom")],
           "kin": [dict(kin_w=0.5), dict(kin_w=1.0)],
           "nopca_umap": [dict(pca=0)]}
    names = list(var) if name == "all" else name.split(",")
    return [dict(DEFAULT, **v) for n in names for v in var[n]]


def detrend(X, t_s, minutes):
    """Each unit's activity minus its centred moving average over `minutes` (in session time):
    removes slow firing-rate drift, which a cluster tied to one part of the session could
    otherwise claim. Uses only the spikes (no labels); behaviour bouts last seconds and survive."""
    half = minutes * 30.0
    cs = np.vstack([np.zeros(X.shape[1]), np.cumsum(X, 0)])
    a, b = np.searchsorted(t_s, t_s - half), np.searchsorted(t_s, t_s + half, side="right")
    return (X - (cs[b] - cs[a]) / (b - a)[:, None]).astype(np.float32)


def onehot(lab, k):
    return np.eye(k, dtype=np.float32)[lab]


def ridge_cv_r2(A_list, Y, folds, buffer, alpha):
    """Pooled blocked-CV R^2 of Y [n, units] from each design matrix in A_list (same folds)."""
    sse = np.zeros(len(A_list))
    sst = 0.0
    for tr, te in nd.blocked_folds(len(Y), folds, buffer):
        mu, sd = Y[tr].mean(0), Y[tr].std(0) + 1e-6
        Ytr, Yte = (Y[tr] - mu) / sd, (Y[te] - mu) / sd
        sst += (Yte ** 2).sum()
        for i, A in enumerate(A_list):
            am, asd = A[tr].mean(0), A[tr].std(0) + 1e-6
            Atr, Ate = (A[tr] - am) / asd, (A[te] - am) / asd
            W = np.linalg.solve(Atr.T @ Atr + alpha * np.eye(A.shape[1]), Atr.T @ Ytr)
            sse[i] += ((Yte - Ate @ W) ** 2).sum()
    return 1 - sse / sst


class Scorer:
    def __init__(self, X, Kp, folds=5, buffer=20, alpha=100.0):
        self.X, self.Kp, self.folds, self.buffer, self.alpha = X, Kp, folds, buffer, alpha
        self.r2_kin = ridge_cv_r2([Kp], X, folds, buffer, alpha)[0]

    def __call__(self, lab):
        k = lab.max() + 1
        H = onehot(lab, k)
        r2_lab, r2_both = ridge_cv_r2([H, np.c_[self.Kp, H]], self.X, self.folds, self.buffer, self.alpha)
        return dict(k=k, r2_lab=r2_lab, r2_kin=self.r2_kin, r2_both=r2_both, dR2=r2_both - self.r2_kin)


def assign_noise(emb, lab):
    if (lab < 0).any() and (lab >= 0).any():
        lab = lab.copy()
        lab[lab < 0] = KNeighborsClassifier(15).fit(emb[lab >= 0], lab[lab >= 0]).predict(emb[lab < 0])
    return np.unique(lab, return_inverse=True)[1]


def embed(feats, K, cfg, seed=0):
    F = StandardScaler().fit_transform(feats)
    F = PCA(cfg["pca"], random_state=seed).fit_transform(F) if cfg["pca"] else F
    if cfg["kin_w"]:
        # each kinematic column gets kin_w times the spread of an average retained PC
        Ks = StandardScaler().fit_transform(K) * cfg["kin_w"] * F.std(0).mean()
        F = np.c_[F, Ks]
    return umap.UMAP(n_components=cfg["umap_dim"], n_neighbors=cfg["nn"], min_dist=cfg["min_dist"],
                     random_state=seed).fit_transform(F)


def cluster(emb, cfg, frac):
    hdb = HDBSCAN(min_cluster_size=max(20, int(frac * len(emb))), min_samples=cfg["min_samples"],
                  cluster_selection_method=cfg["select"], copy=True).fit(emb)
    return hdb.labels_


def cfg_name(cfg):
    d = {k: v for k, v in cfg.items() if DEFAULT.get(k) != v}
    return "default" if not d else ",".join(f"{k}={v}" for k, v in d.items())


def load_inputs(run, emb_dir, seg, quality, feat_keys, pos=False):
    z = np.load(run / emb_dir / "features.npz")
    cfg = json.load(open(run / emb_dir / "config.json"))
    found = pd.read_csv(run / "frames.csv", usecols=["mouse_found"]).mouse_found.to_numpy()
    cs = np.r_[0, np.cumsum(~found)]
    keep = cs[z["stop"]] - cs[z["start"]] == 0  # as cluster_embeddings.py
    start, stop = z["start"][keep], z["stop"][keep]
    sess = nd.load_session(seg, quality)
    hop_s = cfg["hop"] * sess["period"]
    X = nd.sqrt_counts(nd.window_counts(sess, start, stop))
    K, _ = nd.kinematics(run, emb_dir, start, stop, None, pos=pos)
    feats = {k: z[k][keep].astype(np.float32) for k in feat_keys if k in z}
    Kp = nd.poly2(K)
    if pos:  # place beyond a smooth function of x, y: an 8 x 8 grid over the frame, and image heading
        W, H = 2248, 2064
        gx, gy = np.clip((K[:, -2] / W * 8).astype(int), 0, 7), np.clip((K[:, -1] / H * 8).astype(int), 0, 7)
        h = np.deg2rad(z["heading"][keep])
        Kp = np.c_[Kp, np.eye(64, dtype=np.float32)[gy * 8 + gx], np.cos(h), np.sin(h)]
    return dict(X=X, K=K, Kp=Kp, feats=feats, start=start, hop_s=hop_s, sess=sess)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--segment", required=True)
    ap.add_argument("--emb", default="vjepa_window")
    ap.add_argument("--quality", default="rf_good")
    ap.add_argument("--grid", default="all", help="names from grid(), comma separated, or 'all'")
    ap.add_argument("--configs", help="JSON list of configs (overrides --grid); keys as DEFAULT")
    ap.add_argument("--fracs", default=",".join(map(str, FRACS)))
    ap.add_argument("--tag", default="sweep")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kin-pos", action="store_true", help="add arena position (window-mean body centre) to the kinematics")
    ap.add_argument("--detrend-min", type=float, default=0,
                    help="subtract each unit's N-minute moving average from its activity first (slow drift)")
    args = ap.parse_args()

    run = Path(args.run_dir)
    out = run / args.emb / "neural" / f"tune_{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    configs = [dict(DEFAULT, **c) for c in json.load(open(args.configs))] if args.configs else grid(args.grid)
    fracs = [float(f) for f in args.fracs.split(",")]
    feat_keys = sorted({c["feat"] for c in configs})
    d = load_inputs(run, args.emb, args.segment, args.quality, feat_keys, pos=args.kin_pos)
    X = d["X"]
    if args.detrend_min:
        X = detrend(X, d["sess"]["frame_s"][d["start"]], args.detrend_min)
    score = Scorer(X, d["Kp"], buffer=int(round(10 / d["hop_s"])))
    print(f"{len(d['start'])} windows, {X.shape[1]} units{f', detrended {args.detrend_min:g} min' if args.detrend_min else ''}; "
          f"R^2(spikes ~ kin) = {score.r2_kin:.4f}", flush=True)

    rows, labels = [], {}
    # matched-k controls: k-means on kinematics only, and the same labels shifted in time
    Ks = StandardScaler().fit_transform(d["K"])
    rng = np.random.default_rng(args.seed)
    for k in (5, 10, 20, 40, 80):
        lab = KMeans(k, n_init=4, random_state=args.seed).fit_predict(Ks)
        labels[f"km_kin_k{k}"] = lab
        rows.append(dict(config="km_kin", frac=np.nan, noise=0.0, **score(lab)))
        sh = np.roll(lab, rng.integers(len(lab) // 4, 3 * len(lab) // 4))
        rows.append(dict(config="km_kin_shifted", frac=np.nan, noise=0.0, **score(sh)))
        print(rows[-2], rows[-1], flush=True)

    for cfg in configs:
        name = cfg_name(cfg)
        emb = embed(d["feats"][cfg["feat"]], d["K"], cfg, args.seed)
        for frac in fracs:
            raw = cluster(emb, cfg, frac)
            if raw.max() < 1:
                continue
            lab = assign_noise(emb, raw)
            labels[f"{name}|{frac}"] = lab
            rows.append(dict(config=name, frac=frac, noise=np.mean(raw < 0), **score(lab), **{f"cfg_{k}": v for k, v in cfg.items()}))
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items() if not k.startswith("cfg_")}, flush=True)

    df = pd.DataFrame(rows)
    base = df[df.config == "km_kin"].sort_values("k")
    df["excess"] = df.dR2 - np.interp(np.log(df.k), np.log(base.k), base.dR2)
    df.to_csv(out / "results.csv", index=False)
    np.savez_compressed(out / "labels.npz", start=d["start"], **{k.replace("|", "__"): v for k, v in labels.items()})

    summ = (df[~df.config.str.startswith("km_kin")].groupby("config")
            .agg(k_range=("k", lambda s: f"{s.min()}-{s.max()}"), mean_excess=("excess", "mean"),
                 max_dR2=("dR2", "max")).sort_values("mean_excess", ascending=False))
    print(summ.to_string(float_format=lambda v: f"{v:.4f}"))
    (out / "summary.txt").write_text(f"R^2(spikes ~ kin) = {score.r2_kin:.4f}\n\n" + summ.to_string(float_format=lambda v: f"{v:.4f}")
                                     + "\n\n" + df.drop(columns=[c for c in df if c.startswith("cfg_")]).to_string(index=False, float_format=lambda v: f"{v:.4f}") + "\n")

    fig, ax = plt.subplots(figsize=(9, 6))
    for name, g in df.groupby("config"):
        g = g.sort_values("k")
        style = dict(color="k", lw=2) if name == "km_kin" else dict(color="0.6", ls="--") if name == "km_kin_shifted" else {}
        ax.plot(g.k, g.dR2, "o-", label=name, ms=4, **style)
    ax.set_xscale("log")
    ax.set_xlabel("number of clusters k")
    ax.set_ylabel("dR2 = R2(kin + clusters) - R2(kin)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=110)


if __name__ == "__main__":
    main()
