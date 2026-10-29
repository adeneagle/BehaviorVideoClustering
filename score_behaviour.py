"""Score behaviour clusterings without spikes: how much a window's state says about the mouse's
NEXT movements, beyond what its own kinematics already say, at a given state entropy.

    python score_behaviour.py RUN_DIR [--fit-run FIT_RUN_DIR] [--emb vjepa21_rc] [--feat feats]
                              [--labels FILE.npz ...] [--parent NAME] [--tag NAME]

Recommended: --fit-run on another session (e.g. fit on 10-06, score 10-08). Every partition is then
fitted there and only ASSIGNED here (k-means predict; UMAP transform + 15-NN for HDBSCAN), and the
same method refitted here gives a stability score (AMI, frozen vs refit). Without --fit-run the
partitions are fitted on the scored session itself (no targets are used, but less strict).

Use an embedding made with `embed_vjepa.py --centre raw`: with the default crop centring the part
maps show where the mouse is about to go (look-ahead track smoothing), which inflates every score
here up to ~0.7 s ahead.

Windows: those of RUN_DIR/<emb>/features.npz whose frames AND the next 5 s all have the mouse.
t0 = the window's last frame. Measurements come from part_tracks.py (part centroids of the label
maps; independent of V-JEPA and of the clustering). Posture = the 4 paw centroids relative to the
body centroid in that frame's own heading (tail axis, else paw axis), plus log part areas (13).

Targets F (per horizon; displacement in the t0 body frame):
  p0.13 / p0.27 / p0.53 / p1.07   posture at t0 + h, body displacement and heading change since t0 (16)
  s0.7-2 / s2-5                   over (t0 + a, t0 + b]: mean and SD of posture, net displacement,
                                  mean speed, mean |angular speed| (30). Starts >= 0.7 s after the
                                  window, beyond the reach of the crop-smoothing leak.
History H (baseline): tune_neural's window kinematics with pairwise products (77) + posture at t0
(13) + its change over the last 4 frames (13) + end-of-window body velocity / turn (3)
[+ one-hot --parent: score subtypes WITHIN a parent partition, e.g. a behaviour label].

Score, per horizon, in bits per window: held-out Gaussian log-likelihood of the rank-Gaussianised
targets; ridge, blocked 5-fold CV with 10 s buffers; penalty and residual covariance (Ledoit-Wolf)
from inner leave-one-block-out residuals of the training blocks (see Likelihood):
  dLL   add   LL(F | H, S) - LL(F | H)            where the mouse goes next
        int   LL(F | H, S, S x kin) - LL(F | H)    plus state-specific dynamics (different laws of motion)
  ceiling     LL(F | H, PCA-50 features) - LL(F | H): the continuous embedding; dLL / ceiling is the
              share a partition keeps
Controls, at the same windows:
  km_kin_k*   k-means on the history features themselves: excess = dLL minus this curve
              interpolated at the same state entropy H(S) (bits)
  .shift      the partition circularly shifted in time (occupancy and bout lengths kept)
  .perm       bout labels permuted (every boundary kept, which state each bout gets shuffled):
              beating it means the labels matter, not just where the cuts fall

Output in RUN_DIR/<emb>/behaviour/score_<tag>/: results.csv (partition x form x horizon),
summary.txt, curves.png, labels.npz (+ fit_labels.npz with --fit-run: the fit session's own labels,
for abx_behaviour.py), config.json
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
import umap
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

import neural_decode as nd
import tune_neural as tn
from part_tracks import part_tracks

POINTS = (0.133, 0.267, 0.533, 1.067)  # s
SUMMARIES = ((0.7, 2.0), (2.0, 5.0))  # s
MAX_AHEAD = 5.0
KS = (5, 10, 20, 40, 80)
ALPHA_BASE = 10.0  # ridge penalty of the history block (~120 standardised columns, ~20k windows: barely shrunk)
ALPHAS = (1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6)  # grid for the added block (states, embedding), tuned per target
LOG2 = np.log(2)


# ---------------------------------------------------------------- measurements
def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def frame_measures(run):
    """Per frame: body centre (full-res px), heading (rad, NaN filled from neighbours), posture [n, 13]."""
    tr = part_tracks(run)
    fr = pd.read_csv(run / "frames.csv", usecols=["cx", "cy", "mouse_found"])
    xy, area = tr["xy"].astype(np.float64), tr["area"].astype(np.float64)
    body = np.where(np.isnan(xy[:, 0]), fr[["cx", "cy"]].to_numpy(), xy[:, 0])
    h = np.radians(tr["heading"].astype(np.float64))
    ok = ~np.isnan(h)
    t = np.arange(len(h))
    c, s = np.interp(t, t[ok], np.cos(h[ok])), np.interp(t, t[ok], np.sin(h[ok]))
    h = np.arctan2(s, c)
    rel = np.nan_to_num(xy[:, 1:] - xy[:, :1])  # absent paw -> at the body centre (its area says it is absent)
    fwd = rel[..., 0] * np.cos(h)[:, None] + rel[..., 1] * np.sin(h)[:, None]
    lat = -rel[..., 0] * np.sin(h)[:, None] + rel[..., 1] * np.cos(h)[:, None]
    post = np.c_[np.stack([fwd, lat], -1).reshape(len(h), 8) / 100.0, np.log1p(area)]
    return dict(body=body, head=h, post=post)


def to_frame(d, ang):
    """Displacements [n, 2] into the body frame of headings ang [n] (forward, lateral), /100 px."""
    return np.c_[d[:, 0] * np.cos(ang) + d[:, 1] * np.sin(ang), -d[:, 0] * np.sin(ang) + d[:, 1] * np.cos(ang)] / 100.0


def windows(run, emb, fps):
    """Kept windows: mouse found over the window and the next MAX_AHEAD s."""
    z = np.load(run / emb / "features.npz")
    found = pd.read_csv(run / "frames.csv", usecols=["mouse_found"]).mouse_found.to_numpy()
    ahead = int(round(MAX_AHEAD * fps))
    cs = np.r_[0, np.cumsum(~found)]
    stop = z["stop"]
    end = stop + ahead
    keep = (end <= len(found))
    keep[keep] = cs[end[keep]] - cs[z["start"][keep]] == 0
    return keep, z


def history(m, run, emb, start, stop, fps):
    """H: (poly2 window kinematics, linear history features). t0 = stop - 1."""
    K, _ = nd.kinematics(run, emb, start, stop, None)
    t0 = stop - 1
    post0 = m["post"][t0]
    dpost = post0 - m["post"][t0 - 4]
    v = to_frame(m["body"][t0] - m["body"][t0 - 8], m["head"][t0]) * (fps / 8)
    dh = wrap(m["head"][t0] - m["head"][t0 - 8])
    lin = np.c_[K, post0, dpost, v, dh]
    full = np.c_[nd.poly2(K), post0, dpost, v, dh]
    # the few columns that get a state-specific slope in the 'int' form
    dyn = np.c_[K[:, :5], v, dh, np.linalg.norm(dpost[:, :8], axis=1)]
    return full.astype(np.float64), lin.astype(np.float64), dyn.astype(np.float64)


def targets(m, stop, fps):
    """{horizon name: F [n, d]}."""
    t0 = stop - 1
    th0 = m["head"][t0]
    out = {}
    for h in POINTS:
        f = t0 + int(round(h * fps))
        out[f"p{h:.2f}"] = np.c_[to_frame(m["body"][f] - m["body"][t0], th0), wrap(m["head"][f] - m["head"][t0]), m["post"][f]]
    P = m["post"]
    cp, cp2 = np.vstack([np.zeros(P.shape[1]), np.cumsum(P, 0)]), np.vstack([np.zeros(P.shape[1]), np.cumsum(P ** 2, 0)])
    step = np.r_[0, np.linalg.norm(np.diff(m["body"], axis=0), axis=1)]
    turn = np.r_[0, np.abs(wrap(np.diff(m["head"])))]
    cstep, cturn = np.r_[0, np.cumsum(step)], np.r_[0, np.cumsum(turn)]
    for a, b in SUMMARIES:
        fa, fb = t0 + int(round(a * fps)), t0 + int(round(b * fps))  # frames (fa, fb]
        n = (fb - fa)[:, None]
        mu = (cp[fb + 1] - cp[fa + 1]) / n
        sd = np.sqrt(np.maximum((cp2[fb + 1] - cp2[fa + 1]) / n - mu ** 2, 0))
        disp = to_frame(m["body"][fb] - m["body"][fa], th0)
        speed = (cstep[fb + 1] - cstep[fa + 1]) / n[:, 0] * fps / 100.0
        aturn = (cturn[fb + 1] - cturn[fa + 1]) / n[:, 0] * fps
        out[f"s{a:g}-{b:g}"] = np.c_[mu, sd, disp, speed, aturn]
    return out


# ---------------------------------------------------------------- held-out Gaussian likelihood
def rank_gauss(Y):
    """Each column -> standard normal scores of its ranks (ties averaged): the targets have heavy
    tails and point masses (absent paws), which a Gaussian likelihood would otherwise let a few
    windows dominate. Uses no labels, and is the same for every model."""
    from scipy.special import ndtri
    from scipy.stats import rankdata

    Z = np.column_stack([ndtri((rankdata(c) - 0.5) / len(c)) for c in Y.T])
    return (Z - Z.mean(0)) / (Z.std(0) + 1e-9)


def _gauss_ll(R, S):
    """Sum over rows of the log density (nats) of residuals R under N(0, S)."""
    _, logdet = np.linalg.slogdet(S)
    q = np.einsum("ij,ij->i", R @ np.linalg.inv(S), R)
    return -0.5 * (q.sum() + len(R) * (logdet + R.shape[1] * np.log(2 * np.pi)))


class Likelihood:
    """Held-out log-likelihood (bits per window) of every target block from a design matrix.

    5 contiguous blocks; the test block's neighbours lose `buffer` windows next to it. Within each
    training set, leave-one-block-out over the other 4 blocks gives out-of-sample residuals for
    every training window: they pick the added block's ridge penalty (per target block) and give the
    residual covariance (Ledoit-Wolf). The history block keeps ALPHA_BASE, so a useless added block
    can be shrunk away without shrinking the history too. All fits come from per-piece Gram
    matrices."""

    def __init__(self, Ys, buffer, k=5):
        self.names = list(Ys)
        Z = [rank_gauss(Y) for Y in Ys.values()]
        self.dims = np.cumsum([0] + [z.shape[1] for z in Z])
        self.Y = np.hstack(Z)
        n = len(self.Y)
        self.n = n
        edges = np.linspace(0, n, k + 1).astype(int)
        # pieces: (block, kind) with kind 0 head buffer, 1 core, 2 tail buffer
        self.pieces = []
        for j, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            h, t = min(a + buffer, b), max(b - buffer, a)
            for kind, (u, v) in enumerate(((a, h), (h, max(h, t)), (max(h, t), b))):
                self.pieces.append((j, kind, np.arange(u, v)))
        self.k = k
        self.blocks = [np.arange(a, b) for a, b in zip(edges[:-1], edges[1:])]

    def _train_pieces(self, j):
        """Pieces usable for training when block j is the test block."""
        return [i for i, (blk, kind, _) in enumerate(self.pieces)
                if blk != j and not (blk == j - 1 and kind == 2) and not (blk == j + 1 and kind == 0)]

    def __call__(self, A, n_base=None):
        """A [n, d] (None = intercept only, i.e. the marginal); its first n_base columns (default all)
        are the history block -> {target: bits per window}."""
        if A is None:
            A, n_base = np.full((self.n, 1), 10.0), 1
        else:
            n_base = A.shape[1] if n_base is None else n_base
            A = np.c_[np.full(self.n, 10.0), (A - A.mean(0)) / np.maximum(A.std(0), 1e-8)]
            n_base += 1
        # penalty alpha on the added block = ALPHA_BASE on columns scaled by sqrt(ALPHA_BASE / alpha)
        alphas = ALPHAS if A.shape[1] > n_base else (ALPHA_BASE,)
        scales = [np.r_[np.ones(n_base), np.full(A.shape[1] - n_base, np.sqrt(ALPHA_BASE / a))] for a in alphas]
        Y = self.Y
        G = [A[r].T @ A[r] for _, _, r in self.pieces]
        C = [A[r].T @ Y[r] for _, _, r in self.pieces]

        def weights(ps):
            """Weights on the UNscaled columns, one per alpha."""
            Gs, Cs = sum(G[i] for i in ps), sum(C[i] for i in ps)
            out = []
            for d in scales:
                s, V = np.linalg.eigh(Gs * np.outer(d, d))
                out.append(d[:, None] * (V @ ((V.T @ (d[:, None] * Cs)) / (s + ALPHA_BASE)[:, None])))
            return out

        ll = np.zeros(len(self.names))
        for j in range(self.k):
            tr = self._train_pieces(j)
            # inner leave-one-block-out residuals over the training pieces
            res = [[] for _ in alphas]
            for i in sorted({self.pieces[p][0] for p in tr}):
                held = [p for p in tr if self.pieces[p][0] == i]
                rows = np.concatenate([self.pieces[p][2] for p in held])
                for r, W in zip(res, weights([p for p in tr if p not in held])):
                    r.append(Y[rows] - A[rows] @ W)
            res = [np.vstack(r) for r in res]
            Wf = weights(tr)
            te = self.blocks[j]
            for t in range(len(self.names)):
                sl = slice(self.dims[t], self.dims[t + 1])
                best = int(np.argmin([(r[:, sl] ** 2).sum() for r in res]))
                S = LedoitWolf().fit(res[best][:, sl]).covariance_
                ll[t] += _gauss_ll(Y[te, sl] - A[te] @ Wf[best][:, sl], S)
        return dict(zip(self.names, ll / self.n / LOG2))


# ---------------------------------------------------------------- partitions
def onehot(lab):
    lab = np.unique(lab, return_inverse=True)[1]
    return np.eye(lab.max() + 1)[lab]


def entropy_bits(lab):
    p = np.bincount(np.unique(lab, return_inverse=True)[1]) / len(lab)
    return float(-(p * np.log2(p)).sum())


def bouts(lab):
    """(start index of each run of equal labels, its length)."""
    edge = np.r_[0, np.flatnonzero(np.diff(lab)) + 1]
    return edge, np.diff(np.r_[edge, len(lab)])


def shifted(lab, rng):
    return np.roll(lab, rng.integers(len(lab) // 4, 3 * len(lab) // 4))


def bout_permuted(lab, rng):
    edge, length = bouts(lab)
    return np.repeat(rng.permutation(lab[edge]), length)


class Fitted:
    """Partitions fitted on one session's windows, assigned to another's."""

    def __init__(self, fit, score, seed):
        self.fit, self.score, self.seed = fit, score, seed

    def kmeans(self, key, k):
        sc = StandardScaler().fit(self.fit[key])
        km = KMeans(k, n_init=4, random_state=self.seed).fit(sc.transform(self.fit[key]))
        return km.predict(sc.transform(self.score[key])), km.labels_

    def hdbscan(self, fracs):
        cfg = tn.DEFAULT
        sc = StandardScaler().fit(self.fit["feats"])
        pca = PCA(cfg["pca"], random_state=self.seed).fit(sc.transform(self.fit["feats"]))
        um = umap.UMAP(n_components=cfg["umap_dim"], n_neighbors=cfg["nn"], min_dist=cfg["min_dist"], random_state=self.seed)
        e_fit = um.fit_transform(pca.transform(sc.transform(self.fit["feats"])))
        same = self.fit is self.score
        e_sc = e_fit if same else um.transform(pca.transform(sc.transform(self.score["feats"])))
        out = {}
        for frac in fracs:
            raw = tn.cluster(e_fit, cfg, frac)
            if raw.max() < 1:
                continue
            lab = tn.assign_noise(e_fit, raw)
            out[frac] = (lab if same else KNeighborsClassifier(15).fit(e_fit, lab).predict(e_sc), float(np.mean(raw < 0)), lab)
        return out


def load_side(run, emb, feat):
    """Everything per kept window of one session."""
    cfg = json.load(open(run / emb / "config.json"))
    fps = cfg["fps"]
    keep, z = windows(run, emb, fps)
    start, stop = z["start"][keep], z["stop"][keep]
    m = frame_measures(run)
    full, lin, dyn = history(m, run, emb, start, stop, fps)
    return dict(start=start, stop=stop, fps=fps, hop_s=cfg["hop"] / fps, n_all=len(keep),
                feats=z[feat][keep].astype(np.float32), H=full, lin=lin, dyn=dyn, F=targets(m, stop, fps))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--fit-run", help="session to fit the partitions on (default: RUN_DIR itself)")
    ap.add_argument("--emb", default="vjepa21_rc")
    ap.add_argument("--feat", default="feats")
    ap.add_argument("--fracs", default=",".join(map(str, tn.FRACS)), help="HDBSCAN min cluster size as a share of windows")
    ap.add_argument("--labels", nargs="*", default=[],
                    help="extra partitions: .npz with 'start' and one label array per key (tune_neural's labels.npz), for RUN_DIR")
    ap.add_argument("--parent", help="partition name (e.g. km_kin_k5) or 'behaviour' (frames.csv): added to the baseline")
    ap.add_argument("--no-nulls", action="store_true")
    ap.add_argument("--tag", default="default")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t_start = time.time()
    run = Path(args.run_dir)
    out = run / args.emb / "behaviour" / f"score_{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    d = load_side(run, args.emb, args.feat)
    cross = bool(args.fit_run) and Path(args.fit_run).resolve() != run.resolve()
    f = load_side(Path(args.fit_run), args.emb, args.feat) if cross else d
    n = len(d["start"])
    buffer = int(np.ceil(10.0 / d["hop_s"]))
    lik = Likelihood(d["F"], buffer)
    print(f"{n} of {d['n_all']} windows kept ({d['hop_s']:.2f} s hop); partitions fitted on "
          f"{'another session (' + str(len(f['start'])) + ' windows)' if cross else 'this session'}", flush=True)

    # ---- partitions (+ the same method refitted here, for stability)
    rng = np.random.default_rng(args.seed)
    mine = dict(lin=d["lin"], feats=d["feats"])
    on_fit = Fitted(dict(lin=f["lin"], feats=f["feats"]) if cross else mine, mine, args.seed)
    here = Fitted(mine, mine, args.seed) if cross else None
    parts, fit_labels = {}, {}  # name -> (labels, family, noise, ami); name -> labels of the fit session's windows
    for k in KS:
        for fam, key in (("km_kin", "lin"), ("km_emb", "feats")):
            lab, fit_labels[f"{fam}_k{k}"] = on_fit.kmeans(key, k)
            ami = adjusted_mutual_info_score(here.kmeans(key, k)[0], lab) if cross else np.nan
            parts[f"{fam}_k{k}"] = (lab, fam, 0.0, ami)
    fracs = [float(x) for x in args.fracs.split(",")]
    hdb = on_fit.hdbscan(fracs)
    hdb_here = here.hdbscan(fracs) if cross else {}
    for frac, (lab, noise, fit_labels[f"hdb_f{frac:g}"]) in hdb.items():
        ami = adjusted_mutual_info_score(hdb_here[frac][0], lab) if frac in hdb_here else np.nan
        parts[f"hdb_f{frac:g}"] = (lab, "hdb", noise, ami)
    for path in args.labels:
        z = np.load(path)
        row = {int(s): i for i, s in enumerate(z["start"])}
        missing = [s for s in d["start"] if int(s) not in row]
        if missing:
            print(f"  {path}: {len(missing)} kept windows have no label; skipped")
            continue
        idx = np.array([row[int(s)] for s in d["start"]])
        for key in z.files:
            if key != "start":
                parts[f"{Path(path).parent.name}:{key}"] = (z[key][idx], "external", np.nan, np.nan)
    print(f"{len(parts)} partitions ({time.time() - t_start:.0f}s)", flush=True)

    # ---- baseline, marginal, ceiling
    H = d["H"]
    if args.parent:
        if args.parent == "behaviour":
            beh = pd.read_csv(run / "frames.csv", usecols=["behaviour"]).behaviour.to_numpy()
            plab = np.unique(beh[(d["start"] + d["stop"]) // 2], return_inverse=True)[1]
        else:
            plab = parts[args.parent][0]
        H = np.c_[H, onehot(plab)]
        print(f"baseline includes the parent partition {args.parent} ({len(np.unique(plab))} classes)")
    F = d["F"]
    ll_marg = lik(None)
    ll_base = lik(H)
    emb_pca = PCA(50, random_state=args.seed).fit(StandardScaler().fit_transform(f["feats"]))
    Z = emb_pca.transform(StandardScaler().fit(f["feats"]).transform(d["feats"]))
    ll_ceil = lik(np.c_[H, Z], H.shape[1])
    ceiling = {h: ll_ceil[h] - ll_base[h] for h in F}
    print("bits/window   history vs marginal: " + "  ".join(f"{h} {ll_base[h] - ll_marg[h]:.2f}" for h in F))
    print("              ceiling (PCA-50 emb): " + "  ".join(f"{h} {ceiling[h]:.3f}" for h in F), flush=True)

    # ---- every partition (and its nulls), both forms
    rows, saved = [], {}
    dyn = (d["dyn"] - d["dyn"].mean(0)) / (d["dyn"].std(0) + 1e-8)
    for name, (lab, fam, noise, ami) in parts.items():
        variants = [("", lab)]
        if not args.no_nulls:
            variants += [(".shift", shifted(lab, rng)), (".perm", bout_permuted(lab, rng))]
        _, blen = bouts(lab)
        for suffix, L in variants:
            S = onehot(L)
            res = {"add": lik(np.c_[H, S], H.shape[1]),
                   "int": lik(np.c_[H, S, (S[:, :, None] * dyn[:, None, :]).reshape(n, -1)], H.shape[1])}
            for form, ll in res.items():
                for h in F:
                    rows.append(dict(partition=name + suffix, family=fam, null=suffix.strip("."), k=len(np.unique(L)),
                                     H_bits=entropy_bits(L), noise=noise, ami=ami,
                                     bout_med_s=float(np.median(blen)) * d["hop_s"], single_bouts=float(np.mean(blen == 1)),
                                     form=form, horizon=h, dLL=ll[h] - ll_base[h], ceiling=ceiling[h],
                                     history_bits=ll_base[h] - ll_marg[h]))
        saved[name] = lab
        r = [x for x in rows if x["partition"] == name and x["form"] == "add"]
        print(f"  {name:14s} k={r[0]['k']:3d} H={r[0]['H_bits']:.2f}  add dLL " + " ".join(f"{x['dLL']:.3f}" for x in r)
              + f"  ({time.time() - t_start:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df["share_of_ceiling"] = df.dLL / df.ceiling
    df["excess"] = np.nan
    for (form, h), g in df.groupby(["form", "horizon"]):
        base = g[(g.family == "km_kin") & (g.null == "")].sort_values("H_bits")
        df.loc[g.index, "excess"] = g.dLL - np.interp(g.H_bits, base.H_bits, base.dLL)
    real = df[df.null == ""].set_index(["partition", "form", "horizon"])
    for nl in ("shift", "perm"):
        nul = df[df.null == nl].assign(partition=lambda x: x.partition.str.rsplit(".", n=1).str[0]).set_index(["partition", "form", "horizon"])
        df.loc[df.null == "", f"beyond_{nl}"] = (real.dLL - nul.dLL.reindex(real.index)).to_numpy()
    df.to_csv(out / "results.csv", index=False)
    np.savez_compressed(out / "labels.npz", start=d["start"], **{k.replace(":", "__"): v for k, v in saved.items()})
    if cross:
        np.savez_compressed(out / "fit_labels.npz", start=f["start"], **fit_labels)
    json.dump(dict(vars(args), run_dir=str(run), cross=cross), open(out / "config.json", "w"), indent=2)

    # ---- summary: mean over horizons of each kind, per partition
    df["kind"] = np.where(df.horizon.str.startswith("p"), "short", "long")
    r = df[df.null == ""]
    summ = (r.groupby(["partition", "form", "kind"])
            .agg(k=("k", "first"), H=("H_bits", "first"), dLL=("dLL", "mean"), excess=("excess", "mean"),
                 share=("share_of_ceiling", "mean"), beyond_perm=("beyond_perm", "mean"))
            .reset_index().pivot(index=["partition", "k", "H"], columns=["form", "kind"], values=["dLL", "excess", "share", "beyond_perm"]))
    summ.columns = [f"{a}_{b}_{c}" for a, b, c in summ.columns]
    meta = r.groupby("partition").agg(noise=("noise", "first"), ami=("ami", "first"), bout_med_s=("bout_med_s", "first"),
                                      single_bouts=("single_bouts", "first"))
    summ = summ.join(meta, on="partition").sort_values("excess_add_long", ascending=False)
    head = (f"{run.parent.parent.name} / {args.emb} [{args.feat}], {n} windows; partitions fitted on "
            f"{Path(args.fit_run).parent.parent.name if cross else 'this session'}"
            + (f"; parent {args.parent}" if args.parent else "") + "\n"
            + "bits/window, history vs marginal: " + "  ".join(f"{h} {ll_base[h] - ll_marg[h]:.2f}" for h in F) + "\n"
            + "ceiling (kin + PCA-50 embedding):  " + "  ".join(f"{h} {ceiling[h]:.3f}" for h in F) + "\n"
            + "short = p* horizons (<= 1.07 s), long = s* (0.7-5 s); add / int forms; excess vs km_kin at equal H(S); "
              "beyond_perm = dLL minus the bout-permuted null\n\n")
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        txt = summ.to_string(float_format=lambda v: f"{v:.3f}")
    (out / "summary.txt").write_text(head + txt + "\n")
    print(head + txt)
    plot(df, out / "curves.png", head.splitlines()[0])
    print(f"done in {time.time() - t_start:.0f}s -> {out}")


def plot(df, path, title):
    colors = {"km_kin": "#2a78d6", "km_emb": "#eb6834", "hdb": "#1baf7a", "external": "#eda100"}
    hs = list(dict.fromkeys(df.horizon))
    fig, axes = plt.subplots(2, len(hs), figsize=(3.1 * len(hs), 6.2), sharex=True, squeeze=False)
    for i, form in enumerate(("add", "int")):
        for j, h in enumerate(hs):
            ax = axes[i, j]
            g = df[(df.form == form) & (df.horizon == h)]
            for fam, col in colors.items():
                for nl, style in (("", dict(marker="o", ms=4, lw=1.5)), ("perm", dict(marker="", lw=1, ls="--", alpha=0.6))):
                    s = g[(g.family == fam) & (g.null == nl)].sort_values("H_bits")
                    if len(s):
                        ax.plot(s.H_bits, s.dLL, color=col, label=f"{fam}{' (bout-permuted)' if nl else ''}", **style)
            ax.axhline(g.ceiling.iloc[0], color="0.4", lw=1, ls=":", label="ceiling: continuous embedding")
            ax.axhline(0, color="0.8", lw=0.8)
            ax.set_title(f"{h} s, {form}", fontsize=9)
            ax.tick_params(labelsize=8)
            ax.spines[["top", "right"]].set_visible(False)
            if j == 0:
                ax.set_ylabel("dLL beyond history (bits/window)", fontsize=8)
            if i == 1:
                ax.set_xlabel("state entropy H(S) (bits)", fontsize=8)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=110)


if __name__ == "__main__":
    main()
