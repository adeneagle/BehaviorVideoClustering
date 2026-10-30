"""Sticky Gaussian HMM on embed_vjepa.py window features: recurring behaviour states with a
built-in preference to stay in the current state (the MoSeq idea, on V-JEPA features).

    python sticky_hmm.py RUN_DIR [--fit-run FIT_RUN_DIR] [--emb vjepa21_rc] [--feat feats] [--pca 50]
                         [--ks 10,20,40] [--kappas 0,1e4] [--tempers 1,0.3,0.1] [--iters 150]

Model. K states, each a full-covariance Gaussian over the window features (standardised, PCA).
Transitions get a Dirichlet prior with `alpha` pseudo-counts everywhere plus `kappa` on the
diagonal (sticky HDP-HMM style, as keypoint-MoSeq's kappa): larger kappa = longer bouts.
Fitted by MAP expectation-maximisation from a k-means start; each state's covariance is shrunk
towards the global covariance with D + 2 pseudo-windows. --temper t multiplies the emission
log-likelihoods by t (< 1 weakens the per-window evidence against the transition prior; consecutive
windows overlap, so they are not independent evidence).

--fit-run: standardisation, PCA and the HMM are fitted on another session and only applied to
RUN_DIR (frozen parameters), as score_behaviour.py --fit-run does for its partitions.

Labels, per (K, kappa, temper; _t<temper> is left out for temper 1):
  hmm_k<K>_kap<kappa>[_t<temper>]          Viterbi path: uses the whole session (future windows too)
  hmm_k<K>_kap<kappa>[_t<temper>]_causal   argmax of the forward (filtered) posterior: uses windows
                               up to and including this one only, so it can be scored by
                               score_behaviour.py on equal terms with km_emb / hdb (t0 = the
                               window's last frame)

Windows where the localizer lost the mouse in any frame are left out; the sequence is cut there
(each gap-free stretch is its own sequence).

Output in RUN_DIR/<emb>/stickyhmm/ (or --out-name):
  labels.npz    start + one label array per partition (score_behaviour.py --labels, label_video.py)
  summary.txt   per model: log-likelihood, iterations, states used, bout lengths, self-transition
  overview.png  ethograms of a 10 min excerpt
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
from numba import njit
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

LOG2PI = np.log(2 * np.pi)


# ---------------------------------------------------------------- HMM pieces
def emission_loglik(Z, means, covs, dev):
    """log N(z_t | mu_k, Sigma_k) for every window and state: [T, K] (on the GPU, float64)."""
    Zt = torch.as_tensor(Z, dtype=torch.float64, device=dev)
    out = torch.empty(len(Z), len(means), dtype=torch.float64, device=dev)
    for k in range(len(means)):
        L = torch.linalg.cholesky(torch.as_tensor(covs[k], device=dev))
        x = torch.linalg.solve_triangular(L, (Zt - torch.as_tensor(means[k], device=dev)).T, upper=False)
        out[:, k] = -0.5 * ((x * x).sum(0) + 2 * torch.log(torch.diagonal(L)).sum() + Z.shape[1] * LOG2PI)
    return out.cpu().numpy()


@njit(cache=True)
def _forward(B, A, pi):
    """Scaled forward pass. B [T, K] likelihoods (rows rescaled), returns filtered posteriors and scales."""
    T, K = B.shape
    alpha = np.empty((T, K))
    c = np.empty(T)
    a = pi * B[0]
    c[0] = a.sum()
    alpha[0] = a / c[0]
    for t in range(1, T):
        for j in range(K):
            s = 0.0
            for i in range(K):
                s += alpha[t - 1, i] * A[i, j]
            a[j] = s * B[t, j]
        c[t] = a.sum()
        alpha[t] = a / c[t]
    return alpha, c


@njit(cache=True)
def _backward(B, A, c):
    T, K = B.shape
    beta = np.empty((T, K))
    beta[T - 1] = 1.0
    tmp = np.empty(K)
    for t in range(T - 2, -1, -1):
        for j in range(K):
            tmp[j] = B[t + 1, j] * beta[t + 1, j]
        for i in range(K):
            s = 0.0
            for j in range(K):
                s += A[i, j] * tmp[j]
            beta[t, i] = s / c[t + 1]
    return beta


@njit(cache=True)
def _viterbi(logB, logA, logpi):
    T, K = logB.shape
    delta = logpi + logB[0]
    back = np.empty((T, K), np.int64)
    new = np.empty(K)
    for t in range(1, T):
        for j in range(K):
            best, arg = -np.inf, 0
            for i in range(K):
                v = delta[i] + logA[i, j]
                if v > best:
                    best, arg = v, i
            new[j] = best + logB[t, j]
            back[t, j] = arg
        delta[:] = new
    path = np.empty(T, np.int64)
    path[T - 1] = np.argmax(delta)
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def e_step(logB, A, pi, seqs):
    """Forward-backward over every sequence: state posteriors, filtered posteriors, expected transition
    counts and the log-likelihood."""
    T, K = logB.shape
    gamma, filt = np.empty((T, K)), np.empty((T, K))
    xi = np.zeros((K, K))
    ll = 0.0
    for a, b in seqs:
        lb = logB[a:b]
        m = lb.max(1, keepdims=True)
        B = np.exp(lb - m)
        alpha, c = _forward(B, A, pi)
        beta = _backward(B, A, c)
        gamma[a:b] = alpha * beta
        filt[a:b] = alpha
        if b - a > 1:
            xi += A * (alpha[:-1].T @ (B[1:] * beta[1:] / c[1:, None]))
        ll += np.log(c).sum() + m.sum()
    return gamma, filt, xi, ll


def m_step(Z, gamma, xi, kappa, alpha, cov0, lam):
    K, D = gamma.shape[1], Z.shape[1]
    nk = gamma.sum(0)
    means = (gamma.T @ Z) / np.maximum(nk, 1e-12)[:, None]
    covs = np.empty((K, D, D))
    for k in range(K):
        X = Z - means[k]
        covs[k] = ((gamma[:, k, None] * X).T @ X + lam * cov0) / (nk[k] + lam)
    counts = xi + alpha + kappa * np.eye(K)
    return means, covs, counts / counts.sum(1, keepdims=True)


def fit_sticky_hmm(Z, seqs, k, kappa, alpha=1.0, temper=1.0, iters=150, tol=1e-6, seed=0, dev="cuda"):
    """MAP-EM fit from a k-means start. Returns params, Viterbi path, causal (filtered) labels, log-lik
    history."""
    D = Z.shape[1]
    cov0 = np.cov(Z.T)
    lam = D + 2.0
    init = KMeans(k, n_init=4, random_state=seed).fit_predict(Z)
    gamma = np.eye(k)[init]
    xi = np.zeros((k, k))
    for a, b in seqs:
        np.add.at(xi, (init[a : b - 1], init[a + 1 : b]), 1)
    means, covs, A = m_step(Z, gamma, xi, kappa, alpha, cov0, lam)
    pi = np.full(k, 1.0 / k)
    hist = []
    for it in range(iters):
        logB = temper * emission_loglik(Z, means, covs, dev)
        gamma, filt, xi, ll = e_step(logB, A, pi, seqs)
        hist.append(ll)
        if it > 5 and abs(hist[-1] - hist[-2]) < tol * abs(hist[-1]):
            break
        means, covs, A = m_step(Z, gamma, xi, kappa, alpha, cov0, lam)
    prm = dict(means=means, covs=covs, A=A, pi=pi, temper=temper, iters=it + 1)
    path, causal, prm["ll"] = decode(prm, Z, seqs, dev)
    return prm, path, causal, hist


def decode(prm, Z, seqs, dev="cuda"):
    """Viterbi path, causal labels (argmax of the filtered posterior) and log-likelihood of Z under
    fitted parameters."""
    logB = prm["temper"] * emission_loglik(Z, prm["means"], prm["covs"], dev)
    _, filt, _, ll = e_step(logB, prm["A"], prm["pi"], seqs)
    path = np.empty(len(Z), np.int64)
    for a, b in seqs:
        path[a:b] = _viterbi(logB[a:b], np.log(prm["A"]), np.log(prm["pi"]))
    return path, filt.argmax(1), ll


# ---------------------------------------------------------------- helpers
def load_windows(run, emb, feat):
    """Kept windows (mouse found in every frame): start frames, gap flags, sequences, features, fps."""
    z = np.load(run / emb / "features.npz")
    found = pd.read_csv(run / "frames.csv", usecols=["mouse_found"]).mouse_found.to_numpy()
    cs = np.r_[0, np.cumsum(~found)]
    keep = cs[z["stop"]] - cs[z["start"]] == 0
    start = z["start"][keep]
    hop = int(np.median(np.diff(z["start"])))
    gap = np.r_[False, np.diff(start) > hop]
    edges = np.r_[0, np.flatnonzero(gap), len(start)]
    return dict(start=start, gap=gap, seqs=list(zip(edges[:-1], edges[1:])), X=z[feat][keep].astype(np.float32),
                fps=float(z["fps"]), hop_s=hop / float(z["fps"]), n_all=len(keep))



def bout_lengths(lab, gap):
    cut = np.r_[True, (np.diff(lab) != 0) | gap[1:], True]
    return np.diff(np.flatnonzero(cut))


def entropy_bits(lab):
    p = np.bincount(lab) / len(lab)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--fit-run", help="fit on this session and only apply the model to RUN_DIR")
    ap.add_argument("--emb", default="vjepa21_rc", help="embed_vjepa.py output folder in RUN_DIR")
    ap.add_argument("--feat", default="feats", help="feature key in features.npz (feats = last layer)")
    ap.add_argument("--pca", type=int, default=50, help="standardise + PCA to this many dims")
    ap.add_argument("--ks", default="10,20,40", help="numbers of states")
    ap.add_argument("--kappas", default="0,1e4", help="sticky pseudo-counts on the diagonal")
    ap.add_argument("--alpha", type=float, default=1.0, help="pseudo-counts on every transition")
    ap.add_argument("--tempers", default="1,0.3,0.1", help="multiply emission log-likelihoods by these (one model each)")
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--out-name", default="stickyhmm")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    run = Path(args.run_dir)
    out = run / args.emb / args.out_name
    out.mkdir(parents=True, exist_ok=True)
    d = load_windows(run, args.emb, args.feat)
    cross = bool(args.fit_run) and Path(args.fit_run).resolve() != run.resolve()
    f = load_windows(Path(args.fit_run), args.emb, args.feat) if cross else d
    start, gap, seqs, fps, hop_s = d["start"], d["gap"], d["seqs"], d["fps"], d["hop_s"]
    scaler = StandardScaler().fit(f["X"])
    pca = PCA(args.pca, random_state=args.seed).fit(scaler.transform(f["X"]))
    Zf = pca.transform(scaler.transform(f["X"])).astype(np.float64)
    Z = pca.transform(scaler.transform(d["X"])).astype(np.float64) if cross else Zf
    print(f"{len(start)}/{d['n_all']} windows kept ({hop_s:.2f} s hop, {len(seqs)} sequences); PCA {args.pca}: "
          f"{pca.explained_variance_ratio_.sum():.1%} variance; fitted on "
          f"{Path(args.fit_run).parent.parent.name + ' (' + str(len(f['start'])) + ' windows)' if cross else 'this session'}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    parts, rows = {}, []
    configs = [(k, kap, tmp) for tmp in [float(v) for v in args.tempers.split(",")]
               for k in [int(v) for v in args.ks.split(",")] for kap in [float(v) for v in args.kappas.split(",")]]
    for k, kappa, temper in configs:
            t1 = time.time()
            prm, path, causal, hist = fit_sticky_hmm(Zf, f["seqs"], k, kappa, args.alpha, temper, args.iters, seed=args.seed, dev=dev)
            if cross:
                path, causal, prm["ll"] = decode(prm, Z, seqs, dev)
            name = f"hmm_k{k}_kap{kappa:g}" + (f"_t{temper:g}" if temper != 1 else "")
            # number states by first appearance in the Viterbi path, the same numbers in both label sets
            _, first = np.unique(path, return_index=True)
            rank = np.full(k, -1)
            rank[np.unique(path)] = np.argsort(np.argsort(first))
            unseen = rank < 0  # states only in the causal labels go last
            rank[unseen] = rank.max() + 1 + np.arange(unseen.sum())
            path, causal = rank[path], rank[causal]
            parts[name], parts[name + "_causal"] = path, causal
            for nm, lab in ((name, path), (name + "_causal", causal)):
                bl = bout_lengths(lab, gap) * hop_s
                rows.append(dict(partition=nm, K=k, kappa=kappa, temper=temper, states_used=len(np.unique(lab)), H_bits=entropy_bits(lab),
                                 bout_median_s=np.median(bl), bout_mean_s=bl.mean(), single_window_bouts=np.mean(bl <= hop_s),
                                 self_trans=np.mean(np.diag(prm["A"])), ll_per_window=prm["ll"] / len(Z), iters=prm["iters"]))
            agree = np.mean(path == causal)
            print(f"  {name}: {prm['iters']} iters, {time.time() - t1:.0f}s; Viterbi bouts median "
                  f"{rows[-2]['bout_median_s']:.1f} s mean {rows[-2]['bout_mean_s']:.1f} s; causal median "
                  f"{rows[-1]['bout_median_s']:.1f} s mean {rows[-1]['bout_mean_s']:.1f} s; same label {agree:.1%}", flush=True)

    summ = pd.DataFrame(rows).set_index("partition")
    fmt = lambda v: f"{v:.3f}" if abs(v) < 100 else f"{v:.0f}"
    lines = [f"run: {run}", f"embedding: {args.emb} [{args.feat}], standardised, PCA {args.pca} "
             f"({pca.explained_variance_ratio_.sum():.1%} variance)",
             f"windows: {len(start)} of {d['n_all']} kept, hop {hop_s:.2f} s, {len(seqs)} sequences",
             f"fitted on: {args.fit_run if cross else 'this session'}",
             f"alpha {args.alpha:g}; self_trans = mean diagonal of the fitted transition matrix; "
             "ll_per_window = tempered log-likelihood of this session", "",
             summ.to_string(float_format=fmt)]
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-1:]))
    np.savez_compressed(out / "labels.npz", start=start, **parts)
    json.dump(dict(vars(args), run_dir=str(run), cross=cross, n_windows=len(start), hop_s=hop_s), open(out / "config.json", "w"), indent=2)
    plot(parts, start / fps, out / "overview.png", f"{run.parent.parent.name} / {args.emb}")
    print(f"done in {time.time() - t0:.0f}s -> {out}")


def plot(parts, tsec, path, title):
    """Ethograms of a 10 min excerpt from the middle of the session, one row per partition."""
    cmap = plt.get_cmap("tab20")
    names = list(parts)
    mid = tsec[len(tsec) // 2]
    sel = (tsec >= mid) & (tsec < mid + 600)
    fig, ax = plt.subplots(figsize=(16, 1.5 + 0.35 * len(names)))
    for r, n in enumerate(names):
        ax.scatter(tsec[sel] / 60, np.full(sel.sum(), r), c=[cmap(v % 20) for v in parts[n][sel]], marker="|", s=100, lw=1)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.set_ylim(len(names) - 0.5, -0.5)
    ax.set_xlabel("time (min)")
    ax.set_title(f"{title}: sticky HMM states, 10 min excerpt")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


if __name__ == "__main__":
    main()
