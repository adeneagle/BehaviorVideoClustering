"""How well do spikes predict the behaviour clusters? (cluster_embeddings.py output + a Rig2 segment)

    python neural_decode.py RUN_DIR --segment SEG_DIR [--emb vjepa_window] [--quality rf_good]
                            [--folds 10] [--shifts 20]

SEG_DIR is a Rig2 recording segment (.../RealtimeData/Rig2/<animal>/<session>/seg01) with
kilosort4/, phy/cluster_rf.tsv, phy/cluster_area.tsv and alignment/ (align_rig2.py). Spikes and
frame times come from HsuPaper/scripts/bonsai_loader.py (ephys seconds; decoded frame i ->
alignment fit), so the windows are the exact frames V-JEPA saw.

Per window (the frames [start, stop) of embed_vjepa.py, ~1 s, every 0.5 s): spike counts of every
selected unit, sqrt-transformed and z-scored on the training folds. Decoder: XGBoost (GPU hist,
multi:softprob, depth 4), boosting rounds set per fold by early stopping on an inner validation block
(the last 20% of the training windows), then refit on the whole training fold; decode_cv(decoder="logreg") is a multinomial logistic
regression alternative. Cross-validation is
blocked in time (--folds contiguous folds, 10 s of windows either side of each test fold dropped
from training) because neighbouring windows overlap and bouts last seconds.

Scores on held-out windows:
  bal_acc   balanced accuracy, decisions corrected for the class prior (argmax p / prior)
  bits      cross-validated information gain per window, mean log2 p(y) - log2 prior(y); a lower
            bound on I(spikes; cluster), comparable across label sets with different k
  frac_H    bits / H(cluster): share of the label entropy the spikes recover
Chance comes from --shifts circular shifts (fixed rounds) of the labels against the spikes (>= 10 min), which keep
both autocorrelations.

Also: decoding per area, from body-frame kinematics alone (speed, forward / lateral speed, turn,
part areas; degree-2 expansion) and from kinematics + spikes, and the spike/label lag (-3..+3 s, fixed rounds).

Output in RUN_DIR/<emb>/neural/: summary.txt, scores.csv, per_cluster.csv, confusion.png, lag.png
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

HSU_SCRIPTS = Path("/mnt/howlin/mnt/data/samba/yttri/Aden/HsuPaper/scripts")
EPS = 1e-3  # probability floor (mixed with uniform), so a class unseen in training is not -inf bits
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- data
def load_session(seg, quality):
    sys.path.insert(0, str(HSU_SCRIPTS))
    from bonsai_loader import BonsaiSession

    s = BonsaiSession(seg)
    sort = "single" if "single" in s.sorts else next(iter(s.sorts))
    units = s.units(sort)
    ids = s.select(units, quality)
    t, c = s.spikes(sort, cluster_ids=ids)
    area_f = s.sorts[sort].path / "phy" / "cluster_area.tsv"
    area = (pd.read_csv(area_f, sep="\t").set_index("cluster_id")["area"].reindex(ids).fillna("unassigned").to_numpy()
            if area_f.exists() else np.full(len(ids), "all"))
    frame_s = s.frames_to_s(np.arange(s.cam_rel_s.size))
    return dict(t=t, c=c, ids=ids, area=area, frame_s=frame_s, period=s.frame_period_s, sort=sort)


def window_counts(sess, start, stop, lag_s=0.0):
    """Spike counts [windows, units] over frames [start, stop), shifted by lag_s (> 0 = spikes later)."""
    fs = sess["frame_s"]
    t0 = fs[start] + lag_s
    t1 = fs[np.minimum(stop, len(fs) - 1)] + lag_s
    X = np.zeros((len(start), len(sess["ids"])), np.float32)
    order = np.argsort(sess["c"], kind="stable")
    c, t = sess["c"][order], sess["t"][order]
    bounds = np.searchsorted(c, sess["ids"]), np.searchsorted(c, sess["ids"], side="right")
    for j, (a, b) in enumerate(zip(*bounds)):
        tu = t[a:b]
        X[:, j] = np.searchsorted(tu, t1) - np.searchsorted(tu, t0)
    return X


def kinematics(run, emb, start, stop, idx, pos=False):
    """Window-level body-frame motion and part areas (what the clusters could be 'just' encoding).
    pos=True adds the window-mean body centre in the frame (arena position), which raw-video
    features can see (walls, corners) and the part maps cannot."""
    z = np.load(run / emb / "features.npz")
    fr = pd.read_csv(run / "frames.csv")
    row = {int(s): i for i, s in enumerate(z["start"])}
    w = np.array([row[int(s)] for s in start])
    cols = [c for c in fr.columns if c.startswith("area_")]
    cs = np.vstack([np.zeros(len(cols)), np.cumsum(fr[cols].fillna(0).to_numpy(), 0)])
    areas = (cs[stop] - cs[start]) / (stop - start)[:, None]
    xy = fr[["cx", "cy"]].to_numpy()
    cstep = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    speed = (cstep[stop - 1] - cstep[start]) / (stop - 1 - start)
    K = np.c_[speed, z["fwd_speed"][w], z["lat_speed"][w], z["turn_rate"][w], np.abs(z["turn_rate"][w]), areas]
    names = ["speed", "fwd", "lat", "turn", "abs_turn"] + cols
    if pos:
        cxy = np.vstack([np.zeros(2), np.cumsum(xy, 0)])
        K = np.c_[K, (cxy[stop] - cxy[start]) / (stop - start)[:, None]]
        names += ["x", "y"]
    K = np.nan_to_num(K)
    return K, names


# ---------------------------------------------------------------- decoder
def fit_softmax(X, y, k, l2, iters=150):
    n, d = X.shape
    W = torch.zeros(d, k, device=DEV, requires_grad=True)
    b = torch.zeros(k, device=DEV, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(X @ W + b, y) + l2 * (W ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach(), b.detach()


def predict(Wb, X):
    W, b = Wb
    return torch.softmax(X @ W + b, 1)


def blocked_folds(n, folds, buffer):
    edges = np.linspace(0, n, folds + 1).astype(int)
    for a, b in zip(edges[:-1], edges[1:]):
        test = np.arange(a, b)
        train = np.r_[0 : max(a - buffer, 0), min(b + buffer, n) : n]
        yield train, test


def standardize(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    return (Xtr - mu) / sd, (Xte - mu) / sd


XGB_PARAMS = dict(objective="multi:softprob", tree_method="hist", device="cuda", max_bin=64, max_depth=4,
                  learning_rate=0.1, subsample=0.8, colsample_bytree=0.5, min_child_weight=5, reg_lambda=1.0,
                  # one shared starting margin: xgboost >= 2 otherwise estimates a per-class intercept from the
                  # training labels, which sends a class absent from the training fold to p ~ 0
                  base_score=0.5)


def decode_cv(X, y, k, folds=10, buffer=20, decoder="xgb", rounds=None, **kw):
    """Blocked-CV held-out class probabilities [n, k], training-fold priors, and per fold the
    complexity used (xgb: boosting rounds; logreg: L2)."""
    if decoder == "logreg":
        return decode_cv_logreg(X, y, k, folds, buffer, l2=rounds, **kw)
    return decode_cv_xgb(X, y, k, folds, buffer, rounds)


def decode_cv_xgb(X, y, k, folds, buffer, rounds=None, max_rounds=600, patience=30, min_rounds=20):
    """XGBoost per fold. rounds=None: the number of rounds is picked by early stopping on the last
    20% of the training windows (a contiguous-in-time block, buffered; at least min_rounds), then
    the model is refit on the whole training fold, so behaviour seen only late in the training
    windows is still learned. Otherwise `rounds` rounds on the whole training fold (used for the
    nulls and the lag scan)."""
    import xgboost as xgb

    X = np.asarray(X, np.float32)
    P = np.zeros((len(y), k))
    prior = np.zeros((len(y), k))
    used = []
    params = dict(XGB_PARAMS, num_class=k)
    for tr, te in blocked_folds(len(y), folds, buffer):
        dte = xgb.DMatrix(X[te])
        if rounds is None:
            cut = int(0.8 * len(tr))
            a, v = tr[: max(cut - buffer, 1)], tr[cut:]
            dtr, dva = xgb.DMatrix(X[a], label=y[a]), xgb.DMatrix(X[v], label=y[v])
            bst = xgb.train(params, dtr, max_rounds, evals=[(dva, "val")], early_stopping_rounds=patience, verbose_eval=False)
            n_used = max(bst.best_iteration + 1, min_rounds)
        else:
            n_used = rounds
        bst = xgb.train(params, xgb.DMatrix(X[tr], label=y[tr]), n_used)
        P[te] = bst.predict(dte)
        used.append(n_used)
        cnt = np.bincount(y[tr], minlength=k) + 1.0
        prior[te] = cnt / cnt.sum()
    P = (1 - EPS) * P + EPS / k
    return P, prior, used


def decode_cv_logreg(X, y, k, folds=10, buffer=20, l2_grid=(1e-4, 1e-3, 1e-2, 1e-1), l2=None):
    """Blocked-CV held-out class probabilities [n, k] and the L2 used per fold."""
    X = torch.as_tensor(X, dtype=torch.float32, device=DEV)
    yt = torch.as_tensor(y, dtype=torch.long, device=DEV)
    P = torch.zeros(len(y), k, device=DEV)
    prior = torch.zeros(len(y), k, device=DEV)
    used = []
    for tr, te in blocked_folds(len(y), folds, buffer):
        Xtr, Xte = standardize(X[tr], X[te])
        lam = l2
        if lam is None:  # inner split: last 20% of the training windows (contiguous enough) as validation
            cut = int(0.8 * len(tr))
            a, v = slice(0, max(cut - buffer, 1)), slice(cut, None)
            losses = []
            for g in l2_grid:
                p = predict(fit_softmax(Xtr[a], yt[tr][a], k, g, iters=80), Xtr[v])
                p = (1 - EPS) * p + EPS / k
                losses.append(-torch.log(p[torch.arange(p.shape[0]), yt[tr][v]]).mean().item())
            lam = l2_grid[int(np.argmin(losses))]
        used.append(lam)
        P[te] = predict(fit_softmax(Xtr, yt[tr], k, lam), Xte)
        cnt = torch.bincount(yt[tr], minlength=k).float() + 1
        prior[te] = cnt / cnt.sum()
    P = (1 - EPS) * P + EPS / k
    return P.cpu().numpy(), prior.cpu().numpy(), used


def scores(P, prior, y):
    k = P.shape[1]
    n = np.arange(len(y))
    bits = np.mean(np.log2(P[n, y]) - np.log2(prior[n, y]))
    pred = np.argmax(P / prior, 1)
    present = np.unique(y)
    bal = np.mean([np.mean(pred[y == c] == c) for c in present])
    freq = np.bincount(y, minlength=k) / len(y)
    H = -np.sum(freq[freq > 0] * np.log2(freq[freq > 0]))
    return dict(bal_acc=bal, chance_bal=1 / len(present), acc=np.mean(np.argmax(P, 1) == y), bits=bits, H=H, frac_H=bits / H)


def shift_null(X, y, k, n_shift, min_shift, rounds, folds, buffer, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_shift):
        s = rng.integers(min_shift, len(y) - min_shift)
        ys = np.roll(y, s)
        P, prior, _ = decode_cv(X, ys, k, folds, buffer, rounds=rounds)
        out.append(scores(P, prior, ys))
    return pd.DataFrame(out)


def sqrt_counts(X):
    return np.sqrt(X)


def poly2(K):
    K = (K - K.mean(0)) / (K.std(0) + 1e-6)
    iu = np.triu_indices(K.shape[1])
    return np.c_[K, (K[:, :, None] * K[:, None, :])[:, iu[0], iu[1]]]


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--segment", required=True)
    ap.add_argument("--emb", default="vjepa_window")
    ap.add_argument("--quality", default="rf_good", help="bonsai_loader.select(): rf_good, rf, bombcell, ...")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--buffer-s", type=float, default=10)
    ap.add_argument("--shifts", type=int, default=5)
    ap.add_argument("--draws", type=int, default=3, help="random equal-size unit subsets per area")
    args = ap.parse_args()

    run = Path(args.run_dir)
    out = run / args.emb / "neural"
    out.mkdir(exist_ok=True)
    win = pd.read_csv(run / args.emb / "clusters" / "windows.csv")
    cfg = json.load(open(run / args.emb / "config.json"))
    start = win.start.to_numpy()
    stop = start + cfg["span"]
    sess = load_session(args.segment, args.quality)
    fs = sess["frame_s"]
    hop_s = cfg["hop"] * sess["period"]
    buffer = int(round(args.buffer_s / hop_s))
    min_shift = int(round(600 / hop_s))
    print(f"{len(sess['ids'])} units ({args.quality}); {len(win)} windows; frames span ephys "
          f"{fs[0]:.1f}-{fs[-1]:.1f} s; hop {hop_s:.3f} s", flush=True)

    Xall = window_counts(sess, start, stop)
    Kall, knames = kinematics(run, args.emb, start, stop, None)
    lab_all = win.cluster.to_numpy()
    m = lab_all >= 0
    ks = np.unique(lab_all[m])
    y = np.searchsorted(ks, lab_all[m])
    k = len(ks)
    X = sqrt_counts(Xall[m])
    Kp = poly2(Kall[m])
    lines = [f"run: {run} ({args.emb})", f"segment: {args.segment} (sort {sess['sort']})",
             f"units: {len(sess['ids'])} {args.quality}: " + ", ".join(f"{a} {n}" for a, n in zip(*np.unique(sess['area'], return_counts=True))),
             f"windows: {len(win)}, clustered {m.sum()} ({m.mean():.1%}), {k} clusters; blocked {args.folds}-fold CV, "
             f"{args.buffer_s:.0f} s buffer; chance = {args.shifts} circular shifts >= 10 min", ""]

    rows = []

    def run_set(name, F, null=True):
        P, prior, used = decode_cv(F, y, k, args.folds, buffer)
        r = dict(features=name, n_feat=F.shape[1], **scores(P, prior, y), rounds=int(np.median(used)))
        if null and args.shifts:
            nd = shift_null(F, y, k, args.shifts, min_shift, r["rounds"], args.folds, buffer)
            r.update(null_bal_acc=nd.bal_acc.mean(), null_bal_acc_95=nd.bal_acc.quantile(0.95),
                     null_bits=nd.bits.mean(), null_bits_95=nd.bits.quantile(0.95))
        rows.append(r)
        print({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in r.items()}, flush=True)
        return P, prior

    P_all, prior_all = run_set("spikes: all units", X)
    for a in [a for a in np.unique(sess["area"]) if a != "unassigned"]:
        run_set(f"spikes: {a}", X[:, sess["area"] == a], null=False)
    P_kin, prior_kin = run_set("kinematics (deg-2)", Kp)
    run_set("kinematics + spikes", np.c_[Kp, X], null=False)

    # matched unit counts across areas (areas differ in n)
    rng = np.random.default_rng(0)
    nmin = min(np.sum(sess["area"] == a) for a in np.unique(sess["area"]) if a != "unassigned")
    for a in [a for a in np.unique(sess["area"]) if a != "unassigned"]:
        idx = np.flatnonzero(sess["area"] == a)
        bs = []
        for _ in range(args.draws):
            sub = rng.choice(idx, nmin, replace=False)
            P, prior, _ = decode_cv(X[:, sub], y, k, args.folds, buffer)
            bs.append(scores(P, prior, y))
        bs = pd.DataFrame(bs).mean()
        rows.append(dict(features=f"spikes: {a}, {nmin} units ({args.draws} draws)", n_feat=nmin, **bs.to_dict()))

    df = pd.DataFrame(rows)
    df.to_csv(out / "scores.csv", index=False)
    lines += ["decoding the clusters (held-out windows):",
              df.to_string(index=False, float_format=lambda v: f"{v:.3f}"), ""]

    # per cluster: recall from spikes vs from kinematics
    n = np.arange(len(y))
    pred_s, pred_k = np.argmax(P_all / prior_all, 1), np.argmax(P_kin / prior_kin, 1)
    summ = pd.read_csv(run / args.emb / "clusters" / "windows.csv").query("cluster >= 0").groupby("cluster").agg(
        windows=("start", "size"), speed=("speed", "median"), fwd=("fwd_speed", "median"), turn=("turn_rate", "median"))
    summ["recall_spikes"] = [np.mean(pred_s[y == i] == i) for i in range(k)]
    summ["recall_kin"] = [np.mean(pred_k[y == i] == i) for i in range(k)]
    summ["bits_spikes"] = [np.mean(np.log2(P_all[y == i, i]) - np.log2(prior_all[y == i, i])) for i in range(k)]
    summ.to_csv(out / "per_cluster.csv")
    lines += ["per cluster (recall with prior-corrected decisions; chance = 1/k):",
              summ.to_string(float_format=lambda v: f"{v:.2f}"), ""]

    # lag: spikes shifted against the labels
    lags = np.array([-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3])
    lag_rows = []
    for L in lags:
        XL = sqrt_counts(window_counts(sess, start, stop, L)[m])
        P, prior, _ = decode_cv(XL, y, k, args.folds, buffer, rounds=rows[0]["rounds"])
        lag_rows.append(dict(lag_s=L, **scores(P, prior, y)))
    lag_df = pd.DataFrame(lag_rows)
    lag_df.to_csv(out / "lag.csv", index=False)
    lines += ["lag (spike window shifted by lag_s; > 0 = spikes after the behaviour window):",
              lag_df[["lag_s", "bal_acc", "bits"]].to_string(index=False, float_format=lambda v: f"{v:.3f}")]

    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    C = np.zeros((k, k))
    np.add.at(C, (y, pred_s), 1)
    C /= C.sum(1, keepdims=True)
    im = ax[0].imshow(C, vmin=0, vmax=max(0.3, C.diagonal().max()), cmap="viridis")
    ax[0].set_xticks(range(k), ks, fontsize=6)
    ax[0].set_yticks(range(k), ks, fontsize=6)
    ax[0].set_xlabel("decoded from spikes")
    ax[0].set_ylabel("cluster")
    ax[0].set_title(f"confusion (row-normalised), bal. acc {rows[0]['bal_acc']:.2f} (chance {1 / k:.3f})")
    fig.colorbar(im, ax=ax[0])
    ax[1].plot(lag_df.lag_s, lag_df.bits, "o-")
    ax[1].axvline(0, color="k", lw=0.5)
    ax[1].set_xlabel("spike window lag (s); > 0 = spikes after behaviour")
    ax[1].set_ylabel("bits / window (held out)")
    fig.tight_layout()
    fig.savefig(out / "confusion_lag.png", dpi=110)


if __name__ == "__main__":
    main()
