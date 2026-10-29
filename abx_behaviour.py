"""Cross-session ABX for the partitions of a score_behaviour.py --fit-run run (a diagnostic, not a
target: a partition of the measured trajectories themselves would win it).

    python abx_behaviour.py SCORE_DIR [--triplets 10] [--min-windows 20]

SCORE_DIR = RUN_DIR/<emb>/behaviour/score_<tag>/ (needs fit_labels.npz). For states a != b:
A and B from session 1 with states a and b, X from session 2 with state a; correct when X is closer
to A than to B. Both directions (sessions swapped), A, B and X drawn from the same window-speed
tercile so that speed bins cannot win by speed alone. Distance: DTW (Euclidean frames, cost / path
length) between the windows' movement sequences from part_tracks.py, i.e. not the embedding: per
sampled frame, posture (paws in the body frame, log areas), body velocity in the window's frame and
heading change, each standardised over both sessions.

ABX error per pair of states, averaged over pairs: 0.5 = chance, lower = states that look alike
across sessions and differ from each other. excess_err = error minus km_kin's at the same entropy
(negative = better than k-means on the cheap kinematics).

Output: SCORE_DIR/abx.csv, abx.txt
"""

import argparse
import json
from pathlib import Path

import numba
import numpy as np
import pandas as pd

from score_behaviour import entropy_bits, frame_measures, to_frame, wrap


@numba.njit(parallel=True, cache=True)
def dtw_pairs(seqs, i, j):
    """DTW cost / path length between seqs[i[n]] and seqs[j[n]] (seqs [N, T, D])."""
    out = np.empty(len(i))
    T = seqs.shape[1]
    for n in numba.prange(len(i)):
        a, b = seqs[i[n]], seqs[j[n]]
        D = np.full((T + 1, T + 1), np.inf)
        L = np.zeros((T + 1, T + 1))
        D[0, 0] = 0.0
        for p in range(1, T + 1):
            for q in range(1, T + 1):
                c = np.sqrt(((a[p - 1] - b[q - 1]) ** 2).sum())
                best, ln = D[p - 1, q - 1], L[p - 1, q - 1]
                if D[p - 1, q] < best:
                    best, ln = D[p - 1, q], L[p - 1, q]
                if D[p, q - 1] < best:
                    best, ln = D[p, q - 1], L[p, q - 1]
                D[p, q], L[p, q] = best + c, ln + 1
        out[n] = D[T, T] / L[T, T]
    return out


def sequences(run, start, cfg):
    """[n, T, 16] movement sequence per window, and the window speed (px/s)."""
    m = frame_measures(run)
    idx = start[:, None] + np.arange(cfg["frames"])[None] * cfg["step"]
    head0 = m["head"][idx[:, 0]]
    prev = np.maximum(idx - cfg["step"], 0)
    rate = cfg["fps"] / cfg["step"]
    v = np.stack([to_frame(m["body"][idx[:, t]] - m["body"][prev[:, t]], head0) for t in range(idx.shape[1])], 1) * rate
    dh = wrap(m["head"][idx] - m["head"][prev]) * rate
    seq = np.concatenate([m["post"][idx], v, dh[..., None]], -1)
    speed = np.linalg.norm(m["body"][idx[:, -1]] - m["body"][idx[:, 0]], axis=1) / ((idx[:, -1] - idx[:, 0]) / cfg["fps"])
    return seq, speed


def abx(seq, sess, lab, strat, rng, n_trip, min_windows):
    """Mean over ordered state pairs (and both session directions) of the ABX error."""
    states = [s for s in np.unique(lab) if all(((lab == s) & (sess == k)).sum() >= min_windows for k in (0, 1))]
    groups = {}
    for k in (0, 1):
        for s in states:
            for q in np.unique(strat):
                groups[k, s, q] = np.flatnonzero((sess == k) & (lab == s) & (strat == q))
    A, B, X, pair = [], [], [], []
    p = 0
    for k in (0, 1):
        for a in states:
            for b in states:
                if a == b:
                    continue
                qs = [q for q in np.unique(strat) if min(len(groups[k, a, q]), len(groups[k, b, q]), len(groups[1 - k, a, q])) >= 3]
                if not qs:
                    continue
                for _ in range(n_trip):
                    q = qs[rng.integers(len(qs))]
                    A.append(rng.choice(groups[k, a, q]))
                    B.append(rng.choice(groups[k, b, q]))
                    X.append(rng.choice(groups[1 - k, a, q]))
                    pair.append(p)
                p += 1
    if not A:
        return np.nan, 0
    A, B, X, pair = map(np.array, (A, B, X, pair))
    dax, dbx = dtw_pairs(seq, A, X), dtw_pairs(seq, B, X)
    err = (dax > dbx) + 0.5 * (dax == dbx)
    return float(pd.Series(err).groupby(pair).mean().mean()), p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("score_dir")
    ap.add_argument("--triplets", type=int, default=10, help="per ordered pair of states and direction")
    ap.add_argument("--min-windows", type=int, default=20, help="a state needs this many windows in each session")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sd = Path(args.score_dir)
    cfg = json.load(open(sd / "config.json"))
    assert cfg["cross"], "needs a score_behaviour.py run with --fit-run"
    runs = [Path(cfg["fit_run"]), Path(cfg["run_dir"])]
    labs = [np.load(sd / "fit_labels.npz"), np.load(sd / "labels.npz")]
    ecfg = json.load(open(runs[1] / cfg["emb"] / "config.json"))
    seqs, speeds = zip(*(sequences(r, z["start"], ecfg) for r, z in zip(runs, labs)))
    seq = np.concatenate(seqs).astype(np.float64)
    seq = (seq - seq.mean((0, 1))) / (seq.std((0, 1)) + 1e-6)
    speed = np.concatenate(speeds)
    strat = np.digitize(speed, np.quantile(speed, [1 / 3, 2 / 3]))
    sess = np.r_[np.zeros(len(seqs[0]), int), np.ones(len(seqs[1]), int)]
    rng = np.random.default_rng(args.seed)

    rows = []
    for name in labs[0].files:
        if name == "start" or name not in labs[1].files:
            continue
        lab = np.r_[labs[0][name], labs[1][name]]
        err, pairs = abx(seq, sess, lab, strat, rng, args.triplets, args.min_windows)
        rows.append(dict(partition=name, family=name.rsplit("_", 1)[0] if name.startswith("km_") else "hdb",
                         k=len(np.unique(lab)), H_bits=entropy_bits(labs[1][name]), abx_error=err, pairs=pairs))
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    base = df[df.family == "km_kin"].sort_values("H_bits")
    df["excess_err"] = df.abx_error - np.interp(df.H_bits, base.H_bits, base.abx_error)
    df.to_csv(sd / "abx.csv", index=False)
    txt = df.sort_values("excess_err").to_string(index=False, float_format=lambda v: f"{v:.3f}")
    (sd / "abx.txt").write_text(f"cross-session ABX ({runs[0].parent.parent.name} <-> {runs[1].parent.parent.name}), "
                                f"speed-tercile matched, DTW on part trajectories; 0.5 = chance\n\n{txt}\n")
    print(txt)


if __name__ == "__main__":
    main()
