"""XGBoost check of partitions chosen with tune_neural.py: bits decoded from spikes, and beyond kinematics.

    python confirm_neural.py RUN_DIR --segment SEG_DIR --tune TUNE_DIR --keys "KEY1;KEY2" [--emb vjepa_layers]
                             [--folds 5]

KEYs are partitions in TUNE_DIR/labels.npz (config name and min-cluster fraction joined by '__',
e.g. 'default__0.005' or 'feat=feats_L16,kin_w=1.0__0.005'). For each: blocked-CV XGBoost
(neural_decode.decode_cv) from spikes, from kinematics (degree 2) and from both.
  dbits = bits(kin + spikes) - bits(kin): what the spikes say about the partition that the
          kinematics do not. Partitions that are mostly speed/turn bins score ~0.

Output: TUNE_DIR/confirm.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import neural_decode as nd
from tune_neural import load_inputs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--segment", required=True)
    ap.add_argument("--tune", required=True)
    ap.add_argument("--keys", required=True)
    ap.add_argument("--emb", default="vjepa_layers")
    ap.add_argument("--quality", default="rf_good")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--kin-pos", action="store_true", help="add arena position to the kinematics")
    args = ap.parse_args()

    run, tune = Path(args.run_dir), Path(args.tune)
    L = np.load(tune / "labels.npz")
    d = load_inputs(run, args.emb, args.segment, args.quality, [], pos=args.kin_pos)
    assert np.array_equal(L["start"], d["start"]), "labels.npz windows differ from this run's"
    buffer = int(round(10 / d["hop_s"]))
    rows = []
    for key in args.keys.split(";"):
        y = L[key]
        k = y.max() + 1
        r = dict(partition=key, k=k)
        for name, F in [("spikes", d["X"]), ("kin", d["Kp"]), ("kin_spikes", np.c_[d["Kp"], d["X"]])]:
            P, prior, _ = nd.decode_cv(F, y, k, args.folds, buffer)
            s = nd.scores(P, prior, y)
            r.update({f"{name}_bits": s["bits"], f"{name}_bal_acc": s["bal_acc"]})
            r["H"] = s["H"]
        r["dbits"] = r["kin_spikes_bits"] - r["kin_bits"]
        rows.append(r)
        print({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in r.items()}, flush=True)
    df = pd.DataFrame(rows)
    out = tune / "confirm.csv"
    if out.exists():  # keep earlier keys
        df = pd.concat([pd.read_csv(out).query("partition not in @df.partition"), df])
    df.to_csv(out, index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
