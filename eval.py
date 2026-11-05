#!/usr/bin/env python3
import argparse
import pandas as pd
from itertools import combinations

def score(gt_path, pred_path):
    gt = pd.read_csv(gt_path)
    pred = pd.read_csv(pred_path)
    required = {"camera_id", "local_track_id", "predicted_global_id"}
    if not required.issubset(pred.columns):
        raise ValueError(f"prediction must contain {sorted(required)}")

    key = ["camera_id", "local_track_id"]
    d = gt.merge(pred[list(required)], on=key, how="left")
    # Missing prediction behaves as a unique singleton.
    d["predicted_global_id"] = d["predicted_global_id"].astype("object")
    for i in d.index[d.predicted_global_id.isna()]:
        d.at[i, "predicted_global_id"] = f"__MISSING_{i}"

    tp = fp = fn = 0
    rows = list(d.itertuples(index=False))
    for a, b in combinations(rows, 2):
        if a.camera_id == b.camera_id:
            continue
        true_same = a.global_id == b.global_id
        pred_same = a.predicted_global_id == b.predicted_global_id
        if pred_same and true_same:
            tp += 1
        elif pred_same and not true_same:
            fp += 1
        elif not pred_same and true_same:
            fn += 1

    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2*p*r/(p+r) if p+r else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    args = ap.parse_args()
    print(score(args.gt, args.pred))
