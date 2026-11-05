import os
import tempfile
from itertools import combinations

import pandas as pd

from eval import score


def evaluate(gt_path: str, pred_df: pd.DataFrame, summary: dict | None = None) -> dict:
    """
    Returns overall, per-camera-pair, per-class, and temporal-split evaluation.

    Per-class and temporal-split require `summary` (from build_tracklet_summary).

    Per-class note: FPs from cross-class over-merging (e.g. a car merged with a truck)
    are not counted within per-class precision; per-class precision may therefore be
    higher than overall precision.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        pred_df.to_csv(f, index=False)
        pred_tmp = f.name

    try:
        overall = score(gt_path, pred_tmp)
    finally:
        os.unlink(pred_tmp)

    gt = pd.read_csv(gt_path)

    merged = gt.merge(pred_df, on=["camera_id", "local_track_id"], how="left")
    id_counts = {
        "gt_global_ids":   int(gt["global_id"].nunique()),
        "pred_global_ids": int(merged["predicted_global_id"].nunique()),
        "labeled_tracks":  len(gt),
        "pred_total_ids":  int(pred_df["predicted_global_id"].nunique()),
    }

    cameras = sorted(gt["camera_id"].unique())
    per_pair = {}
    for cam_a, cam_b in combinations(cameras, 2):
        gt_pair = gt[gt["camera_id"].isin([cam_a, cam_b])]
        pred_pair = pred_df[pred_df["camera_id"].isin([cam_a, cam_b])]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            gt_pair.to_csv(f, index=False)
            gt_tmp = f.name
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            pred_pair.to_csv(f, index=False)
            pred_tmp = f.name

        try:
            per_pair[f"{cam_a}-{cam_b}"] = score(gt_tmp, pred_tmp)
        finally:
            os.unlink(gt_tmp)
            os.unlink(pred_tmp)

    result = {
        "overall": overall,
        "id_counts": id_counts,
        "per_pair": per_pair,
    }

    if summary is None:
        return result

    # --- Per-class evaluation ---
    track_cls = {(cam, tid): summary[(cam, tid)]["cls"] for (cam, tid) in summary}
    gt_with_cls = gt.copy()
    gt_with_cls["cls"] = gt_with_cls.apply(
        lambda r: track_cls.get((r["camera_id"], r["local_track_id"]), "unknown"), axis=1
    )

    per_class = {}
    for cls_name, grp in gt_with_cls.groupby("cls"):
        gt_cls_df = grp.drop(columns=["cls"])
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            gt_cls_df.to_csv(f, index=False)
            gt_tmp = f.name
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            pred_df.to_csv(f, index=False)
            pred_tmp = f.name
        try:
            per_class[cls_name] = score(gt_tmp, pred_tmp)
        finally:
            os.unlink(gt_tmp)
            os.unlink(pred_tmp)

    result["per_class"] = per_class

    # --- Sequential vs Overlapping pair recall ---
    pred_id = dict(zip(
        zip(pred_df["camera_id"], pred_df["local_track_id"]),
        pred_df["predicted_global_id"],
    ))

    seq_tp = seq_fn = ov_tp = ov_fn = 0
    for gid, grp in gt.groupby("global_id"):
        keys = [(r["camera_id"], r["local_track_id"]) for _, r in grp.iterrows()]
        for a_key, b_key in combinations(keys, 2):
            if a_key[0] == b_key[0]:
                continue
            if a_key not in summary or b_key not in summary:
                continue

            a_track, b_track = summary[a_key], summary[b_key]
            gap = max(a_track["t_start"], b_track["t_start"]) - min(a_track["t_end"], b_track["t_end"])
            pa, pb = pred_id.get(a_key), pred_id.get(b_key)
            is_tp = pa is not None and pb is not None and pa == pb

            if gap > 0:
                seq_tp += int(is_tp)
                seq_fn += int(not is_tp)
            else:
                ov_tp += int(is_tp)
                ov_fn += int(not is_tp)

    total_seq = seq_tp + seq_fn
    total_ov = ov_tp + ov_fn
    result["temporal_split"] = {
        "sequential": {
            "total_pairs": total_seq,
            "tp": seq_tp,
            "fn": seq_fn,
            "recall": seq_tp / total_seq if total_seq else 0.0,
        },
        "overlapping": {
            "total_pairs": total_ov,
            "tp": ov_tp,
            "fn": ov_fn,
            "recall": ov_tp / total_ov if total_ov else 0.0,
        },
    }

    return result
