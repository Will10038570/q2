from itertools import combinations

import pandas as pd

from baseline import compute_endpoint_dist, compute_pair_dist
from data_loader import build_tracklet_summary, load_tracklets


def calibrate_thresholds(
    summary: dict,
    gt_path: str,
    pct_overlap: float = 90,
    pct_seq_ep: float = 90,
    pct_gap: float = 90,
    pct_vmax: float = 95,
) -> dict:
    """Calibrate association thresholds from GT data.

    Returns:
        {
            "T_max_gap": float,
            "D_overlap_thresh_per_pair": {pair_key: float},
            "D_seq_thresh_per_pair": {pair_key: float},
            "V_max": float,
        }
    """
    gt = pd.read_csv(gt_path)

    overlap_dists: dict[str, list[float]] = {}
    seq_ep_dists: dict[str, list[float]] = {}
    all_gaps: list[float] = []
    all_speeds: list[float] = []

    for gid, grp in gt.groupby("global_id"):
        keys = [(r["camera_id"], r["local_track_id"]) for _, r in grp.iterrows()]
        for a_key, b_key in combinations(keys, 2):
            if a_key[0] == b_key[0]:
                continue
            if a_key not in summary or b_key not in summary:
                continue

            a, b = summary[a_key], summary[b_key]
            pair_key = "-".join(sorted([a_key[0], b_key[0]]))
            gap = max(a["t_start"], b["t_start"]) - min(a["t_end"], b["t_end"])

            if gap <= 0:
                d = compute_pair_dist(a, b, max_dt=0.5) + compute_pair_dist(b, a, max_dt=0.5)
                if d:
                    overlap_dists.setdefault(pair_key, []).extend(d)
            else:
                all_gaps.append(gap)
                earlier, later = (a, b) if a["t_end"] <= b["t_start"] else (b, a)
                ep_dist = compute_endpoint_dist(earlier, later)
                if ep_dist is not None:
                    seq_ep_dists.setdefault(pair_key, []).append(ep_dist)
                    all_speeds.append(ep_dist / gap)

    def pct(lst, p):
        s = sorted(lst)
        idx = max(0, int(p / 100 * (len(s) - 1)))
        return round(s[idx], 3)

    T_max_gap = pct(all_gaps, pct_gap) if all_gaps else 15.0
    V_max = pct(all_speeds, pct_vmax) if all_speeds else 25.0
    D_overlap = {pk: pct(dists, pct_overlap) for pk, dists in overlap_dists.items()}
    D_seq = {pk: pct(dists, pct_seq_ep) for pk, dists in seq_ep_dists.items()}

    return {
        "T_max_gap": T_max_gap,
        "D_overlap_thresh_per_pair": D_overlap,
        "D_seq_thresh_per_pair": D_seq,
        "V_max": V_max,
    }


if __name__ == "__main__":
    df = load_tracklets("dev/tracklets.csv")
    summary = build_tracklet_summary(df)
    t = calibrate_thresholds(summary, "dev/ground_truth.csv")

    T_max_gap = t["T_max_gap"]
    D_overlap = t["D_overlap_thresh_per_pair"]
    D_seq = t["D_seq_thresh_per_pair"]
    V_max = t["V_max"]

    print("=" * 60)
    print("Calibrated thresholds from GT (p90 distances, p95 V_max)")
    print("=" * 60)
    print(f"\nT_max_gap : {T_max_gap:.2f} s  (p90 of GT sequential gaps)")
    print(f"V_max     : {V_max:.2f} m/s  (p95 of GT sequential required speeds)")

    print("\nD_overlap_thresh_per_pair  (p90 of GT overlapping per-frame distances):")
    for pk in sorted(D_overlap):
        print(f"  {pk}: {D_overlap[pk]:.2f} m")

    print("\nD_seq_thresh_per_pair  (p90 of GT sequential endpoint distances):")
    for pk in sorted(D_seq):
        print(f"  {pk}: {D_seq[pk]:.2f} m")

    print("\n" + "=" * 60)
    print("Copy-paste config:")
    print("=" * 60)
    print(f"T_MAX_GAP = {T_max_gap}")
    print(f"V_MAX = {V_max}")
    print("D_OVERLAP_THRESH_PER_PAIR = {")
    for k, v in sorted(D_overlap.items()):
        print(f'    "{k}": {v},')
    print("}")
    print("D_SEQ_THRESH_PER_PAIR = {")
    for k, v in sorted(D_seq.items()):
        print(f'    "{k}": {v},')
    print("}")
