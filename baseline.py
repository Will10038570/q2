import math
from collections import defaultdict

import pandas as pd


class _UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px != py:
            self.parent[px] = py


_DISTRACTOR_CLASSES = {"person", "scooter"}


def _is_distractor(track: dict, n_frames_thresh: int, conf_thresh: float) -> bool:
    """Phase 1 distractor filter.

    Gates (OR logic — any single gate fires → distractor):
      1. cls in DISTRACTOR_CLASSES : person / scooter are never in GT.
      2. n_frames <= n_frames_thresh : very short tracks; no GT track has < 6 frames.
      3. world_valid_ratio == 0      : zero usable world coords; never seen in GT.
      4. mean_conf < conf_thresh     : consistently low-confidence detections.
    """
    if track["cls"] in _DISTRACTOR_CLASSES:
        return True
    frames = track["frames"]
    n = len(frames)
    if n <= n_frames_thresh:
        return True
    if sum(f["world_valid"] for f in frames) == 0:
        return True
    mean_conf = sum(f["confidence"] for f in frames) / n
    if mean_conf < conf_thresh:
        return True
    return False


def compute_pair_dist(a: dict, b: dict, max_dt: float = 1.0) -> list[float]:
    """Per-frame world-coord distances between two overlapping tracks.

    For each world_valid=1 frame in track a, finds the nearest-timestamp
    world_valid=1 frame in track b within max_dt seconds and computes the
    euclidean distance. Returns a list of distances (metres); empty when no
    frame pairs are found.
    """
    b_pts = [
        (f["timestamp"], f["world_x"], f["world_y"])
        for f in b["frames"] if f["world_valid"] == 1
    ]
    if not b_pts:
        return []

    dists = []
    for f in a["frames"]:
        if f["world_valid"] != 1:
            continue
        at, ax, ay = f["timestamp"], f["world_x"], f["world_y"]
        candidates = [(bt, bx, by) for bt, bx, by in b_pts if abs(bt - at) <= max_dt]
        if not candidates:
            continue
        _, bx, by = min(candidates, key=lambda p: abs(p[0] - at))
        dists.append(math.sqrt((ax - bx) ** 2 + (ay - by) ** 2))
    return dists


def compute_endpoint_dist(earlier: dict, later: dict) -> float | None:
    """Euclidean distance from the last world_valid position of `earlier`
    to the first world_valid position of `later`.

    Returns None when either track has no world_valid frames (gate is skipped).
    """
    last_valid = [
        (f["timestamp"], f["world_x"], f["world_y"])
        for f in earlier["frames"] if f["world_valid"] == 1
    ]
    first_valid = [
        (f["timestamp"], f["world_x"], f["world_y"])
        for f in later["frames"] if f["world_valid"] == 1
    ]
    if not last_valid or not first_valid:
        return None
    _, lx, ly = max(last_valid, key=lambda p: p[0])
    _, fx, fy = min(first_valid, key=lambda p: p[0])
    return math.sqrt((lx - fx) ** 2 + (ly - fy) ** 2)


def run_baseline(
    summary: dict,
    D_overlap_thresh_per_pair: dict[str, float] | None = None,
    D_seq_thresh_per_pair: dict[str, float] | None = None,
    T_max_gap: float = 15.0,
    V_max: float = 25.0,
    n_frames_thresh: int = 5,
    conf_thresh: float = 0.45,
    max_dt: float = 1.0,
) -> tuple[pd.DataFrame, dict]:
    """
    Associate cross-camera tracklets using class + temporal + spatial gates,
    followed by greedy per-camera-pair 1:1 matching and Union-Find propagation.

    Two tracks are candidates for association when ALL of the following hold:
      1. Same class.
      2. Temporal gap <= T_max_gap  (gap < 0 = temporal overlap; allows sequential pairs).
      3a. [Overlapping, gap <= 0]  Mean per-frame world-coord distance <= D_overlap_thresh.
          Gate skipped when no entry for this pair in D_overlap_thresh_per_pair,
          or when no frame pairs within max_dt seconds can be found.
      3b. [Sequential, gap > 0]  Endpoint-to-startpoint distance <= D_seq_thresh AND
          required travel speed (endpoint_dist / gap) <= V_max.
          Gate skipped when no entry for this pair in D_seq_thresh_per_pair,
          or when either track has no world_valid frames.

    Clustering strategy: greedy 1:1 matching per camera pair (sorted by spatial score
    ascending), then Union-Find to propagate matches across cameras. This avoids the
    cascade failure of complete-linkage clique clustering under high FP rates.

    Args:
        D_overlap_thresh_per_pair: per-camera-pair spatial threshold for overlapping
            pairs (metres), keyed "camera_X-camera_Y" (alphabetical).
        D_seq_thresh_per_pair: per-camera-pair endpoint distance threshold for
            sequential pairs (metres).
        T_max_gap: maximum allowed temporal gap in seconds. Pairs with gap > T_max_gap
            are rejected. Negative gap means temporal overlap (always passes).
        V_max: maximum plausible travel speed (m/s) for sequential pairs.
        n_frames_thresh: tracks with <= this many frames are marked distractor.
        conf_thresh: tracks with mean confidence < this are marked distractor.
        max_dt: max timestamp gap (seconds) when matching frames for the overlapping gate.

    Returns:
        (pred_df, stats) where pred_df has columns
        [camera_id, local_track_id, predicted_global_id] for every track.
    """
    all_keys = list(summary.keys())
    valid_keys = [k for k in all_keys if not _is_distractor(summary[k], n_frames_thresh, conf_thresh)]
    distractor_keys = [k for k in all_keys if _is_distractor(summary[k], n_frames_thresh, conf_thresh)]

    # Build compat: set of cross-camera pairs that pass all gates.
    compat: set[tuple] = set()
    n_overlapping_tested = n_sequential_tested = 0

    for i in range(len(valid_keys)):
        for j in range(i + 1, len(valid_keys)):
            a_key, b_key = valid_keys[i], valid_keys[j]
            if a_key[0] == b_key[0]:
                continue

            a, b = summary[a_key], summary[b_key]

            if a["cls"] != b["cls"]:
                continue

            gap = max(a["t_start"], b["t_start"]) - min(a["t_end"], b["t_end"])
            if gap > T_max_gap:
                continue

            pair_key = "-".join(sorted([a_key[0], b_key[0]]))

            if gap <= 0:
                n_overlapping_tested += 1
                dists = compute_pair_dist(a, b, max_dt)
                if dists:
                    mean_d = sum(dists) / len(dists)
                    d_overlap = (D_overlap_thresh_per_pair or {}).get(pair_key)
                    if d_overlap is not None and mean_d > d_overlap:
                        continue
            else:
                n_sequential_tested += 1
                earlier, later = (a, b) if a["t_end"] <= b["t_start"] else (b, a)
                ep_dist = compute_endpoint_dist(earlier, later)
                if ep_dist is not None:
                    if ep_dist / gap > V_max:
                        continue
                    d_seq = (D_seq_thresh_per_pair or {}).get(pair_key)
                    if d_seq is not None and ep_dist > d_seq:
                        continue

            compat.add((a_key, b_key))

    # clique-based clustering (complete linkage)
    # merge two clusters only when ALL cross-cluster pairs are in compat
    cluster_of = {key: key for key in valid_keys}
    members: dict = {key: {key} for key in valid_keys}

    changed = True
    while changed:
        changed = False
        for a_key, b_key in compat:
            ca, cb = cluster_of[a_key], cluster_of[b_key]
            if ca == cb:
                continue
            if all(
                (x, y) in compat or (y, x) in compat
                for x in members[ca] for y in members[cb]
            ):
                if len(members[ca]) < len(members[cb]):
                    ca, cb = cb, ca
                for key in members[cb]:
                    cluster_of[key] = ca
                members[ca] |= members[cb]
                del members[cb]
                changed = True
                break

    clusters = members

    rows = []
    gid = 0
    for root in sorted(clusters.keys(), key=str):
        gid += 1
        global_id = f"G{gid:04d}"
        for camera_id, local_track_id in clusters[root]:
            rows.append({
                "camera_id": camera_id,
                "local_track_id": local_track_id,
                "predicted_global_id": global_id,
            })

    for camera_id, local_track_id in distractor_keys:
        gid += 1
        rows.append({
            "camera_id": camera_id,
            "local_track_id": local_track_id,
            "predicted_global_id": f"G{gid:04d}",
        })

    stats = {
        "total_tracks": len(all_keys),
        "valid_tracks": len(valid_keys),
        "distractor_tracks": len(distractor_keys),
        "overlapping_pairs_tested": n_overlapping_tested,
        "sequential_pairs_tested": n_sequential_tested,
        "compat_pairs": len(compat),
        "association_clusters": len(clusters),
    }
    return pd.DataFrame(rows), stats
