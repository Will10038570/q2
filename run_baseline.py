from analyze_pair_dist import calibrate_thresholds
from baseline import run_baseline
from data_loader import build_tracklet_summary, load_tracklets
from evaluator import evaluate

df = load_tracklets("dev/tracklets.csv")
summary = build_tracklet_summary(df)
print(f"Tracks loaded: {len(summary)}")

thresholds = calibrate_thresholds(summary, "dev/ground_truth.csv")
T_MAX_GAP = thresholds["T_max_gap"]
V_MAX = thresholds["V_max"]
D_OVERLAP_THRESH_PER_PAIR = thresholds["D_overlap_thresh_per_pair"]
D_SEQ_THRESH_PER_PAIR = thresholds["D_seq_thresh_per_pair"]

print(f"\nCalibrated thresholds:")
print(f"  T_max_gap = {T_MAX_GAP:.2f} s")
print(f"  V_max     = {V_MAX:.2f} m/s")
print(f"  D_overlap = { {k: round(v, 2) for k, v in sorted(D_OVERLAP_THRESH_PER_PAIR.items())} }")
print(f"  D_seq     = { {k: round(v, 2) for k, v in sorted(D_SEQ_THRESH_PER_PAIR.items())} }")

pred, stats = run_baseline(
    summary,
    D_overlap_thresh_per_pair=D_OVERLAP_THRESH_PER_PAIR,
    D_seq_thresh_per_pair=D_SEQ_THRESH_PER_PAIR,
    T_max_gap=T_MAX_GAP,
    V_max=V_MAX,
)

print(f"\n=== Phase 1 Filter ===")
print(f"  Total tracks     : {stats['total_tracks']}")
print(f"  Distractor       : {stats['distractor_tracks']}")
print(f"  Valid (entering) : {stats['valid_tracks']}")

print(f"\n=== Phase 2 Gate Stats ===")
print(f"  Overlapping pairs tested : {stats['overlapping_pairs_tested']}")
print(f"  Sequential pairs tested  : {stats['sequential_pairs_tested']}")
print(f"  Compatible pairs         : {stats['compat_pairs']}")
print(f"  Matched pairs (greedy)   : {stats['matched_pairs']}")
print(f"  Clusters formed          : {stats['association_clusters']}")
print(f"  Total predicted IDs      : {pred['predicted_global_id'].nunique()}")

result = evaluate("dev/ground_truth.csv", pred, summary=summary)

print("\n=== ID Counts ===")
for k, v in result["id_counts"].items():
    print(f"  {k}: {v}")

print("\n=== Overall ===")
for k, v in result["overall"].items():
    print(f"  {k}: {round(v, 4) if isinstance(v, float) else v}")

print("\n=== Per Camera Pair ===")
for pair, m in result["per_pair"].items():
    print(f"  {pair}: P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")

print("\n=== Per Class ===")
for cls, m in sorted(result.get("per_class", {}).items()):
    print(f"  {cls}: P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  "
          f"TP={m['tp']}  FP={m['fp']}  FN={m['fn']}")

print("\n=== Temporal Split Recall ===")
for split, m in result.get("temporal_split", {}).items():
    print(f"  {split}: recall={m['recall']:.3f}  TP={m['tp']}  FN={m['fn']}  "
          f"total={m['total_pairs']}")
