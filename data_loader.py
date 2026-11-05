import pandas as pd


def load_tracklets(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def build_tracklet_summary(df: pd.DataFrame) -> dict:
    """
    Returns Dict[(camera_id, local_track_id), {t_start, t_end, cls, world_centroid, frames}]
    world_centroid is (x, y) mean over world_valid=1 frames, or None if none exist.
    """
    summary = {}
    for (camera_id, local_track_id), group in df.groupby(["camera_id", "local_track_id"]):
        t_start = float(group["timestamp"].min())
        t_end = float(group["timestamp"].max())

        cls = group["class"].mode()[0]

        valid = group[group["world_valid"] == 1]
        world_centroid = (
            (float(valid["world_x"].mean()), float(valid["world_y"].mean()))
            if len(valid) > 0
            else None
        )

        frames = group[[
            "frame_id", "timestamp", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
            "confidence", "world_x", "world_y", "world_valid"
        ]].to_dict("records")

        summary[(camera_id, local_track_id)] = {
            "t_start": t_start,
            "t_end": t_end,
            "cls": cls,
            "world_centroid": world_centroid,
            "frames": frames,
        }
    return summary
