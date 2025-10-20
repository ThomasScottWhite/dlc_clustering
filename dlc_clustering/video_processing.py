# video_processing.py
import cv2
import math
import polars as pl
from pathlib import Path
from typing import List, Optional, Tuple
from dlc_clustering.projects import Project
from tqdm import tqdm
import numpy as np

def open_video_captures(video_paths: List[str]) -> Tuple[List[cv2.VideoCapture], int, int, float]:
    caps = []
    width = height = fps = None
    for p in video_paths:
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            raise ValueError(f"Error: Could not open video {p}")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        if width is None:
            width, height, fps = w, h, f
        else:
            if (w, h) != (width, height):
                raise ValueError(f"All videos must share resolution. Got {(w,h)} vs {(width,height)}.")
            if abs(f - fps) > 1e-6:
                raise ValueError(f"All videos must share FPS. Got {f} vs {fps}.")
        caps.append(cap)
    return caps, width, height, fps

def tile_frames(frames: List[np.ndarray], width: int, height: int, cols: int) -> np.ndarray:
    """Tile frames into rows x cols grid without resizing; pad black frames to fill grid."""
    n = len(frames)
    rows = math.ceil(n / cols)

    # pad with black if needed
    black = np.zeros((height, width, 3), dtype=np.uint8)
    frames_padded = frames + [black] * (rows * cols - n)

    row_imgs = []
    for r in range(rows):
        row_imgs.append(cv2.hconcat(frames_padded[r*cols:(r+1)*cols]))
    return cv2.vconcat(row_imgs) if rows > 1 else row_imgs[0]

def render_videos(
    video_paths: List[str],
    clustering_output: pl.DataFrame,
    out_dir: str,
    time_series_data: Optional[pl.DataFrame],
    camera_order: Optional[List[str]] = None,
    layout_cols: Optional[int] = None,
    label_each_pane: bool = True
):
    """
    Render N camera streams into a single composite frame per timestep.
    - No resizing; asserts equal FPS and resolution.
    - layout_cols: number of columns in the grid (default: 2 for N=2, else ceil(sqrt(N))).
    """
    if not video_paths:
        print("No video paths provided.")
        return

    captures, width, height, fps = open_video_captures(video_paths)
    n = len(captures)
    if layout_cols is None:
        layout_cols = 2 if n == 2 else math.ceil(math.sqrt(n))
    rows = math.ceil(n / layout_cols)

    # determine minimum available frames across streams, this should be redundant as videos should be the same frame length
    frame_counts = []
    for cap in captures:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_counts.append(total if total > 0 else 10**12)
    max_render_frames = min(frame_counts)

    # make sure there is no null values which may cause errors
    clustering_output = clustering_output.with_columns([
    pl.col("cluster").cast(pl.Int32).fill_null(-1),
    pl.col("bout_id").cast(pl.Int32).fill_null(-1),
    ])

    cluster_list = clustering_output["cluster"].to_list()
    bout_list    = clustering_output["bout_id"].to_list()
    total_frames = min(max_render_frames, len(cluster_list))

    if time_series_data is not None:
        if time_series_data.shape[0] < total_frames:
            total_frames = time_series_data.shape[0]
        text_cols = [c for c in time_series_data.columns if c.endswith("_text")]
        flag_cols = [c for c in time_series_data.columns if c.endswith("_flag")]
        flag_positions = {
            col: time_series_data.select(pl.col(col)).to_series().to_numpy().nonzero()[0]
            for col in flag_cols
        }
    else:
        text_cols, flag_cols, flag_positions = [], [], {}

    composite_w = layout_cols * width
    composite_h = rows * height

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")


    # filter out invalids before sorting
    valid_clusters = sorted({c for c in cluster_list if c is not None and c != -1})
    cluster_writers = {
        cid: cv2.VideoWriter(str(Path(out_dir) / f"cluster_{cid}.mp4"), fourcc, fps, (composite_w, composite_h))
        for cid in valid_clusters
    }
    all_writer_path = Path(out_dir) / "all_clusters.mp4"
    all_writer = cv2.VideoWriter(str(all_writer_path), fourcc, fps, (composite_w, composite_h))

    camera_order = camera_order if (camera_order and len(camera_order) == n) else _infer_cam_names(video_paths)

    with tqdm(total=total_frames, desc="Rendering multi-cam") as pbar:
        for frame_idx in range(total_frames):
            raw_frames = []
            ok_all = True
            for cap in captures:
                ok, fr = cap.read()
                if not ok:
                    ok_all = False
                    break
                raw_frames.append(fr)
            if not ok_all:
                break

            composite = tile_frames(raw_frames, width, height, cols=layout_cols)

            # labels per pane
            if label_each_pane:
                font, scale, th = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                for i, name in enumerate(camera_order):
                    row = i // layout_cols
                    col = i % layout_cols
                    x0 = col * width + 10
                    y0 = row * height + 30
                    cv2.putText(composite, name, (x0, y0), font, scale, (255, 255, 255), th, cv2.LINE_AA)

            # global overlays cluster/bout + time-series, this should be done once per composite
            cluster_val = cluster_list[frame_idx]
            bout_val = bout_list[frame_idx]
            if cluster_val != -1:
                y_offset = 40
                cv2.putText(composite, f"Cluster: {cluster_val}   Bout: {bout_val}", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                y_offset += 36

                for col in text_cols:
                    val = time_series_data[col][frame_idx]
                    if val is not None and str(val).strip() != "":
                        cv2.putText(composite, f"{col[:-5]}: {val}", (10, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                        y_offset += 28

                for col in flag_cols:
                    positions = flag_positions[col]
                    prev = positions[positions < frame_idx]
                    next_ = positions[positions > frame_idx]
                    since_prev = (frame_idx - prev[-1]) / fps if len(prev) else None
                    until_next = (next_[0] - frame_idx) / fps if len(next_) else None

                    display = f"{col[:-5]}: "
                    if since_prev is not None:
                        display += f"{since_prev:.1f}s since last "
                    if until_next is not None:
                        display += f"{until_next:.1f}s until next"
                    cv2.putText(composite, display, (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
                    y_offset += 28

                # write
                cluster_writers[cluster_val].write(composite)
                all_writer.write(composite)

            pbar.update(1)

    # memory cleanup
    for cap in captures:
        cap.release()
    all_writer.release()
    for w in cluster_writers.values():
        w.release()

    print("Videos written to:")
    for cid in valid_clusters:
        print(f"  Cluster {cid}: {Path(out_dir) / f'cluster_{cid}.mp4'}")
    print(f"  All clusters: {all_writer_path}  ({composite_w}x{composite_h} @ {fps}fps)")

def render_cluster_videos(project: Project):
    if not project.video_data:
        print("No video data available for rendering.")
        return

    for video in project.video_data:
        clustering_output = video["clustering_output"]
        video_paths = video.get("video_paths") or []
        time_series_data = video["time_series_data"]
        camera_order = video.get("camera_order")

        if not video_paths or clustering_output is None:
            continue

        out_dir = project.output_path / "videos" / f"{video['video_name']}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # For camA|camB side-by-side: 2 columns.
        layout_cols = 2 if len(video_paths) == 2 else None

        render_videos(
            video_paths=video_paths,
            clustering_output=clustering_output,
            out_dir=str(out_dir),
            time_series_data=time_series_data,
            camera_order=camera_order,
            layout_cols=layout_cols,
            label_each_pane=True,
        )
