import cv2
import polars as pl
from pathlib import Path
from dlc_clustering.projects import Project
from tqdm import tqdm

def overlay_cluster_text(video_path: str, clustering_output: pl.DataFrame, output_dir: str, time_series_data: pl.DataFrame):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    cluster_list = clustering_output["cluster"].to_list()
    bout_list = clustering_output["bout_id"].to_list()
    total_frames = len(cluster_list)

    # Verify alignment
    if time_series_data is not None:
        if time_series_data.shape[0] != total_frames:
            raise ValueError("Time series data does not match video frame count.")
        text_cols = [col for col in time_series_data.columns if col.endswith("_text")]
        flag_cols = [col for col in time_series_data.columns if col.endswith("_flag")]
    else:
        text_cols = []
        flag_cols = []
        # If no time series data, we won't overlay text or flags

    cluster_writers = {}
    output_paths = {}

    valid_clusters = sorted(set(cluster_list) - {-1})
    for cluster_id in valid_clusters:
        out_path = Path(output_dir) / f"cluster_{cluster_id}.mp4"
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        cluster_writers[cluster_id] = writer
        output_paths[cluster_id] = out_path

    all_clusters_path = Path(output_dir) / "all_clusters.mp4"
    all_writer = cv2.VideoWriter(str(all_clusters_path), fourcc, fps, (width, height))

    # Precompute flag timestamps for each flag column
    flag_positions = {
        col: time_series_data.select(pl.col(col)).to_series().to_numpy().nonzero()[0]
        for col in flag_cols
    }

    frame_idx = 0
    with tqdm(total=total_frames, desc="Processing video frames") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_idx >= total_frames:
                break

            cluster_val = cluster_list[frame_idx]
            bout_val = bout_list[frame_idx]

            if cluster_val != -1:
                y_offset = 50

                # Overlay cluster info
                text = f"Cluster: {cluster_val}, Bout: {bout_val}"
                cv2.putText(frame, text, (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                y_offset += 40

                # Draw _text values
                for col in text_cols:
                    val = time_series_data[col][frame_idx]
                    if val is not None and str(val).strip() != "":
                        cv2.putText(frame, f"{col[:-5]}: {val}", (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        y_offset += 30

                # Draw _flag timing deltas
                for col in flag_cols:
                    positions = flag_positions[col]
                    prev = positions[positions < frame_idx]
                    next_ = positions[positions > frame_idx]

                    seconds_since_prev = (frame_idx - prev[-1]) / fps if len(prev) else None
                    seconds_until_next = (next_[0] - frame_idx) / fps if len(next_) else None

                    display = f"{col[:-5]}: "
                    if seconds_since_prev is not None:
                        display += f"{seconds_since_prev:.1f}s since last "
                    if seconds_until_next is not None:
                        display += f"{seconds_until_next:.1f}s until next"
                    cv2.putText(frame, display, (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                    y_offset += 30

                cluster_writers[cluster_val].write(frame)
                all_writer.write(frame)

            frame_idx += 1
            pbar.update(1)

    cap.release()
    all_writer.release()
    for writer in cluster_writers.values():
        writer.release()

    print("Videos written to:")
    for cluster_id, path in output_paths.items():
        print(f"  Cluster {cluster_id}: {path}")
    print(f"  All clusters: {all_clusters_path}")


def render_cluster_videos(project: Project):
    """
    Renders videos with cluster information overlaid on each frame.
    
    Args:
        project: Project object containing video data and clustering output.
    """
    if not project.video_data:
        print("No video data available for rendering.")
        return
    
    for video in project.video_data:
        clustering_output = video["clustering_output"]
        video_path = video["video_path"]
        time_series_data = video["time_series_data"]
        output_path = project.output_path / "videos" / f"{video['video_name']}" / f"{video['video_name']}.avi"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if video_path and clustering_output is not None:
            overlay_cluster_text(video_path, clustering_output, output_path, time_series_data)

def render_all(project: Project):
    render_cluster_videos(project)