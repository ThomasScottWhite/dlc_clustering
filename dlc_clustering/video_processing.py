import cv2
import polars as pl
from pathlib import Path
from dlc_clustering.projects import Project
from tqdm import tqdm

def overlay_cluster_text(video_path: str, clustering_output: pl.DataFrame, output_dir: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Extract values
    cluster_list = clustering_output["cluster"].to_list()
    bout_list = clustering_output["bout_id"].to_list()
    total_frames = len(cluster_list)

    # Initialize writers for each cluster
    valid_clusters = sorted(set(cluster_list) - {-1})
    cluster_writers = {}
    output_paths = {}

    for cluster_id in valid_clusters:
        out_path = Path(output_dir) / f"cluster_{cluster_id}.mp4"
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        cluster_writers[cluster_id] = writer
        output_paths[cluster_id] = out_path

    all_clusters_path = Path(output_dir) / "all_clusters.mp4"
    all_writer = cv2.VideoWriter(str(all_clusters_path), fourcc, fps, (width, height))

    frame_idx = 0
    with tqdm(total=total_frames, desc="Processing video frames") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_idx >= total_frames:
                break

            cluster_val = cluster_list[frame_idx]
            bout_val = bout_list[frame_idx]

            if cluster_val != -1:
                # Add overlay
                text = f"Cluster: {cluster_val}, Bout: {bout_val}"
                cv2.putText(
                    frame, text, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA
                )

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
        output_path = project.output_path / "videos" / f"{video['video_name']}" / f"{video['video_name']}.avi"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # overlay_cluster_text(video_path, clustering_output, output_path)
        if video_path and clustering_output is not None:
            overlay_cluster_text(video_path, clustering_output, output_path)

def render_all(project: Project):
    render_cluster_videos(project)