# The following Python script generates a time series file with:

# - A column displaying `"This is a test"` on every frame.
# - A flag that is active (`True`) at frames 300 and 600.

from pathlib import Path
import cv2
import polars as pl
import pandas as pd
example_project = Path("/home/thomas/Documents/dlc_clustering/data/example_project")
video_paths = example_project / "videos"
time_series_paths = example_project / "time_series"

for video_path in video_paths.glob("*.avi"):
    video = cv2.VideoCapture(str(video_path))
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    print(f"Video: {video_path.name}, FPS: {fps}, Frame Count: {frame_count}, Duration: {duration:.2f} seconds")

    df = pd.DataFrame({
        "text_example_text": ["This is a test"] * frame_count,
        "flag_example_flag": [False] * frame_count,
    })
    df.loc[6000, "flag_example_flag"] = True
    df.loc[12000, "flag_example_flag"] = True

    df.to_csv(time_series_paths / f"{video_path.stem}.csv", index=False)