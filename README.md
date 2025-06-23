# DLC Clustering

This project does clustering, graphing and video rendering of deeplabcut outputs

An example project is provided in [`example.py`](example.ipynb).

## Installation Instructions

Activate your desired **conda environment**, then run:

```bash
git clone https://github.com/ThomasScottWhite/dlc_clustering.git
cd dlc_clustering
pip install -e .
```
Afterwords you should be able to import `dlc_clustering` and create your own projects

## Project Layout

Place your DLC data in a project directory with the following structure:
```
example_project/
├── dlc_data
│   └── video1.h5
├── time_series(optional)
│   └── video1.csv
└── videos(optional)
    └── video1.avi
```
- `.h5`, `.avi` and `.csv` filenames **must match exactly**.
- If the `videos/` folder is omitted, the pipeline will still function, but **video rendering outputs will be skipped**.

## Time Series Data

Time series data allows you to overlay contextual information on videos (and graphs when implemented).  
This is useful for marking when specific events occur, such as foot shocks, tones, or other stimuli.

### Format Requirements

- Each `.csv` file must have **the same number of rows as frames** in the corresponding video.
- Columns in the CSV can follow specific suffix conventions to control behavior:

| Suffix       | Behavior                                                                 |
|--------------|--------------------------------------------------------------------------|
| `_text`      | Displays the text from that row over the video frame.                    |
| `_flag`      | Displays time **since the last** and **until the next** `True` value.    |

**Example:**  
A column named `shock_flag` will show countdowns or time since a shock occurred.  
A column named `note_text` will overlay text messages on the video.

### Example Script

The following Python script generates a time series file with:

- A column displaying `"This is a test"` on every frame.
- A flag that is active (`True`) at frames 300 and 600.

```python
from pathlib import Path
import cv2
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
    df.loc[300, "flag_example_flag"] = True
    df.loc[600, "flag_example_flag"] = True

    df.to_csv(time_series_paths / f"{video_path.stem}.csv", index=False)