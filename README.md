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
An example script for generating text and flags for videos can be found in ```example_generate_flag.py```


## Additional Documentation

For more information on data processing and clustering strategies, refer to the following documentation files:

- [Data Processing Strategies](data_processing.md): Learn how pose estimation data is normalized, filtered, or converted into features like velocity or angles.
- [Clustering Strategies](clustering.md): Explore available methods for grouping behavior patterns using techniques such as PCA + KMeans or UMAP + HDBSCAN.

