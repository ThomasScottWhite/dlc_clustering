# DLC Clustering

This repository is a work-in-progress rewrite of scripts for clustering, graphing, and rendering DeepLabCut videos.

An example project is provided in [`example.ipynb`](example.ipynb).

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
example_project
    ├── dlc_data
    │   └── video1.h5
    └── videos(optional)
        └── video1.avi
```
- `.h5` and `.avi` filenames **must match exactly**.
- If the `videos/` folder is omitted, the pipeline will still function, but **video rendering outputs will be skipped** (when implemented).

