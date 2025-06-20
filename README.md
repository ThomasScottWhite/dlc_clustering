# DLC Clustering

This repository is a work-in-progress rewrite of scripts for clustering, graphing, and rendering DeepLabCut videos.

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
example_project
    ├── dlc_data
    │   └── video1.h5
    └── videos(optional)
        └── video1.avi
```
- `.h5` and `.avi` filenames **must match exactly**.
- If the `videos/` folder is omitted, the pipeline will still function, but **video rendering outputs will be skipped** (when implemented).

## UMAP and HDBSCAN Parameters

### UMAP (Dimensionality Reduction)

| Parameter      | Description                                             | Higher Value                      | Lower Value                          | Typical Range |
|----------------|---------------------------------------------------------|-----------------------------------|---------------------------------------|----------------|
| `n_neighbors`  | Number of neighbors to preserve local/global structure  | Emphasizes global structure       | Emphasizes local detail               | 5–50          |
| `min_dist`     | Min distance between points in embedding                | Broad, spread clusters            | Tight, packed clusters                | 0.1–0.8       |
| `n_components` | Output dimensions                                       | Preserves more information        | More compressed representation        | 2–20          |
| `metric`       | Distance function (`'euclidean'`, etc.)                 | Controls how neighbors are chosen | Can expose nonlinear structure        | -             |

### t-SNE (Dimensionality Reduction)
| Parameter                   | Default       | Description                                                                                 |
|-----------------------------|---------------|---------------------------------------------------------------------------------------------|
| **n_components**            | 2             | Dimension of the embedded space.                                                            |
| **perplexity**              | 30.0          | The perplexity balances attention between local and global aspects; roughly the # of neighbors. |
| **early_exaggeration**      | 12.0          | Controls tightness of natural clusters in early optimization.                               |
| **learning_rate**           | `'warn'`      | Step size for optimization; a warning default selects an appropriate value automatically.   |
| **n_iter**                  | 1000          | Maximum number of iterations for optimization.                                              |
| **n_iter_without_progress** | 300           | Maximum iterations without progress before aborting.                                        |
| **min_grad_norm**           | 1e-07         | If the gradient norm falls below this threshold, optimization is stopped early.             |
| **metric**                  | `'euclidean'` | The metric to use when calculating distance between instances.                              |
| **metric_params**           | `None`        | Additional keyword arguments for the metric function.                                       |
| **init**                    | `'warn'`      | Initialization of embedding: `'random'`, `'pca'`, or a user-provided array (warns by default). |
| **verbose**                 | 0             | Verbosity level: 0 (silent) or higher for more messages.                                    |
| **random_state**            | `None`        | Seed or `RandomState` for reproducible results.                                             |
| **method**                  | `'barnes_hut'`| Algorithm to use: `'barnes_hut'` (approximate, faster) or `'exact'` (slower, precise).      |
| **angle**                   | 0.5           | Trade‐off parameter for Barnes-Hut: lower values give more accurate results at increased cost. |
| **n_jobs**                  | `None`        | The number of parallel jobs to run for neighbors search (if applicable).                    |
| **square_distances**        | `'deprecated'`| Deprecated parameter: distances are squared by default under the hood.                       |

### HDBSCAN (Clustering)

| Parameter                   | Description                                       | Higher Value                       | Lower Value                           | Typical Range   |
|----------------------------|---------------------------------------------------|------------------------------------|----------------------------------------|-----------------|
| `min_cluster_size`         | Min points to form a cluster                      | Fewer, larger clusters             | More, smaller clusters                 | 20–150          |
| `min_samples`              | Min density in core neighborhood                  | Tighter clusters                   | Looser clusters, more outliers         | 5–100           |
| `cluster_selection_method` | `'eom'` (default) or `'leaf'`                     | Coarser, broader clusters          | More granular splits                   | `'eom'`, `'leaf'`|
| `cluster_selection_epsilon`| Distance threshold for merging nearby clusters    | Merges clusters                    | Keeps them separate                    | 0.0–0.5         |


### Recommended Starting Point for Behavioral Clustering

```python
umap_args = {
    "n_neighbors": 10,
    "min_dist": 0.3,
    "n_components": 3,
    "metric": "euclidean"
}

hdbscan_args = {
    "min_cluster_size": 75,
    "min_samples": 50,
    "cluster_selection_method": "eom"
}

