# DLC Clustering - Clustering Strategies

This document describes the available clustering strategies for behavior analysis using DeepLabCut data. These strategies work on segmented sequences of frames and apply dimensionality reduction and clustering algorithms to group similar behaviors.

Each strategy transforms the data and assigns a `cluster` label and a `bout_id` to each frame in the dataset.

## What is a Bout?

A **bout** is a short, continuous sequence of frames (e.g., 15 frames) used as the unit of clustering. Instead of clustering single frames, we extract temporal patterns over bouts to better capture motion and behavior.

---

## Shared Parameters

- **`bout_length`**: Number of consecutive frames per bout (default: 15).
- **`stride`**: Step size to move the sliding window when extracting bouts (default: 1).

Frames excluded by a `_filter` column are ignored during clustering and receive a `cluster` of `-1`.


## Available Strategies

### `PCAKMeansBoutStrategy`

Clusters frame bouts using **Principal Component Analysis (PCA)** followed by **K-Means**.
This a fast, but very simple clustering algorithm.

**Key Parameters**:
- `n_components`: Number of PCA components (default: 2).
- `n_clusters`: Number of K-Means clusters (default: 5).

---

### `UmapHdbscanBoutStrategy`

Clusters frame bouts using **UMAP** (nonlinear dimensionality reduction) followed by **HDBSCAN** (density-based clustering).


**Key Parameters**:
- `n_components`: Number of UMAP components (default: 10).
- `umap_args`: Optional UMAP settings (e.g., `n_neighbors`, `min_dist`).
- `hdbscan_args`: Optional HDBSCAN settings (e.g., `min_cluster_size`).

---

### `PCAHDBScanBoutStrategy`

Clusters frame bouts using **PCA** for dimensionality reduction, then **HDBSCAN**.


**Key Parameters**:
- `n_components`: Number of PCA components (default: 10).
- `hdbscan_args`: Optional HDBSCAN parameters.

---

## Common Output Columns

- `cluster`: Cluster assignment per frame.  
  `-1` means excluded or unassigned (e.g., filtered out).
- `bout_id`: Integer ID for which bout the frame belongs to.

---

## Notes

- **Filtering**: Any column ending in `_filter` will be used to exclude frames.
- **Missing Values**: All missing or NaNs are converted to zero before processing.
- **Dimensionality Reduction** is crucial for clustering performance—choose UMAP for non-linear reduction, PCA for speed.
- **Clustering Outputs** You can access clustering outputs though the `Project` function `project.get_cluster_output()` or save the cluster the outputs to csvs with `project.save_clustering_output()`
---

## Example Usage

```python
strategy = PCAKMeansBoutStrategy(n_components=2, n_clusters=5, bout_length=15, stride=1)
clustered_df = strategy.process(df)
