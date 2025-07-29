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


# Unsupervised Clustering Strategies

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


# Supervised Clustering Strategies

These strategies use labeled data to train a supervised classifier on frame bouts. Once trained, the classifier is used to predict the cluster label for each bout based on its features. These approaches may yield higher quality clustering if labeled data is available.

### Shared Parameters

- **`bout_length`**: Number of consecutive frames per bout (default: 15).
- **`stride`**: Step size to move the sliding window when extracting bouts (default: 5).
- **`prepare_supervised_dataset(project)`**: Prepares a labeled dataset from all project videos for training.

---

### `RandomForestBoutClassifier`

Uses a **Random Forest** classifier to assign cluster labels to frame bouts.

**Key Parameters**:
- `n_estimators`: Number of trees (default: 100).
- `random_state`: Seed for reproducibility.

---

### `GradientBoostingBoutClassifier`

Uses a **Gradient Boosting Classifier** for labeling bouts. Often more accurate than Random Forest for complex patterns.

**Key Parameters**:
- `random_state`: Seed for reproducibility.

---

### `KNearestBoutClassifier`

Uses **K-Nearest Neighbors (KNN)** to classify bouts based on distance in feature space.

**Key Parameters**:
- `n_neighbors`: Number of neighbors to consider (default: 5).

---

### `SVMBoutClassifier`

Uses a **Support Vector Machine (SVM)** to classify bout features with a specified kernel.

**Key Parameters**:
- `kernel`: Kernel type (`"linear"`, `"rbf"`, etc.) (default: `"rbf"`).
- `C`: Regularization parameter (default: `1.0`).

---

### `MLPBoutClassifier`

Uses a **Multi-Layer Perceptron (MLP)** neural network for clustering.

**Key Parameters**:
- `hidden_layer_sizes`: Shape of hidden layers (default: `(100,)`).
- `max_iter`: Maximum training iterations (default: 300).

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
