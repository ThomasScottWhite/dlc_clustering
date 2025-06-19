import polars as pl
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from collections import defaultdict
from typing import Protocol
from sklearn.cluster import HDBSCAN
import umap

class ClusteringStrategy(Protocol):
    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Process the input DataFrame and return a clustered DataFrame.
        
        Parameters:
        - df: Input DataFrame containing the data to be clustered.
        
        Returns:
        - A DataFrame with clustering results.
        """
        pass

class PCAKMeansBoutStrategy:
    def __init__(self, n_components: int = 2, n_clusters: int = 5, bout_length: int = 15, stride: int = 1):
        self.n_components = n_components
        self.n_clusters = n_clusters
        self.bout_length = bout_length
        self.stride = stride

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        # Pipeline for clustering bouts in a DataFrame using PCA and KMeans
        def get_filtered_mask(df: pl.DataFrame) -> np.ndarray:
            filter_cols = [col for col in df.columns if col.endswith("_filter")]
            if not filter_cols:
                return np.zeros(len(df), dtype=bool)
            return (
                df.select(pl.any_horizontal([pl.col(col) for col in filter_cols]))
                .to_series()
                .to_numpy()
            )

        def get_valid_bouts(X: np.ndarray, valid_indices: list[int], bout_length: int, stride: int):
            bouts, bout_to_frames = [], []
            for i in range(0, len(valid_indices) - bout_length + 1, stride):
                frame_indices = valid_indices[i:i + bout_length]
                bouts.append(X[frame_indices].flatten())
                bout_to_frames.append(frame_indices)
            return bouts, bout_to_frames

        def cluster_bouts(bouts: list[np.ndarray]):
            if not bouts:
                return [], []
            X_bouts = np.stack(bouts)
            bouts_pca = PCA(n_components=self.n_components).fit_transform(X_bouts)
            labels = KMeans(n_clusters=self.n_clusters, n_init="auto").fit(bouts_pca).labels_
            return labels, bouts

        def assign_clusters_to_frames(n_frames, filtered_mask, bout_to_frames, labels):
            cluster_map, bout_map = defaultdict(list), defaultdict(list)
            for bout_id, (frames, label) in enumerate(zip(bout_to_frames, labels)):
                for idx in frames:
                    cluster_map[idx].append(label)
                    bout_map[idx].append(bout_id)

            cluster_per_frame, bout_id_per_frame = [], []
            for i in range(n_frames):
                if filtered_mask[i] or i not in cluster_map:
                    cluster_per_frame.append(-1)
                    bout_id_per_frame.append(-1)
                else:
                    cluster_per_frame.append(cluster_map[i][-1])
                    bout_id_per_frame.append(bout_map[i][-1])
            return cluster_per_frame, bout_id_per_frame

        # === Pipeline ===
        filtered_mask = get_filtered_mask(df)

        usable_cols = [col for col in df.columns if not col.endswith("_filter")]
        X = np.nan_to_num(df.select(usable_cols).to_numpy())

        valid_indices = [i for i, valid in enumerate(filtered_mask) if not valid]
        bouts, bout_to_frames = get_valid_bouts(X, valid_indices, self.bout_length, self.stride)
        labels, _ = cluster_bouts(bouts)

        cluster_per_frame, bout_id_per_frame = assign_clusters_to_frames(
            n_frames=len(df),
            filtered_mask=filtered_mask,
            bout_to_frames=bout_to_frames,
            labels=labels
        )

        clustered_output = df.with_columns([
            pl.Series("cluster", cluster_per_frame),
            pl.Series("bout_id", bout_id_per_frame),
        ])
        return clustered_output


"""
ADDED BY HUGO
"""

class UmapHdbscanBoutStrategy:
  def __init__(self, n_components: int = 10, bout_length: int = 15, stride: int = 1, umap_args=None, hdbscan_args=None):
        self.n_components = n_components
        self.bout_length = bout_length
        self.stride = stride
        self.umap_args = umap_args or {}
        self.hdbscan_args = hdbscan_args or {}

  def process(self, df: pl.DataFrame) -> pl.DataFrame:
    # Pipeline for clustering bouts in a DataFrame using PCA and KMeans
    def get_filtered_mask(df: pl.DataFrame) -> np.ndarray:
        filter_cols = [col for col in df.columns if col.endswith("_filter")]
        if not filter_cols:
            return np.zeros(len(df), dtype=bool)
        return (
            df.select(pl.any_horizontal([pl.col(col) for col in filter_cols]))
            .to_series()
            .to_numpy()
        )

    def get_valid_bouts(X: np.ndarray, valid_indices: list[int], bout_length: int, stride: int):
        bouts, bout_to_frames = [], []
        for i in range(0, len(valid_indices) - bout_length + 1, stride):
            frame_indices = valid_indices[i:i + bout_length]
            bouts.append(X[frame_indices].flatten())
            bout_to_frames.append(frame_indices)
        return bouts, bout_to_frames

    def cluster_bouts(bouts: list[np.ndarray]):
        if not bouts:
            return [], []
        X_bouts = np.stack(bouts)
        
        dim_red = umap.UMAP(n_components=self.n_components, **self.umap_args)
        X_umap = dim_red.fit_transform(X_bouts)

        clust = HDBSCAN(**self.hdbscan_args)
        labels = clust.fit_predict(X_umap)

        return labels, bouts

    def assign_clusters_to_frames(n_frames, filtered_mask, bout_to_frames, labels):
        cluster_map, bout_map = defaultdict(list), defaultdict(list)
        for bout_id, (frames, label) in enumerate(zip(bout_to_frames, labels)):
            for idx in frames:
                cluster_map[idx].append(label)
                bout_map[idx].append(bout_id)

        cluster_per_frame, bout_id_per_frame = [], []
        for i in range(n_frames):
            if filtered_mask[i] or i not in cluster_map:
                cluster_per_frame.append(-1)
                bout_id_per_frame.append(-1)
            else:
                cluster_per_frame.append(cluster_map[i][-1])
                bout_id_per_frame.append(bout_map[i][-1])
        return cluster_per_frame, bout_id_per_frame

    # === Pipeline ===
    filtered_mask = get_filtered_mask(df)

    usable_cols = [col for col in df.columns if not col.endswith("_filter")]
    X = np.nan_to_num(df.select(usable_cols).to_numpy())

    valid_indices = [i for i, valid in enumerate(filtered_mask) if not valid]
    bouts, bout_to_frames = get_valid_bouts(X, valid_indices, self.bout_length, self.stride)
    labels, _ = cluster_bouts(bouts)

    cluster_per_frame, bout_id_per_frame = assign_clusters_to_frames(
        n_frames=len(df),
        filtered_mask=filtered_mask,
        bout_to_frames=bout_to_frames,
        labels=labels
    )

    clustered_output = df.with_columns([
        pl.Series("cluster", np.array(cluster_per_frame, dtype=np.int32)),
        pl.Series("bout_id", np.array(bout_id_per_frame, dtype=np.int32)),
    ])
    
    return clustered_output
  
class PCAHDBScanBoutStrategy:
  def __init__(self, n_components: int = 10, bout_length: int = 15, stride: int = 1, umap_args=None, hdbscan_args=None):
        self.n_components = n_components
        self.bout_length = bout_length
        self.stride = stride
        self.umap_args = umap_args or {}
        self.hdbscan_args = hdbscan_args or {}

  def process(self, df: pl.DataFrame) -> pl.DataFrame:
    # Pipeline for clustering bouts in a DataFrame using PCA and KMeans
    def get_filtered_mask(df: pl.DataFrame) -> np.ndarray:
        filter_cols = [col for col in df.columns if col.endswith("_filter")]
        if not filter_cols:
            return np.zeros(len(df), dtype=bool)
        return (
            df.select(pl.any_horizontal([pl.col(col) for col in filter_cols]))
            .to_series()
            .to_numpy()
        )

    def get_valid_bouts(X: np.ndarray, valid_indices: list[int], bout_length: int, stride: int):
        bouts, bout_to_frames = [], []
        for i in range(0, len(valid_indices) - bout_length + 1, stride):
            frame_indices = valid_indices[i:i + bout_length]
            bouts.append(X[frame_indices].flatten())
            bout_to_frames.append(frame_indices)
        return bouts, bout_to_frames

    def cluster_bouts(bouts: list[np.ndarray], n_components: int = 2, hdbscan_args: dict = {}):
        if not bouts:
            return [], []
        
        X_bouts = np.stack(bouts)

        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_bouts)

        clust = HDBSCAN(**hdbscan_args)
        labels = clust.fit_predict(X_pca)

        return labels, bouts
        
    def assign_clusters_to_frames(n_frames, filtered_mask, bout_to_frames, labels):
        cluster_map, bout_map = defaultdict(list), defaultdict(list)
        for bout_id, (frames, label) in enumerate(zip(bout_to_frames, labels)):
            for idx in frames:
                cluster_map[idx].append(label)
                bout_map[idx].append(bout_id)

        cluster_per_frame, bout_id_per_frame = [], []
        for i in range(n_frames):
            if filtered_mask[i] or i not in cluster_map:
                cluster_per_frame.append(-1)
                bout_id_per_frame.append(-1)
            else:
                cluster_per_frame.append(cluster_map[i][-1])
                bout_id_per_frame.append(bout_map[i][-1])
        return cluster_per_frame, bout_id_per_frame

    # === Pipeline ===
    filtered_mask = get_filtered_mask(df)

    usable_cols = [col for col in df.columns if not col.endswith("_filter")]
    X = np.nan_to_num(df.select(usable_cols).to_numpy())

    valid_indices = [i for i, valid in enumerate(filtered_mask) if not valid]
    bouts, bout_to_frames = get_valid_bouts(X, valid_indices, self.bout_length, self.stride)
    labels, _ = cluster_bouts(bouts)

    cluster_per_frame, bout_id_per_frame = assign_clusters_to_frames(
        n_frames=len(df),
        filtered_mask=filtered_mask,
        bout_to_frames=bout_to_frames,
        labels=labels
    )

    clustered_output = df.with_columns([
        pl.Series("cluster", cluster_per_frame),
        pl.Series("bout_id", bout_id_per_frame),
    ])
    return clustered_output