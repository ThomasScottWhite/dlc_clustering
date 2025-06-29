import polars as pl
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from collections import defaultdict
from typing import Protocol
from sklearn.cluster import HDBSCAN
import umap
from dataclasses import dataclass
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def get_bouts(df: pl.DataFrame, bout_length: int, stride: int = 1) -> list[dict]:
    bouts = []
    for video_name in df["video_name"].unique().to_list():
        video_df = df.filter(pl.col("video_name") == video_name)
        bout_id = 1
        for i in range(0, len(video_df) - bout_length + 1, stride):
            bout = video_df.slice(i, bout_length)
            row_indices = bout["row_idx"].to_list()
            bouts.append({
                "video_name": video_name,
                "row_indices": row_indices,
                "features": bout.drop(["video_name", "row_idx"]),  # drop metadata
                "bout_id": bout_id,
            })
            bout_id += 1
    return bouts


def apply_clustering(project, clustering_strategy, bout_length=10, stride=5):
    all_data = []
    for video_data in project.video_data:
        if not video_data["processed_dlc_data"]:
            continue

        combined_data = video_data.get("combined_data")
        if combined_data is None:
            continue

        combined_data = combined_data.with_row_index(name="row_idx")  # <- moved here

        combined_data = combined_data.with_columns([
            pl.lit(video_data["video_name"]).alias("video_name")
        ])
        all_data.append(combined_data)

    df = pl.concat(all_data, how="vertical")


    # Filter out invalid rows
    filter_cols = [col for col in df.columns if col.endswith("_filter")]
    if filter_cols:
        mask = df.select(pl.any_horizontal([pl.col(col) for col in filter_cols])).to_series()
        df = df.filter(~mask)

    bouts = get_bouts(df, bout_length=15, stride=5)
    cluster_labels = clustering_strategy(bouts)

    for bout, label in zip(bouts, cluster_labels):
        bout["cluster"] = label


    # This will hold tuples like (video_name, row_idx) → cluster
    row_to_cluster = []

    for bout in bouts:
        for row_idx in bout["row_indices"]:
            row_to_cluster.append({
                "video_name": bout["video_name"],
                "row_idx": row_idx,
                "cluster": bout["cluster"],
                "bout_id": bout["bout_id"], 
            })

    cluster_df = pl.DataFrame(row_to_cluster)
    cluster_df = cluster_df.unique(subset=["video_name", "row_idx"], keep="first")
    
    for video_data in project.video_data:
        video_name = video_data["video_name"]
        combined_data = video_data.get("combined_data")
        if combined_data is None:
            continue

        # Restore row indices
        combined_data = combined_data.with_row_index(name="row_idx")

        # Join with cluster assignments

        clustered_video_data = cluster_df.filter(pl.col("video_name") == video_name)
        updated_data = combined_data.join(
            clustered_video_data,
            on=["row_idx"],
            how="left"
        )

        video_data["clustering_output"] = updated_data


class ClusteringStrategy(Protocol):
    n_components: int
    n_clusters: int
    bout_length: int
    stride: int

    def process(self, bouts: list[dict], project) -> list[int]:
        """Process bouts and return cluster labels."""
        pass

@dataclass
class PCAKMeansBoutStrategy(ClusteringStrategy):
    n_components: int
    n_clusters: int
    bout_length: int
    stride: int

    def process(self, project) -> list[int]:
        """Process bouts and return cluster labels."""

        def clustering_strategy(bouts: list[dict]) -> list[int]:
            X = np.array([b["features"].to_numpy().flatten() for b in bouts])
            X = PCA(n_components=self.n_components).fit_transform(X)  # Optional dimensionality reduction
            cluster_labels = KMeans(n_clusters=self.n_clusters).fit_predict(X)
            return cluster_labels
        
        apply_clustering(project, clustering_strategy, bout_length=self.bout_length, stride=self.stride)


@dataclass
class HDBSCANBoutStrategy(ClusteringStrategy):
    n_components: int
    n_clusters: int
    bout_length: int
    stride: int

    def process(self, project) -> list[int]:
        """Process bouts and return cluster labels."""

        def clustering_strategy(bouts: list[dict]) -> list[int]:
            X = np.array([b["features"].to_numpy().flatten() for b in bouts])
            X = PCA(n_components=self.n_components).fit_transform(X)
            cluster_labels = HDBSCAN(min_cluster_size=self.n_clusters).fit_predict(X)
            return cluster_labels
        
        apply_clustering(project, clustering_strategy, bout_length=self.bout_length, stride=self.stride)

@dataclass
class UMAPKMeansBoutStrategy(ClusteringStrategy):
    n_components: int
    n_clusters: int
    bout_length: int
    stride: int

    def process(self, project) -> list[int]:
        """Process bouts and return cluster labels."""

        def clustering_strategy(bouts: list[dict]) -> list[int]:
            X = np.array([b["features"].to_numpy().flatten() for b in bouts])
            X = umap.UMAP(n_components=self.n_components).fit_transform(X)
            cluster_labels = KMeans(n_clusters=self.n_clusters).fit_predict(X)
            return cluster_labels
        
        apply_clustering(project, clustering_strategy, bout_length=self.bout_length, stride=self.stride)