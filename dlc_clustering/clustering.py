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

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


def get_bouts(df: pl.DataFrame, bout_length: int, stride: int = 1, supervised_labels=False) -> list[dict]:
    bouts = []
    for video_name in df["video_name"].unique().to_list():
        video_df = df.filter(pl.col("video_name") == video_name)
        bout_id = 1
        for i in range(0, len(video_df) - bout_length + 1, stride):
            bout = video_df.slice(i, bout_length)
            row_indices = bout["row_idx"].to_list()

            if not supervised_labels:
                bouts.append({
                    "video_name": video_name,
                    "row_indices": row_indices,
                    "features": bout.drop(["video_name", "row_idx"]),  # drop metadata
                    "bout_id": bout_id,
                })
                bout_id += 1
                continue
            
            if supervised_labels:
                # If supervised labels are present, filter out bouts with -1 label
                if "label" in bout.columns and (bout["label"] == -1).all():
                    continue

                # If label is present, get the most common non -1 label
                label = None
                if "label" in bout.columns:
                    non_negative_labels = bout.filter(pl.col("label") != -1)["label"]
                    if non_negative_labels.len() > 0:
                        label = (
                            non_negative_labels
                            .value_counts()
                            .sort("count", descending=True)
                            .select("label")
                            .to_series()[0]  # Get the most common label
                        )

                if label is not None:
                    bouts.append({
                        "video_name": video_name,
                        "row_indices": row_indices,
                        "features": bout.drop(["video_name", "row_idx", "label"]),  # drop metadata
                        "bout_id": bout_id,
                        "label": label,
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

        combined_data = combined_data.with_row_index(name="row_idx")

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
        df = df.drop(filter_cols)

    bouts = get_bouts(df, bout_length=15, stride=5)
    cluster_labels = clustering_strategy(bouts)

    for bout, label in zip(bouts, cluster_labels):
        bout["cluster"] = label


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
            X = PCA(n_components=self.n_components).fit_transform(X) 
            cluster_labels = KMeans(n_clusters=self.n_clusters).fit_predict(X)
            return cluster_labels
        
        apply_clustering(project, clustering_strategy, bout_length=self.bout_length, stride=self.stride)


@dataclass
class HDBSCANBoutStrategy(ClusteringStrategy):
    n_components: int = 2
    n_clusters: int = 5 
    bout_length: int = 15
    stride: int = 1

    # HDBSCAN parameters
    min_samples: int | None = None
    cluster_selection_epsilon: float = 0.0
    max_cluster_size: int | None = None
    metric: str = "euclidean"
    metric_params: dict | None = None
    alpha: float = 1.0
    algorithm: str = "auto"
    leaf_size: int = 40
    n_jobs: int | None = None
    cluster_selection_method: str = "eom"
    allow_single_cluster: bool = False
    store_centers: bool | None = None
    copy: bool = False

    def process(self, project) -> list[int]:
        """Process bouts and return cluster labels using HDBSCAN."""

        def clustering_strategy(bouts: list[dict]) -> list[int]:
            X = np.array([b["features"].to_numpy().flatten() for b in bouts])
            X = PCA(n_components=self.n_components).fit_transform(X)
            clusterer = HDBSCAN(
                min_cluster_size=self.n_clusters,
                min_samples=self.min_samples,
                cluster_selection_epsilon=self.cluster_selection_epsilon,
                max_cluster_size=self.max_cluster_size,
                metric=self.metric,
                metric_params=self.metric_params,
                alpha=self.alpha,
                algorithm=self.algorithm,
                leaf_size=self.leaf_size,
                n_jobs=self.n_jobs,
                cluster_selection_method=self.cluster_selection_method,
                allow_single_cluster=self.allow_single_cluster,
                store_centers=self.store_centers,
                copy=self.copy
            )
            cluster_labels = clusterer.fit_predict(X)
            return cluster_labels

        return apply_clustering(project, clustering_strategy, bout_length=self.bout_length, stride=self.stride)
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

# Supervised Clustering

def prepare_supervised_dataset(project):
    all_data = []
    for video_data in project.video_data:
        if not video_data["processed_dlc_data"]:
            continue

        combined_data = video_data.get("combined_data")
        if combined_data is None:
            continue

        supervised_labels = video_data.get("supervised_labels")
        if supervised_labels is None:
            continue

        combined_data = pl.concat([
            combined_data,
            supervised_labels
        ], how="horizontal")

        combined_data = combined_data.with_row_index(name="row_idx") 


        combined_data = combined_data.with_columns([
            pl.lit(video_data["video_name"]).alias("video_name"),
        ])
        all_data.append(combined_data)

    df = pl.concat(all_data, how="vertical")


    # Filter out invalid rows
    filter_cols = [col for col in df.columns if col.endswith("_filter")]
    if filter_cols:
        mask = df.select(pl.any_horizontal([pl.col(col) for col in filter_cols])).to_series()
        df = df.filter(~mask)
        # Remove filter columns
        df = df.drop(filter_cols)

    # Filter out rows where label is -1
    # if "label" in df.columns:
    #     df = df.filter(pl.col("label") != -1)

    supervised_bouts = get_bouts(df, bout_length=15, stride=5, supervised_labels=True)
    X = np.array([
        bout["features"].to_numpy().flatten()
        for bout in supervised_bouts
    ])
    y = np.array([bout.get("label") for bout in supervised_bouts])

    return X, y
class RandomForestBoutClassifier:
    bout_length = 15
    stride = 5
    n_estimators = 100
    random_state = None

    def __init__(self):
        from sklearn.ensemble import RandomForestClassifier as SklearnRF
        self.model = SklearnRF(n_estimators=self.n_estimators, random_state=self.random_state)

    def train(self, project):


        X, y = prepare_supervised_dataset(project)
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred))

    def process(self, project) -> list[int]:
        """Process bouts and return cluster labels."""

        self.train(project)

        def clustering_strategy(bouts: list[dict]) -> list[int]:
            X = np.array([b["features"].to_numpy().flatten() for b in bouts])
            cluster_labels = self.model.predict(X)
            return cluster_labels
        
        apply_clustering(project, clustering_strategy, bout_length=self.bout_length, stride=self.stride)


class GradientBoostingBoutClassifier:
    bout_length = 15
    stride = 5

    def __init__(self):
        self.model = GradientBoostingClassifier(random_state=42)

    def train(self, project):
        X, y = prepare_supervised_dataset(project)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred))

    def process(self, project) -> list[int]:
        self.train(project)
        def clustering_strategy(bouts: list[dict]) -> list[int]:
            X = np.array([b["features"].to_numpy().flatten() for b in bouts])
            return self.model.predict(X)
        apply_clustering(project, clustering_strategy, bout_length=self.bout_length, stride=self.stride)


class KNearestBoutClassifier:
    bout_length = 15
    stride = 5

    def __init__(self, n_neighbors=5):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def train(self, project):
        X, y = prepare_supervised_dataset(project)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred))

    def process(self, project) -> list[int]:
        self.train(project)
        def clustering_strategy(bouts: list[dict]) -> list[int]:
            X = np.array([b["features"].to_numpy().flatten() for b in bouts])
            return self.model.predict(X)
        apply_clustering(project, clustering_strategy, bout_length=self.bout_length, stride=self.stride)


class SVMBoutClassifier:
    bout_length = 15
    stride = 5

    def __init__(self, kernel="rbf", C=1.0):
        self.model = SVC(kernel=kernel, C=C)

    def train(self, project):
        X, y = prepare_supervised_dataset(project)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred))

    def process(self, project) -> list[int]:
        self.train(project)
        def clustering_strategy(bouts: list[dict]) -> list[int]:
            X = np.array([b["features"].to_numpy().flatten() for b in bouts])
            return self.model.predict(X)
        apply_clustering(project, clustering_strategy, bout_length=self.bout_length, stride=self.stride)



class MLPBoutClassifier:
    bout_length = 15
    stride = 5

    def __init__(self, hidden_layer_sizes=(100,), max_iter=300):
        self.model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=42)

    def train(self, project):
        X, y = prepare_supervised_dataset(project)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred))

    def process(self, project) -> list[int]:
        self.train(project)
        def clustering_strategy(bouts: list[dict]) -> list[int]:
            X = np.array([b["features"].to_numpy().flatten() for b in bouts])
            return self.model.predict(X)
        apply_clustering(project, clustering_strategy, bout_length=self.bout_length, stride=self.stride)

