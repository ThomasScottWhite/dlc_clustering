from __future__ import annotations

import polars as pl
import numpy as np
from dataclasses import dataclass, field
from typing import Protocol, List, Optional, Callable, Sequence

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import umap 
from hdbscan import HDBSCAN  


NUMERIC_DTYPES = {pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64}

def select_numeric_feature_columns(df: pl.DataFrame, exclude: Sequence[str]) -> List[str]:
    """Deterministically select columns that are numeric and not in exclude."""
    return [
        c for c in df.columns
        if c not in set(exclude) and df.schema[c] in NUMERIC_DTYPES
    ]

def flatten_features(bouts: list[dict], feature_columns: Optional[List[str]] = None) -> np.ndarray:
    """
    Turn each bout's `features` DataFrame into a 1D vector. If feature_columns is
    provided, select those columns in order for determinism.
    """
    X_rows = []
    for b in bouts:
        f: pl.DataFrame = b["features"]
        if feature_columns is not None:
            f = f.select(feature_columns)
        X_rows.append(f.to_numpy().ravel())  # (bout_len * n_features,)
    return np.stack(X_rows, axis=0)

def get_bouts(df: pl.DataFrame, bout_length: int, stride: int = 1, supervised_labels: bool = False) -> list[dict]:
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
                    "features": bout.drop(["video_name", "row_idx"]),
                    "bout_id": bout_id,
                })
                bout_id += 1
                continue

            # supervised path
            if "label" in bout.columns and (bout["label"] == -1).all():
                continue

            label = None
            if "label" in bout.columns:
                non_negative = bout.filter(pl.col("label") != -1)["label"]
                if non_negative.len() > 0:
                    label = (
                        non_negative
                        .value_counts()
                        .sort("count", descending=True)
                        .select("label")
                        .to_series()[0]
                    )
            if label is not None:
                bouts.append({
                    "video_name": video_name,
                    "row_indices": row_indices,
                    "features": bout.drop(["video_name", "row_idx", "label"]),
                    "bout_id": bout_id,
                    "label": label,
                })
                bout_id += 1
    return bouts


# -----------------------------
# Core join-back helper
# -----------------------------
def apply_clustering(project, clustering_fn: Callable[[list[dict], List[str]], List[int]], bout_length=10, stride=5, supervised: bool = False) -> List[int]:
    all_data = []
    for video_data in project.video_data:
        if not video_data.get("processed_dlc_data"):
            continue
        combined_data = video_data.get("combined_data")
        if combined_data is None:
            continue

        if supervised:
            # If supervised path, try to bring in labels (if present)
            supervised_labels = video_data.get("supervised_labels")
            if supervised_labels is not None:
                combined_data = pl.concat([combined_data, supervised_labels], how="horizontal")

        combined_data = combined_data.with_row_index(name="row_idx")
        combined_data = combined_data.with_columns(pl.lit(video_data["video_name"]).alias("video_name"))
        all_data.append(combined_data)

    if not all_data:
        return []

    df = pl.concat(all_data, how="vertical")

    # drop *_filter rows and columns
    filter_cols = [c for c in df.columns if c.endswith("_filter")]
    if filter_cols:
        mask = df.select(pl.any_horizontal([pl.col(c) for c in filter_cols])).to_series()
        df = df.filter(~mask).drop(filter_cols)

    # Build bouts & deterministic numeric feature selection
    bouts = get_bouts(df, bout_length=bout_length, stride=stride, supervised_labels=supervised)
    if not bouts:
        return []

    # Determine numeric feature columns from first bout for deterministic stacking
    first_feat_df: pl.DataFrame = bouts[0]["features"]
    feature_cols = select_numeric_feature_columns(first_feat_df, exclude=[])

    labels = clustering_fn(bouts, feature_cols)

    # Attach labels to bouts for row->cluster mapping
    for b, lab in zip(bouts, labels):
        b["cluster"] = int(lab) if lab is not None else -1

    row_to_cluster = []
    for b in bouts:
        for row_idx in b["row_indices"]:
            row_to_cluster.append({
                "video_name": b["video_name"],
                "row_idx": row_idx,
                "cluster": b["cluster"],
                "bout_id": b["bout_id"],
            })
    cluster_df = pl.DataFrame(row_to_cluster).unique(subset=["video_name", "row_idx"], keep="first")

    # join back per video
    for video_data in project.video_data:
        video_name = video_data["video_name"]
        combined_data = video_data.get("combined_data")
        if combined_data is None:
            continue
        combined_data = combined_data.with_row_index(name="row_idx")
        clustered_video_data = cluster_df.filter(pl.col("video_name") == video_name)
        updated_data = combined_data.join(clustered_video_data, on=["row_idx"], how="left")
        video_data["clustering_output"] = updated_data

    return labels


class Preprocessor(Protocol):
    def fit(self, X: np.ndarray) -> "Preprocessor": ...
    def transform(self, X: np.ndarray) -> np.ndarray: ...

class Clusterer(Protocol):
    def fit_predict(self, X: np.ndarray) -> np.ndarray: ...

class Classifier(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> "Classifier": ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...


@dataclass
class SKStandardScaler(Preprocessor):
    scaler: StandardScaler = field(default_factory=StandardScaler)

    def fit(self, X: np.ndarray) -> "SKStandardScaler":
        self.scaler.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.transform(X)

@dataclass
class SKPCA(Preprocessor):
    n_components: int
    pca: Optional[PCA] = field(default=None, init=False)

    def fit(self, X: np.ndarray) -> "SKPCA":
        self.pca = PCA(n_components=self.n_components).fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.pca is not None
        return self.pca.transform(X)

@dataclass
class UMAPReducer(Preprocessor):
    n_components: int = 2
    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = "euclidean"
    umap_model: Optional[umap.UMAP] = field(default=None, init=False)

    def fit(self, X: np.ndarray) -> "UMAPReducer":
        self.umap_model = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
        ).fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.umap_model is not None
        return self.umap_model.transform(X)


@dataclass
class SKKMeans(Clusterer):
    n_clusters: int
    random_state: Optional[int] = None

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return KMeans(n_clusters=self.n_clusters, random_state=self.random_state).fit_predict(X)

@dataclass
class HDBSCANClusterer(Clusterer):
    min_cluster_size: int = 15
    min_samples: Optional[int] = None
    metric: str = "euclidean"
    cluster_selection_method: str = "eom"
    allow_single_cluster: bool = False

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        model = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            cluster_selection_method=self.cluster_selection_method,
            allow_single_cluster=self.allow_single_cluster,
        )
        labels = model.fit_predict(X)  # noise points labeled -1
        return labels


@dataclass
class SKRandomForest(Classifier):
    n_estimators: int = 200
    random_state: Optional[int] = 42
    model: RandomForestClassifier = field(init=False)

    def __post_init__(self):
        self.model = RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SKRandomForest":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

@dataclass
class SKGradientBoosting(Classifier):
    model: GradientBoostingClassifier = field(default_factory=lambda: GradientBoostingClassifier(random_state=42))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SKGradientBoosting":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

@dataclass
class SKSVM(Classifier):
    kernel: str = "rbf"
    C: float = 1.0
    model: SVC = field(init=False)

    def __post_init__(self):
        self.model = SVC(kernel=self.kernel, C=self.C)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SKSVM":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

@dataclass
class SKKNN(Classifier):
    n_neighbors: int = 5
    model: KNeighborsClassifier = field(init=False)

    def __post_init__(self):
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SKKNN":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

@dataclass
class SKMLP(Classifier):
    hidden_layer_sizes: tuple[int, ...] = (100,)
    max_iter: int = 400
    random_state: int = 42
    model: MLPClassifier = field(init=False)

    def __post_init__(self):
        self.model = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes, max_iter=self.max_iter, random_state=self.random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SKMLP":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


@dataclass
class UnsupervisedBoutPipeline:
    preprocessors: List[Preprocessor]
    clusterer: Clusterer
    bout_length: int = 15
    stride: int = 5

    def _fit_transform(self, X: np.ndarray) -> np.ndarray:
        for p in self.preprocessors:
            p.fit(X)
            X = p.transform(X)
        return X

    def process(self, project) -> List[int]:
        def _fn(bouts: list[dict], feature_cols: List[str]) -> List[int]:
            X = flatten_features(bouts, feature_cols)
            Xr = self._fit_transform(X)
            labels = self.clusterer.fit_predict(Xr)
            return labels.tolist()
        return apply_clustering(project, _fn, bout_length=self.bout_length, stride=self.stride, supervised=False)


def prepare_supervised_dataset(project, bout_length: int = 15, stride: int = 5):
    all_data = []
    for vd in project.video_data:
        if not vd.get("processed_dlc_data"):
            continue
        combined = vd.get("combined_data")
        if combined is None:
            continue
        labels = vd.get("supervised_labels")
        if labels is None:
            continue
        df = pl.concat([combined, labels], how="horizontal").with_row_index("row_idx")
        df = df.with_columns(pl.lit(vd["video_name"]).alias("video_name"))
        all_data.append(df)

    if not all_data:
        return np.empty((0,)), np.empty((0,))

    df = pl.concat(all_data, how="vertical")
    filter_cols = [c for c in df.columns if c.endswith("_filter")]
    if filter_cols:
        mask = df.select(pl.any_horizontal([pl.col(c) for c in filter_cols])).to_series()
        df = df.filter(~mask).drop(filter_cols)

    # NOTE: we keep rows with label == -1 in raw df but will be filtered out by get_bouts()
    bouts = get_bouts(df, bout_length=bout_length, stride=stride, supervised_labels=True)
    if not bouts:
        return np.empty((0,)), np.empty((0,))

    feat_cols = select_numeric_feature_columns(bouts[0]["features"], exclude=[])
    X = flatten_features(bouts, feat_cols)
    y = np.array([b["label"] for b in bouts])
    return X, y


@dataclass
class SupervisedBoutPipeline:
    preprocessors: List[Preprocessor]
    classifier: Classifier
    bout_length: int = 15
    stride: int = 5
    test_size: float = 0.2
    random_state: int = 42
    print_report: bool = True

    def _fit_transform_train(self, X: np.ndarray) -> np.ndarray:
        for p in self.preprocessors:
            p.fit(X)
            X = p.transform(X)
        return X

    def _transform_only(self, X: np.ndarray) -> np.ndarray:
        for p in self.preprocessors:
            X = p.transform(X)
        return X

    def train(self, project) -> None:
        X, y = prepare_supervised_dataset(project, self.bout_length, self.stride)
        if X.size == 0 or y.size == 0:
            print("No supervised bouts found.")
            return
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state, stratify=y)
        X_tr = self._fit_transform_train(X_tr)
        self.classifier.fit(X_tr, y_tr)
        X_te = self._transform_only(X_te)
        y_pred = self.classifier.predict(X_te)
        if self.print_report:
            print(classification_report(y_te, y_pred))

    def process(self, project) -> List[int]:
        self.train(project)

        def _fn(bouts: list[dict], feature_cols: List[str]) -> List[int]:
            X = flatten_features(bouts, feature_cols)
            X = self._transform_only(X)
            preds = self.classifier.predict(X)
            return preds.astype(int).tolist()
        return apply_clustering(project, _fn, bout_length=self.bout_length, stride=self.stride, supervised=False)
