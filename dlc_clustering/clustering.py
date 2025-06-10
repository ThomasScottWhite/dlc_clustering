import polars as pl
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from collections import defaultdict

class PCAKMeansBoutStrategy:
    def __init__(self, n_components: int = 2, n_clusters: int = 5, bout_length: int = 15, stride: int = 1):
        self.n_components = n_components
        self.n_clusters = n_clusters
        self.bout_length = bout_length
        self.stride = stride

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        X = df.to_numpy()
        X = np.nan_to_num(X)

        bouts = []
        bout_to_frames = []  

        for start in range(0, len(X) - self.bout_length + 1, self.stride):
            end = start + self.bout_length
            bout = X[start:end].flatten()
            bouts.append(bout)
            bout_to_frames.append(list(range(start, end)))

        bouts = np.stack(bouts)

        # PCA
        pca = PCA(n_components=self.n_components)
        bouts_pca = pca.fit_transform(bouts)

        # KMeans
        clusterer = KMeans(n_clusters=self.n_clusters, n_init="auto").fit(bouts_pca)
        labels = clusterer.labels_

        # Frame-to-bout and cluster mapping
        frame_cluster_map = defaultdict(list)
        frame_bout_map = defaultdict(list)

        for bout_id, (frames, cluster_label) in enumerate(zip(bout_to_frames, labels)):
            for frame_idx in frames:
                frame_cluster_map[frame_idx].append(cluster_label)
                frame_bout_map[frame_idx].append(bout_id)

        # Resolve each frame to last cluster and bout ID
        n_frames = len(df)
        cluster_per_frame = []
        bout_id_per_frame = []

        for i in range(n_frames):
            if i in frame_cluster_map:
                cluster_per_frame.append(frame_cluster_map[i][-1])  
                bout_id_per_frame.append(frame_bout_map[i][-1])
            else:
                cluster_per_frame.append(-1)
                bout_id_per_frame.append(-1)

        clustered_output = df.with_columns([
            pl.Series(name="cluster", values=cluster_per_frame),
            pl.Series(name="bout_id", values=bout_id_per_frame)
        ])

        return clustered_output
