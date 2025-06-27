from dlc_clustering.data_types import ProjectType, VideoData2D
from dlc_clustering.data_processing import KeepOriginalStrategy
from dlc_clustering.data_processing import DataProcessingStrategy
from dlc_clustering.clustering import ClusteringStrategy   

from polars import Float64, Float32
from pathlib import Path
from typing import List, Optional, Type
import polars as pl
import glob
import warnings
from dlc_clustering.clustering import PCAKMeansBoutStrategy
import pandas as pd
import numpy as np

def read_hdf(csv_path: str) -> pl.DataFrame:
    """
    Reads an HDF5 file and processes the DataFrame.
    
    Returns:
        pd.DataFrame: Processed DataFrame with multi-level columns flattened.
    """
    df = pd.read_hdf(csv_path)
    df.columns = df.columns.droplevel(0)
    df.columns = ['_'.join(col).strip() for col in df.columns.values]
    df = pl.from_pandas(df)
    return df


def convert_str_to_paths(video_paths: List[str]) -> List[Path]:
    """
    Convert a list of string paths to Path objects.
    """
    return [Path(path) for path in video_paths]


def get_paths(directory: str, pattern: str, required: bool = True) -> List[Path]:
    paths = [Path(p) for p in glob.glob(str(directory / pattern))]
    if not paths:
        if required:
            raise ValueError(f"No files matching '{pattern}' found in {directory}")
        else:
            warnings.warn(f"No files found in {directory}. Ignore if intentional.")
    return paths


def load_supervised_labels(csv_dir: Path) -> pd.DataFrame:
    csv_files = glob.glob(str(csv_dir / "*.csv"))
    all_dfs = [pd.read_csv(p) for p in csv_files]
    return pd.concat(all_dfs, ignore_index=True)


def extract_labels_for_dlc(dlc_stem: str, all_labels: pd.DataFrame) -> pd.DataFrame:
    filtered_rows = [
        row for _, row in all_labels.iterrows()
        if any(dlc_stem in src for src in str(row["Source"]).split("|"))
    ]
    return pd.DataFrame(filtered_rows)


def make_behavior_label_vector(df: pd.DataFrame, behavior_to_id, total_frames: int) -> pd.DataFrame:
    behavior_df = df[["Behavior", "Behavior type", "Image index"]].dropna()
    unique_behaviors = behavior_df["Behavior"].unique()

    label_vector = np.zeros(total_frames, dtype=int)

    for behavior in unique_behaviors:
        sub_df = behavior_df[behavior_df["Behavior"] == behavior].reset_index(drop=True)
        for i in range(0, len(sub_df), 2):
            if (
                i + 1 < len(sub_df)
                and sub_df.loc[i, "Behavior type"] == "START"
                and sub_df.loc[i + 1, "Behavior type"] == "STOP"
            ):
                start_idx = sub_df.loc[i, "Image index"]
                stop_idx = sub_df.loc[i + 1, "Image index"]
                label_vector[start_idx:stop_idx + 1] = behavior_to_id[behavior]

    return pd.DataFrame({"label": label_vector})

def populate_video_data(project_path: str) -> List[VideoData2D]:

    project_path = Path(project_path)
    if not project_path.exists():
        raise ValueError(f"Project path {project_path} does not exist. Please provide a valid path.")

    dlc_data_dir = project_path / "dlc_data"
    video_dir = project_path / "videos"
    time_series_dir = project_path / "time_series"
    supervised_label_dir = project_path / "supervised_labels"


    dlc_h5_paths = get_paths(dlc_data_dir, "*.h5", required=True)

    supervised_labels_all = load_supervised_labels(supervised_label_dir)
    behavior_to_id = {b: i + 1 for i, b in enumerate(supervised_labels_all["Behavior"].unique())}

    video_data = []

    for dlc_path in dlc_h5_paths:
        video_path = video_dir / f"{dlc_path.stem}.avi"
        if not video_path.exists():
            # warnings.warn(f"Video file {video_path} does not exist. Ignoring.")
            video_path = None

        time_series_path = time_series_dir / f"{dlc_path.stem}.csv"
        if not time_series_path.exists():
            # warnings.warn(f"Time series file {time_series_path} does not exist. Ignoring.")
            time_series_path = None

        # Read and cast DLC HDF5 data
        dlc_data = read_hdf(str(dlc_path))
        dlc_data = dlc_data.with_columns([
            pl.col(col).cast(pl.Float32) if dtype == pl.Float64 else pl.col(col)
            for col, dtype in zip(dlc_data.columns, dlc_data.dtypes)
        ])

        # Extract and encode supervised labels
        supervised_labels = extract_labels_for_dlc(dlc_path.stem, supervised_labels_all)
        if supervised_labels.empty:
            # warnings.warn(f"No labels found for {dlc_path.stem}.")
            label_df = None
        else:
            total_frames = dlc_data.height
            label_df = make_behavior_label_vector(supervised_labels, behavior_to_id, total_frames)
            label_df = pl.from_pandas(label_df).with_columns(
                pl.col("label").cast(pl.Int32)
            )

        # Load time series if available
        time_series_data = pl.read_csv(str(time_series_path)) if time_series_path else None

        video_data.append(VideoData2D(
            video_name=dlc_path.stem,
            video_path=str(video_path) if video_path else None,
            dlc_path=str(dlc_path),
            original_dlc_data=dlc_data,
            time_series_data=time_series_data,
            processed_dlc_data=[],
            supervised_labels=label_df,
        ))

    return video_data

class Project:
    project_name: str
    project_path: str
    output_path: Path
    data_processing_strategies: List[DataProcessingStrategy]
    clustering_strategy: ClusteringStrategy
    project_type: ProjectType
    video_data: List[VideoData2D]

    def __init__(
        self,
        project_name: str,
        project_path: str,
        data_processing_strategies: Optional[List[DataProcessingStrategy]] = [KeepOriginalStrategy(include_likelihood=False)],
        clustering_strategy: Optional[ClusteringStrategy] = PCAKMeansBoutStrategy(n_components=2, n_clusters=5, bout_length=15, stride=1),
        output_path: Optional[str] = None,
    ):
        self.project_name = project_name
        self.project_path = project_path
        self.project_type = ProjectType.D2
        self.output_path = Path(output_path or f"./output/{project_name}")
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.clustering_strategy = clustering_strategy
        self.data_processing_strategies = data_processing_strategies

        self.video_data = populate_video_data(project_path)

    def exclude_columns(self, columns: List[str]) -> None:
        """
        Exclude specific columns from the original DLC data for all videos.

        Parameters:
        - columns: List of column names to exclude.
        """
        pattern = "|".join(columns)
        for video_data in self.video_data:
            video_data["original_dlc_data"] = video_data["original_dlc_data"].select(
                pl.exclude("^.*(" + pattern + ").*$")
            )

    def include_columns(self, columns: List[str]) -> None:
        """
        Include only specific columns in the original DLC data for all videos,
        matched by regex pattern.

        Parameters:
        - columns: List of substrings or regex patterns to match column names for inclusion.
        """
        pattern = "|".join(columns)  # e.g., "nose|jaw"
        for video_data in self.video_data:
            video_data["original_dlc_data"] = video_data["original_dlc_data"].select(
                pl.col("^.*(" + pattern + ").*$")
            )

    def process_data(self) -> None:
        """Process the DLC data using the defined strategies."""
        for video_data in self.video_data:
            original_data = video_data["original_dlc_data"]
            for strategy in self.data_processing_strategies:
                processed_data = strategy.process(original_data)
                video_data["processed_dlc_data"].append({
                    "strategy": strategy,
                    "result": processed_data,
                    "completed": True,
                })

            # Combine all processed outputs horizontally
            combined = [
                output["result"]
                for output in video_data["processed_dlc_data"]
                if output["result"] is not None
            ]
            video_data["combined_data"] = pl.concat(combined, how="horizontal") if combined else None

    def cluster_data(self) -> None:
        """Apply the clustering strategy to the processed data."""
        for video_data in self.video_data:
            if not video_data["processed_dlc_data"]:
                continue

            combined_data = video_data.get("combined_data")
            if combined_data is None:
                continue

            video_data["clustering_output"] = self.clustering_strategy.process(combined_data)

    def get_cluster_output(self, combined: bool = True, drop_excess_rows: bool = True) -> pl.DataFrame | List[pl.DataFrame]:
        """
        Get the clustering output for each video.

        Parameters:
        - combined: Whether to return a single concatenated DataFrame.
        - drop_excess_rows: If True, drops rows with cluster == -1.
        """
        cluster_outputs = []

        for video_data in self.video_data:
            cluster_output = video_data["clustering_output"].clone()

            # Add video name and enforce i32 types
            cluster_output = cluster_output.with_columns([
                pl.lit(Path(video_data["dlc_path"]).stem).alias("video_name"),
                pl.col("cluster").cast(pl.Int32),
                pl.col("bout_id").cast(pl.Int32),
            ])

            if drop_excess_rows:
                cluster_output = cluster_output.filter(pl.col("cluster") != -1)

            cluster_outputs.append(cluster_output)

        return pl.concat(cluster_outputs, how="vertical") if combined else cluster_outputs

    def is_using_data_processing_strategy(self, strategy_type: Type) -> bool:
        """Check if the project is using a specific data processing strategy type."""
        return any(isinstance(s, strategy_type) for s in self.data_processing_strategies)
    
    def save_clustering_output(self, drop_excess_rows=True) -> None:
        """
        Save the clustering output to CSV files in the output directory.
        The output will include a combined CSV file and individual CSV files for each video.
        The combined CSV will contain all clustering results, while individual CSVs will be named after each video.
        """
        
        self.output_path.mkdir(parents=True, exist_ok=True)

        combined_df = self.get_cluster_output(combined=True, drop_excess_rows=drop_excess_rows)
        Path(self.output_path / "csvs").mkdir(parents=True, exist_ok=True)
        combined_df.write_csv(self.output_path / "csvs" / "clustering_output.csv")
        
        for unique_video_name in combined_df["video_name"].unique().to_list():
            video_df = combined_df.filter(pl.col("video_name") == unique_video_name)
            video_df.write_csv(self.output_path / "csvs" / f"{unique_video_name}_clustering_output.csv")