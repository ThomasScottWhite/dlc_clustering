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

def populate_video_data(project_path: str) -> List[VideoData2D]:
    project_path = Path(project_path)
    if not project_path.exists():
        raise ValueError(f"Project path {project_path} does not exist. Please provide a valid path.")

    def get_paths(subdir: str, pattern: str, required: bool = True) -> List[Path]:
        paths = convert_str_to_paths(glob.glob(str(project_path / subdir / pattern)))
        if not paths and required:
            raise ValueError(f"No files matching '{pattern}' found in {project_path / subdir}")
        if not paths and not required:
            warnings.warn(f"No files found in {project_path / subdir}. Ignore if intentional.")
        return paths

    dlc_h5_paths = get_paths("dlc_data", "*.h5", required=True)
    video_paths = get_paths("videos", "*.avi", required=False)
    time_series_paths = get_paths("time_series", "*.csv", required=False)

    video_dir = Path(project_path / "videos")
    time_series_dir = Path(project_path / "time_series")

    video_data = []


    for dlc_path in dlc_h5_paths:

        video_path = None
        time_series_path = None
        
        # Verifies if video_paths exist and matches the DLC file name
        if video_paths:
            video_path = video_dir / f"{dlc_path.stem}.avi"
            if not video_path.exists():
                warnings.warn(f"Video file {video_path} does not exist for DLC data {dlc_path}. Ignoring.")
                video_path = None

        if time_series_paths:
            time_series_path = time_series_dir / f"{dlc_path.stem}.csv"
            if not video_path.exists():
                warnings.warn(f"Video file {video_path} does not exist for DLC data {dlc_path}. Ignoring.")
                video_path = None

        # Read the DLC data from the HDF5 file and force float32 for compatibility
        dlc_data = read_hdf(str(dlc_path))

        dlc_data = dlc_data.with_columns([
            pl.col(col).cast(Float32) if dtype == Float64 else pl.col(col)
            for col, dtype in zip(dlc_data.columns, dlc_data.dtypes)
        ])
        time_series_data = pl.read_csv(str(time_series_path)) if time_series_path else None
        
        video_data.append(VideoData2D(
            video_name=dlc_path.stem,
            video_path=str(video_path),
            dlc_path=str(dlc_path),
            original_dlc_data=dlc_data,
            time_series_data=time_series_data,
            processed_dlc_data=[]
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