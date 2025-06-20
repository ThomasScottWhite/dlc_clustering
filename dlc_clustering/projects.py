from dlc_clustering.data_types import ProjectType, VideoData2D
from dlc_clustering.data_processing import read_hdf, KeepOriginalStrategy
from dlc_clustering.data_processing import DataProcessingStrategy
from dlc_clustering.clustering import ClusteringStrategy        
from polars import Float64, Float32
from pathlib import Path
from typing import List, Optional, Type
import polars as pl
import glob
import warnings

def convert_str_to_paths(video_paths: List[str]) -> List[Path]:
    """
    Convert a list of string paths to Path objects.
    """
    return [Path(path) for path in video_paths]

def populate_video_data(video_paths, dlc_h5_paths):
    video_data = []
    if video_paths:
        video_dir = video_paths[0].parent
    else:
        video_dir = Path("./this_directory_does_not_exist")

    h5_to_video_map = {}
    for dlc_h5_path in dlc_h5_paths:
        video_map_path = video_dir / (dlc_h5_path.stem + ".avi")
        h5_to_video_map[dlc_h5_path] = video_map_path


    for delc_path in h5_to_video_map.keys():
        video_path = h5_to_video_map[delc_path]
        if not video_path.exists():
            warnings.warn(f"Video file {video_path} does not exist for DLC data {delc_path}. Ignore if you have done this intentionally.")
            video_path = None

        original_dlc_data = read_hdf(str(delc_path))
        
        # Force all Float64 columns to Float32
        original_dlc_data = original_dlc_data.with_columns([
            pl.col(col).cast(Float32) if dtype == Float64 else pl.col(col)
            for col, dtype in zip(original_dlc_data.columns, original_dlc_data.dtypes)
        ])


        video_data_2d = VideoData2D(
            video_name=delc_path.stem,
            video_path=str(video_path),
            dlc_path=str(delc_path),
            original_dlc_data=original_dlc_data,
            processed_dlc_data=[]
        )
    
        video_data.append(video_data_2d)

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
        data_processing_strategies: Optional[List[DataProcessingStrategy]] = None,
        clustering_strategy: Optional[ClusteringStrategy] = None,
        output_path: Optional[str] = None,
    ):
        self.project_name = project_name
        self.project_path = project_path
        self.project_type = ProjectType.D2
        self.output_path = Path(output_path or f"./output/{project_name}")
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.data_processing_strategies = (
            data_processing_strategies or [KeepOriginalStrategy(include_likelihood=False)]
        )

        if clustering_strategy is None:
            from dlc_clustering.clustering import PCAKMeansBoutStrategy
            clustering_strategy = PCAKMeansBoutStrategy(
                n_components=2, n_clusters=5, bout_length=15, stride=1
            )

        self.clustering_strategy = clustering_strategy

        if not Path(project_path).exists():
            raise ValueError(
                f"Project path {project_path} does not exist. Please provide a valid path."
            )

        dlc_h5_paths = convert_str_to_paths(glob.glob(f"{project_path}/dlc_data/*.h5"))
        if not dlc_h5_paths:
            raise ValueError(
                f"No DLC data found in {project_path}/dlc_data/. Please ensure the directory contains .h5 files."
            )

        video_paths = convert_str_to_paths(glob.glob(f"{project_path}/videos/*"))
        if not video_paths:
            warnings.warn(
                f"No video files found in {project_path}/videos/. Ignore if you have done this intentionally."
            )

        self.video_data = populate_video_data(video_paths, dlc_h5_paths)

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