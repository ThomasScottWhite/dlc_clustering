from dlc_clustering.data_types import ProjectType, VideoData2D
from dlc_clustering.data_processing import KeepOriginalStrategy
from dlc_clustering.data_processing import DataProcessingStrategy

from pathlib import Path
from typing import List, Optional, Type, Tuple
import polars as pl
import glob
import warnings
import pandas as pd
import numpy as np
import re
from pathlib import Path

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

    label_vector = np.full(total_frames, fill_value=-1, dtype=int)

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





def populate_video_data(project_path: str, merge_cameras: bool = True) -> Tuple[list[VideoData2D], dict[str, int]]:

    if merge_cameras:
        CAM_SUFFIX_RE = re.compile(r"_cam[A-Za-z0-9]+$")
        def base_key(stem: str) -> str:
            return CAM_SUFFIX_RE.sub("", stem)
    else:
        # keep full stem so nothing is merged
        def base_key(stem: str) -> str:
            return stem


    project_path = Path(project_path)
    dlc_dir = project_path / "dlc_data"
    video_dir = project_path / "videos"
    time_series_dir  = project_path / "time_series"
    supervised_dir  = project_path / "supervised_labels"

    dlc_paths = get_paths(dlc_dir, "*.h5", required=True)

    # group by base key (e.g., removing _camA, _camB, etc.)
    grouped: dict[str, list[Path]] = {}
    for path in dlc_paths:
        grouped.setdefault(base_key(path.stem), []).append(path)

    # supervised labels are optional
    has_labels = supervised_dir.exists() and any(supervised_dir.glob("*"))
    if has_labels:
        supervised_labels_all = load_supervised_labels(supervised_dir)
        # handle empty dataframe too
        if getattr(supervised_labels_all, "empty", False):
            has_labels = False
    else:
        supervised_labels_all = None

    behavior_to_id: dict[str, int] = {}
    if has_labels:
        behavior_to_id = {b: i + 1 for i, b in enumerate(supervised_labels_all["Behavior"].unique())}

    video_data: list[VideoData2D] = []

    for key, cam_paths in grouped.items():
        video_paths: list[str] = []
        dlc_paths_list: list[str] = []
        original_dlc_data: list[pl.DataFrame] = []
        camera_order: list[str] = []

        for path in sorted(cam_paths):
            dlc_paths_list.append(str(path))

            cam_name = CAM_SUFFIX_RE.search(path.stem)

            camera_order.append(cam_name.group(0).split("_")[-1] if cam_name else "")

            raw_df = read_hdf(str(path))
            # downcast float64 to float32 to fix later data type issues
            float64_cols = [c for c, t in zip(raw_df.columns, raw_df.dtypes) if t == pl.Float64]
            dlc_df = raw_df.with_columns([pl.col(c).cast(pl.Float32) for c in float64_cols]) 
            original_dlc_data.append(dlc_df)

            video_path = video_dir / f"{path.stem}.avi"
            if video_path.exists():
                video_paths.append(str(video_path))

            time_series_path = time_series_dir / f"{path.stem}.csv"
            time_series_data = (pl.read_csv(str(time_series_path)) if time_series_path.exists() else None)

        # checks for same frame count across cams
        heights = [df.height for df in original_dlc_data]
        if len(set(heights)) != 1:
            print(f"Frame count mismatch for {key}: {heights}, in videos: {video_paths}, dlc files: {dlc_paths_list}")
            print("Skipping this set of cameras.")
            continue

        # labels are optional and keyed by base
        label_df = None
        if has_labels:
            supervised_labels = extract_labels_for_dlc(key, supervised_labels_all)
            if not getattr(supervised_labels, "empty", True):
                label_df = pl.from_pandas(
                    make_behavior_label_vector(supervised_labels, behavior_to_id, heights[0])
                ).with_columns(pl.col("label").cast(pl.Int32))

        video_data.append(VideoData2D(
            video_name=key,
            video_paths=video_paths or None,
            dlc_paths=dlc_paths_list,
            original_dlc_data=original_dlc_data,
            time_series_data=time_series_data,
            processed_dlc_data=[],
            processed_dlc_data_components=[],
            combined_data=None,
            clustering_output=[],
            supervised_labels=label_df,
            camera_order=camera_order,
            group=[],
        ))

    return video_data, behavior_to_id

class Project:
    project_name: str
    project_path: str
    output_path: Path
    data_processing_strategies: List[DataProcessingStrategy]
    clustering_strategy: any
    project_type: ProjectType
    video_data: List[VideoData2D]

    def __init__(
        self,
        project_name: str,
        project_path: str,
        data_processing_strategies: Optional[List[DataProcessingStrategy]] = [KeepOriginalStrategy(include_likelihood=False)],
        clustering_strategy = None,
        output_path: Optional[str] = None,
        merge_cameras = True
    ):
        self.project_name = project_name
        self.project_path = project_path
        self.project_type = ProjectType.D2
        self.output_path = Path(output_path or f"./output/{project_name}")
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.clustering_strategy = clustering_strategy
        self.data_processing_strategies = data_processing_strategies
        self.merge_cameras = merge_cameras
        self.groups: set = set()

        self.video_data, self.behaviour_to_id = populate_video_data(project_path, merge_cameras=merge_cameras)

    def exclude_columns(self, columns: List[str]) -> None:
        """
        Exclude specific columns from the original DLC data for all videos.

        Parameters:
        - columns: List of column names to exclude.
        """
        pattern = "|".join(columns)
        for video in self.video_data:
            video["original_dlc_data"] = [
                df.select(pl.exclude("^.*(" + pattern + ").*$")) for df in video["original_dlc_data"]
            ]


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

        for video in self.video_data:
            video["original_dlc_data"] = [
                df.select(pl.col("^.*(" + pattern + ").*$")) for df in video["original_dlc_data"]
            ]

    def process_data(self) -> None:
        """Process each camera's DLC DataFrame with all strategies."""
        for video in self.video_data:
            video["processed_dlc_data"] = []
            video["processed_dlc_data_components"] = [] 
            video["combined_data"] = []


            for cam_idx, original_df in enumerate(video["original_dlc_data"]):
                result_dfs = []

                for strategy in self.data_processing_strategies:
                    result = strategy.process(original_df)

                    result_dfs.append({
                        "strategy": strategy,
                        "result": result,
                    })

                # save per-cam outputs
                video["processed_dlc_data_components"].append(result_dfs)

                # horizontally combine results for this cam
                result_dfs = [r["result"] for r in result_dfs if r["result"] is not None]
                combined = pl.concat(result_dfs, how="horizontal")

                # prefix columns with cam name, if cam name exists
                if video["camera_order"][cam_idx] != "":
                    cam_prefix = video["camera_order"][cam_idx].lstrip("_") + "_"
                    combined = combined.rename({c: f"{cam_prefix}{c}" for c in combined.columns})

                video["processed_dlc_data"].append(combined)
            
            # Horizontally combine all cams' processed data
            combined_dfs = [df for df in video["processed_dlc_data"] if df is not None]
            if combined_dfs:
                video["combined_data"] = pl.concat(combined_dfs, how="horizontal")

    def cluster_data(self) -> None:
        """Apply the clustering strategy to the processed data."""
        self.clustering_strategy.process(self)


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
                pl.lit(Path(video_data["video_name"]).stem).alias("video_name"),
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

    def group_videos(self, group_title, video_name_regex):
        self.groups.add(group_title)
        for video in self.video_data:
            if video_name_regex in video["video_name"]:
                video["group"].append(group_title)