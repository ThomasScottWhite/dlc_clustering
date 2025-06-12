from typing import List, Dict, Any, TypedDict
from enum import Enum
import polars as pl
from dlc_clustering.data_processing import DataProcessingStrategy
class ProjectType(str, Enum):
    D2 = "2d"
    D3 = "3d"


class DataProcessingOutput(TypedDict):
    strategy: DataProcessingStrategy
    result: pl.DataFrame | None
    completed: bool

class VideoData2D(TypedDict):
    video_name: str
    video_path: str | None
    dlc_path: str
    original_dlc_data: pl.DataFrame
    processed_dlc_data: List[DataProcessingOutput]  # safer for older Python
    combined_data: pl.DataFrame | None
    clustering_output: pl.DataFrame | None  # rename for consistency
class VideoData3D(TypedDict):
    video_paths: List[str]
    dlc_data: pl.DataFrame
