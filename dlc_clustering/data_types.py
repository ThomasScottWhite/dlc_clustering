from typing import List, Dict, Any, TypedDict
from enum import Enum
import polars as pl

class ProjectType(str, Enum):
    D2 = "2d"
    D3 = "3d"


class DataProcessingStrategy(TypedDict):
    name: str
    parameters: Dict[str, Any]



class DataProcessingOutput(TypedDict):
    strategy: DataProcessingStrategy
    result: pl.DataFrame | None
    completed: bool


class VideoData2D(TypedDict):
    video_path: str | None
    dlc_path: str
    original_dlc_data: pl.DataFrame
    processed_dlc_data: list[DataProcessingOutput]
    combined_data: pl.DataFrame | None

class VideoData3D(TypedDict):
    video_paths: List[str]
    dlc_data: pl.DataFrame


class ProjectData(TypedDict):
    project_path: str
    project_name: str
    video_data: List[VideoData2D] | List[VideoData3D]
    project_type: ProjectType
    data_processing_strategies: List[DataProcessingStrategy]
