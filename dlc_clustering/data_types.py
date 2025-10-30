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
    video_paths: List[str] | None
    dlc_paths: List[str]
    original_dlc_data: List[pl.DataFrame]
    time_series_data: pl.DataFrame | None
    processed_dlc_data: List[pl.DataFrame]
    processed_dlc_data_components: List[DataProcessingOutput] 
    combined_data: pl.DataFrame | None
    clustering_output: pl.DataFrame | None  
    supervised_labels: pl.DataFrame | None
    camera_order: List[str] | None
    group: List[str] | None
    
class VideoData3D(TypedDict):
    video_paths: List[str]
    dlc_data: pl.DataFrame
