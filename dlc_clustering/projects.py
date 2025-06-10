from dlc_clustering.data_types import ProjectData, ProjectType, VideoData2D
from dlc_clustering.data_processing import read_hdf, KeepOriginalStrategy
from pathlib import Path
from typing import List
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
    video_dir = video_paths[0].parent
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
        
        video_data_2d = VideoData2D(
            video_path=str(video_path),
            dlc_path=str(dlc_h5_path),
            original_dlc_data=original_dlc_data,
            processed_dlc_data=[]
        )
    
    video_data.append(video_data_2d)

    return video_data

class Project():

    def __init__(self, project_name: str, project_path: str, data_processing_strategies=None, clustering_strategy=None):
        self.project_name = project_name
        self.project_path = project_path
        self.data_processing_strategies = data_processing_strategies if data_processing_strategies is not None else [KeepOriginalStrategy(include_likelihood=False)]
        self.clustering_strategy = clustering_strategy
        self.project_type=ProjectType.D2,
        self.video_data = []

        if clustering_strategy is None:
            from dlc_clustering.clustering import PCAKMeansBoutStrategy
            clustering_strategy = PCAKMeansBoutStrategy(n_components=2, n_clusters=5, bout_length=15, stride=1)
            
        if not Path(project_path).exists():
            raise ValueError(f"Project path {project_path} does not exist. Please provide a valid path.")

        dlc_h5_paths = convert_str_to_paths(glob.glob(f"{project_path}/dlc_data/*.h5"))
        if len(dlc_h5_paths) == 0:
            raise ValueError(f"No DLC data found in {project_path}/dlc_data/. Please ensure the directory contains .h5 files.")
        
        video_paths = convert_str_to_paths(glob.glob(f"{project_path}/videos/*"))
        if len(video_paths) == 0:
            warnings.warn(f"No video files found in {project_path}/videos/. Ignore if you have done this intentionally.")

        self.video_data = populate_video_data(video_paths, dlc_h5_paths)

    def process_data(self):
        """
        Process the DLC data using the defined strategies.
        """
        for video_data in self.video_data:
            original_data = video_data['original_dlc_data']
            for strategy in self.data_processing_strategies:
                processed_data = strategy.process(original_data)
                video_data['processed_dlc_data'].append({
                    'strategy': strategy,
                    'result': processed_data,
                    'completed': True
                })
            video_data["combined_data"] = pl.concat([output['result'] for output in video_data['processed_dlc_data'] if output['result'] is not None], how='horizontal')

    def cluster_data(self):
        """
        Apply the clustering strategy to the processed data.
        """
        for video_data in self.video_data:
            if not video_data['processed_dlc_data']:
                continue
            
            combined_data = video_data['combined_data']
            if combined_data is None:
                continue
            
            clustered_output = self.clustering_strategy.process(combined_data)
            video_data['clustering_output'] = clustered_output