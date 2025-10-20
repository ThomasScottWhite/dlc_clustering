from dlc_clustering.projects import Project
from dlc_clustering.clustering import UnsupervisedBoutPipeline, SKStandardScaler, SKPCA, SKKMeans
from dlc_clustering.data_processing import CentroidDiffStrategy, VelocityStrategy, VelocityFilterStrategy, ConfidenceFilterStrategy
from dlc_clustering.data_processing import AnglesStrategy
from dlc_clustering.graphing import graph_all
from dlc_clustering.video_processing import render_cluster_videos


# Define the data processing strategies and clustering strategy
data_processing_strategies = []
data_processing_strategies.append(AnglesStrategy([["nose", "tail_base", "tail_end"]]))
data_processing_strategies.append(CentroidDiffStrategy())
data_processing_strategies.append(VelocityStrategy())
data_processing_strategies.append(VelocityFilterStrategy(percentile_threshold=0.75))
data_processing_strategies.append(ConfidenceFilterStrategy(percentile_threshold=0.75))

clustering_strategy = UnsupervisedBoutPipeline(
    preprocessors=[SKStandardScaler(), SKPCA(n_components=10)],
    clusterer=SKKMeans(n_clusters=8, random_state=42),
    bout_length=15, stride=5
)

project_path = "/home/thomas/Documents/dlc_clustering/data/example_project"
# The files in the project directory should be in the format:
#example_project
#├── dlc_data
#│   └── video1.h5
#└── videos
#    └── video1.avi

output_path = f"./output/example_project"
project = Project("Example Project", project_path, data_processing_strategies, clustering_strategy, output_path=output_path)
project.include_columns(["nose", "tail_base", "tail_end", "left_ear", "right_ear"])
project.process_data()
project.cluster_data()
graph_all(project)
render_cluster_videos(project)
