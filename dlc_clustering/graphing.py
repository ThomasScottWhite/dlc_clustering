from dlc_clustering.projects import Project
from dlc_clustering.data_processing import CentroidDiffStrategy, VelocityStrategy
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import polars as pl
import numpy as np
import copy

def graph_combined_cluster_counts(project: Project):
    combined_df: pl.DataFrame = project.get_cluster_output()
    output_path = project.output_path / "graphs" / "cluster_counts"
    output_path.mkdir(parents=True, exist_ok=True)

    # Gets the cluster for each bout
    bout_clusters = combined_df.group_by(["bout_id"]).agg(
        pl.col("cluster").mode().first().alias("cluster"),
    )
    # Counts the number of bouts per cluster
    bouts_per_cluster = bout_clusters.group_by("cluster").len().rename({"len": "num_bouts"})


    df = bouts_per_cluster.to_pandas()
    df = df.sort_values("cluster")
    # Plot using cluster as x-axis
    plt.figure(figsize=(6, 4))
    plt.bar(df["cluster"].astype(str), df["num_bouts"])
    plt.xlabel("Cluster Label")
    plt.ylabel("Number of Bouts")
    plt.title("Cluster Membership Counts")
    plt.xticks(rotation=0)
    plt.grid(axis='y')
    plt.tight_layout()

    plt.savefig(output_path / "combined_cluster_counts.png")
    plt.close()


def graph_video_cluster_counts(project: Project):
    combined_df: pl.DataFrame = project.get_cluster_output()
    output_path = project.output_path / "graphs" / "cluster_counts"
    output_path.mkdir(parents=True, exist_ok=True)

    bout_clusters = (
        combined_df
        .group_by(["video_name", "bout_id"])
        .agg(pl.col("cluster").mode().first().alias("cluster"))
    )
    bouts_per_cluster = (
        bout_clusters
        .group_by(["video_name", "cluster"])
        .len()
        .rename({"len": "num_bouts"})
        .to_pandas()
    )

    for video_name, group in bouts_per_cluster.groupby("video_name"):
        group = group.sort_values("cluster")
        plt.figure(figsize=(6, 4))
        plt.bar(group["cluster"].astype(str), group["num_bouts"])
        plt.xlabel("Cluster Label")
        plt.ylabel("Number of Bouts")
        plt.title(f"Cluster Counts for Video: {video_name}")
        plt.xticks(rotation=0)
        plt.grid(axis='y')
        plt.tight_layout()

        # Save with a sanitized filename
        file_path = output_path / f"{video_name}.png"
        plt.savefig(file_path)
        plt.close()

def graph_cluster_specific_velocity(project: Project):
    if not project.is_using_data_processing_strategy(VelocityStrategy):
        print("VelocityStrategy is not used in this project. Skipping velocity graphing.")
        return
    output_path = project.output_path / "graphs" / "velocity" / "combined_cluster_velocity"
    output_path.mkdir(parents=True, exist_ok=True)

    combined_df = project.get_cluster_output()

    #  Get the unique cluster per (video, bout)
    bout_clusters = combined_df.group_by(["video_name", "cluster"]).agg(
        [
        pl.col("^.*_velocity$").mean(),

        ]
    )

    bout_clusters = bout_clusters.to_pandas()

    velocity_cols = [col for col in bout_clusters.columns if col.endswith("_velocity")]

    for col in velocity_cols:
        plt.figure(figsize=(8, 4))
        # X-axis: cluster (you could also include video_name if you want finer granularity)
        x = bout_clusters["cluster"].astype(str)  # Convert to string to avoid int labels
        y = bout_clusters[col]

        plt.bar(x, y)
        plt.xlabel("Cluster")
        plt.ylabel("Mean " + col)
        plt.title(f"Mean {col} per Cluster")
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig(output_path / f"{col}.png")
        plt.close()

def graph_video_cluster_specific_velocity(project: Project):
    if not project.is_using_data_processing_strategy(VelocityStrategy):
        print("VelocityStrategy is not used in this project. Skipping velocity graphing.")
        return

    output_path = project.output_path / "graphs" / "velocity" / "video_cluster_velocity"
    output_path.mkdir(parents=True, exist_ok=True)

    combined_df = project.get_cluster_output()

    # Aggregate mean velocities per video_name and cluster
    bout_clusters = combined_df.group_by(["video_name", "cluster"]).agg(
        [pl.col("^.*_velocity$").mean()]
    )
    bout_clusters = bout_clusters.to_pandas()

    #  Get all velocity columns
    velocity_cols = [col for col in bout_clusters.columns if col.endswith("_velocity")]

    # Group by video_name and plot separately
    for video_name, group in bout_clusters.groupby("video_name"):
        group = group.sort_values("cluster")

        for col in velocity_cols:
            plt.figure(figsize=(8, 4))
            x = group["cluster"].astype(str)
            y = group[col]

            plt.bar(x, y)
            plt.xlabel("Cluster")
            plt.ylabel(f"Mean {col}")
            plt.title(f"{col} by Cluster for {video_name}")
            plt.grid(axis='y')
            plt.tight_layout()

            # Sanitize file name
            filename = f"{col}.png"
            video_folder = output_path / video_name
            video_folder.mkdir(parents=True, exist_ok=True)
            plt.savefig(video_folder / filename)
            plt.close()

def graph_video_specific_velocity(project: Project):
    if not project.is_using_data_processing_strategy(VelocityStrategy):
        print("VelocityStrategy is not used in this project. Skipping velocity graphing.")
        return

    output_path = project.output_path / "graphs" / "velocity" / "video_specific_velocity"
    output_path.mkdir(parents=True, exist_ok=True)

    combined_df = project.get_cluster_output()

    # Aggregate mean velocities per video_name only
    video_velocities = combined_df.group_by("video_name").agg(
        [pl.col("^.*_velocity$").mean()]
    ).to_pandas()

    # Identify velocity columns
    velocity_cols = [col for col in video_velocities.columns if col.endswith("_velocity")]

    # Plot one graph per video
    for _, row in video_velocities.iterrows():
        video_name = row["video_name"]
        x = velocity_cols
        y = [row[col] for col in velocity_cols]

        plt.figure(figsize=(10, 5))
        positions = np.arange(len(x))  # Use numerical positions for the bars
        plt.bar(positions, y)

        plt.xlabel("Body Part")
        plt.ylabel("Mean Velocity")
        plt.title(f"Mean Body Part Velocities for {video_name}")
        plt.xticks(positions, x, rotation=45, ha="right")  # Set tick positions and labels
        plt.grid(axis='y')
        plt.tight_layout()

        filename = f"{video_name}.png"
        plt.savefig(output_path / filename)
        plt.close()

def plot_cluster_heatmap(project : Project):
    # Step 1: Aggregate cluster data across videos
    output_path = project.output_path / "graphs" / "heatmaps"
    output_path.mkdir(parents=True, exist_ok=True)
    dfs = []
    for video in project.video_data:
        df: pl.DataFrame = video["clustering_output"]
        df = df.select(["cluster", "bout_id"]).with_columns(
            pl.lit(video["video_name"]).alias("video_name")
        )
        dfs.append(df)

    # Step 2: Combine into a single DataFrame
    df = pl.concat(dfs, how="vertical")

    # Step 3: Pivot to create matrix for heatmap
    pivoted = df.pivot(
        values="cluster",
        index="video_name",
        columns="bout_id",
        aggregate_function="first"  # assumes unique video_name-bout_id pairs
    )

    # Step 4: Optional – fill nulls (e.g., with -1)
    pivoted = pivoted.fill_null(-1)

    # Step 5: Convert to pandas and plot
    df_pandas = pivoted.to_pandas().set_index("video_name")

    plt.figure(figsize=(12, 6))
    sns.heatmap(df_pandas, annot=False, fmt="d", cmap="viridis", cbar=True)
    plt.title("Cluster Heatmap by Video and Bout")
    plt.xlabel("Bout ID")
    plt.ylabel("Video Name")
    plt.tight_layout()
    plt.savefig(output_path / "cluster_heatmap.png")
    plt.close()

def graph_all(project: Project):
    """
    Generate all graphs for the project.
    """
    graph_combined_cluster_counts(project)
    graph_video_cluster_counts(project)
    graph_cluster_specific_velocity(project)
    graph_video_cluster_specific_velocity(project)
    graph_video_specific_velocity(project)
    plot_cluster_heatmap(project)

    for group in project.groups:
        group_project = copy.deepcopy(project)
        group_project.output_path = project.output_path / "groups" / group
        group_project.video_data = [
            video for video in project.video_data if group in video["group"]
        ]
        graph_combined_cluster_counts(group_project)
        graph_video_cluster_counts(group_project)
        graph_cluster_specific_velocity(group_project)
        graph_video_cluster_specific_velocity(group_project)
        graph_video_specific_velocity(group_project)
        plot_cluster_heatmap(group_project)