from dlc_clustering.projects import Project
from dlc_clustering.data_processing import VelocityStrategy, ConfidenceFilterStrategy
import matplotlib.pyplot as plt
import polars as pl
import numpy as np
import polars.selectors as cs
import seaborn as sns
import matplotlib.pyplot as plt

def process_data(project: Project, strategies: list) -> None:
    project_data = project.video_data
    """Process each camera's DLC DataFrame with all strategies."""
    video_combined_dfs = []
    for video in project_data:
        video["processed_dlc_data"] = []
        video["processed_dlc_data_components"] = [] 
        video["combined_data"] = []


        for cam_idx, original_df in enumerate(video["original_dlc_data"]):
            result_dfs = []

            
            for strategy in strategies:
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
            combined_dfs = pl.concat(combined_dfs, how="horizontal")
            video["combined_data"] = combined_dfs

    return project.video_data

def collect_data(project : Project):

    project_video_data = process_data(project, [
        ConfidenceFilterStrategy(),
        VelocityStrategy(),
    ])
    dfs = []
    for video in project_video_data:
        combined_data = video["combined_data"]
        video_name = video["video_name"]
        combined_data = combined_data.with_columns(
            total_speed = pl.sum_horizontal(cs.ends_with("_speed")),
            total_confidence = pl.sum_horizontal(cs.ends_with("confidence_filter")),
            video_name = pl.lit(video_name)
        )
        dfs.append(combined_data)

    collected_data = pl.concat(dfs, how="vertical")
    return collected_data



def graph_total_speed(project: Project):
    collected_data = collect_data(project)
    
    collected_data_log = collected_data.with_columns(
        log_total_speed = (pl.col("total_speed"))
    )
    plt.figure(figsize=(10, 6)) 
    plt.hist(collected_data_log["log_total_speed"], bins=30, edgecolor='black', alpha=0.7)
    plt.title("Distribution of Total Speed", fontsize=16)
    plt.xlabel("Total Speed (units)", fontsize=12)
    plt.ylabel("Frequency (Count)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()



def graph_log_total_speed(project: Project):
    collected_data = collect_data(project)

    collected_data_log = collected_data.with_columns(
        log_total_speed = (pl.col("total_speed") + 1).log()
    )
    plt.figure(figsize=(10, 6))
    plt.hist(collected_data_log["log_total_speed"], bins=30, edgecolor='black', alpha=0.7)
    plt.title("Distribution of Log Total Speed", fontsize=16)
    plt.xlabel("Log Total Speed (units)", fontsize=12)
    plt.ylabel("Frequency (Count)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def graph_log_total_speed_fixed(project: Project):
    collected_data = collect_data(project)
    collected_data_log = collected_data.with_columns(
        log_total_speed = (pl.col("total_speed") + 1).log()
    )
    plt.figure(figsize=(10, 6)) 
    plt.hist(collected_data_log["log_total_speed"], bins=30, edgecolor='black', alpha=0.7)
    ax = plt.gca()

    ticks = ax.get_xticks()
    labels = [f"{np.exp(t) - 1:.1f}" for t in ticks]

    ax.set_xticklabels(labels)
    plt.title("Distribution of Total Speed", fontsize=16)
    plt.xlabel("Total Speed (on a Log(x+1) Scale)", fontsize=12)
    plt.ylabel("Frequency (Count)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()



def graph_binned_log_mean_speed(project: Project, frames_per_bin: int = 100):
    collected_data = collect_data(project)
    df_indexed = collected_data.sort(["video_name"]).with_columns(
        frame_index = pl.int_range(0, pl.len()).over("video_name")
    )

    frames_per_bin = 100 

    df_binned = df_indexed.with_columns(
        frame_bin = (pl.col("frame_index") // frames_per_bin) * frames_per_bin
    )

    df_grouped = df_binned.group_by(["video_name", "frame_bin"]).agg(
        pl.col("total_speed").mean().alias("mean_speed")
    )

    collected_data_log = df_grouped.with_columns(
        log_mean_speed = (pl.col("mean_speed") + 1).log()
    )

    plt.figure(figsize=(10, 6)) 

    num_bins = 30
    counts, bin_edges, patches = plt.hist(
        collected_data_log["log_mean_speed"], 
        bins=num_bins, 
        edgecolor='black', 
        alpha=0.7
    )

    ax = plt.gca()

    bin_centers_log = (bin_edges[:-1] + bin_edges[1:]) / 2

    num_labels = 6
    indices = np.linspace(0, len(bin_centers_log) - 1, num_labels, dtype=int)

    tick_locations = bin_centers_log[indices]
    tick_labels = [f"{np.exp(t) - 1:.1f}" for t in tick_locations]

    ax.set_xticks(tick_locations)
    ax.set_xticklabels(tick_labels)

    plt.title(f"Distribution of Mean Speed (per {frames_per_bin} Frame Bins)", fontsize=16)
    plt.xlabel("Mean Speed (on a Log(x+1) Scale, labels at bin centers)", fontsize=12)
    plt.ylabel("Frequency (Count of Bins)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def graph_heatmap_mean_speed(project: Project, frames_per_bin: int = 100):
    collected_data = collect_data(project)
    df_indexed = collected_data.sort(["video_name"]).with_columns(
        frame_index = pl.int_range(0, pl.len()).over("video_name")
    )

    df_binned = df_indexed.with_columns(
        frame_bin = (pl.col("frame_index") // frames_per_bin) * frames_per_bin
    )

    df_grouped = df_binned.group_by(["video_name", "frame_bin"]).agg(
        pl.col("total_speed").mean().alias("mean_speed")
    )

    df_pivot_binned = df_grouped.pivot(
        index="video_name",
        columns="frame_bin",
        values="mean_speed"
    ).sort("video_name")

    df_pivot_binned_pd = df_pivot_binned.to_pandas().set_index("video_name")

    df_pivot_binned_pd.columns = df_pivot_binned_pd.columns.astype(int)
    df_pivot_binned_pd = df_pivot_binned_pd.sort_index(axis=1) 

    plt.figure(figsize=(20, 8))
    sns.heatmap(
        df_pivot_binned_pd,
        mask=df_pivot_binned_pd.isnull(),
        cmap="viridis",
        cbar_kws={'label': 'Mean Total Speed'},
        xticklabels=4 
    )
    plt.title(f"Heatmap of Mean Speed (Binned every {frames_per_bin} frames)", fontsize=16)
    plt.xlabel(f"Frame Bin (Start Frame #)", fontsize=12)
    plt.ylabel("Video Name", fontsize=12)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def graph_heatmap_categorical_speed(project: Project, frames_per_bin: int = 100, low_threshold: float = 3.0, high_threshold: float = 6.2):
    collected_data = collect_data(project)
    df_indexed = collected_data.sort(["video_name"]).with_columns(
        frame_index = pl.int_range(0, pl.len()).over("video_name")
    )

    df_binned = df_indexed.with_columns(
        frame_bin = (pl.col("frame_index") // frames_per_bin) * frames_per_bin
    )

    df_grouped = df_binned.group_by(["video_name", "frame_bin"]).agg(
        pl.col("total_speed").mean().alias("mean_speed")
    )

    df_pivot_binned = df_grouped.pivot(
        index="video_name",
        columns="frame_bin",
        values="mean_speed"
    ).sort("video_name")

    df_pivot_binned_pd = df_pivot_binned.to_pandas().set_index("video_name")

    df_pivot_binned_pd.columns = df_pivot_binned_pd.columns.astype(int)
    df_pivot_binned_pd = df_pivot_binned_pd.sort_index(axis=1) 

    # This function converts a speed value into a category number
    def classify_speed_numeric(speed_val):
        if pd.isna(speed_val):
            return np.nan  # Keep missing values as-is
        
        if speed_val < low_threshold:
            return 0  # Represents "Low"
        elif speed_val < high_threshold:
            return 1  # Represents "Middle"
        else:
            return 2  # Represents "High"

    df_categorical_numeric = df_pivot_binned_pd.map(classify_speed_numeric)
    categorical_cmap = ["#4c72b0", "#f5964f", "#c44e52"]

    plt.figure(figsize=(20, 8))

    ax = sns.heatmap(
        df_categorical_numeric, 
        mask=df_categorical_numeric.isnull(),
        cmap=categorical_cmap,  
        vmin=0, 
        vmax=2,
        cbar_kws={
            'label': 'Speed Category',
            'ticks': [0, 1, 2] 
        },
        xticklabels=4 
    )

    cbar = ax.collections[0].colorbar
    cbar.set_ticklabels(['Low', 'Middle', 'High'])

    plt.title(f"Heatmap of Speed Category (Binned every {frames_per_bin} frames)", fontsize=16)
    plt.xlabel(f"Frame Bin (Start Frame #)", fontsize=12)
    plt.ylabel("Video Name", fontsize=12)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()