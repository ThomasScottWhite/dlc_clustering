from dlc_clustering.projects import Project
from dlc_clustering.data_processing import CentroidVelocityStrategy, ConfidenceFilterStrategy
import matplotlib.pyplot as plt
import polars as pl
import pandas as pd
import numpy as np
import polars.selectors as cs
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch

def process_data(project: Project, strategies: list) -> None:
    project_data = project.video_data
    """Process each camera's DLC DataFrame with all strategies."""
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

def collect_freezing_data(project: Project) -> pl.DataFrame:
    project_video_data = process_data(project, [
        ConfidenceFilterStrategy(),
        CentroidVelocityStrategy(),
    ])
    dfs = []
    for video in project_video_data:
        combined_data = video["combined_data"]
        video_name = video["video_name"]
        time_series_data = video["time_series_data"]

        combined_data = combined_data.with_columns(
            total_speed = pl.sum_horizontal(cs.ends_with("_speed")),
            total_confidence = pl.sum_horizontal(cs.ends_with("confidence_filter")),
            video_name = pl.lit(video_name)
        )
        # Combine with time series data
        time_series_data = time_series_data.with_row_index("frame_index")
        combined_data = combined_data.with_row_index("frame_index")
        combined_data = combined_data.join(time_series_data, on="frame_index")
        dfs.append(combined_data)

    columns_set = set()
    for df in dfs:
        columns_set.update(df.columns)

    ordered_columns = sorted(list(columns_set))
    aligned_dfs = []

    for df in dfs:
        current_cols = df.columns
        cols_to_add = [
            pl.lit(0).alias(col) for col in ordered_columns if col not in current_cols
        ]
        aligned_df = df.with_columns(cols_to_add).select(ordered_columns)
        aligned_dfs.append(aligned_df)

    collected_data = pl.concat(aligned_dfs, how="vertical")

    return collected_data

def graph_total_speed(project: Project):
    collected_data = collect_freezing_data(project)
    
    collected_data_log = collected_data.with_columns(
        log_total_speed = (pl.col("total_speed"))
    )
    plt.figure(figsize=(10, 6)) 
    plt.hist(collected_data_log["log_total_speed"], bins=30, edgecolor='black', alpha=0.7)
    plt.title("Distribution of Total Speed", fontsize=16)
    plt.xlabel("Total Speed (units)", fontsize=12)
    plt.ylabel("Frequency (Count)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    output_path = Path(project.output_path) / "freezing_analysis" / "total_speed_distribution.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


# def graph_log_total_speed(project: Project):
#     collected_data = collect_freezing_data(project)

#     collected_data_log = collected_data.with_columns(
#         log_total_speed = (pl.col("total_speed") + 1).log()
#     )
#     plt.figure(figsize=(10, 6))
#     plt.hist(collected_data_log["log_total_speed"], bins=30, edgecolor='black', alpha=0.7)
#     plt.title("Distribution of Log Total Speed", fontsize=16)
#     plt.xlabel("Log Total Speed (units)", fontsize=12)
#     plt.ylabel("Frequency (Count)", fontsize=12)
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
#     plt.show()


def graph_log_total_speed_fixed(project: Project):
    collected_data = collect_freezing_data(project)
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
    output_path = Path(project.output_path) / "freezing_analysis" / "log_total_speed_distribution.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()



def graph_binned_log_mean_speed(project: Project, frames_per_bin: int = 100):
    collected_data = collect_freezing_data(project)
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
    output_path = Path(project.output_path) / "freezing_analysis" / "binned_log_mean_speed.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def graph_heatmap_mean_speed(project: Project, frames_per_bin: int = 100):
    collected_data = collect_freezing_data(project)

    frames_per_bin = 500


    FLAG_COLORS = {
        "VoidTiming_flag": "red",
        "ToneOffsetframe_flag": "blue",
        "ToneONsetframe_flag": "green",
        "ShockONsetframe_flag": "orange",
        "ShockOffsetframe_flag": "purple",
    }

    all_flag_cols = [col for col in collected_data.columns if col.endswith("_flag")]

    default_colors = ['magenta', 'cyan', 'yellow', 'brown', 'black', 'grey'] # Fallback colors
    color_idx = 0
    for flag_col in all_flag_cols:
        if flag_col not in FLAG_COLORS:
            if color_idx < len(default_colors):
                FLAG_COLORS[flag_col] = default_colors[color_idx]
                color_idx += 1
            else:
                FLAG_COLORS[flag_col] = 'black' # Default if all fallback colors are used

    print(f"Using flag colors: {FLAG_COLORS}")


    df_indexed = collected_data.sort(["video_name"]).with_columns(
        frame_index = pl.int_range(0, pl.len()).over("video_name")
    )
    df_binned = df_indexed.with_columns(
        frame_bin = (pl.col("frame_index") // frames_per_bin) * frames_per_bin
    )
    flag_aggregations = [pl.col(col).max().alias(f"max_{col}") for col in all_flag_cols]

    df_grouped = df_binned.group_by(["video_name", "frame_bin"]).agg(
        pl.col("total_speed").mean().alias("mean_speed"),
        *flag_aggregations 
    )


    df_pivot_binned = df_grouped.pivot(
        index="video_name",
        columns="frame_bin",
        values="mean_speed"
    ).sort("video_name")
    df_pivot_binned_pd = df_pivot_binned.to_pandas().set_index("video_name")
    df_pivot_binned_pd.columns = df_pivot_binned_pd.columns.astype(int)
    df_pivot_binned_pd = df_pivot_binned_pd.sort_index(axis=1)


    df_pivot_flags_pd_dict = {}
    for flag_col in all_flag_cols:
        df_pivot_flag = df_grouped.pivot(
            index="video_name",
            columns="frame_bin",
            values=f"max_{flag_col}"
        ).sort("video_name")
        
        df_pivot_flag_pd = df_pivot_flag.to_pandas().set_index("video_name")
        df_pivot_flag_pd.columns = df_pivot_flag_pd.columns.astype(int)
        df_pivot_flag_pd = df_pivot_flag_pd.sort_index(axis=1)
        df_pivot_flag_pd = df_pivot_flag_pd.reindex_like(df_pivot_binned_pd).fillna(0).astype(int)
        
        df_pivot_flags_pd_dict[flag_col] = df_pivot_flag_pd

    plt.figure(figsize=(20, 10)) 
    ax = sns.heatmap(
        df_pivot_binned_pd,
        mask=df_pivot_binned_pd.isnull(),
        cmap="viridis",
        cbar_kws={'label': 'Mean Total Speed'},
        xticklabels=4,
    )

    y_labels = df_pivot_binned_pd.index
    x_labels = df_pivot_binned_pd.columns

    for flag_col, flag_df_pd in df_pivot_flags_pd_dict.items():
        for i, video_name in enumerate(y_labels):
            for j, frame_bin in enumerate(x_labels):
                if not np.isnan(flag_df_pd.loc[video_name, frame_bin]) and flag_df_pd.loc[video_name, frame_bin] == 1:
                    ax.text(
                        j + 0.5,
                        i + 0.5,
                        'x',
                        color=FLAG_COLORS[flag_col],
                        ha='center', va='center',
                        fontsize=24,
                        fontweight='bold'
                    )

    legend_handles = []
    for flag_col, color in FLAG_COLORS.items():
        if flag_col in all_flag_cols:
            legend_handles.append(Patch(color=color, label=flag_col.replace("_flag", ""))) # Customize label if desired

    ax.legend(handles=legend_handles, title="Active Flags", bbox_to_anchor=(1.15, 1), loc='upper left', borderaxespad=0.)


    plt.title(f"Heatmap of Mean Speed with Active Flags (Binned every {frames_per_bin} frames)", fontsize=16)
    plt.xlabel(f"Frame Bin (Start Frame #)", fontsize=12)
    plt.ylabel("Video Name", fontsize=12)
    plt.yticks(rotation=0)
    plt.tight_layout(rect=[0, 0, 0.88, 1])

    output_path = Path(project.output_path) / "freezing_analysis" / "heatmap_mean_speed.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def graph_heatmap_categorical_speed(project: Project, frames_per_bin: int = 100, percentile=None, thresholds=None):
    # An example function call for percentile-based thresholds:
    # graph_heatmap_categorical_speed(project, frames_per_bin=100, percentile=(25, 75))

    # An example function call for fixed thresholds:
    # graph_heatmap_categorical_speed(project, frames_per_bin=100, thresholds=(3.0, 6.2))

    if thresholds is not None:
        low_threshold, high_threshold = thresholds
    elif percentile is not None:
        collected_data = collect_freezing_data(project)
        speed_values = collected_data["total_speed"].to_numpy()
        low_threshold = np.percentile(speed_values, percentile[0])
        high_threshold = np.percentile(speed_values, percentile[1])
    else:
        low_threshold = 3.0
        high_threshold = 6.2

    collected_data = collect_freezing_data(project)

    frames_per_bin = 500


    FLAG_COLORS = {
        "VoidTiming_flag": "red",
        "ToneOffsetframe_flag": "blue",
        "ToneONsetframe_flag": "green",
        "ShockONsetframe_flag": "orange",
        "ShockOffsetframe_flag": "purple",
    }

    all_flag_cols = [col for col in collected_data.columns if col.endswith("_flag")]

    default_colors = ['magenta', 'cyan', 'yellow', 'brown', 'black', 'grey'] # Fallback colors
    color_idx = 0
    for flag_col in all_flag_cols:
        if flag_col not in FLAG_COLORS:
            if color_idx < len(default_colors):
                FLAG_COLORS[flag_col] = default_colors[color_idx]
                color_idx += 1
            else:
                FLAG_COLORS[flag_col] = 'black' # Default if all fallback colors are used

    print(f"Using flag colors: {FLAG_COLORS}")


    df_indexed = collected_data.sort(["video_name"]).with_columns(
        frame_index = pl.int_range(0, pl.len()).over("video_name")
    )
    df_binned = df_indexed.with_columns(
        frame_bin = (pl.col("frame_index") // frames_per_bin) * frames_per_bin
    )
    flag_aggregations = [pl.col(col).max().alias(f"max_{col}") for col in all_flag_cols]

    df_grouped = df_binned.group_by(["video_name", "frame_bin"]).agg(
        pl.col("total_speed").mean().alias("mean_speed"),
        *flag_aggregations 
    )


    df_pivot_binned = df_grouped.pivot(
        index="video_name",
        columns="frame_bin",
        values="mean_speed"
    ).sort("video_name")


    df_pivot_binned_pd = df_pivot_binned.to_pandas().set_index("video_name")
    df_pivot_binned_pd.columns = df_pivot_binned_pd.columns.astype(int)
    df_pivot_binned_pd = df_pivot_binned_pd.sort_index(axis=1)


    df_pivot_flags_pd_dict = {}
    for flag_col in all_flag_cols:
        df_pivot_flag = df_grouped.pivot(
            index="video_name",
            columns="frame_bin",
            values=f"max_{flag_col}"
        ).sort("video_name")
        
        df_pivot_flag_pd = df_pivot_flag.to_pandas().set_index("video_name")
        df_pivot_flag_pd.columns = df_pivot_flag_pd.columns.astype(int)
        df_pivot_flag_pd = df_pivot_flag_pd.sort_index(axis=1)
        df_pivot_flag_pd = df_pivot_flag_pd.reindex_like(df_pivot_binned_pd).fillna(0).astype(int)
        
        df_pivot_flags_pd_dict[flag_col] = df_pivot_flag_pd
        
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

    plt.figure(figsize=(20, 10)) 

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

    y_labels = df_pivot_binned_pd.index
    x_labels = df_pivot_binned_pd.columns

    for flag_col, flag_df_pd in df_pivot_flags_pd_dict.items():
        for i, video_name in enumerate(y_labels):
            for j, frame_bin in enumerate(x_labels):
                if not np.isnan(flag_df_pd.loc[video_name, frame_bin]) and flag_df_pd.loc[video_name, frame_bin] == 1:
                    ax.text(
                        j + 0.5,
                        i + 0.5,
                        'x',
                        color=FLAG_COLORS[flag_col],
                        ha='center', va='center',
                        fontsize=24,
                        fontweight='bold'
                    )

    legend_handles = []
    for flag_col, color in FLAG_COLORS.items():
        if flag_col in all_flag_cols:
            legend_handles.append(Patch(color=color, label=flag_col.replace("_flag", ""))) # Customize label if desired

    ax.legend(handles=legend_handles, title="Active Flags", bbox_to_anchor=(1.15, 1), loc='upper left', borderaxespad=0.)


    plt.title(f"Heatmap of Mean Speed with Active Flags (Binned every {frames_per_bin} frames)", fontsize=16)
    plt.xlabel(f"Frame Bin (Start Frame #)", fontsize=12)
    plt.ylabel("Video Name", fontsize=12)
    plt.yticks(rotation=0)
    plt.tight_layout(rect=[0, 0, 0.88, 1])

    output_path = Path(project.output_path) / "freezing_analysis" / "heatmap_mean_speed.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def graph_all_freezing_analysis(project: Project, frames_per_bin: int = 100):
    graph_total_speed(project)
    graph_log_total_speed_fixed(project)
    graph_binned_log_mean_speed(project, frames_per_bin=frames_per_bin)
    graph_heatmap_mean_speed(project, frames_per_bin=frames_per_bin)
    graph_heatmap_categorical_speed(project, frames_per_bin=frames_per_bin, percentile=(10, 50))