import pandas as pd
import polars as pl
import polars as pl
from dataclasses import dataclass
from typing import Protocol

def read_hdf(csv_path: str) -> pd.DataFrame:
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



class DataProcessingStrategy(Protocol):
    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Process the input DataFrame and return a new DataFrame.
        
        Args:
            df (pl.DataFrame): Input DataFrame to process.
        
        Returns:
            pl.DataFrame: Processed DataFrame.
        """
        pass

@dataclass
class KeepOriginalStrategy:
    include_likelihood: bool = False

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        regex = "^.*(_x|_y|_z|_likelihood)$" if self.include_likelihood else "^.*(_x|_y|_z)$"
        return df.select(pl.col(regex))


@dataclass
class CentroidDiffStrategy:
    include_centroid: bool = False

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        """Finds the centroid difference for each _x, _y, (_z) column."""

        # Exclude _likelihood columns
        df = df.select([col for col in df.columns if not col.endswith("_likelihood")])

        # Add centroid columns
        df = df.with_columns(
            centroid_x=pl.concat_list(pl.col("^.*_x$")).list.mean(),
            centroid_y=pl.concat_list(pl.col("^.*_y$")).list.mean(),
        )

        if df.select(pl.col("^.*_z$")).width > 0:
            df = df.with_columns(
                centroid_z=pl.concat_list(pl.col("^.*_z$")).list.mean(),
            )

        # Subtract centroid from matching columns
        for axis in ['x', 'y', 'z']:
            axis_cols = df.select(pl.col(f"^.*_{axis}$")).columns
            if not axis_cols:
                continue

            df = df.with_columns([
                (pl.col(col) - pl.col(f"centroid_{axis}")).alias(col) for col in axis_cols
            ])

        # Optionally drop centroid columns
        if not self.include_centroid:
            df = df.drop(["centroid_x", "centroid_y", "centroid_z"], strict=False)

        return df

@dataclass
class VelocityStrategy:
    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        """Returns a DataFrame containing only the velocity (Euclidean) for each body part."""

        # Identify all body part prefixes (excluding _likelihood)
        body_parts = set()
        for col in df.columns:
            if col.endswith("_x") and not col.endswith("_likelihood"):
                body_parts.add(col[:-2])

        velocity_exprs = []

        for part in body_parts:
            x_col = f"{part}_x"
            y_col = f"{part}_y"
            has_z = f"{part}_z" in df.columns

            dx = pl.col(x_col).diff()
            dy = pl.col(y_col).diff()

            if has_z:
                dz = pl.col(f"{part}_z").diff()
                velocity = (dx**2 + dy**2 + dz**2).sqrt().alias(f"{part}_velocity")
            else:
                velocity = (dx**2 + dy**2).sqrt().alias(f"{part}_velocity")

            velocity_exprs.append(velocity)

        # Return only velocity columns and fill missing values
        df = df.select(velocity_exprs)
        df = df.fill_null(strategy="backward")
        df = df.fill_null(strategy="forward")

        return df
    
@dataclass
class VelocityFilterStrategy:
    percentile_threshold: float = 0.75

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        """Filters out rows where the velocity is below a certain threshold."""
        body_parts = [col[:-2] for col in df.columns if col.endswith("_x")]

        # Compute per-part velocity columns
        velocity_exprs = []
        for part in body_parts:
            dx = pl.col(f"{part}_x").diff()
            dy = pl.col(f"{part}_y").diff()

            if f"{part}_z" in df.columns:
                dz = pl.col(f"{part}_z").diff()
                vel = ((dx**2 + dy**2 + dz**2).sqrt()).alias(f"{part}_velocity")
            else:
                vel = ((dx**2 + dy**2).sqrt()).alias(f"{part}_velocity")
            
            velocity_exprs.append(vel)

        # Add velocity columns
        df = df.with_columns(velocity_exprs)

        # Compute row-wise mean velocity across all parts
        velocity_cols = [f"{part}_velocity" for part in body_parts]
        avg_velocity : pl.DataFrame = df.select(pl.mean_horizontal([pl.col(col) for col in velocity_cols]).alias("average_velocity"))


        q3 = avg_velocity.select(pl.col("average_velocity").quantile(self.percentile_threshold, interpolation="nearest")).item()
        mask = avg_velocity.select((pl.col("average_velocity") < q3).alias("avg_velocity_filter")).fill_null(False)

        
        return mask
        

@dataclass
class PairwiseDistanceStrategy:
    pairwise_list: list[list[str]]

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        """Finds the Euclidean distance for each keypoint pair in 2D or 3D if _z is present."""

        distance_exprs = []

        for pair in self.pairwise_list:
            first_keypoint = pair[0]
            second_keypoint = pair[1]

            first_x = pl.col(f"{first_keypoint}_x")
            first_y = pl.col(f"{first_keypoint}_y")
            second_x = pl.col(f"{second_keypoint}_x")
            second_y = pl.col(f"{second_keypoint}_y")

            # Start with 2D distance
            squared_diff = (first_x - second_x) ** 2 + (first_y - second_y) ** 2

            # Add 3D distance if _z columns exist
            z1_col = f"{first_keypoint}_z"
            z2_col = f"{second_keypoint}_z"
            if z1_col in df.columns and z2_col in df.columns:
                first_z = pl.col(z1_col)
                second_z = pl.col(z2_col)
                squared_diff += (first_z - second_z) ** 2

            distance = squared_diff.sqrt().alias(f"{first_keypoint}-{second_keypoint}_pairwise_distance")
            distance_exprs.append(distance)

        # Return only the distance columns
        return df.select(distance_exprs)