import pandas as pd
import polars as pl
import polars as pl
from dataclasses import dataclass

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