import polars as pl
import polars.selectors as cs
from dataclasses import dataclass
from typing import Protocol
from typing import List
import numpy as np
from scipy.signal import savgol_filter
from scipy.signal import ShortTimeFFT
from typing import List, Optional
    


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
class CentroidVelocityStrategy:
    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Returns a DataFrame containing only the velocity components
        (x, y, z) and speed (Euclidean) for the body's centroid.
        """

        has_z = any(col.endswith("_z") for col in df.columns)

        cvx = pl.mean_horizontal(cs.ends_with("_x")).diff()
        cvy = pl.mean_horizontal(cs.ends_with("_y")).diff()

        select_exprs = [
            cvx.alias("centroid_velocity_x"),
            cvy.alias("centroid_velocity_y"),
        ]

        if has_z:
            cvz = pl.mean_horizontal(cs.ends_with("_z")).diff()
            select_exprs.append(cvz.alias("centroid_velocity_z"))
            
            speed_expr = (
                (cvx**2 + cvy**2 + cvz**2)
                .sqrt()
                .alias("centroid_speed")
            )
        else:
            speed_expr = (
                (cvx**2 + cvy**2)
                .sqrt()
                .alias("centroid_speed")
            )
            
        select_exprs.append(speed_expr)

        df = df.select(select_exprs)

        df = df.fill_null(strategy="backward")
        df = df.fill_null(strategy="forward")

        return df

@dataclass
class VelocityFilterStrategy:
    percentile_threshold: Optional[float] = None
    velocity_threshold: Optional[float] = None

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        """Filters out rows based on both percentile and fixed velocity thresholds.
        Returns a mask where rows meet all specified velocity criteria.
        """

        if self.percentile_threshold is None and self.velocity_threshold is None:
            raise ValueError("Either `percentile_threshold` or `velocity_threshold` must be set.")

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

        # Compute average velocity per frame
        velocity_cols = [f"{part}_velocity" for part in body_parts]
        avg_velocity_df = df.select(
            pl.mean_horizontal([pl.col(col) for col in velocity_cols]).alias("average_velocity")
        )

        # Start with all True
        mask = pl.Series("velocity_filter", [True] * avg_velocity_df.height)

        # Percentile-based filtering
        if self.percentile_threshold is not None:
            q = avg_velocity_df.select(
                pl.col("average_velocity").quantile(self.percentile_threshold, interpolation="nearest")
            ).item()

            percentile_mask = avg_velocity_df.select(
                (pl.col("average_velocity") >= q).alias("percentile_filter")
            ).fill_null(False)

            mask = mask & percentile_mask["percentile_filter"]

        # Fixed-threshold filtering
        if self.velocity_threshold is not None:
            threshold_mask = avg_velocity_df.select(
                (pl.col("average_velocity") >= self.velocity_threshold).alias("threshold_filter")
            ).fill_null(False)

            mask = mask & threshold_mask["threshold_filter"]

        return pl.DataFrame({"velocity_filter": mask})

        

@dataclass
class ConfidenceFilterStrategy:
    percentile_threshold: Optional[float] = None
    likelihood_threshold: Optional[float] = None

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        """Filters out rows based on both percentile and fixed likelihood threshold.
        Returns a mask where rows meet all specified confidence criteria.
        """

        # Extract all likelihood columns
        likelihood_cols = [col for col in df.columns if col.endswith("_likelihood")]
        if not likelihood_cols:
            raise ValueError("No likelihood columns found in DataFrame.")

        # Compute row-wise mean likelihood
        avg_likelihood_df = df.select(
            pl.mean_horizontal([pl.col(col) for col in likelihood_cols]).alias("average_likelihood")
        )

        mask = pl.Series("confidence_filter", [True] * avg_likelihood_df.height)

        if self.percentile_threshold is not None:
            q_threshold = avg_likelihood_df.select(
                pl.col("average_likelihood").quantile(self.percentile_threshold, interpolation="nearest")
            ).item()

            percentile_mask = avg_likelihood_df.select(
                (pl.col("average_likelihood") >= q_threshold).alias("percentile_filter")
            ).fill_null(False)

            mask = mask & percentile_mask["percentile_filter"]

        if self.likelihood_threshold is not None:
            threshold_mask = avg_likelihood_df.select(
                (pl.col("average_likelihood") >= self.likelihood_threshold).alias("threshold_filter")
            ).fill_null(False)

            mask = mask & threshold_mask["threshold_filter"]

        return pl.DataFrame({"confidence_filter": mask})    
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



@dataclass
class AnglesStrategy:
    angles_list: list[list[str]]
    normalize: bool = True  # If True, normalize angle to [0, 1]

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        angle_exprs = []
        epsilon = 1e-6  # avoid division by zero

        for a, b, c in self.angles_list:
            # Vector BA
            ba_x = pl.col(f"{a}_x") - pl.col(f"{b}_x")
            ba_y = pl.col(f"{a}_y") - pl.col(f"{b}_y")

            # Vector BC
            bc_x = pl.col(f"{c}_x") - pl.col(f"{b}_x")
            bc_y = pl.col(f"{c}_y") - pl.col(f"{b}_y")

            # Dot product and norms
            dot = ba_x * bc_x + ba_y * bc_y
            norm_ba = (ba_x ** 2 + ba_y ** 2).sqrt()
            norm_bc = (bc_x ** 2 + bc_y ** 2).sqrt()

            # Cosine of angle
            cos_theta = (dot / (norm_ba * norm_bc + epsilon)).clip(-1, 1)
            angle_rad = cos_theta.arccos()

            angle_col_name = f"angle_{a}_{b}_{c}"
            angle_expr = (
                (angle_rad / np.pi if self.normalize else angle_rad)
                .alias(angle_col_name)
            )
            angle_exprs.append(angle_expr)

        return df.select(angle_exprs)



@dataclass
class SpectrogramBoutStrategy:
    keypoints: List[str]
    fs: int = 30
    segment_length: int = 15
    num_overlap_samples: int = 10
    bout_length: int = 15
    stride: int = 1
    smoothing: bool = True
    smoothing_window_size: int = 7
    smoothing_polynomial_order: int = 2
    angles: Optional[List[List[str]]] = None  # Optional list of angles A-B-C

    # This code is crazy and should be refactored
    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        def get_bouts(n_frames: int) -> List[List[int]]:
            return [
                list(range(i, i + self.bout_length))
                for i in range(0, n_frames - self.bout_length + 1, self.stride)
            ]

        def compute_angle_series(a, b, c, df: pl.DataFrame) -> np.ndarray:
            # Create vectors BA and BC
            ba = np.stack([
                df[f"{a}_x"].to_numpy() - df[f"{b}_x"].to_numpy(),
                df[f"{a}_y"].to_numpy() - df[f"{b}_y"].to_numpy()
            ], axis=1)
            bc = np.stack([
                df[f"{c}_x"].to_numpy() - df[f"{b}_x"].to_numpy(),
                df[f"{c}_y"].to_numpy() - df[f"{b}_y"].to_numpy()
            ], axis=1)

            if all(f"{pt}_z" in df.columns for pt in [a, b, c]):
                ba = np.concatenate([ba, (df[f"{a}_z"].to_numpy() - df[f"{b}_z"].to_numpy())[:, None]], axis=1)
                bc = np.concatenate([bc, (df[f"{c}_z"].to_numpy() - df[f"{b}_z"].to_numpy())[:, None]], axis=1)

            dot = np.sum(ba * bc, axis=1)
            norm_ba = np.linalg.norm(ba, axis=1)
            norm_bc = np.linalg.norm(bc, axis=1)
            cos_theta = np.clip(dot / (norm_ba * norm_bc), -1.0, 1.0)
            angle_deg = np.arccos(cos_theta) * 180 / np.pi
            return angle_deg

        frames_features = []
        for frame_indices in get_bouts(len(df)):
            bout = df[frame_indices]
            feature_vector = []

            # Keypoint coordinate features
            for kp in self.keypoints:
                for axis in ["x", "y"]:
                    col = f"{kp}_{axis}"
                    if col not in bout.columns:
                        continue
                    signal = bout[col].to_numpy()

                    if self.smoothing and len(signal) >= self.smoothing_window_size:
                        signal = savgol_filter(
                            signal,
                            window_length=self.smoothing_window_size,
                            polyorder=self.smoothing_polynomial_order,
                        )

                    stfft = ShortTimeFFT.from_window(
                        ('tukey', 0.25),
                        fs=self.fs,
                        nperseg=self.segment_length,
                        noverlap=self.num_overlap_samples,
                        fft_mode='onesided',
                        scale_to='magnitude'
                    )
                    Sxx = stfft.spectrogram(signal, detr='constant')
                    Sxx_mean = Sxx.mean(axis=1)
                    feature_vector.extend(Sxx_mean)

            # Angle features
            if self.angles:
                for a, b, c in self.angles:
                    if any(f"{pt}_x" not in bout.columns or f"{pt}_y" not in bout.columns for pt in [a, b, c]):
                        continue
                    signal = compute_angle_series(a, b, c, bout)

                    if self.smoothing and len(signal) >= self.smoothing_window_size:
                        signal = savgol_filter(
                            signal,
                            window_length=self.smoothing_window_size,
                            polyorder=self.smoothing_polynomial_order,
                        )

                    stfft = ShortTimeFFT.from_window(
                        ('tukey', 0.25),
                        fs=self.fs,
                        nperseg=self.segment_length,
                        noverlap=self.num_overlap_samples,
                        fft_mode='onesided',
                        scale_to='magnitude'
                    )
                    Sxx = stfft.spectrogram(signal, detr='constant')
                    Sxx_mean = Sxx.mean(axis=1)
                    feature_vector.extend(Sxx_mean)

            repeated = {
                f"spectro_{i}": [val] * len(frame_indices) for i, val in enumerate(feature_vector)
            }
            repeated["frame_idx"] = frame_indices
            frames_features.append(pl.DataFrame(repeated))

        features_df = (
            pl.concat(frames_features, how="vertical")
            .sort("frame_idx")
            .unique(subset="frame_idx", keep="last")
        )

        # Ensure full coverage and interpolate
        full_frame_idx = pl.DataFrame({"frame_idx": list(range(len(df)))})
        full_df = full_frame_idx.join(features_df, on="frame_idx", how="left")

        # Maybe using polars was a mistake
        return full_df.select([
            pl.col("frame_idx"),
            *[
                pl.col(col).interpolate().fill_null(strategy="forward")
                for col in full_df.columns if col.startswith("spectro_")
            ]
        ])
