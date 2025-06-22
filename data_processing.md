# DLC Clustering - Data Processing Strategies

This document outlines the various data processing strategies implemented for handling DeepLabCut pose estimation data. These strategies are modular, allowing users to build flexible pipelines for filtering, transforming, and extracting features from pose estimation time series.

Each strategy implements a `process(df: pl.DataFrame) -> pl.DataFrame` method, and they can be composed together in a list passed to a `Project`.

## How Strategies Are Used

Strategies are applied sequentially on each video’s pose DataFrame via the `Project` pipeline. You configure them using the `data_processing_strategies` list, which is passed to a `Project` instance. Filtering strategies typically return boolean masks, which are used to filter frames, while transformation strategies modify or enrich the input DataFrame.


## **Data Processing Strategies**

### `KeepOriginalStrategy`

Keeps the original coordinates (`_x`, `_y`, `_z`) and optionally `_likelihood` columns.
`CentroidDiffStrategy` should generally be used over this for clustering

**Parameters**:
- `include_likelihood`: If `True`, includes likelihood values.

---

### `CentroidDiffStrategy`

Centers each frame’s coordinates by subtracting the centroid position of all visible keypoints.

**Parameters**:
- `include_centroid`: Whether to retain centroid columns like `centroid_x`.

---

### `VelocityStrategy`

Calculates frame-to-frame Euclidean velocity for each keypoint.

**Returns**:
- A DataFrame with `[keypoint]_velocity` columns.

---

### `PairwiseDistanceStrategy`

Computes the pairwise Euclidean distance between selected keypoints.

**Parameters**:
- `pairwise_list`: A list of `[keypoint1, keypoint2]` pairs.

**Returns**:
- A DataFrame with columns like `nose-tail_base_pairwise_distance`.

---

### `AnglesStrategy`

Calculates angles between three keypoints (A-B-C), where B is the vertex.

**Parameters**:
- `angles_list`: Triplets like `[['nose', 'tail_base', 'tail_end']]`.
- `normalize`: If `True`, angle is normalized to range [0, 1].

---

### `SpectrogramBoutStrategy`

Generates frequency-domain features using FFT-based spectrograms over short pose sequences (bouts). Can be applied to both keypoints and angles.

**Parameters**:
- `keypoints`: Keypoints to analyze.
- `fs`: Sampling rate in Hz.
- `segment_length`: FFT segment window size.
- `num_overlap_samples`: Number of overlapping samples.
- `bout_length`: Bout size in frames.
- `stride`: Step size between bouts.
- `smoothing`: Whether to apply Savitzky-Golay filter.
- `smoothing_window_size`: Size of smoothing window.
- `smoothing_polynomial_order`: Order of the smoothing polynomial.
- `angles`: Optional angle triplets to process spectrally.

**Returns**:
- A DataFrame with `spectro_*` columns interpolated to match the original frame count.

---
## **Filtering Strategies**

### `VelocityFilterStrategy`

Filters out frames with low overall movement based on average velocity, using:
- a **percentile threshold**, and/or
- a **fixed velocity threshold**.

**Parameters**:
- `percentile_threshold` *(Optional[float])*: Drops frames below this velocity percentile.
- `velocity_threshold` *(Optional[float])*: Drops frames below this fixed velocity.

**Returns**:
- A boolean mask DataFrame (`velocity_filter`) marking valid frames.

---

### `ConfidenceFilterStrategy`

Filters out frames with low confidence based on keypoint likelihood values, using:
- a **percentile threshold**, and/or
- a **fixed likelihood threshold**.

**Parameters**:
- `percentile_threshold` *(Optional[float])*: Drops frames below this likelihood percentile.
- `likelihood_threshold` *(Optional[float])*: Drops frames below this fixed threshold.

**Returns**:
- A boolean mask DataFrame (`confidence_filter`) marking valid frames.
