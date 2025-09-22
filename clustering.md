# DLC Clustering - Clustering Pipelines
The document describes the DLC data clustering pipelines composed of modular **preprocessors**, **clusterers**, and **classifiers**. These pipelines work on segmented sequences of frames (bouts) to group similar behaviors.

Each pipeline processes the data and assigns a `cluster` label and a `bout_id` to each relevant frame in the dataset.


## Core Concepts

### What is a Bout?
A bout is a short, continuous sequence of frames (e.g., 15 frames) that serves as the fundamental unit for clustering. Instead of clustering individual frames, we extract temporal patterns over these bouts to better capture motion and behavior.

### What is a Pipeline?
A pipeline is a sequence of data processing steps. You can combine different preprocessing steps (like scaling and dimensionality reduction) with a final clustering or classification algorithm to create a custom clustering workflow.

* `UnsupervisedBoutPipeline`: For discovering patterns in data without pre-existing labels.
* `SupervisedBoutPipeline`: For classifying behaviors using a model trained on user provided labels.


## Unsupervised Clustering Pipeline

The `UnsupervisedBoutPipeline` is designed for exploratory data analysis where you want the algorithm to discover behavioral groups automatically. It chains together one or more **preprocessors** with a final **clusterer**.

### Pipeline Steps
1.  **Bout Extraction**: Data is segmented into bouts of a specified `bout_length` and `stride`.
2.  **Feature Flattening**: The time series data for each bout is flattened into a single feature vector.
3.  **Preprocessing**: Each preprocessor is fit and applied to the feature vectors (e.g., scaling, then PCA).
4.  **Clustering**: The final clusterer (`KMeans` or `HDBSCAN`) assigns a cluster label to each bout's processed feature vector.

### Key Parameters
-   `preprocessors`: A list of preprocessor objects to apply in sequence.
-   `clusterer`: The clustering algorithm object.
-   `bout_length`: Number of consecutive frames per bout (default: 15).
-   `stride`: Step size for the sliding window when extracting bouts (default: 5).


## Supervised Clustering Pipeline

The `SupervisedBoutPipeline` uses labeled data to train a classifier. Once trained, this classifier can predict behavioral labels for all the data in the project. This approach is powerful for consistently identifying known behaviors.

### Pipeline Steps
1.  **Prepare Dataset**: The `train()` method first calls `prepare_supervised_dataset`, which extracts bouts that have corresponding user-provided labels.
2.  **Train-Test Split**: The labeled data is split into training and testing sets to evaluate model performance.
3.  **Fit Preprocessors**: Preprocessors are fit on the training data.
4.  **Train Classifier**: The classifier is trained on the preprocessed training data and labels. A classification report is printed to show performance on the test set.
5.  **Process Project Data**: After training, the `process()` method runs the entire project's data (labeled and unlabeled) through the fitted preprocessors and the trained classifier to assign a cluster label to every bout.

### Key Parameters
-   `preprocessors`: A list of preprocessor objects.
-   `classifier`: The classification algorithm object.
-   `bout_length`: Number of consecutive frames per bout.
-   `stride`: Step size for the sliding window.
-   `test_size`: Fraction of labeled data to use for the test set.


## Pipeline Components

You can mix and match these components to build your desired pipeline.

### Preprocessors
These modules transform the feature space. A standard first step is `SKStandardScaler`.
-   `SKStandardScaler`: Standardizes features by removing the mean and scaling to unit variance. **Highly recommended.**
-   `SKPCA`: **Principal Component Analysis**. A fast, linear dimensionality reduction technique.
    -   `n_components`: Number of principal components to keep.
-   `UMAPReducer`: **Uniform Manifold Approximation and Projection**. A powerful non-linear technique for dimensionality reduction.
    -   `n_components`: Number of dimensions for the embedded space.
    -   `n_neighbors`: The number of neighboring points used for manifold approximation.
    -   `min_dist`: The minimum distance between embedded points.

### Unsupervised Clusterers
These are used in the `UnsupervisedBoutPipeline`.
-   `SKKMeans`: **K-Means Clustering**. Partitions data into a pre-determined number of clusters.
    -   `n_clusters`: The number of clusters to form.
-   `HDBSCANClusterer`: **Hierarchical Density-Based Spatial Clustering**. A robust algorithm that can find clusters of varying shapes and densities, and marks noise points as unclustered (`-1`).
    -   `min_cluster_size`: The minimum size of a group to be considered a cluster.

### Supervised Classifiers
These are used in the `SupervisedBoutPipeline`.
-   `SKRandomForest`: A **Random Forest** classifier.
    -   `n_estimators`: The number of trees in the forest.
-   `SKGradientBoosting`: A **Gradient Boosting** classifier, often more accurate but slower to train than Random Forest.
-   `SKSVM`: A **Support Vector Machine** classifier.
    -   `kernel`: The type of kernel to use (`'rbf'`, `'linear'`, etc.).
    -   `C`: Regularization parameter.
-   `SKKNN`: A **K-Nearest Neighbors** classifier.
    -   `n_neighbors`: Number of neighbors to use for classification.
-   `SKMLP`: A **Multi-Layer Perceptron** (neural network) classifier.
    -   `hidden_layer_sizes`: A tuple defining the number of neurons in each hidden layer (e.g., `(100, 50)`).
    -   `max_iter`: Maximum number of training iterations.


## Common Output Columns
-   `cluster`: The final cluster assignment for each frame. A value of `-1` means the frame was unassigned (e.g., filtered out or marked as noise by HDBSCAN).
-   `bout_id`: An integer ID identifying which bout a frame belongs to.


## Notes
-   **Filtering**: Any column in your data ending with `_filter` will be used to exclude frames before bout creation.
-   **Missing Values**: All missing values (NaNs) in numeric feature columns are converted to zero before processing.
-   **Accessing Outputs**: You can access clustering results via `project.get_cluster_output()` or save them to CSV files with `project.save_clustering_output()`.


## Example Usage

### Unsupervised: PCA + K-Means
```python
from your_module import UnsupervisedBoutPipeline, SKStandardScaler, SKPCA, SKKMeans

# Define the pipeline
pipeline = UnsupervisedBoutPipeline(
    preprocessors=[SKStandardScaler(), SKPCA(n_components=2)],
    clusterer=SKKMeans(n_clusters=5),
    bout_length=15,
    stride=1
)

# Run the pipeline on your project data
pipeline.process(project)