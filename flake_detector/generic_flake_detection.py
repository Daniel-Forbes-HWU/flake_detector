"""
This module provides a a method (`find_flakes`) to detect clusters within an image using HDBSCAN clustering.
The result of `find_flakes` in a dataclass (`FlakeFindingResult`),
 which provides access to the data of each cluster and methods to manipulate them.
Dan Forbes - March 2025
"""
from time import perf_counter
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy import stats

try:
    _start = perf_counter()
    from fast_hdbscan_cached import HDBSCAN  # This imports quickly (after 1st run) and runs very quickly.
    _t_taken = perf_counter() - _start
    _mode = "fast_hdbscan_cached"
except ImportError:
    _start = perf_counter()
    from fast_hdbscan import HDBSCAN  # This is slow to import, but runs very quickly.
    _t_taken = perf_counter() - _start
    _mode = "fast_hdbscan"
    # from sklearn.cluster import HDBSCAN  # This is fast to import, but runs much more slowly.
    # Sklearn has min_samples -= 1 vs fast_hdbscan

print(f"Using HDBSCAN implementation: {_mode} (import time: {_t_taken:.2f}s)")
if _mode == "fast_hdbscan":
    print("Cached Implementation unavailable. Try installing 'fast_hdbscan_cached' for faster import times. See README.md for more information.")


@dataclass
class FlakeFindingResult:
    """
    Stores the result of a flake finding operation using HDBSCAN clustering.

    The error mask, or pixels classified as noise, are labelled as -1.

    Attributes:
    label_img: A 2D array containing cluster labels for each pixel in the image.
    probabilities_img: A 2D array containing the probability of each pixel belonging to it's cluster.
    fit_result: The HDBSCAN object returned from the clustering operation (HDBSCAN().fit()).
    source_image: A copy of the original image used for clustering, not including normalisation.
        If not provided, this is None.

    Computed properties:
    n_clusters: The number of clusters found in the image.
    labels: The cluster labels sorted by decreasing cluster size.
    cluster_sizes: The size of each cluster sorted by decreasing cluster size.

    Methods:
    masks(return_error_mask: bool = False) -> list[np.ndarray[bool]]: Returns a list of masks for each cluster in the image, sorted by decreasing cluster size.
        By default, the error mask (-1) is excluded from the list of masks.
    mask(label) -> np.ndarray[bool]: Returns a mask for the specified cluster label.

    get_cluster_values(label) -> np.ndarray[uint8 | uint16]: Returns a flat array of values underlying the mask of the specified cluster from the original image
        Raises an AttributeError if the source image is not available.

    remove_cluster(label): Sets the cluster with the specified label to the error mask.

    merge_clusters(old_label, new_label): Changes the label of the specified cluster to the new label.
        NOTE This operation invalidates the cluster probabilities of merged clusters.

    filter_by_size(min_size_px): Sets clusters with an area less than `min_size_px` to the error mask.

    find_substrate_by_img_mode() -> tuple[int, np.ndarray[uint8 | uint16]]: Attempts to locate the substrate peak by finding the cluster with
        mean value being closest to the source_image's modal value.
        Returns the label of the cluster closest to the mode and the mode value.

    find_alternate_substrate_labels(substrate_label) -> list[int]: Returns a list of labels which are similar in appearance to the substrate cluster.
        NOTE This is relatively slow compared to the other methods with the current implementation!

    Note:
    The cluster size ordering of the labels and cluster_sizes properties may not be maintained when clusters are removed or merged.
    """
    # Init properties
    label_img: npt.NDArray[np.integer] = field(repr=False)
    probabilities_img: npt.NDArray[np.floating] = field(repr=False)
    fit_result: HDBSCAN
    source_image: npt.NDArray[np.uint8 | np.uint16] | None = field(default=None, repr=False)

    # Computed properties
    n_clusters: int = field(init=False)
    labels: npt.NDArray[np.integer] = field(init=False)  # Unique labels sorted by decreasing cluster size
    cluster_sizes: npt.NDArray[np.integer] = field(init=False)  # Cluster sizes sorted by decreasing cluster size

    def __post_init__(self):
        unique_labels, cluster_sizes = np.unique(self.label_img, return_counts=True)

        # Sort unique labels and clusters sizes by decreasing cluster size
        sort_idx = np.argsort(cluster_sizes)[::-1]
        self.labels = unique_labels[sort_idx]
        self.cluster_sizes = cluster_sizes[sort_idx]

        self.n_clusters = len(self.labels)

    def masks(self, return_error_mask: bool = True) -> list[npt.NDArray[np.bool_]]:
        """Returns a list of masks for each cluster in the image."""
        unique_labels: list[int] = self.labels.tolist()
        if not return_error_mask:
            unique_labels.remove(-1)
        return [
            self.label_img == label
            for label in unique_labels
        ]

    def mask(self, label: int) -> npt.NDArray[np.bool_]:
        return self.label_img == label

    def get_cluster_values(self, label: int) -> npt.NDArray[np.uint8 | np.uint16]:
        """Returns a flat array of values underlying the mask of the specified cluster from the original image.
        Raises an AttributeError if the source image is not available"""
        if self.source_image is None:
            raise AttributeError("Source image not provided.")
        return self.source_image[self.mask(label)]

    def remove_cluster(self, label: int):
        """Sets the cluster with the specified label to the error mask."""
        # Set the cluster to the error mask
        label_mask = self.mask(label)
        self.label_img[label_mask] = -1
        self.probabilities_img[label_mask] = 0

        cluster_idx = self.labels == label

        # Update the error cluster size
        self.cluster_sizes[self.labels == -1] += self.cluster_sizes[cluster_idx]

        # Remove the cluster from the list of labels and cluster sizes
        self.labels = np.delete(self.labels, cluster_idx)
        self.cluster_sizes = np.delete(self.cluster_sizes, cluster_idx)
        self.n_clusters -= 1

    def merge_clusters(self, old_label: int, new_label: int):
        """Changes the label of the specified cluster to the new label.
        NOTE This operation invalidates the cluster probabilities of merged clusters."""
        # Check if the labels are valid
        if old_label not in self.labels:
            raise ValueError(f"Old label {old_label} is not a valid cluster label.")
        if new_label not in self.labels:
            raise ValueError(f"New label {new_label} is not a valid cluster label.")

        # Update the label image
        self.label_img[self.mask(old_label)] = new_label

        # Update target cluster_size
        old_cluster_idx, new_cluster_idx = (self.labels == old_label), (self.labels == new_label)
        old_cluster_size = self.cluster_sizes[old_cluster_idx]
        self.cluster_sizes[new_cluster_idx] += old_cluster_size

        # Remove the old cluster
        self.labels = np.delete(self.labels, old_cluster_idx)
        self.cluster_sizes = np.delete(self.cluster_sizes, old_cluster_idx)
        self.n_clusters -= 1

    def filter_by_size(self, min_size_px: int):
        """Sets clusters with an area less than `min_size_px` to the error mask."""
        # Iterate over all clusters
        for label, size in zip(self.labels, self.cluster_sizes):
            if (size < min_size_px) and (label != -1):
                self.remove_cluster(label)

    def find_substrate_by_img_mode(self) -> tuple[int, npt.NDArray[np.uint8 | np.uint16]]:
        """Attempts to locate the substrate peak by finding the cluster with
        mean value being closest to the source_image's modal value.
        Returns the label of the cluster closest to the mode and the mode value."""
        mode = stats.mode(self.source_image, axis=(0, 1)).mode
        cluster_means = np.array([np.mean(self.get_cluster_values(label), axis=0) for label in self.labels])
        assert len(self.labels) == len(cluster_means), f"Size mismatch: {len(self.labels)} != {len(cluster_means)}"
        substrate_label_idx: int = np.argmin(np.abs(cluster_means - mode), axis=0)[0]  # Only take the scalar index
        return self.labels[substrate_label_idx], mode

    def find_alternate_substrate_labels(self, substrate_label: int) -> list[int]:
        """Returns a list of labels which are similar in appearance to the substrate cluster."""
        substrate_values = self.get_cluster_values(substrate_label)
        substrate_mean = np.percentile(substrate_values, 50, axis=0)
        substrate_stdev = np.std(substrate_values, axis=0)
        return [
            label
            for label in self.labels
            if label != substrate_label
            and np.allclose(
                substrate_mean,
                np.mean(self.get_cluster_values(label), axis=0),
                atol=3*substrate_stdev
            )
        ]


def get_indices_for_image_shape(
        image_shape: tuple[int, int],
        dtype: npt.DTypeLike = np.uint8) -> np.ndarray:
    """Returns an (m*n, 2) array of indices for an image of shape image_shape=(m, n).
    Can be included in feature vectors for clustering, etc."""
    return np.indices(image_shape, dtype=dtype).transpose(1, 2, 0)


def find_flakes(
        image: np.ndarray,
        min_cluster_size: int | None = 13,
        min_samples: int | None = 30,
        include_pixel_index: bool = True,
        norm_image: bool = False,
        norm_pixel_index: bool = False,
        return_image: bool = True,
        normalised_dtype: npt.DTypeLike = np.float64,
        mask: npt.NDArray[np.bool_] | None = None,
        **cluster_kwargs
        ) -> FlakeFindingResult:
    """
    Performs HDBSCAN clustering to find regions of similar appearance within the image.
    For more detailed information, see the [HDBSCAN documentation](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html)


    Args:
    image: The image to cluster.
    min_cluster_size: The smallest sized grouping of pixels which is considered a cluster.
    min_samples: Restricts the minimum density of allowed clusters, higher values will result in fewer clusters.
    include_pixel_index: Whether to include the pixel index in the feature vector for clustering.
        If True (default), the clustering operates 'spectrally' (RGB) and spatially (XY).
        If False, the clustering operates 'spectrally' (RGB) only.
    norm_image: Whether to normalise the image to the range [0, 1] before clustering.
    norm_pixel_index: Whether to normalise the pixel index to the range [0, 1] before clustering.
    return_image: Whether to return the source image in the result object,
        disabling this can improve multiprocessed performance by avoiding pickling the image in order
        to send it across process boundaries.
    normalised_dtype: The dtype to use for normalised images.
    mask: A boolean mask of the same shape as the image, where True values indicate pixels to include in the clustering.

    **cluster_kwargs: Additional keyword arguments to pass to the HDBSCAN constructor.

    Returns:
    FlakeFindingResult: A dataclass containing the results of the clustering operation.
    NOTE: When a mask is used, the cluster labels for excluded pixels is set to -2.
    """
    # Convert to float64 and normalise the image
    if norm_image:
        to_fit_image = image.astype(normalised_dtype, copy=True)
        _min, _max = to_fit_image.min(), to_fit_image.max()
        to_fit_image -= _min
        to_fit_image /= (_max - _min)
    else:
        to_fit_image = image

    # Generate pixel indices, if used
    if include_pixel_index:
        image_indices = get_indices_for_image_shape(to_fit_image.shape[:2], dtype=to_fit_image.dtype)

        # Normalise the pixel index
        if norm_pixel_index:
            image_indices /= to_fit_image.shape[:2]

    # Get the mask for used pixels
    if mask is None:
        mask = np.ones(to_fit_image.shape[:2], dtype=bool)
    else:  # Validate the mask
        if mask.shape[:2] != to_fit_image.shape[:2]:
            raise ValueError("Mask shape does not match image shape.")

    if include_pixel_index:
        to_fit = np.column_stack((to_fit_image[mask], image_indices[mask]))
    else:
        to_fit = to_fit_image[mask]

    # Fit the HDBSCAN model
    result = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        # metric="euclidean",
        **cluster_kwargs,
    ).fit(to_fit)

    # Reshape results into original image shape
    label_img = np.full(to_fit_image.shape[:2], -2, dtype=result.labels_.dtype)
    label_img[mask] = result.labels_
    prob_img = np.zeros(to_fit_image.shape[:2], dtype=result.probabilities_.dtype)
    prob_img[mask] = result.probabilities_

    return FlakeFindingResult(
        label_img=label_img,
        probabilities_img=prob_img,
        fit_result=result,
        source_image=image.copy() if return_image else None
    )
