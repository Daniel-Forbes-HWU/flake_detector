from time import perf_counter

import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt

from flake_detector import find_flakes, FlakeFindingResult


def example_detect():
    # Load the image
    image = tiff.imread("A2_O7_5x_Unprocessed White Light.tif")

    # Generate a mask
    mask: None | np.ndarray = np.zeros(image.shape[:2], dtype=bool)
    mask[20:170, 20:200] = True  # Square mask

    # Detect flakes
    time_before = perf_counter()
    result: FlakeFindingResult = find_flakes(
        image,
        min_cluster_size=13, min_samples=30,  # The defaults in `find_flakes`
        include_pixel_index=True,  # Includes the XY information for each pixel
        norm_image=True, norm_pixel_index=True,  # Normalises the cluster inputs
        return_image=True,  # Adds the input image to the returned cluster results. Can make things slow if the result passed across process boundaries, and will take up more space, but it's convenient.
        mask=mask  # Only detect pixels where the mask is True. Excluded pixels are labelled -2.
    )
    time_taken = perf_counter() - time_before
    print(f"Clustered image with {image.shape[0] * image.shape[1]:,} px in {time_taken} s")
    if mask is not None:
        print(f"With mask, {mask.sum():,} px clustered.")
    print(f"Detected {result.n_clusters} clusters!")

    # Unique labels, sorted by decreasing cluster size
    unique_labels = result.labels

    # Gets all the the masks, sorted by decreasing cluster size
    mask_list = result.masks()

    # You can find the substrate cluster by finding the most common RGB value
    substrate_label, substrate_mode = result.find_substrate_by_img_mode()

    # You can get a specific mask
    substrate_mask = result.mask(substrate_label)

    # You can access the original hdbscan result
    HDBSCAN_result = result.fit_result

    # This is VERY slow, but can find multiple clusters with similar appearance to the substrate
    # alternative_substrate_labels = result.find_alternate_substrate_labels(substrate_label)

    # You can get a flattened array of cluster pixel values
    substrate_pixels = result.get_cluster_values(substrate_label)

    # You can remove a cluster - Set to error cluster
    # result.remove_cluster(substrate_label)

    # You can access the source image, if `return_image=True` when calling `find_flakes`
    source_image: np.ndarray | None = result.source_image
    assert np.array_equal(source_image, image), "Source image is not the same as the input image"

    # You could get the (flattened) probabilities for a cluster
    substrate_probs = result.probabilities_img[result.mask(substrate_label)]

    # You can filter the result by size (inplace)
    # result.filter_by_size(100)

    # You can merge clusters, this does not modify the HDBSCAN_result above, and invalidates probabilities somewhat
    # result.merge_clusters(1, 2)

    # Display the results
    fig, (ax_l, ax_r) = plt.subplots(1, 2)
    ax_l.imshow(image)
    ax_r.imshow(result.label_img, alpha=result.probabilities_img, cmap="tab20")  # Be careful with having more than 20 clusters here!
    plt.show()


if __name__ == "__main__":
    example_detect()
