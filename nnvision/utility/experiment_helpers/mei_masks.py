import torch
import numpy as np
import scipy
from scipy import ndimage
from skimage import morphology
from skimage.transform import rescale


def generate_mask(mei, gaussian_sigma=1, zscore_thresh=0.25, closing_iters=2, return_centroids=False):

    if type(mei) == torch.Tensor:
        mei = mei.detach().cpu().numpy().squeeze()
        is_tensor = True
    else:
        is_tensor = False

    norm_mei = (mei - mei.mean()) / mei.std()
    thresholded = np.abs(norm_mei) > zscore_thresh

    # Remove small holes in the thresholded image and connect any stranding pixels
    closed = ndimage.binary_closing(thresholded, iterations=closing_iters)
    # Remove any remaining small objects
    labeled = morphology.label(closed, connectivity=2)
    most_frequent = np.argmax(np.bincount(labeled.ravel())[1:]) + 1
    oneobject = labeled == most_frequent
    # Create convex hull just to close any remaining holes and so it doesn't look weird

    hull = morphology.convex_hull_image(oneobject)
    # Smooth edges
    smoothed = ndimage.gaussian_filter(hull.astype(np.float32), sigma=gaussian_sigma)
    mask = smoothed  # final mask
    if not return_centroids:
        return torch.tensor(mask)[None, None, ...].cuda() if is_tensor else mask
    # Compute mask centroid
    px_y, px_x = (coords.mean() + 0.5 for coords in np.nonzero(hull))
    mask_y, mask_x = px_y - mask.shape[0] / 2, px_x - mask.shape[1] / 2
    # Compute MEI std inside the mask
    mei_mean = (mei * mask).sum() / mask.sum()
    mei_std = np.sqrt((((mei - mei_mean) ** 2) * mask).sum() / mask.sum())

    return mask, px_x, px_y, mask_x, mask_y