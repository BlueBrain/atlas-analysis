""" Collection of free functions to visualize atlases and derived files
"""
import logging
import warnings
import numpy as np
from skimage.transform import downscale_local_mean
import plotly.graph_objects as go

from atlas_analysis import AtlasAnalysisError

L = logging.getLogger(__name__)
logging.captureWarnings(True)

# Flatmap


def downscale(raw, resolution):
    """
    Downscale a 2D scalar array.

    Args:
        raw(numpy.ndarray): bool|int|float 2D array.
        resolution(int): number of pixels requested for the width of the downscaled array. The
             height is downscaled using the same scaling factor. The aspect ratio is thus
             preserved, 'as musch as possible'. (The aspect ratio is left unchanged only if the
             uniform scaling factor divides both dimensions).
    Returns:
        2D float numpy.narray
    Raises:
        AtlasAnalysisError if `resolution` is larger than `raw.shape[0]`, i.e., the user wrongly
         requested an upscale.
    """
    assert isinstance(resolution, int) and resolution > 0
    scaling_factor = raw.shape[0] // resolution
    if scaling_factor == 0:
        warnings.warn(
            'The requested resolution exceeds the length along the x-axis of the array to '
            f'downscale:\n resolution: {resolution}, length: {raw.shape[0]}.\n'
            'Using full length resolution.',
            UserWarning,
        )
        return raw
    return downscale_local_mean(raw, (scaling_factor, scaling_factor))


def compute_flatmap_image(flatmap_raw):
    """ Compute the binary image of the flatmap.

    This function returns a 2D boolean array representing the flatmap image.
    A pixel is white (True) if it is the image of a voxel by the
    flatmap. Otherwise the pixel is black (False).

    Args:
       flatmap_raw(numpy.ndarray): integer array of shape
            (l, w, h, 2), to be interpreted as a map from a 3D volume
            to a 2D rectangle image.
    Returns:
        boolean numpy.ndarray of shape (W, H) where W and H are
        the maximal values, augmented by 1,
        of the flatmap with respect to its last axis.
    """
    assert issubclass(flatmap_raw.dtype.type, np.integer)
    in_mask = np.all(flatmap_raw >= 0, axis=-1)
    pixel_indices = flatmap_raw[in_mask].T
    image_shape = np.max(pixel_indices, axis=1) + 1
    image = np.zeros(image_shape, dtype=np.bool)
    image[tuple(pixel_indices)] = True
    return image


def compute_flatmap_histogram(flatmap_raw):
    """ Compute the fiber volume histogram of the flatmap.

    We call the set of voxels mapping to the same pixel the `fiber`
    of the flatmap over this pixel.

    This function returns a 2D integer array representing the volumes of the
    flatmap fibers over each pixel in its image.

    Args:
       flatmap_raw(numpy.ndarray): integer array of shape
            (l, w, h, 2), to be interpreted as a map from a 3D volume
            to a 2D rectangle image.
    Returns:
        integer array of shape (L, W) where L and W are
        the maximal values, augmented by 1,
        of the flatmap with respect to its last axis.
    """
    assert issubclass(flatmap_raw.dtype.type, np.integer)
    in_mask = np.all(flatmap_raw >= 0, axis=-1)
    pixel_coordinates, counts = np.unique(
        flatmap_raw[in_mask], axis=0, return_counts=True
    )
    image_shape = np.max(pixel_coordinates, axis=0) + 1
    histogram = np.zeros(image_shape, dtype=np.int)
    histogram[tuple(pixel_coordinates.T)] = counts
    return histogram


def flatmap_image_figure(voxeldata, resolution=None):
    """ Display the flatmap image.

    Display the flatmap image as a black and white picture.
    If the flatmap image is downsized by specifying a lower resolution,
    the displayed image is a greyscale image.

    The `resolution` parameter is the target width of the image to display, i.e.,
    its length in terms of pixels along the x-axis.

    Args:
        voxeldata(VoxelData): object holding the multi-dimensional array of the flatmap.
        resolution(None|int): Optional resolution parameter used to downsample the image.
            If the resolution is not None, the flatmap image will be downsampled to a rectangular
            grid of dimensions (`resolution`, `int(aspect_ratio * resolution)`)
            with square bins of equal size. The float `aspect_ratio` is the aspect ratio of the
            flatmap image, i.e. height / width.
    """
    assert resolution is None or resolution > 0
    image = compute_flatmap_image(voxeldata.raw)
    if resolution is not None:
        image = downscale(image, resolution)
    # Boolean arrays are not rendered properly by go.Heatmap.
    if image.dtype == 'bool':
        image = image.astype(int)

    L.info(f'Flatmap image dimensions: {image.shape}')

    return go.Figure(data=go.Heatmap(z=image))


def flatmap_volume_histogram(voxeldata, resolution=None):
    """ Display the histogram of volumes lying over pixels.

    The volume, i.e., the number of voxels, which is mapped to the same pixel is depicted
    by means of an elevation grid.

    If the flatmap image is downsized by specifying a lower resolution,
    the bar heights of the displayed histogram are non-negative floats.

    The `resolution` parameter is the target width of the image to display, i.e.,
    its length in terms of pixels along the x-axis.

    Args:
        voxeldata(VoxelData): object holding the multi-dimensional array of the flatmap.
        resolution(int|None): Optional resolution parameter used to downsample the histogram.
            If the resolution is not None, the flatmap image will be downscaled to a rectangular
            grid of dimensions (`resolution`, `int(aspect_ratio * resolution)`)
            with square bins of equal size. The float `aspect_ratio` is the aspect ratio of the
            flatmap image, i.e., height / width.
    """
    assert resolution is None or resolution > 0
    histogram = compute_flatmap_histogram(voxeldata.raw)
    if resolution is not None:
        histogram = downscale(histogram, resolution)
    figure = go.Figure(data=[go.Surface(z=histogram,)])
    figure.update_layout(
        title='Flatmap Fibers Elevation Grid', autosize=True,
    )
    return figure
