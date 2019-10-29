""" Collection of free functions to perform curation operations on atlases """
import numpy as np
from scipy import ndimage
from scipy.ndimage.morphology import generate_binary_structure
import voxcell


def remove_connected_components(voxeldata, threshold_size, connectivity=1):
    """ Remove the connected components whose sizes are below a specified threshold

      Args:
          voxeldata(VoxelData): VoxelData object holding the multi-dimensional array
          to be processed.
          threshold_size(int): Every connected components with no more than
          threshold_size voxels will be removed.
          connectivity(int): optional, integer value which defines what connected voxels are.
          Two voxels are connected if their squared distance
          is not greater than connectivity.
          If connectivity is 1, i.e., the default value, then
          two voxels are connected only if they share a common face, see
          https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.morphology.generate_binary_structure.html
          and
          https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.label.html
      Returns:
          filtered_voxeldata(VoxelData): a copy of the initial VoxelData object
                                         deprived of its 'small' connected components.
    """
    # As a raw array loaded by nrrd.read() is immutable when the input file is compressed
    # if pynrrd's version < 0.3.4, we need to make a deep copy of it.
    raw = np.copy(voxeldata.raw)

    # Extract all connected components
    structure = generate_binary_structure(3, connectivity)
    labeled_components, _ = ndimage.label(raw, structure=structure)

    # Compute the mask of the connected components to remove
    unique_labels, counts = np.unique(labeled_components, return_counts=True)
    labels_counts = np.array((unique_labels, counts)).T
    # pylint: disable=unsubscriptable-object
    labels_mask = labels_counts[:, 1] <= threshold_size
    labels_to_remove = labels_counts[labels_mask][:, 0]
    raw_mask = np.where(np.isin(labeled_components, labels_to_remove))

    # Removes all connected components with a size <= size_threshold
    raw[raw_mask] = 0

    return voxeldata.with_data(raw)


def create_aabbs(voxeldata):
    """ Create an Axis Aligned Bounding Box (https://en.wikipedia.org/wiki/Minimum_bounding_box)
        for each non-zero voxel label of the input image file.

      Args:
          voxeldata(VoxelData): VoxelData object holding the multi-dimensional array
                                to be processed.
      Returns:
          aabbs(dict): a dictionary whose integer keys are the non-zero unique labels of the input
          image. The dictionary values are the smallest
          AABBs enclosing the regions corresponding to the label keys. An AABB is
          of the form (bottom, top) where bottom and top are the two
          3D integer vectors defining the bottom and the top of the AABB in index
          coordinates.
    """

    raw = voxeldata.raw
    labels = np.unique(raw)
    labels = labels[np.nonzero(labels)]  # Remove the background label
    aabbs = dict()
    for label in labels:
        region_indices = np.nonzero(raw == label)
        aabb = np.min(region_indices, axis=1), np.max(region_indices, axis=1)
        aabbs[label] = aabb

    return aabbs


def clip_region(label, voxeldata, aabb):
    """ Extract from a VoxelData object the region with the specified label and clip it using
        the provided axis aligned bounding box.

      Args:
          label(int): the label of the region of interest
          voxeldata(VoxelData): VoxelData object holding the multi-dimensional array
          to be processed.
          aabb(tuple): Axis Aligned Bounding Box (AABB) used to clip the specified region.
          An AABB is of the form (bottom, top) where bottom and top are the two
          3D integer vectors defining the bottom and the top of the AABB in index
          coordinates.
      Returns:
          region(VoxelData): VoxelData object containing the specified region only.
          The dimensions of underlying array are set using the specified bounding box.
    """

    region_raw = voxcell.math_utils.clip(voxeldata.raw, aabb)
    off_mask = np.nonzero(region_raw != label)
    region_raw[off_mask] = 0
    dimensions = voxeldata.voxel_dimensions
    offset = aabb[0] * dimensions
    region = voxcell.VoxelData(region_raw, dimensions, voxeldata.offset + offset)
    return region


def _add_margin(raw, margin):
    return np.pad(raw, margin, 'constant', constant_values=0)


def median_filter(voxel_data, filter_size, closing_size):
    """ Apply a median filter to the input image with a filter of the specified size.

        This size, given in terms of voxels, is the edge length of the cube inside
        which the median is computed.
        A dilation is performed before the application of the median filter and an erosion
        is performed afterwards. Both operations use a box whose edge length is the
        specified closing size. This combination, which is a morphological closing
        with a filter in the middle, has proved useful to fill holes in shapes with
        large openings.
        See https://en.wikipedia.org/wiki/Mathematical_morphology
        for definitions.
        Note: this function does not preserve the volume and is likely to expand it.
    Args:
        voxeldata(VoxelData): VoxelData object holding the multi-dimensional array
        to be processed.
        filter_size(int): edge length of the box used for the median filter
        https://en.wikipedia.org/wiki/Median_filter
        closing_size(int): edge length of the box used to dilate the input image
        before median-filtering and to erode it afterwards.
    Returns:
        voxeldata(VoxelData): VoxelData object whose array has been filtered.
        Each dimension of the array has been increased by
        2 * (filter_size + closing_size + 1) to take into account
        volume expansion. The offset is adjusted accordingly.
    """

    raw = np.copy(voxel_data.raw)  # in-place is not possible as dimensions will be changed
    label_dtype = raw.dtype
    labels = np.unique(raw)
    label = np.max(labels)  # zero only if the 3D image is fully black
    binary_mask = raw > 0
    del raw  # free memory
    margin = filter_size + closing_size + 1
    binary_mask = _add_margin(binary_mask, margin)
    cube = np.full([closing_size] * 3, 1)
    binary_mask = ndimage.morphology.binary_dilation(binary_mask, structure=cube)
    binary_mask = ndimage.median_filter(binary_mask, size=filter_size)
    binary_mask = ndimage.morphology.binary_erosion(binary_mask, structure=cube)
    output_raw = np.zeros(binary_mask.shape, dtype=label_dtype)
    output_raw[binary_mask] = label
    # We do not remove the margin because the image has been expanded
    offset = voxel_data.offset - margin  # adjusted because dimensions have changed
    return voxcell.VoxelData(output_raw, voxel_data.voxel_dimensions, offset=offset)
