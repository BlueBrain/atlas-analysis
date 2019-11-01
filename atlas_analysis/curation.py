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


def set_region(arr, region, aabb):
    """ Fill in-place the region of the input array enclosed in the specified box.

        The values to be assigned are the values of the specified region.
        Note: the function assumes that the dimensions of the specified region coincide
        with the edge lengths of the specified bounding box.

    Args:
        arr(numpy.ndarray): 3D input array
        region(numpy.ndarray): 3D array holding the value to be set.
        aabb(list): axis aligned bounding box defining the target region of the input array.
        It is of the form (bottom, top) where bottom and top are 3D integer vectors.
    """

    bottom = aabb[0]
    top = aabb[1] + 1
    arr[bottom[0]:top[0], bottom[1]:top[1], bottom[2]:top[2]] = region


def merge(input_voxeldata, output_voxeldata, overlap_label):
    """ Merge the input volumetric image into the output one.

        If a non-void voxel of the input corresponds to a non-void
        voxel of the output before merging, the voxel value of the output is
        set with the specified overlap label.

      Args:
          input_voxeldata(VoxelData): the VoxelData object whose underlying array
          will be merged into the array of the specified output VoxelData.
          output_voxeldata(VoxelData): VoxelData object holding
          the multi-dimensional array into which the input array is merged.
          overlap_label(int): the value indicating that
          the output voxel was already assigned a non-zero label before
          merging.
    """

    input_raw = input_voxeldata.raw
    input_shape = np.array(input_raw.shape)
    # Get the input image offset wrt to the output image in index coordinates
    voxel_offset = (input_voxeldata.offset / input_voxeldata.voxel_dimensions).astype(int)
    # Create the bounding box in which the merge operation will be performed
    aabb = (voxel_offset, input_shape + voxel_offset - 1)
    # Merge
    output_region_raw = voxcell.math_utils.clip(output_voxeldata.raw, aabb)
    overlap_mask = np.logical_and(input_raw > 0, output_region_raw > 0)
    input_mask = input_raw > 0
    output_region_raw[input_mask] = input_raw[input_mask]
    output_region_raw[overlap_mask] = overlap_label
    set_region(output_voxeldata.raw, output_region_raw, aabb)


def pick_closest_voxel(voxel_index, voxeldata):
    """ Pick a voxel with a different label among the closest voxels to the input one.

        The distance in use is the Euclidean distance between
        3D indices. The selected voxel of the input array must be non-void
        and must have a different label from the input voxel.
        If no such voxel exists, the function returns the input voxel.

    Args:
        voxel(numpy.array): 3D integer vector holding voxel indices
        voxeldata(VoxelData): VoxelData object holding the 3D array
        to wich the input voxel belongs.
    Returns:
        voxel_index(numpy.array): 3D integer vector holding the voxel multi-index
    """
    raw = voxeldata.raw
    shape = np.array(raw.shape)
    label = raw[tuple(voxel_index)]
    side_lengths = np.full((3), 1)
    full_array_was_visited = False  # True only if the full array has been already visited
    while True:
        if full_array_was_visited:
            break
        # Create a cube centered at the input voxel.
        # This cube is defined as an axis-aligned bounding box, with a bottom and a top voxel.
        bottom = voxel_index - side_lengths
        top = voxel_index + side_lengths
        full_array_was_visited = np.all(top >= shape) and np.all(bottom <= 0)
        # Clip the cube dimensions to the ambient array dimensions
        top = np.min([top, shape - 1], axis=0)
        bottom = np.max([bottom, [0, 0, 0]], axis=0)
        # Visit the cube, looking for all possible matches
        aabb = (bottom, top)
        visited_cube = voxcell.math_utils.clip(raw, aabb)
        matches = np.where((visited_cube > 0) & (visited_cube != label))
        matches = np.array(matches).T
        # If no match, continue with twice the previous edge length
        if matches.shape[0] == 0:
            side_lengths = side_lengths * 2
            continue
        # Pick one of the closest matches
        distances = np.linalg.norm(matches - voxel_index)
        closest_match = np.argmin(distances)
        closest_voxel_index = bottom + \
            matches[closest_match]  # pylint: disable=unsubscriptable-object
        return closest_voxel_index
    return voxel_index


def assign_to_closest_region(voxeldata, label):
    """ Assign in-place voxels with the specified label to their closest region.

        For each voxel of the input volumetric image bearing the specified label,
        the algorithm selects one of the closest voxels with a different but non-zero label.
        After assignment, the region identified by the specified label is
        entirely distributed accross the other regions of the input volumetric
        image.

    Args:
        voxeldata(VoxelData): the VoxelData object whose voxels are
        going to be re-assigned to their closest region.
        label(int): the label of the region to be redistributed.
    """

    raw = voxeldata.raw
    indices_to_be_assigned = np.where(raw == label)
    indices_to_be_assigned = np.array(indices_to_be_assigned).T
    for voxel in indices_to_be_assigned:  # pylint: disable=not-an-iterable
        closest_voxel_index = pick_closest_voxel(voxel, voxeldata)
        raw[tuple(voxel)] = raw[tuple(closest_voxel_index)]
