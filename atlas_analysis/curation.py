""" Collection of free functions to perform curation operations on atlases """
import numpy as np
from scipy import ndimage
from scipy.ndimage.morphology import generate_binary_structure


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
    # As a raw array loaded by nrrd.read() is immutable when the input file is compressed,
    # we need to make a deep copy of it.
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
