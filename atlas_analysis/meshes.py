""" Module that takes care of creating meshes from a voxeldata object """
import numpy as np

from atlas_analysis import vtk_utils
from atlas_analysis.exceptions import AtlasAnalysisError
from atlas_analysis.atlas import extract_labels, indices_to_voxel_centers

MARCHING_CUBES = 'marching_cubes'
ALPHA_HULL = 'alpha_hull'
ALGORITHMS = [MARCHING_CUBES, ALPHA_HULL]


def alpha_hull(voxel_data, alpha=None, tol=None, labels=None):
    """ Convert a voxelData object to vtkUnstructuredGrid using alpha_hull

    Args:
        voxel_data: a VoxelData object to convert.
        alpha: alpha used in the alpha hull algorithm
        tol: tolerance for the alpha hull algorithm
        labels: list of labels (or single label) you want to make a mesh of

    Returns:
        an vtkUnstructuredGrid object representing the data voxel.
    """
    if labels is not None:
        voxel_data = extract_labels(voxel_data, labels)

    if alpha is None:
        alpha = max(voxel_data.voxel_dimensions)
    if tol is None:
        tol = (max(voxel_data.voxel_dimensions)) * 2
    nz_idx = np.array(np.nonzero(voxel_data.raw)).T
    pos = indices_to_voxel_centers(voxel_data, nz_idx)
    return vtk_utils.alpha_hull(pos, alpha=alpha, tol=tol)


def marching_cubes(voxel_data, iso_value=None, labels=None):
    """ Convert a voxelData object to vtkPolyData using marching cube algorithm

    Args:
        voxel_data: a VoxelData object to convert.
        iso_value: iso value used in the marching cubes algorithm
        labels: list of labels (or single label) you want to make a mesh of

    Returns:
        an vtkPolyData object representing the data voxel contour

    Notes:
        the iso value is set to 0.5 as default
    """
    if labels is not None:
        voxel_data = extract_labels(voxel_data, labels)

    if iso_value is None:
        iso_value = 0.5

    image_data = vtk_utils.voxeldata_to_vtkImageData(voxel_data)
    return vtk_utils.marching_cubes(image_data, iso_value=iso_value)


def create_meshes(voxel_data, mesh_properties, algorithm=MARCHING_CUBES):
    """ Create meshes from a voxeldata object

    Args:
        voxel_data: the input voxel data object
        mesh_properties: dictionary with meshe's names as key and corresponding labels as values
        algorithm: The algorithm to use. Either 'marching_cubes' or 'alpha_hull'. You can use
        the atlas_analysis.meshes.ALPHA_HULL or the atlas_analysis.meshes.MARCHING_CUBES
        constants instead of the bare strings.

    Returns:
        a dictionary with name as key and the vtkObject object as value
    """
    if algorithm not in ALGORITHMS:
        raise AtlasAnalysisError(f'{algorithm} unsupported mesh algorithm')
    algo_function = {ALPHA_HULL: alpha_hull, MARCHING_CUBES: marching_cubes}
    res = {}
    for name, labels in mesh_properties.items():
        res[name] = algo_function[algorithm](voxel_data, labels=labels)
    return res
