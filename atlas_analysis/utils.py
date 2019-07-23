""" Utility module for atlas analyses """
import itertools
import logging
import collections
from pathlib import Path

import six
import numpy as np

import voxcell

from atlas_analysis.exceptions import AtlasError

L = logging.getLogger(__name__)
L.setLevel(logging.INFO)


def add_suffix(file_path, to_add, force=False):
    """ Add a suffix to a file name: /dir1/dir2/file.ext --> /dir1/dir2/file<to_add>.ext"""
    if not force and not to_add:
        raise AtlasError('to_add arg cannot be empty. Would override original file')
    path = Path(file_path)
    return str(Path(path.parent, path.name.replace(path.suffix, to_add + path.suffix)))


def ensure_list(value):
    """ Convert iterable / wrap scalar into list (strings are considered scalar). """
    if isinstance(value, collections.Iterable) and not isinstance(value, (
            six.string_types, collections.Mapping)):
        return list(value)
    return [value]


def assert_safe_cast(value, expected_type):
    """ Check if we can safely cast a scalar or array scalar into a given type"""
    if not np.can_cast(value, expected_type, 'safe'):
        raise AtlasError('Cannot cast {} into {}'.format(value, expected_type))


def pairwise(iterable):
    """v -> (v0,v1), (v1,v2), (v2, v3), ..."""
    v1, v2 = itertools.tee(iterable)
    next(v2, None)
    return zip(v1, v2)


def save_raw(file_path, raw, ref_voxcell_data, is_orientation=False):
    """ Save a numpy array to a nrrd file

    Args:
        file_path: path to the file
        raw: the atlas data to store in a voxeldata file
        ref_voxcell_data: the ref voxcell data used to define the offset and voxel sizes
        is_orientation: boolean for orientation files

    Returns:
        The file path of the newly create file
    """
    voxel_data = ref_voxcell_data.with_data(raw)
    if not is_orientation:
        voxel_data.save_nrrd(file_path)
    else:
        voxcell.OrientationField.save_nrrd(voxel_data, file_path)
    L.info('%s has been created', file_path)
    return file_path


def compare_all(data_sets, fun, comp):
    """ Compares using comp all values extracted from data_sets using the fun access function

    Ex:
        compare_all(atlases, lambda x: x.raw.shape, comp=np.allclose)
    """
    try:
        res = all(comp(fun(data_sets[0]), fun(other)) for other in data_sets[1:])
    except Exception as e:
        raise AtlasError("Bad operation during comparing") from e
    return res


def between(array, down, top):
    """ return the mask or boolean if a value is between down and up """
    return (array >= down) & (array <= top)
