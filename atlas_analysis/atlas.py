""" Collection of free functions to perform basic operations on atlases """
from pathlib import Path

import numpy as np
import nrrd

from atlas_analysis.exceptions import AtlasError
from atlas_analysis.utils import (ensure_list, add_suffix, assert_safe_cast,
                                  compare_all)

VALID_ENCODING_NRRD = {'raw', 'ascii', 'text', 'txt', 'gzip', 'bzip2'}


def _non_zero(a, negative_is_zero):
    """ Return np.less_equal(a, 0) if negative_is_zero else return np.not_equal(a, 0) """
    return np.greater(a, 0) if negative_is_zero else np.not_equal(a, 0)


def safe_cast_atlas(voxel_data, new_type):
    """ Safely cast an atlas into a new data type """
    if voxel_data.raw.dtype == new_type:
        return voxel_data
    raw = voxel_data.raw
    if not np.can_cast(raw.dtype, new_type):
        max_data = np.max(raw)
        min_data = np.min(raw)
        if not np.can_cast(max_data, new_type):
            raise AtlasError("Cannot cast atlas in {}. Max value is {}".format(new_type, max_data))
        if not np.can_cast(min_data, new_type):
            raise AtlasError("Cannot cast atlas in {}. Min value is {}".format(new_type, min_data))
    return voxel_data.with_data(raw.astype(new_type))


def homogenize_atlas_types(atlases, cast='safe'):
    """ Automatic cast of multiple atlases so they can safely share the same dtype

    Args:
       atlases: a list of voxeldata
       cast: switch for atlases casting ("strict"|"minimal"|"safe")

    Returns:
        A list of VoxelData objects. The output order is the same as atlases.

    Raises:
       if strict mode and not all atlases are of same dtype.
       if the casting strategy does not exist

    Notes:
       In case of different dtypes for atlases you can use different cast strategies:
       "strict": force all input atlases to have the same dtype.
       "safe": will find the safe cast that can include all values from all dtypes.
       "minimum": Will find the smallest casting type wrt the min and max of all atlases
    """
    atlases = ensure_list(atlases)
    same_types = all(atlases[0].raw.dtype == atlas.raw.dtype for atlas in atlases[1:])
    if same_types:
        return atlases
    if cast == 'strict':
        raise AtlasError('All atlases must have the same '
                         'dtype when using "strict" cast mode')
    if cast == 'safe':
        global_type = np.result_type(*list(map(lambda x: x.raw.dtype, atlases)))
    elif cast == 'minimal':
        data_max = max([atlas.raw.max() for atlas in atlases])
        data_min = min([atlas.raw.min() for atlas in atlases])
        max_type = np.min_scalar_type(data_max)
        min_type = np.min_scalar_type(data_min)
        # TODO: if max is a int32 70000 min_scalar returns uint32 and if min is negative
        # TODO: the code goes to promote instead of if and return int64 when int32 is wanted
        if np.can_cast(data_min, max_type):
            global_type = max_type
        elif np.can_cast(data_max, min_type):
            global_type = min_type
        else:
            global_type = np.promote_types(max_type, min_type)
    else:
        raise AtlasError('Unknown cast type {}. Should be '
                         '"strict", "minimal" or "safe"'.format(cast))
    casted_atlases = []
    for atlas in atlases:
        if atlas.raw.dtype == global_type:
            casted_atlases.append(atlas)
        else:
            casted_atlases.append(atlas.with_data(atlas.raw.astype(global_type)))
    return casted_atlases


def assert_properties(atlases):
    """ Assert that all atlases properties match

    Args:
        atlases: a list of voxeldata

    Raises:
        if one of the property is not shared by all data files
    """
    atlases = ensure_list(atlases)
    if not compare_all(atlases, lambda x: x.raw.shape, comp=np.allclose):
        raise AtlasError('Need to have the same shape for all files')
    if not compare_all(atlases, lambda x: x.voxel_dimensions, comp=np.allclose):
        raise AtlasError('Need to have the same voxel_dimensions for all files')
    if not compare_all(atlases, lambda x: x.offset, comp=np.allclose):
        raise AtlasError('Need to have the same offset for all files')


def coherent_atlases(atlases, cast='safe'):
    """ Homogenizes atlases and check if all the properties match

    Args:
        atlases: a list of voxeldata
        cast: switch for atlas casting ("strict"|"minimal"|"safe")

    Returns:
         A list of VoxelData objects. The order is the same as atlases.

    Raises:
        if one of the property is not shared by all data files

    Notes:
        In case of different dtypes for atlases you can use different cast strategies:
        "strict": force all input atlases to have the same dtype.
        "safe": will find the safe cast that can include all values from all dtypes.
        "minimum": Will find the smallest casting type wrt the min and max of all atlases
    """
    atlases = ensure_list(atlases)
    atlases = homogenize_atlas_types(atlases, cast=cast)
    assert_properties(atlases)
    return atlases


def extract_labels(voxel_data, segmentation_labels, new_label=None):
    """ Create a VoxelData object keeping only voxels with value in segmentation labels.

    Args:
        voxel_data: the voxel_data input file.
        segmentation_labels: the segmentation labels from the input file you want to keep in
        the filtered VoxelData (list or set or tuple).
        new_label: The new label value to replace all the original labels if needed

    Returns:
        A filtered VoxelData object containing only the wanted labels.
    """
    if new_label is not None:
        assert_safe_cast(new_label, voxel_data.raw.dtype)
    mask = np.nonzero(np.isin(voxel_data.raw, ensure_list(segmentation_labels)))
    res = np.zeros_like(voxel_data.raw)
    # pylint: disable=unsupported-assignment-operation
    res[mask] = voxel_data.raw[mask] if new_label is None else new_label
    return voxel_data.with_data(res)


def reset_all_values(voxel_data, value, negative_is_zero=True):
    """ Reset all values from a VoxelData object to value

    Args:
        voxel_data: the VoxelData from which you want to redefine the values
        value: the value you want to set
        negative_is_zero: consider negative values as zeros

    Returns:
        a VoxelData object containing value only

    Raises:
        if value cannot be cast safely into the voxel_data dtype
    """
    assert_safe_cast(value, voxel_data.raw.dtype)
    res = np.copy(voxel_data.raw)
    res[_non_zero(res, negative_is_zero)] = value
    return voxel_data.with_data(res)


def regroup_atlases(atlases, new_label=None, cast='safe', negative_is_zero=True):
    """ Regroup multiple atlases in one if all properties are similar

    Args:
        atlases: list of VoxelData
        new_label: set a global label to all parts
        cast: switch for atlases casting ("strict"|"minimal"|"safe")
        negative_is_zero: consider negative values as zeros

    Returns:
        the VoxelData of the regrouped atlases

    Notes:
        Superpose all the atlases. So in case of overlap between atlases the last one will
        win.
        In case of different dtypes for atlases you can use different cast strategies:
        "strict": force all input atlases to have the same dtype.
        "safe": will find the safe cast that can include all values from all dtypes.
        "minimum": Will find the smallest casting type wrt the min and max of all atlases
    """
    atlases = coherent_atlases(atlases, cast=cast)
    if new_label:
        assert_safe_cast(new_label, atlases[0].raw.dtype)
    res = np.zeros_like(atlases[0].raw)
    for atlas in atlases:
        mask = _non_zero(atlas.raw, negative_is_zero)
        # pylint: disable=unsupported-assignment-operation
        res[mask] = atlas.raw[mask] if new_label is None else new_label
    return atlases[0].with_data(res)


def logical_and(atlases, logical_and_label, cast='safe', negative_is_zero=True):
    """ Logical and multiple atlases between themselves

    Ex with logical_and_label == 2:
    input1   |  input2   | input3   | returned
    1 1 1 0  |  0 0 1 0  | 0 0 1 0  | 0 0 2 0
    0 0 0 0  |  0 0 0 0  | 0 0 1 0  | 0 0 0 0
    1 1 0 0  |  0 1 0 1  | 1 1 1 1  | 0 2 0 0

    Args:
        atlases: list of VoxelData
        logical_and_label: the label you want to set to the clipped area
        cast: switch for atlases casting ("strict"|"minimal"|"safe")
        negative_is_zero: consider negative values as zeros

    Returns:
        the VoxelData of the logical and atlas

    Notes:
        In case of different dtypes for atlases you can use different cast strategies:
        "strict": force all input atlases to have the same dtype.
        "safe": will find the safe cast that can include all values from all dtypes.
        "minimum": Will find the smallest casting type wrt the min and max of all atlases
        All np.nan are considered as zeros.
    """
    atlases = coherent_atlases(atlases, cast=cast)
    ref_raw = atlases[0].raw
    assert_safe_cast(logical_and_label, ref_raw.dtype)
    res = np.zeros_like(ref_raw)
    mask = _non_zero(ref_raw, negative_is_zero)
    for atlas in atlases[1:]:
        mask = np.logical_and(mask, _non_zero(atlas.raw, negative_is_zero))
    # pylint: disable=unsupported-assignment-operation
    res[mask] = logical_and_label
    return atlases[0].with_data(res)


def voxel_mask(input_data, mask_data, masked_off=False, negative_is_zero=True):
    """ Mask the input VoxelData using the mask VoxelData

    Will set 0 in the input atlas where the cropping atlas is not zero.

    Ex with negative_mask == True:
    input          mask          result
    0 2 1 8    |   0 1 0 0   |    0 0 1 8
    0 0 1 5    |   0 0 0 1   |    0 0 1 0
    0 0 0 1    |   0 0 0 0   |    0 0 0 1

    Ex with negative_mask == False:
    input          mask          result
    0 2 1 8    |   0 1 0 0   |    0 2 0 0
    0 0 1 5    |   0 0 0 1   |    0 0 0 5
    0 0 0 1    |   0 0 0 0   |    0 0 0 0

    Args:
        input_data: the VoxelData to mask
        mask_data: the VoxelData containing the cropping volume
        masked_off: define if the mask is used to select or reject voxels
        negative_is_zero: consider negative values as zeros

    Returns:
        the VoxelData object of the masked atlas

    Notes:
        All np.nan in the mask_data are considered as zeros. Values from the mask are not
        taken into account, just the presence of a value or not.
    """
    input_data, mask_data = coherent_atlases([input_data, mask_data], cast='safe')
    input_raw = input_data.raw
    res = np.copy(input_raw)
    mask = _non_zero(mask_data.raw, negative_is_zero)
    mask = mask if masked_off else ~mask
    res[mask] = 0
    return input_data.with_data(res)


def indices_to_voxel_centers(voxel_data, idx):
    """ Indices to middle of voxel positons """
    return voxel_data.indices_to_positions(idx + 0.5)


def sample_positions_from_voxeldata(voxel_data, nb_voxels=-1):
    """ Returns locs from a VoxelData object where the voxels are not 0.

    Args:
        voxel_data: the voxel data object.
        nb_voxels: the number of positions for non zero voxels you want to retrieve.
        If nb_voxels is -1 or if nb_voxels superior to the number of voxel in voxel_data
        all voxels are used, if nb_voxel == 0 returns an empty array,
        if  0 < nb_voxel < nb voxel in voxel_data a random sample of

    Returns:
        Positions of the voxels from the VoxelData object (array([[x1 ,y1, z1],...,[xn ,yn, zn]]).
    """
    nz_idx = np.array(np.nonzero(voxel_data.raw)).T
    if nb_voxels < 0 or nb_voxels >= nz_idx.shape[0]:
        return indices_to_voxel_centers(voxel_data, nz_idx)
    if nb_voxels == 0:
        return np.empty(0)
    nb_voxels = min(nb_voxels, nz_idx.shape[0])
    sampling = np.random.choice(nz_idx.shape[0], nb_voxels, replace=False)
    return indices_to_voxel_centers(voxel_data, nz_idx[sampling])


def change_encoding(nrrd_path, output=None, encoding='gzip', suffix='_gzip'):
    """ Change the opt['encoding'] from a nrrd file and add a suffix to the created file

    Args:
        nrrd_path: input path of the nrrd file
        output: output path for the reencoded atlas
        encoding: the chosen encoding for your new file
        suffix: suffix added before extension. name.nrrd --> name<suffix>.nrrd

    Notes:
        Regiodesic needs 'raw' encoding

    Raises:
        if encoding not in the valid encoding for pynrrd
        if suffix is an empty string (raised by add suffix)
    """
    encoding = encoding.lower()
    if encoding not in VALID_ENCODING_NRRD:
        raise AtlasError('Encoding {} not in {}'.format(encoding, ','.join(VALID_ENCODING_NRRD)))
    raw, opt = nrrd.read(nrrd_path)
    opt['encoding'] = encoding
    if output is None:
        output = add_suffix(nrrd_path, suffix)
    nrrd.write(output, raw, options=opt)
    return str(Path(output).absolute())
