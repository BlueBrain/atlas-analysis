import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import nose.tools as nt
import numpy.testing as npt
import numpy as np
import itertools
from mock import patch
import nrrd

import voxcell
import atlas_analysis.atlas as tested
from atlas_analysis.exceptions import AtlasAnalysisError

from utils import load_nrrd, path, load_nrrds


def choice(result, sample, replace=False):
    return np.arange(sample)


def test__non_zero():
    expected = np.zeros((2, 1, 1), dtype=np.bool)
    expected[0, 0, 0] = False
    expected[1, 0, 0] = True

    neg_int8 = load_nrrd("negative_positive_int8.nrrd")
    res = tested._non_zero(neg_int8.raw, True)
    npt.assert_array_equal(res, expected)

    expected = np.zeros((2, 1, 1), dtype=np.bool)
    expected[0, 0, 0] = True
    expected[1, 0, 0] = True
    res = tested._non_zero(neg_int8.raw, False)
    npt.assert_array_equal(res, expected)

    expected = np.zeros((3, 3, 3), dtype=np.uint8)
    expected[:, 0, 0] = True
    expected[:, 1, 0] = True
    v1 = load_nrrd("1.nrrd")
    res = tested._non_zero(v1.raw, True)
    npt.assert_array_equal(res, expected)


def test_safe_cast_atlas_1():
    neg_int8 = load_nrrd("negative_int8.nrrd")
    npt.assert_array_equal(tested.safe_cast_atlas(neg_int8, np.int8).raw, [[[-1]]])
    npt.assert_array_equal(tested.safe_cast_atlas(neg_int8, np.int16).raw, [[[-1]]])

    small_uint32 = load_nrrd("small_uint32.nrrd")
    int8_cast = tested.safe_cast_atlas(small_uint32, np.int8)
    npt.assert_array_equal(int8_cast.raw, [[[2]]])


@nt.raises(AtlasAnalysisError)
def test_safe_cast_atlas_2():
    neg_int8 = load_nrrd("negative_positive_int8.nrrd")
    tested.safe_cast_atlas(neg_int8, np.uint8)


@nt.raises(AtlasAnalysisError)
def test_safe_cast_atlas_3():
    large_uint8 = load_nrrd("large_uint8.nrrd")
    tested.safe_cast_atlas(large_uint8, np.int8)


def test_homogenize_atlas_types_1():
    # same dtype
    v1 = load_nrrd("1.nrrd")
    v2 = load_nrrd("2.nrrd")
    res = tested.homogenize_atlas_types([v1, v2], cast='safe')
    npt.assert_equal(res[0].raw.dtype, res[1].raw.dtype)
    npt.assert_equal(res[0].raw.dtype, np.uint8)

    # unknown cast but same entry type anyway
    res = tested.homogenize_atlas_types([v1, v2], cast='dummy')
    npt.assert_equal(res[0].raw.dtype, res[1].raw.dtype)
    npt.assert_equal(res[0].raw.dtype, np.uint8)

    # (int8 and uint8) safe cast returns int16
    v1 = load_nrrd("negative_int8.nrrd")
    v2 = load_nrrd("1.nrrd")
    res = tested.homogenize_atlas_types([v1, v2], cast='safe')
    npt.assert_equal(res[0].raw.dtype, res[1].raw.dtype)
    npt.assert_equal(res[0].raw.dtype, np.int16)

    # (int8 with negative values and int32 with large values) minimal cast returns int32
    v1 = load_nrrd("negative_int8.nrrd")
    v2 = load_nrrd("large_int32.nrrd")
    res = tested.homogenize_atlas_types([v1, v2], cast='minimal')
    npt.assert_equal(res[0].raw.dtype, res[1].raw.dtype)
    npt.assert_equal(res[0].raw.dtype, np.int32)

    # (int8 with negative values and uint8 with small values) minimal cast returns int8
    v1 = load_nrrd("negative_int8.nrrd")
    v2 = load_nrrd("1.nrrd")
    res = tested.homogenize_atlas_types([v1, v2], cast='minimal')
    npt.assert_equal(res[0].raw.dtype, res[1].raw.dtype)
    npt.assert_equal(res[0].raw.dtype, np.int8)

    # (int8 with negative values and uint8 with large values) minimal cast returns int16
    v1 = load_nrrd("negative_int8.nrrd")
    v2 = load_nrrd("large_uint8.nrrd")
    res = tested.homogenize_atlas_types([v1, v2], cast='minimal')
    npt.assert_equal(res[0].raw.dtype, res[1].raw.dtype)
    npt.assert_equal(res[0].raw.dtype, np.int16)


@nt.raises(AtlasAnalysisError)
def test_homogenize_atlas_types_2():
    v1 = load_nrrd("negative_int8.nrrd")
    v2 = load_nrrd("large_uint8.nrrd")
    tested.homogenize_atlas_types([v1, v2], cast='strict')


@nt.raises(AtlasAnalysisError)
def test_homogenize_atlas_types_2():
    v1 = load_nrrd("negative_int8.nrrd")
    v2 = load_nrrd("large_uint8.nrrd")
    tested.homogenize_atlas_types([v1, v2], cast='dummy')


def compare(fun, data_sets, comp=np.allclose):
    return all(comp(fun(data_sets[0]), fun(other)) for other in data_sets[1:])


def compare_all(testee):
    nt.ok_(compare(lambda x: x.raw.dtype, testee, comp=np.equal))
    nt.ok_(compare(lambda x: x.raw.shape, testee))
    nt.ok_(compare(lambda x: x.voxel_dimensions, testee))
    nt.ok_(compare(lambda x: x.offset, testee))


def test_assert_properties_1():
    atlases = load_nrrds(["1.nrrd", "1.nrrd"])
    tested.assert_properties(atlases)


@nt.raises(AtlasAnalysisError)
def test_assert_properties_1():
    atlases = load_nrrds(["1.nrrd", "1_shape.nrrd"])
    tested.assert_properties(atlases)


@nt.raises(AtlasAnalysisError)
def test_assert_properties_2():
    atlases = load_nrrds(["1.nrrd", "1_voxel_dimensions.nrrd"])
    tested.assert_properties(atlases)


@nt.raises(AtlasAnalysisError)
def test_assert_properties_3():
    atlases = load_nrrds(["1.nrrd", "1_offset.nrrd"])
    tested.assert_properties(atlases)


def test_coherent_atlases_1():
    atlases = load_nrrds(["negative_int8.nrrd", "large_uint8.nrrd",
                          "large_int32.nrrd", "small_uint32.nrrd"])
    compare_all(tested.coherent_atlases(atlases, cast='minimal'))

    atlases = load_nrrds(["1.nrrd", "1_type.nrrd"])
    compare_all(tested.coherent_atlases(atlases, cast='safe'))


@nt.raises(AtlasAnalysisError)
def test_load_coherent_atlases_2():
    files = load_nrrds(["1.nrrd", "1_type.nrrd"])
    tested.coherent_atlases(files, cast='strict')


@nt.raises(AtlasAnalysisError)
def test_load_coherent_atlases_3():
    files = load_nrrds(["1.nrrd", "1_shape.nrrd"])
    tested.coherent_atlases(files, cast='strict')


@nt.raises(AtlasAnalysisError)
def test_load_coherent_atlases_4():
    files = load_nrrds(["1.nrrd", "1_voxel_dimensions.nrrd"])
    tested.coherent_atlases(files, cast='strict')


@nt.raises(AtlasAnalysisError)
def test_load_coherent_atlases_5():
    files = load_nrrds(["1.nrrd", "1_offset.nrrd"])
    tested.coherent_atlases(files, cast='strict')


def test_extract_labels():
    expected = np.zeros((3, 3, 3), dtype=np.uint8)
    expected[0, 0, 0] = 1

    v = load_nrrd("data.nrrd")
    res = tested.extract_labels(v, 1)
    npt.assert_array_equal(res.raw, expected)

    res = tested.extract_labels(v, [1])
    npt.assert_array_equal(res.raw, expected)

    res = tested.extract_labels(v, {1})
    npt.assert_array_equal(res.raw, expected)

    expected = np.zeros((3, 3, 3), dtype=np.uint8)
    expected[0, 0, 0] = 1
    expected[0, 2, 0] = 3

    res = tested.extract_labels(v, [1, 3])
    npt.assert_array_equal(res.raw, expected)

    res = tested.extract_labels(v, {1, 3})
    npt.assert_array_equal(res.raw, expected)

    expected = np.zeros((3, 3, 3), dtype=np.uint8)
    expected[0, 0, 0] = 12
    expected[0, 2, 0] = 12

    res = tested.extract_labels(v, [1, 3], new_label=12)
    npt.assert_array_equal(res.raw, expected)


def test_reset_all_values():
    expected = np.zeros((3, 3, 3), dtype=np.uint8)
    expected[:, 0, 0] = 12
    expected[:, 1, 0] = 12

    v = load_nrrd("1.nrrd")
    res = tested.reset_all_values(v, 12)
    npt.assert_array_equal(res.raw, expected)


def test_regroup_atlases_1():
    expected = np.zeros((3, 3, 3), dtype=np.uint8)
    expected[:, 0, 0] = 1
    expected[:, 1, 0] = 2
    expected[0, :, 0] = 1
    expected[1, :, 0] = 1

    files = list(map(load_nrrd, ["1.nrrd", "2.nrrd"]))
    res = tested.regroup_atlases(files)
    npt.assert_array_equal(res.raw, expected)

    expected = np.zeros((3, 3, 3), dtype=np.uint8)
    expected[:, 0, 0] = 12
    expected[:, 1, 0] = 12
    expected[0, :, 0] = 12
    expected[1, :, 0] = 12

    res = tested.regroup_atlases(files, new_label=12)
    npt.assert_array_equal(res.raw, expected)


@nt.raises(AtlasAnalysisError)
def test_regroup_atlases_2():
    # uint8 for both "1.nrrd" and "2.nrrd"
    files = map(load_nrrd, ["1.nrrd", "2.nrrd"])
    tested.regroup_atlases(files, new_label=-1)


@nt.raises(AtlasAnalysisError)
def test_regroup_atlases_3():
    files = map(load_nrrd, ["1.nrrd", "1_offset.nrrd"])
    tested.regroup_atlases(files, new_label=1)


@nt.raises(AtlasAnalysisError)
def test_regroup_atlases_4():
    files = map(load_nrrd, ["1.nrrd", "1_type.nrrd"])
    tested.regroup_atlases(files, new_label=1, cast='strict')


def test_logical_and_1():
    logical_and = np.zeros((3, 3, 3), dtype=np.uint8)
    logical_and[:, 0, 0] = 12
    logical_and[:, 1, 0] = 12
    logical_and[2, 0, 0] = 0
    logical_and[2, 1, 0] = 0

    files = load_nrrds(["1.nrrd", "2.nrrd"])
    tested.logical_and(files, 12)


@nt.raises(AtlasAnalysisError)
def test_logical_and_2():
    files = load_nrrds(["1.nrrd", "1_offset.nrrd"])
    tested.logical_and(files, 12)


@nt.raises(AtlasAnalysisError)
def test_logical_and_3():
    files = load_nrrds(["1.nrrd", "1_type.nrrd"])
    tested.logical_and(files, 12, cast='strict')


def test_voxel_mask_1():
    expected = np.zeros((3, 3, 3), dtype=np.uint8)
    expected[2, 0, 0] = 1
    expected[2, 1, 0] = 2

    v1 = load_nrrd("1.nrrd")
    v2 = load_nrrd("2.nrrd")
    res = tested.voxel_mask(v1, v2, masked_off=True)
    npt.assert_array_equal(res.raw, expected)

    res = tested.voxel_mask(v1, v2)
    expected = np.zeros((3, 3, 3), dtype=np.uint8)
    expected[0:2, 0, 0] = 1
    expected[0:2, 1, 0] = 2
    npt.assert_array_equal(res.raw, expected)


@nt.raises(AtlasAnalysisError)
def test_voxel_mask_2():
    v1, v2 = load_nrrds(["1.nrrd", "1_offset.nrrd"])
    tested.voxel_mask(v1, v2)


def test_voxel_mask_3():
    # should not raise no need of same dtype here just same properties
    v1, v2 = load_nrrds(["1.nrrd", "1_type.nrrd"])
    tested.voxel_mask(v1, v2)


def test_indices_to_voxel_centers():
    expected = [[4., 4., 4.],
                [4., 14., 4.],
                [14., 4., 4.],
                [14, 14, 4.],
                [24, 4., 4.],
                [24., 14., 4.]]

    v1 = load_nrrd("1.nrrd")
    res = tested.indices_to_voxel_centers(v1, np.array(np.nonzero(v1.raw)).T)
    npt.assert_array_almost_equal(res, expected)


@patch('numpy.random.choice', choice)
def test_sample_positions_from_voxeldata():
    expected = [[4., 4., 4.],
                [4., 14., 4.],
                [14., 4., 4.],
                [14, 14, 4.],
                [24, 4., 4.],
                [24., 14., 4.]]

    v1 = load_nrrd("1.nrrd")
    res = tested.sample_positions_from_voxeldata(v1)

    # should have only position for nonzero values of v1
    npt.assert_equal(len(res), len(np.nonzero(v1.raw)[0]))
    npt.assert_array_almost_equal(res, expected)

    res = tested.sample_positions_from_voxeldata(v1, voxel_count=0)
    npt.assert_array_almost_equal(res, [])

    res = tested.sample_positions_from_voxeldata(v1, voxel_count=12)
    npt.assert_equal(len(res), len(np.nonzero(v1.raw)[0]))
    npt.assert_array_almost_equal(res, expected)

    res = tested.sample_positions_from_voxeldata(v1, voxel_count=4)
    npt.assert_equal(len(res), 4)
    npt.assert_array_almost_equal(res, expected[:4])


def test_change_encoding_1():
    raw_ref, opt_ref = nrrd.read(path("1.nrrd"))

    encoding = 'raw'
    with TemporaryDirectory() as directory:
        file_path = shutil.copyfile(path("1.nrrd"), Path(directory, "1.nrrd"))
        output_path = str(Path(directory, "1_raw.nrrd"))
        returned_path = tested.change_encoding(file_path, output=output_path, encoding=encoding)
        raw, opt = nrrd.read(output_path)
        nt.assert_equal(opt.pop('encoding'), encoding)
        opt_ref.pop('encoding')
        npt.assert_array_almost_equal(opt_ref.pop('sizes'), opt.pop('sizes'))
        npt.assert_array_almost_equal(opt_ref.pop('space directions'), opt.pop('space directions'))
        npt.assert_array_almost_equal(opt_ref.pop('space origin'), opt.pop('space origin'))
        nt.assert_dict_equal(opt_ref, opt)
        nt.assert_equal(output_path, returned_path)

        returned_path = tested.change_encoding(file_path, encoding=encoding)
        nt.assert_equal(returned_path, returned_path)


@nt.raises(AtlasAnalysisError)
def test_change_encoding_2():
    with TemporaryDirectory() as directory:
        file_path = shutil.copyfile(path("1.nrrd"), Path(directory, "1.nrrd"))
        tested.change_encoding(file_path, encoding="dummy")
