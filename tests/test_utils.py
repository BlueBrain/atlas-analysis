from pathlib import Path
from tempfile import TemporaryDirectory

import nose.tools as nt
import numpy.testing as npt
import numpy as np

from mock import patch

import atlas_analysis.utils as tested
from atlas_analysis.exceptions import AtlasAnalysisError

from tests.utils import load_nrrd, path, load_orientation


def test_add_suffix_1():
    expected = str(path("1_suffix.nrrd"))
    res = tested.add_suffix(path("1.nrrd"), "_suffix")
    nt.eq_(res, expected)

    expected = str(path("1.nrrd"))
    res = tested.add_suffix(path("1.nrrd"), "", force=True)
    nt.eq_(res, expected)


@nt.raises(AtlasAnalysisError)
def test_add_suffix_1():
    tested.add_suffix(path("1.nrrd"), "")


def test_ensure_list():
    expected = ["astring"]
    res = tested.ensure_list("astring")
    nt.eq_(res, expected)

    expected = [1]
    res = tested.ensure_list(1)
    nt.eq_(res, expected)

    expected = [1, 2, 3]
    res = tested.ensure_list([1, 2, 3])
    nt.eq_(res, expected)

    expected = [1]
    res = tested.ensure_list([1])
    nt.eq_(res, expected)

    expected = [{"key": "value"}]
    res = tested.ensure_list({"key": "value"})
    nt.eq_(res, expected)

    expected = [{"key": "value"}]
    res = tested.ensure_list([{"key": "value"}])
    nt.eq_(res, expected)


def test_assert_safe_cast_1():
    tested.assert_safe_cast(1, np.uint8)


@nt.raises(AtlasAnalysisError)
def test_assert_safe_cast_2():
    tested.assert_safe_cast(-1, np.uint8)


@nt.raises(AtlasAnalysisError)
def test_assert_safe_cast_3():
    tested.assert_safe_cast(300, np.uint8)


def test_assert_safe_cast_4():
    tested.assert_safe_cast(1, str)


def test_pairwise():
    expected = [(1, 2), (2, 3), (3, 4)]
    res = list(tested.pairwise([1, 2, 3, 4]))
    nt.assert_list_equal(res, expected)

    expected = [("ab", "cd"), ("cd", "ef")]
    res = list(tested.pairwise(["ab", "cd", "ef"]))
    nt.assert_list_equal(res, expected)


def test_save_raw():
    import voxcell

    v1 = load_nrrd("1.nrrd")
    orient = load_orientation("orientation.nrrd")
    with TemporaryDirectory() as directory:
        file_name = str(Path(directory, 'file.nrrd'))
        with patch('atlas_analysis.utils.L') as mock_logger:
            file_path = tested.save_raw(file_name, v1.raw, v1)
            nt.eq_(mock_logger.info.call_count, 1)
            nt.eq_(file_path, file_name)
        res = voxcell.VoxelData.load_nrrd(file_path)
        npt.assert_array_equal(res.raw, v1.raw)

        file_name = str(Path(directory, 'orientation.nrrd'))
        with patch('atlas_analysis.utils.L') as mock_logger:
            file_path = tested.save_raw(file_name, orient.raw, orient, is_orientation=True)
            nt.eq_(mock_logger.info.call_count, 1)
            nt.eq_(file_path, file_name)
        res = voxcell.OrientationField.load_nrrd(file_path)
        npt.assert_array_equal(res.raw, orient.raw)


def test_compare_all_1():
    v1 = load_nrrd("1.nrrd")
    v2 = load_nrrd("2.nrrd")
    v3 = load_nrrd("data.nrrd")
    atlases = [v1, v2, v3]
    res = tested.compare_all(atlases, lambda x: x.raw.shape, comp=np.allclose)
    nt.ok_(res)

    v2 = load_nrrd("1_shape.nrrd")
    atlases = [v1, v2, v3]
    res = tested.compare_all(atlases, lambda x: x.raw.shape, comp=np.allclose)
    nt.assert_false(res)


@nt.raises(AtlasAnalysisError)
def test_compare_all_2():
    v1 = load_nrrd("1.nrrd")
    v2 = load_nrrd("2.nrrd")
    v3 = load_nrrd("data.nrrd")
    atlases = [v1, v2, v3]
    tested.compare_all(atlases, lambda x: x.raw.shape, comp=np.equal)


def test_between():
    v1 = load_nrrd("data.nrrd")

    expected = np.zeros((3, 3, 3), dtype=np.uint8).astype(bool)
    expected[0, 0, 0] = True
    expected[0, 1, 0] = True
    res = tested.between(v1.raw, 1, 2)
    npt.assert_array_equal(res, expected)

    expected = np.zeros((3, 3, 3), dtype=np.uint8).astype(bool)
    expected[0, 0, 0] = True
    res = tested.between(v1.raw, 1, 1)
    npt.assert_array_equal(res, expected)

    expected = np.zeros((3, 3, 3), dtype=np.uint8).astype(bool)
    expected[0, 0:4, 0] = True
    res = tested.between(v1.raw, 1, 3)
    npt.assert_array_equal(res, expected)

    expected = np.zeros((3, 3, 3), dtype=np.uint8).astype(bool)
    expected[0, 1, 0] = True
    res = tested.between(v1.raw, 2, 2)
    npt.assert_array_equal(res, expected)


def test_string_to_type_converter():
    nt.assert_equal(tested.string_to_type_converter('bool')('False'), True)
    nt.assert_equal(tested.string_to_type_converter('bool')(''), False)
    nt.assert_equal(tested.string_to_type_converter('int')('4'), 4)
    nt.assert_equal(tested.string_to_type_converter('int')('-15'), -15)
    nt.assert_equal(tested.string_to_type_converter('float')('3.14159'), 3.14159)
    nt.assert_equal(tested.string_to_type_converter('float')('inf'), float('inf'))
    nt.assert_equal(tested.string_to_type_converter('str')('Some text'), 'Some text')
