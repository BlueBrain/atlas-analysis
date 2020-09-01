import numpy.testing as npt
import numpy as np
import nose.tools as nt
from pyquaternion import Quaternion

from atlas_analysis.maths import normalize_vector
import atlas_analysis.planes.maths as tested


def test_get_normal():
    expected = [0, -1, 0]
    a = Quaternion(1, 1, 0, 0)
    res = tested.get_normal(a)
    npt.assert_allclose(res, expected)


def test_get_plane_quaternion():
    plane = [1, 2, 3, 0.5, 0.5, 0.5, 0.5]
    expected = [0.5, 0.5, 0.5, 0.5]
    res = tested.get_plane_quaternion(plane)
    nt.assert_equal(res, expected)

    plane = [1, 2, 3, 1, 1, 0, 0]
    expected = [0.7071067811865475, 0.7071067811865475, 0, 0]
    res = tested.get_plane_quaternion(plane)
    npt.assert_allclose(res.elements, expected)


def test_split_plane_elements():
    plane = [1, 2, 3, 0.5, 0.5, 0.5, 0.5]
    expected_xyz = plane[:3]
    expected_quat = [0.5, 0.5, 0.5, 0.5]
    res_xyz, res_quat = tested.split_plane_elements(plane)
    nt.assert_equal(res_xyz, expected_xyz)
    nt.assert_equal(res_quat, expected_quat)


def test_get_normals():
    planes = [
        [0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 1],
    ]
    expected = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]
    npt.assert_allclose(tested.get_normals(planes), expected)


def test_distances_to_plane():
    planes = [[0, 0, 0, 1, 0, 1, 0], [1, 0, 0, 1, 0, 1, 0], [2, 0, 0, 1, 0, 1, 0]]
    npt.assert_allclose(tested.distances_to_planes([3, 0, 0], planes), [3, 2, 1])


def test_quaternion_from_vectors():
    expected = np.array([1.0 + np.sqrt(3), -1.0, 1.0, 0.0])
    actual = tested.quaternion_from_vectors([0.0, 0.0, 1.0], [1.0, 1.0, 1.0])
    npt.assert_array_almost_equal(actual, expected)

    expected = np.array([2.0 + np.sqrt(5), 0.0, -1.0, 0.0])
    actual = tested.quaternion_from_vectors([0.0, 0.0, 1.0], [-1.0, 0.0, 2.0])
    npt.assert_array_almost_equal(actual, expected)

    v = [1.0, 2.0, 3.0]
    w = [-3.0, 2.0, 1.0]
    quaternion = Quaternion(tested.quaternion_from_vectors(v, w))
    npt.assert_array_almost_equal(quaternion.rotate(v), w)

    v = normalize_vector([0.1, 20.0, 30.1])
    w = normalize_vector([13.0, -7.0, 100.0])
    quaternion = Quaternion(tested.quaternion_from_vectors(v, w))
    npt.assert_array_almost_equal(quaternion.rotate(v), w)


def test_quaternion_format_to_equation():
    planes = [
        [0.0, 0.0, 1.0, 1.0 + np.sqrt(3), -1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0 + np.sqrt(3), -1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 1.0 + np.sqrt(3), -1.0, 1.0, 0.0],
    ]
    expected = [
        [
            0.0,
            0.0,
            1.0,
            1.0 / np.sqrt(3),
            1.0 / np.sqrt(3),
            1.0 / np.sqrt(3),
            1.0 / np.sqrt(3),
        ],
        [
            0.0,
            1.0,
            0.0,
            1.0 / np.sqrt(3),
            1.0 / np.sqrt(3),
            1.0 / np.sqrt(3),
            1.0 / np.sqrt(3),
        ],
        [
            1.0,
            0.0,
            0.0,
            1.0 / np.sqrt(3),
            1.0 / np.sqrt(3),
            1.0 / np.sqrt(3),
            1.0 / np.sqrt(3),
        ],
    ]
    npt.assert_array_almost_equal(
        tested.quaternion_format_to_equation(planes), expected
    )

    planes = [
        [0.0, 0.0, 1.0, 2.0 + np.sqrt(5), 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 2.0 + np.sqrt(5), 0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0, 2.0 + np.sqrt(5), 0.0, -1.0, 0.0],
    ]
    expected = [
        [0.0, 0.0, 1.0, -1.0 / np.sqrt(5), 0.0, 2.0 / np.sqrt(5), 2.0 / np.sqrt(5)],
        [0.0, 1.0, 0.0, -1.0 / np.sqrt(5), 0.0, 2.0 / np.sqrt(5), 0.0],
        [1.0, 0.0, 0.0, -1.0 / np.sqrt(5), 0.0, 2.0 / np.sqrt(5), -1.0 / np.sqrt(5)],
    ]
    npt.assert_array_almost_equal(
        tested.quaternion_format_to_equation(planes), expected
    )


def test_equation_format_to_quaternion():
    planes = [
        [
            0.0,
            0.0,
            1.0,
            1.0 / np.sqrt(3),
            1.0 / np.sqrt(3),
            1.0 / np.sqrt(3),
            1.0 / np.sqrt(3),
        ],
        [
            0.0,
            1.0,
            0.0,
            1.0 / np.sqrt(3),
            1.0 / np.sqrt(3),
            1.0 / np.sqrt(3),
            1.0 / np.sqrt(3),
        ],
        [
            1.0,
            0.0,
            0.0,
            1.0 / np.sqrt(3),
            1.0 / np.sqrt(3),
            1.0 / np.sqrt(3),
            1.0 / np.sqrt(3),
        ],
    ]
    quaternion = normalize_vector([1.0 + np.sqrt(3), -1.0, 1.0, 0.0])
    expected = [
        np.concatenate([[0.0, 0.0, 1.0], quaternion]),
        np.concatenate([[0.0, 1.0, 0.0], quaternion]),
        np.concatenate([[1.0, 0.0, 0.0], quaternion]),
    ]
    npt.assert_array_almost_equal(
        tested.equation_format_to_quaternion(planes), expected
    )

    planes = [
        [0.0, 0.0, 1.0, -1.0 / np.sqrt(5), 0.0, 2.0 / np.sqrt(5), 2.0 / np.sqrt(5)],
        [0.0, 1.0, 0.0, -1.0 / np.sqrt(5), 0.0, 2.0 / np.sqrt(5), 0.0],
        [1.0, 0.0, 0.0, -1.0 / np.sqrt(5), 0.0, 2.0 / np.sqrt(5), -1.0 / np.sqrt(5)],
    ]
    quaternion = normalize_vector([2.0 + np.sqrt(5), 0.0, -1.0, 0.0])
    expected = [
        np.concatenate([[0.0, 0.0, 1.0], quaternion]),
        np.concatenate([[0.0, 1.0, 0.0], quaternion]),
        np.concatenate([[1.0, 0.0, 0.0], quaternion]),
    ]
    npt.assert_array_almost_equal(
        tested.equation_format_to_quaternion(planes), expected
    )
