import warnings
import numpy.testing as npt
import numpy as np
import nose.tools as nt
import unittest
from pyquaternion import Quaternion

from atlas_analysis.exceptions import AtlasAnalysisError
from atlas_analysis.maths import normalize_vector
from atlas_analysis.constants import XVECTOR, YVECTOR, ZVECTOR
import atlas_analysis.planes.maths as tested


class Test_Plane(unittest.TestCase):
    def test_constructor(self):
        plane = tested.Plane([1, 1, 2], [4.0, 4.0, 2.0])
        npt.assert_equal(plane.point, [1, 1, 2])
        npt.assert_equal(plane.normal, [2.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0])

        # Zero normal vector as input
        with nt.assert_raises(AtlasAnalysisError):
            tested.Plane([1, 1, 2], [0.0, 0.0, 0.0])

        # Wrong input length for the anchor point
        with nt.assert_raises(AtlasAnalysisError):
            tested.Plane([0, 1, 1, 2], [1.0, 1.0, 0.0])

        # Wrong input length for the normal vector
        with nt.assert_raises(AtlasAnalysisError):
            tested.Plane([1, 1, 2], [1.0, 1.0])

    def test_get_equation(self):
        plane = tested.Plane([-1, -1, 2], [4.0, 4.0, 2.0])
        npt.assert_array_almost_equal(
            plane.get_equation(), [2.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0, -2.0 / 3.0]
        )

        plane = tested.Plane([-1, 0, 1], [8.0, 6.0, 0.0])
        npt.assert_array_almost_equal(
            plane.get_equation(), [4.0 / 5.0, 3.0 / 5.0, 0.0, -4.0 / 5.0]
        )

        plane = tested.Plane([0, 3, 3], [8.0, 4.0, 8.0])
        npt.assert_array_almost_equal(
            plane.get_equation(), [2.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0, 3.0]
        )

    def test_normalize(self):
        plane = tested.Plane([-1, 1, 2], [4.0, 4.0, 1.0])
        plane.normal = [4.0, 4.0, 2.0]
        plane.normalize()
        npt.assert_array_almost_equal(plane.normal, [2.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0])
        npt.assert_array_almost_equal(
            plane.get_equation(), [2.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0]
        )

    def test_get_quaternion(self):
        plane = tested.Plane([0.0, 1.0, 0.0], np.array([1.0, 1.0, 1.0]) / np.sqrt(3))
        expected = normalize_vector([1.0 + np.sqrt(3), -1.0, 1.0, 0.0])
        npt.assert_array_almost_equal(plane.get_quaternion([0, 0, 1]).q, expected)
        plane = tested.Plane(
            [0.0, 0.0, 1.0],
            [-1.0 / np.sqrt(5), 0.0, 2.0 / np.sqrt(5)],
        )
        expected = normalize_vector([2.0 + np.sqrt(5), 0.0, -1.0, 0.0])
        npt.assert_array_almost_equal(plane.get_quaternion([0, 0, 1]).q, expected)
        plane = tested.Plane([0.0, 0.0, 0.0], [-1.0, 1.0, 0.0])
        expected = [np.sqrt(2) / 2.0, 0.0, 0.0, np.sqrt(2) / 2.0]
        npt.assert_array_almost_equal(plane.get_quaternion([1, 1, 0]).q, expected)

    def test_get_basis(self):
        plane = tested.Plane([1.0, 0.0, 3.0], [-1.0, 1.0, 3.0])
        basis = plane.get_basis()
        q = plane.get_quaternion(ZVECTOR)
        npt.assert_array_equal(basis, [q.rotate(XVECTOR), q.rotate(YVECTOR)])

        plane = tested.Plane([-1.0, 1.0, 3.0], [-1.0, 1.0, 3.0])
        basis = plane.get_basis(np.array([YVECTOR, ZVECTOR, XVECTOR]))
        q = plane.get_quaternion(XVECTOR)
        npt.assert_array_equal(basis, [q.rotate(YVECTOR), q.rotate(ZVECTOR)])

        plane = tested.Plane([10.0, -1.0, 0.0], [0.5, 0.3, 311.0])
        with nt.assert_raises(
            AtlasAnalysisError
        ):  # The basis vectors are not normalized
            plane.get_basis(np.array([XVECTOR, [0, 1.0, 1.0], [0.0, 1.0, -1.0]]))

        reference_basis = np.array(
            [
                XVECTOR,
                [0.0, 1.0 / np.sqrt(2), 1.0 / np.sqrt(2)],
                [0.0, -1.0 / np.sqrt(2), 1.0 / np.sqrt(2)],
            ]
        )
        basis = plane.get_basis(reference_basis)
        npt.assert_equal(np.linalg.norm(basis, axis=1), [1.0, 1.0])
        assert np.isclose(np.dot(basis[0], basis[1]), 0.0)
        npt.assert_array_almost_equal(np.cross(basis[0], basis[1]), plane.normal)

    def test_from_quaternion(self):
        quaternion = [1.0 + np.sqrt(3), -1.0, 1.0, 0.0]
        plane = tested.Plane.from_quaternion([0, 1, 0], quaternion)
        expected_equation = np.array([1.0, 1.0, 1.0, 1.0]) / np.sqrt(3)
        npt.assert_array_almost_equal(plane.get_equation(), expected_equation)

        quaternion = normalize_vector([-1.0, 20.0, -300.0, 4000.0])
        plane = tested.Plane.from_quaternion(
            [-5.0, 1, 6.0], quaternion, reference_vector=YVECTOR
        )
        npt.assert_array_almost_equal(
            plane.normal, Quaternion(quaternion).rotate(YVECTOR)
        )

        # Wrong quaternionic input
        with nt.assert_raises(AtlasAnalysisError):
            tested.Plane.from_quaternion([0, 1, 0], [1, 1, 1, 1, 1])

    def test_to_numpy(self):
        plane = tested.Plane([-5, 2, 7], [1.0 / np.sqrt(2), 0, 1.0 / np.sqrt(2)])
        npt.assert_array_almost_equal(
            plane.to_numpy(),
            [-5, 2, 7, 1.0 / np.sqrt(2), 0, 1.0 / np.sqrt(2)],
        )

        plane = tested.Plane(
            [1, 1, 1],
            [1.0 / np.sqrt(3), 1.0 / np.sqrt(3), 1.0 / np.sqrt(3)],
        )
        npt.assert_array_almost_equal(
            plane.to_numpy(),
            [1, 1, 1, 1.0 / np.sqrt(3), 1.0 / np.sqrt(3), 1.0 / np.sqrt(3)],
        )

    def test_get_best_alignment_basis(self):
        plane = tested.Plane([0, 0, 0], [0, 0, 1])
        actual = plane.get_best_alignment_basis([1, 1, 0])
        npt.assert_array_almost_equal(actual[0], np.array([1, 1, 0]) / np.sqrt(2.0))
        actual = plane.get_best_alignment_basis([1, 1, 0], axis=1)
        npt.assert_array_almost_equal(actual[1], np.array([1, 1, 0]) / np.sqrt(2.0))

        plane = tested.Plane([-1, 10, -1], [1, 0, 0])
        actual = plane.get_best_alignment_basis([0, 0, 1], axis=0)
        npt.assert_array_almost_equal(actual[0], np.array([0, 0, 1]))
        actual = plane.get_best_alignment_basis([0, 0, 1], axis=1)
        npt.assert_array_almost_equal(actual[1], np.array([0, 0, 1]))

        plane = tested.Plane([1, -2, 3], [-15, 0.24, 120])
        target = np.array([11, 12, -10])
        actual = plane.get_best_alignment_basis(target)
        basis = plane.get_basis()
        target = target / np.linalg.norm(target)
        min_dist = float('inf')
        for angle in np.linspace(0.0, 2 * np.pi, 30):
            quaternion = Quaternion(axis=plane.normal, angle=angle)
            dist = np.linalg.norm(quaternion.rotate(basis[0]) - target)
            min_dist = min(dist, min_dist)

        assert np.linalg.norm(actual[0] - target) <= min_dist

        plane = tested.Plane([9.9, 2.1, 3.3], [7.5, 16.33, -60.0])
        target = np.array([4, 5, 10])
        actual = plane.get_best_alignment_basis(target, axis=1)
        basis = plane.get_basis()
        target = target / np.linalg.norm(target)
        min_dist = float('inf')
        for angle in np.linspace(0.0, 2 * np.pi, 30):
            quaternion = Quaternion(axis=plane.normal, angle=angle)
            dist = np.linalg.norm(quaternion.rotate(basis[0]) - target)
            min_dist = min(dist, min_dist)

        assert np.linalg.norm(actual[1] - target) <= min_dist

    def test_get_best_alignment_basis_raises(self):
        plane = tested.Plane([1, 1, 1], [1, 0, 0])
        with nt.assert_raises(AtlasAnalysisError):
            plane.get_best_alignment_basis([0, 0, 0])

        plane = tested.Plane([1, 1, 1], [0, 10, 100])
        with nt.assert_raises(AtlasAnalysisError):
            plane.get_best_alignment_basis([1, 1, 0], axis=-1)

        plane = tested.Plane([1, 1, 1], [0.1, 1.0, 0.2])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            basis = plane.get_best_alignment_basis([0.2, 2.0, 0.4], axis=1)
            assert issubclass(w[-1].category, UserWarning)
            assert 'colinear' in str(w[-1].message)
            npt.assert_array_equal(basis, plane.get_basis())


def test_distances_to_plane():
    planes = [
        tested.Plane([0, 0, 0], [1, 0, 1]),
        tested.Plane([1, 0, 0], [1, 0, 1]),
        tested.Plane([2, 0, 0], [1, 0, 1]),
    ]
    npt.assert_allclose(
        tested.distances_to_planes([3, 0, 0], planes),
        [3 / np.sqrt(2), 2 / np.sqrt(2), 1 / np.sqrt(2)],
    )


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


def test_create_orthogonal_planes():
    actual = tested.create_orthogonal_planes([1, 0, 0], [0, 0, 1], [0, 0.5, 1])
    actual_points = [plane.point for plane in actual]
    npt.assert_array_equal(
        actual_points, [[1.0, 0.0, 0.0], [0.5, 0.0, 0.5], [0.0, 0.0, 1.0]]
    )
    actual_normals = [plane.normal for plane in actual]
    npt.assert_array_almost_equal(
        actual_normals, [[-1.0 / np.sqrt(2), 0.0, 1.0 / np.sqrt(2)]] * 3
    )

    actual = tested.create_orthogonal_planes([0, 1, 2], [3, -1, 1], [1, 2, 3, 7])
    actual_points = [plane.point for plane in actual]
    npt.assert_array_almost_equal(
        actual_points,
        [
            [0, 1, 2],
            [0.5, 2.0 / 3.0, 11.0 / 6.0],
            [1.0, 1.0 / 3.0, 5.0 / 3.0],
            [3, -1, 1],
        ],
    )
    actual_normals = [plane.normal for plane in actual]
    expected_normal = np.array([3.0, -2.0, -1.0]) / np.sqrt(14.0)
    npt.assert_array_almost_equal(actual_normals, [expected_normal] * 4)


def test_create_orthogonal_planes_raises():
    with nt.assert_raises(AtlasAnalysisError):
        tested.create_orthogonal_planes(
            [1, 0, 0], [0, 0, 1], [[0, 0.5, 1]]
        )  # points must be 1D

    with nt.assert_raises(AtlasAnalysisError):
        tested.create_orthogonal_planes(
            [1, 0, 0], [1, 0, 0], [0, 1, 2]
        )  # end != start required
