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
            [0.0, 0.0, 1.0], [-1.0 / np.sqrt(5), 0.0, 2.0 / np.sqrt(5)],
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
            plane.to_numpy(), [-5, 2, 7, 1.0 / np.sqrt(2), 0, 1.0 / np.sqrt(2)],
        )

        plane = tested.Plane(
            [1, 1, 1], [1.0 / np.sqrt(3), 1.0 / np.sqrt(3), 1.0 / np.sqrt(3)],
        )
        npt.assert_array_almost_equal(
            plane.to_numpy(),
            [1, 1, 1, 1.0 / np.sqrt(3), 1.0 / np.sqrt(3), 1.0 / np.sqrt(3)],
        )


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
