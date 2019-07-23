import numpy.testing as npt
import numpy as np
from pyquaternion import Quaternion

import atlas_analysis.maths as tested


def test_normalize_vectors():
    vectors = [[0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 2, 3]]
    expected = np.array([[0., 0., 1., ],
                         [0., 0.70710678, 0.70710678],
                         [0.57735027, 0.57735027, 0.57735027],
                         [0.26726124, 0.53452248, 0.80178373]])

    res = tested.normalize_vectors(vectors)
    npt.assert_allclose(res, expected)


def test_normalize_vector():
    vectors = [[0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 2, 3]]
    expected = np.array([[0., 0., 1., ],
                         [0., 0.70710678, 0.70710678],
                         [0.57735027, 0.57735027, 0.57735027],
                         [0.26726124, 0.53452248, 0.80178373]])

    for vector, exp in zip(vectors, expected):
        res = tested.normalize_vector(vector)
        npt.assert_allclose(res, exp)


def test_get_middle():
    a = [1, 2]
    res = tested.get_middle(a)
    npt.assert_almost_equal(res, 1.5)


def test_get_normal():
    expected = [0, -1, 0]
    a = Quaternion(1, 1, 0, 0)
    res = tested.get_normal(a)
    npt.assert_allclose(res, expected)
