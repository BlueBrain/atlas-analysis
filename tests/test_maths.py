import numpy.testing as npt
import numpy as np

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

