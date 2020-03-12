import numpy.testing as npt
import numpy as np

from voxcell import VoxelData
import atlas_analysis.visualization as tested


def test_downscale():
    raw = np.ones((4, 6), dtype=np.int)
    result = tested.downscale(raw, 2)
    npt.assert_array_equal(result, np.ones((2, 3), dtype=np.float))
    raw = np.array([[1, 2, 3, 4], [0, 2, 3, 4],])
    result = tested.downscale(raw, 1)
    npt.assert_array_equal(result, [[5.0 / 4.0, 14.0 / 4.0]])


def test_compute_flatmap_image():
    flatmap_raw = np.array(
        [
            [[[1, 2], [-1, -1], [2, 0]], [[2, 1], [-1, -1], [0, 7]]],
            [[[2, 1], [1, 2], [-1, 0]], [[2, 1], [1, 2], [-1, 0]]],
            [[[0, -2], [4, 5], [0, 6]], [[-2, 0], [5, 5], [6, 1]]],
        ]
    )
    result = tested.compute_flatmap_image(flatmap_raw)
    npt.assert_array_equal(
        result,
        [
            [False, False, False, False, False, False, True, True],
            [False, False, True, False, False, False, False, False],
            [True, True, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False],
            [False, False, False, False, False, True, False, False],
            [False, False, False, False, False, True, False, False],
            [False, True, False, False, False, False, False, False],
        ],
    )


def test_compute_flatmap_histogram():
    flatmap_raw = np.array(
        [
            [[[1, 2], [-1, -1], [2, 0]], [[2, 1], [-1, -1], [0, 7]]],
            [[[2, 1], [1, 2], [-1, 0]], [[2, 1], [1, 2], [-1, 0]]],
            [[[0, -2], [4, 5], [0, 6]], [[-2, 0], [5, 5], [6, 1]]],
        ]
    )
    result = tested.compute_flatmap_histogram(flatmap_raw)
    npt.assert_array_equal(
        result,
        [
            [0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 3, 0, 0, 0, 0, 0],
            [1, 3, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
        ],
    )


def test_flatmap_image_figure():
    flatmap_raw = np.array(
        [
            [[[1, 2], [1, 1], [2, 0]], [[2, 1], [-1, -1], [0, 5]]],
            [[[2, 1], [1, 2], [-1, 0]], [[2, 1], [1, 2], [-1, 0]]],
            [[[0, -2], [1, 5], [0, 5]], [[-2, 0], [2, 2], [2, 1]]],
        ]
    )
    # Full resolution
    figure = tested.flatmap_image_figure(VoxelData(flatmap_raw, [10.0] * 3))
    npt.assert_array_equal(
        figure['data'][0]['z'],
        [
            [False, False, False, False, False, True],
            [False, True, True, False, False, True],
            [True, True, True, False, False, False],
        ],
    )
    # Downscaled to shape (1, 2)
    figure = tested.flatmap_image_figure(
        VoxelData(flatmap_raw, [10.0] * 3), resolution=1
    )
    npt.assert_array_equal(figure['data'][0]['z'], [[5.0 / 9.0, 2.0 / 9.0]])


def test_flatmap_volume_histogram():
    flatmap_raw = np.array(
        [
            [[[1, 2], [1, 1], [2, 0]], [[2, 1], [-1, -1], [0, 5]]],
            [[[2, 1], [1, 2], [-1, 0]], [[2, 1], [1, 2], [-1, 0]]],
            [[[0, -2], [1, 5], [0, 5]], [[-2, 0], [2, 2], [2, 1]]],
        ]
    )
    # Full resolution
    figure = tested.flatmap_volume_histogram(VoxelData(flatmap_raw, [10.0] * 3))
    npt.assert_array_equal(
        figure['data'][0]['z'],
        [[0, 0, 0, 0, 0, 2], [0, 1, 3, 0, 0, 1], [1, 4, 1, 0, 0, 0],],
    )
    # Downscaled to shape (1, 2)
    figure = tested.flatmap_volume_histogram(VoxelData(flatmap_raw, [10.0] * 3), resolution=1)
    npt.assert_array_equal(figure['data'][0]['z'], [[10.0 / 9.0, 1.0 / 3.0]])
