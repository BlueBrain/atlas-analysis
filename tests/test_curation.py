import numpy.testing as npt
import nose.tools as nt
import numpy as np
import voxcell

import atlas_analysis.curation as tested

def test_remove_connected_components_default_struct():
    initial_raw = np.zeros((3, 3, 3), dtype=np.uint32)
    # One connected component with 3 voxels
    initial_raw[0, 0, 0] = 4
    initial_raw[1, 0, 0] = 4
    initial_raw[2, 0, 0] = 4
    # One connected component with 2 voxels
    initial_raw[0, 2, 0] = 2
    initial_raw[0, 2, 1] = 2
    # One connected component with 1 voxel
    initial_raw[0, 0, 2] = 3
    expected_offset = [11.0, 22.0, 33.0]
    expected_voxel_dimensions = [15.0, 15.0, 16.0]
    voxeldata = voxcell.VoxelData(initial_raw, expected_voxel_dimensions, expected_offset)
    res = tested.remove_connected_components(voxeldata, 2)
    expected_raw = np.copy(initial_raw)
    expected_raw[0, 2, 0] = 0
    expected_raw[0, 2, 1] = 0
    expected_raw[0, 0, 2] = 0
    npt.assert_array_equal(res.raw, expected_raw)
    npt.assert_array_equal(res.offset, expected_offset)
    npt.assert_array_equal(res.voxel_dimensions, expected_voxel_dimensions)

def test_create_aabbs():
    raw = np.zeros(shape=(3, 3, 3), dtype=np.uint32)
    raw[1][0][1:3] = 1
    raw[2][0][2] = 1
    raw[1][1][2] = 1
    raw[2][2][2] = 2
    raw[0][2][1] = 2
    raw[0][1][0] = 3
    raw[0][2][0] = 3
    voxel_data = voxcell.VoxelData(raw, (1.0, 1.0, 1.0))
    aabbs = tested.create_aabbs(voxel_data)
    nt.assert_equal(len(aabbs), 3)
    expected_aabbs = {
        1: (np.array([1, 0, 1]), np.array([2, 1, 2])),
        2: (np.array([0, 2, 1]), np.array([2, 2, 2])),
        3: (np.array([0, 1, 0]), np.array([0, 2, 0])),
    }
    for label, box in aabbs.items():
        npt.assert_array_equal(box[0], expected_aabbs[label][0])
        npt.assert_array_equal(box[1], expected_aabbs[label][1])

def test_clip_region():
    raw = np.zeros(shape=(3, 3, 3), dtype=np.uint32)
    raw[0, :, :] = 1
    raw[1, 1:3, 1] = 2
    voxel_data = voxcell.VoxelData(raw, (15.0, 15.0, 15.0))
    # test label 1
    aabb = [0, 0, 0], [0, 2, 2]
    clipped = tested.clip_region(1, voxel_data, aabb)
    npt.assert_array_equal(clipped.offset, (0.0, 0.0, 0.0))
    expected_raw = np.array([[[1, 1, 1], [1, 1, 1], [1, 1, 1]]], dtype=np.uint32)
    npt.assert_array_equal(expected_raw, clipped.raw)
    # test label 2
    aabb = [1, 1, 1], [1, 2, 1]
    clipped = tested.clip_region(2, voxel_data, aabb)
    npt.assert_array_equal(clipped.offset, (15.0, 15.0, 15.0))
    expected_raw = np.array([[[2], [2]]], dtype=np.uint32)
    npt.assert_array_equal(expected_raw, clipped.raw)
