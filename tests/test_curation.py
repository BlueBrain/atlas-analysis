import numpy.testing as npt
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

def test_remove_connected_components_custom_struct():
    initial_raw = np.zeros((3, 3, 3), dtype=np.uint32)
    # One connected component with 3 voxels
    initial_raw[0, 0, 0] = 4
    initial_raw[1, 0, 0] = 4
    initial_raw[2, 0, 0] = 4
    # Another one connected with 3 voxels
    initial_raw[0, 2, 0] = 2
    initial_raw[0, 2, 1] = 2
    initial_raw[0, 1, 2] = 2 # diagonal-type of touch
    
    # One connected component with 1 voxel
    initial_raw[2, 2, 2] = 3

    expected_offset = [0.0, 0.0, 0.0]
    expected_voxel_dimensions = [10.0, 10.0, 10.0]
    voxeldata = voxcell.VoxelData(initial_raw, expected_voxel_dimensions, expected_offset)
    res = tested.remove_connected_components(voxeldata, 2, connectivity=2)
    expected_raw = np.copy(initial_raw)
    expected_raw[2, 2, 2] = 0
    npt.assert_array_equal(res.raw, expected_raw)
    npt.assert_array_equal(res.offset, expected_offset)
    npt.assert_array_equal(res.voxel_dimensions, expected_voxel_dimensions)