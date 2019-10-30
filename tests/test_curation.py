import nose.tools as nt
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

def test_median_filter():
    label = 3
    shape = np.array([50, 60, 40])
    initial_raw = np.full(shape, label)
    filter_size = 7
    closing_size = 14
    x = shape[0] // 2
    y = shape[1] // 2
    z = shape[2] // 2
    r = closing_size // 4
    # Splits the box into four sub-boxes
    initial_raw[(x - r):(x + r), :, :] = 0
    initial_raw[:, (y - r):(y + r), :] = 0
    initial_raw[:, :, (z - r):(z + r)] = 0
    x = shape[0] // 4
    y = shape[1] // 4
    z = shape[2] // 4
    # Creates a hole
    initial_raw[(x - r):(x + r), (y - r):(y + r), (z - r):(z + r)] = 0
    voxel_data = voxcell.VoxelData(initial_raw, (1.0, 1.0, 1.0))
    # Getting the ouput
    output = tested.median_filter(voxel_data, filter_size, closing_size)
    # Comparing with expected result
    margin = 2 * (filter_size + closing_size + 1)
    expanded_shape = shape + margin
    npt.assert_array_equal(output.raw.shape, expanded_shape)
    expected_raw = np.full(shape, label)
    expected_raw = np.pad(expected_raw, margin // 2, 'constant', constant_values=0)
    average_mismatch = np.mean(expected_raw != output.raw)
    npt.assert_almost_equal(average_mismatch, 0.0034, decimal=4)

def test_merge_without_offset():
    # Create regions
    voxel_dimensions = (25.0, 25.0, 25.0)
    raw_1 = np.zeros(shape=(3, 3, 3), dtype=np.uint32)
    raw_1[0, :, :] = 1
    raw_1[1, 1:3, 1] = 1
    raw_2 = np.zeros(shape=(3, 3, 3), dtype=np.uint32)
    raw_2[:, 0, :] = 2
    raw_2[1:3, 1, 1] = 2
    raw_3 = np.zeros(shape=(3, 3, 3), dtype=np.uint32)
    raw_3[:, :, 0] = 3
    raw_3[1, 1, 1:3] = 3
    raws = [raw_1, raw_2, raw_3]
    regions = [voxcell.VoxelData(raw, voxel_dimensions) for raw in raws]
    # Create output VoxelData object
    raw = np.zeros(shape=(3, 3, 3), dtype=np.uint32)
    merge_output = voxcell.VoxelData(raw, voxel_dimensions)
    # Merge regions into the output object
    overlap_label = 666
    for region in regions:
        tested.merge(region, merge_output, overlap_label)
    expected_raw = np.copy(raw_1)
    expected_raw[:, 0, :] = 2
    expected_raw[1:3, 1, 1] = 2
    expected_raw[:, :, 0] = 3
    expected_raw[1, 1, 1:3] = 3
    expected_raw[0, 0, :] = overlap_label
    expected_raw[0, :, 0] = overlap_label
    expected_raw[:, 0, 0] = overlap_label
    expected_raw[1, 1, 1] = overlap_label
    npt.assert_array_equal(expected_raw, merge_output.raw)

def test_merge_with_offset():
    # Create regions
    voxel_dimensions = (25.0, 25.0, 25.0)
    raw_1 = np.full((2, 2, 3), 1, dtype=np.uint32)
    raw_2 = np.full((2, 2, 3), 2, dtype=np.uint32)
    region_1 = voxcell.VoxelData(raw_1, voxel_dimensions, offset=(0.0, voxel_dimensions[1], 0.0))
    region_2 = voxcell.VoxelData(raw_2, voxel_dimensions, offset=(voxel_dimensions[0], 0.0, 0.0))
    # Create output VoxelData object
    raw = np.zeros(shape=(3, 3, 3), dtype=np.uint32)
    merge_output = voxcell.VoxelData(raw, voxel_dimensions)
    # Merge regions into the output object
    overlap_label = 666
    for region in [region_1, region_2]:
        tested.merge(region, merge_output, overlap_label)
    expected_raw = np.zeros(shape=(3, 3, 3))
    expected_raw[0:2, 1:3, :] = 1
    expected_raw[1:3, 0:2, :] = 2
    expected_raw[1, 1, :] = overlap_label
    npt.assert_array_equal(expected_raw, merge_output.raw)
