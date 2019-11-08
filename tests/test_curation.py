import os
from pathlib import Path
import tempfile
import nose.tools as nt
import numpy.testing as npt
import numpy as np
from scipy import ndimage
from scipy.ndimage.morphology import generate_binary_structure
import voxcell
from utils import load_nrrd, path, load_nrrds
from atlas_analysis.exceptions import AtlasAnalysisError
from atlas_analysis.curation import NEAREST_NEIGHBOR_INTERPOLATION,\
    COMPETITIVE_NEAREST_NEIGHBOR_INTERPOLATION
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
    tested.remove_connected_components(voxeldata, 2)
    expected_raw = np.copy(initial_raw)
    expected_raw[0, 2, 0] = 0
    expected_raw[0, 2, 1] = 0
    expected_raw[0, 0, 2] = 0
    npt.assert_array_equal(voxeldata.raw, expected_raw)
    npt.assert_array_equal(voxeldata.offset, expected_offset)
    npt.assert_array_equal(voxeldata.voxel_dimensions, expected_voxel_dimensions)

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
    tested.remove_connected_components(voxeldata, 2, connectivity=2)
    expected_raw = np.copy(initial_raw)
    expected_raw[2, 2, 2] = 0
    npt.assert_array_equal(voxeldata.raw, expected_raw)
    npt.assert_array_equal(voxeldata.offset, expected_offset)
    npt.assert_array_equal(voxeldata.voxel_dimensions, expected_voxel_dimensions)

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
    offset = np.array((50.0, 50.0, 100.0))
    voxel_dimensions = np.array((25.0, 25.0, 25.0))
    voxel_data = voxcell.VoxelData(initial_raw, voxel_dimensions=voxel_dimensions, offset=offset)
    # Getting the ouput
    output = tested.median_filter(voxel_data, filter_size, closing_size)
    # Comparing with expected result
    margin = filter_size + closing_size + 1
    expanded_shape = shape + 2 * margin
    npt.assert_array_equal(output.raw.shape, expanded_shape)
    expected_raw = np.full(shape, label)
    expected_raw = np.pad(expected_raw, margin, 'constant', constant_values=0)
    expected_offset = offset - margin * voxel_dimensions
    average_mismatch = np.mean(expected_raw != output.raw)
    npt.assert_almost_equal(average_mismatch, 0.0034, decimal=4)
    npt.assert_array_equal(expected_offset, output.offset)

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
    target_offset = np.array((100.0, 10.0, 20.0))
    voxel_dimensions = np.array((25.0, 25.0, 25.0))
    raw_1 = np.full((2, 2, 3), 1, dtype=np.uint32)
    raw_2 = np.full((2, 2, 3), 2, dtype=np.uint32)
    offset_1 = target_offset + np.array((0.0, voxel_dimensions[1], 0.0))
    offset_2 = target_offset + np.array((voxel_dimensions[0], 0.0, 0.0))
    region_1 = voxcell.VoxelData(raw_1, voxel_dimensions, offset=offset_1)
    region_2 = voxcell.VoxelData(raw_2, voxel_dimensions, offset=offset_2)
    # Create output VoxelData object
    raw = np.zeros(shape=(3, 3, 3), dtype=np.uint32)
    merge_output = voxcell.VoxelData(raw, voxel_dimensions, target_offset)
    # Merge regions into the output object
    overlap_label = 666
    for region in [region_1, region_2]:
        tested.merge(region, merge_output, overlap_label)
    expected_raw = np.zeros(shape=(3, 3, 3))
    expected_raw[0:2, 1:3, :] = 1
    expected_raw[1:3, 0:2, :] = 2
    expected_raw[1, 1, :] = overlap_label
    npt.assert_array_equal(expected_raw, merge_output.raw)


def test_pick_closest_voxel():
    raw = np.zeros(shape=(6, 6, 6), dtype=np.uint32)
    voxeldata = voxcell.VoxelData(raw, voxel_dimensions=(15.0, 15.0, 15.0))
    raw[5, 5, 5] = 1
    raw[2, 2, 3] = 666
    raw[3, 2, 4] = 666
    raw[1, 2, 3] = 666
    voxel_index = [2, 2, 2]
    raw[tuple(voxel_index)] = 666
    closest_voxel_index = tested.pick_closest_voxel(voxel_index, voxeldata)
    npt.assert_array_equal(closest_voxel_index, [5, 5, 5])
    raw[5, 5, 5] = 0
    closest_voxel_index = tested.pick_closest_voxel(voxel_index, voxeldata)
    npt.assert_array_equal(closest_voxel_index, voxel_index)
    raw[4, 3, 1] = 2
    raw[3, 4, 1] = 2
    raw[2, 3, 1] = 4
    closest_voxel_index = tested.pick_closest_voxel(voxel_index, voxeldata)
    npt.assert_array_equal(closest_voxel_index, [2, 3, 1])
    raw[3, 2, 1] = 5
    closest_voxel_index = tested.pick_closest_voxel(voxel_index, voxeldata)
    npt.assert_array_equal(closest_voxel_index, [2, 3, 1])
    voxel_index = [1, 1, 5]
    raw[tuple(voxel_index)] = 666
    closest_voxel_index = tested.pick_closest_voxel(voxel_index, voxeldata)
    npt.assert_array_equal(closest_voxel_index, [2, 3, 1])
    raw[2, 3, 1] = 0
    closest_voxel_index = tested.pick_closest_voxel(voxel_index, voxeldata)
    npt.assert_array_equal(closest_voxel_index, [3, 2, 1])

def create_volume_for_assignment(overlap_value):
    # Create volume
    voxel_dimensions = (1.0, 1.0, 1.0)
    side_length = 12
    raw = np.zeros(shape=[side_length] * 3, dtype=np.uint32)
    a = side_length // 3
    b = 2 * side_length // 3
    c = side_length // 2
    # Slice the box into 3 thirds with different labels
    raw[0:a, :, :] = 1
    raw[a:b, :, :] = overlap_value
    raw[b:side_length, :, :] = 2
    raw[a:b, (c - 1):(c + 1), (c - 1):(c + 1)] = 2 # create a hole in the middle layer
    raw[0][0][0] = 2 # exceptional value, disconnected from its region
    raw[[side_length - 1] * 3] = 1 # idem
    ## Create VoxelData object
    voxeldata = voxcell.VoxelData(raw, voxel_dimensions)
    return voxeldata

def test_assign_to_closest_region():
    overlap_value = 666
    voxeldata = create_volume_for_assignment(overlap_value)
    tested.assign_to_closest_region(voxeldata, overlap_value)
    remains = np.any(voxeldata.raw == overlap_value)
    # Check that all voxels have been assigned a label different from the overla value
    nt.assert_equal(not remains, True)
    structure = generate_binary_structure(3, 1)
    labeled_components, _ = ndimage.label(voxeldata.raw, structure=structure)
    labels = np.unique(labeled_components)
    # Check that there are only two connected components
    nt.assert_equal(len(labels) + 1, 2)

def test_assign_competitively_to_closest_region():
    overlap_value = 666
    voxeldata = create_volume_for_assignment(overlap_value)
    tested.assign_to_closest_region(voxeldata, overlap_value, \
        algorithm=COMPETITIVE_NEAREST_NEIGHBOR_INTERPOLATION)
    remains = np.any(voxeldata.raw == overlap_value)
    # Check that all voxels have been assigned a label different from the overla value
    nt.assert_equal(not remains, True)
    structure = generate_binary_structure(3, 1)
    labeled_components, _ = ndimage.label(voxeldata.raw, structure=structure)
    labels = np.unique(labeled_components)
    # Check that there are only two connected components
    nt.assert_equal(len(labels) + 1, 2)

def test_assign_to_closest_region_wrong_algo():
    overlap_value = 666
    voxeldata = create_volume_for_assignment(overlap_value)
    with nt.assert_raises(AtlasAnalysisError):
        tested.assign_to_closest_region(voxeldata, overlap_value, \
            algorithm='ultimate-death-star-optimizer')

def test_crop():
    # One voxel only
    voxel_dimensions = (1.0, 2.0, 1.0)
    offset = (5.0, 2.0, 10.0)
    raw = np.zeros(shape=[3] * 3, dtype=np.uint32)
    voxeldata = voxcell.VoxelData(raw, voxel_dimensions, offset=offset)
    raw[1, 1, 1] = 1
    tested.crop(voxeldata)
    npt.assert_equal(voxeldata.offset, (6, 4, 11))
    npt.assert_equal(voxeldata.raw.shape, (1, 1, 1))
    # Two voxels located at opposite corners
    voxel_dimensions = (3.0, 1.0, 0.0)
    offset = (1.0, 2.0, 1.0)
    raw = np.zeros(shape=[3] * 3, dtype=np.uint32)
    voxeldata = voxcell.VoxelData(raw, voxel_dimensions, offset=offset)
    raw[0, 0, 0] = 2
    raw[2, 2, 2] = 2
    expected_raw = np.copy(raw)
    tested.crop(voxeldata)
    npt.assert_equal(voxeldata.offset, (1.0, 2.0, 1.0))
    npt.assert_equal(expected_raw, voxeldata.raw)
    # Three voxels
    voxel_dimensions = (3.0, 1.0, 0.0)
    offset = (1.0, 2.0, 1.0)
    raw = np.zeros(shape=[3] * 3, dtype=np.uint32)
    voxeldata = voxcell.VoxelData(raw, voxel_dimensions, offset=offset)
    raw[1, 1, 1] = 3
    raw[2, 0, 1] = 3
    raw[1, 0, 0] = 4
    tested.crop(voxeldata)
    npt.assert_equal(voxeldata.offset, (4.0, 2.0, 1.0))
    expected_raw = np.zeros((2, 2, 2))
    expected_raw[0, 1, 1] = 3
    expected_raw[1, 0, 1] = 3
    expected_raw[0, 0, 0] = 4
    npt.assert_equal(expected_raw, voxeldata.raw)

def test_split_into_region_files():
    voxel_dimensions = (1.0, 2.0, 1.0)
    offset = (5.0, 2.0, 10.0)
    edge_length = 12
    raw = np.zeros(shape=[edge_length] * 3, dtype=np.uint32)
    raw[0, 0, 0] = 1
    raw[edge_length - 1, edge_length - 1, edge_length - 1] = 2
    a = edge_length // 4
    raw[1:a, 1:a, 1:a] = 1
    b = 2 * a + 1
    raw[a:b, a:b, a:b] = 2
    m = edge_length // 2
    raw[m, m, m] = 3
    voxeldata = voxcell.VoxelData(raw, voxel_dimensions, offset=offset)
    with tempfile.TemporaryDirectory() as tempdir:
        # Splitting
        tested.split_into_region_files(voxeldata, tempdir)
        dirpath = Path(tempdir)
        # Region 1
        voxeldata_1 = voxcell.VoxelData.load_nrrd(dirpath.joinpath('1.nrrd'))
        npt.assert_array_equal(offset, voxeldata_1.offset)
        npt.assert_array_equal([a] * 3, voxeldata_1.raw.shape)
        # Region 2
        voxeldata_2 = voxcell.VoxelData.load_nrrd(dirpath.joinpath('2.nrrd'))
        expected_offset = np.array(offset) + np.array(voxel_dimensions) * a
        npt.assert_array_equal(expected_offset, voxeldata_2.offset)
        npt.assert_array_equal([3 * a] * 3, voxeldata_2.raw.shape)
        # Region 3
        voxeldata_3 = voxcell.VoxelData.load_nrrd(dirpath.joinpath('3.nrrd'))
        expected_offset = np.array(offset) + np.array(voxel_dimensions) * m
        npt.assert_array_equal(expected_offset, voxeldata_3.offset)
        npt.assert_array_equal([1] * 3, voxeldata_3.raw.shape)
        # Merging back
        raw_merge = np.zeros(shape=[edge_length] * 3, dtype=np.uint32)
        voxeldata_merge = voxcell.VoxelData(raw_merge, voxel_dimensions, offset=offset)
        overlap_label = 666
        tested.merge_regions(tempdir, voxeldata_merge, overlap_label)
        npt.assert_equal(voxeldata.offset, voxeldata_merge.offset)
        npt.assert_equal(voxeldata.raw, voxeldata_merge.raw)

def test_merge_regions():
    voxel_dimensions = np.array((1.0, 2.0, 1.0))
    offset = np.array((5.0, 2.0, 10.0))
    edge_length = 12
    a = 2 * (edge_length // 3)
    raw_1 = np.full([a] * 3, 1, dtype=np.uint32)
    voxeldata_1 = voxcell.VoxelData(raw_1, voxel_dimensions, offset=offset)
    raw_2 = np.full([a] * 3, 2, dtype=np.uint32)
    b = edge_length // 3
    offset_2 = np.array(offset) + b * np.array(voxel_dimensions)
    voxeldata_2 = voxcell.VoxelData(raw_2, voxel_dimensions, offset=offset_2)
    raw = np.zeros(shape=[edge_length] * 3, dtype=np.uint32)
    voxeldata = voxcell.VoxelData(raw, voxel_dimensions, offset=offset)
    expected_raw = np.zeros(shape=[edge_length] * 3, dtype=np.uint32)
    expected_raw[0:a, 0:a, 0:a] = 1
    expected_raw[b:edge_length, b:edge_length, b:edge_length] = 2
    overlap_value = 666
    expected_raw[b:a, b:a, b:a] = overlap_value
    with tempfile.TemporaryDirectory() as tempdir:
        dirpath = Path(tempdir)
        voxeldata_1.save_nrrd(str(dirpath.joinpath('1.nrrd')))
        voxeldata_2.save_nrrd(str(dirpath.joinpath('2.nrrd')))
        tested.merge_regions(tempdir, voxeldata, overlap_value)
        npt.assert_array_equal(expected_raw, voxeldata.raw)
        npt.assert_array_equal(offset, voxeldata.offset)


def test_smooth():
    voxeldata = load_nrrd('three_brain_regions.nrrd')
    with tempfile.TemporaryDirectory() as tempdir:
        threshold_size = 100
        filter_size = 3
        closing_size = 6
        margin = filter_size + closing_size + 1
        expected_shape = np.array((202, 143, 183)) + 2 * margin
        expected_offset = (np.array((191, 152, 137)) - margin) * voxeldata.voxel_dimensions
        tested.smooth(voxeldata, tempdir, threshold_size, filter_size, closing_size)
        npt.assert_array_equal(expected_shape, voxeldata.raw.shape)
        npt.assert_array_equal(expected_offset, voxeldata.offset)
        labels = np.unique(voxeldata.raw)
        labels = labels[np.nonzero(labels)]
        expected_counts = {54: 3186, 260: 7131, 606826663: 1025}
        npt.assert_array_equal(list(expected_counts.keys()), labels)
        for label, expected_count in expected_counts.items():
            count = (voxeldata.raw == label).sum()
            diff = abs(expected_count - count) / expected_count
            npt.assert_almost_equal(0.0, diff, decimal=1)
