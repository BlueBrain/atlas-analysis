from pathlib import Path
from tempfile import TemporaryDirectory

import nose.tools as nt
import numpy.testing as npt
import numpy as np
import itertools
from mock import patch
import nrrd

import voxcell
from atlas_analysis.exceptions import AtlasAnalysisError
import atlas_analysis.reporting as tested

nt.assert_equal.__self__.maxDiff = None

def test_density_report_from_dict():
    density_dictionary = {
        '1': {
            'cell_density': 0.001,
            'glia_density': 0.004,
            'exc_density': 0.004
        },
        '2': {
            'cell_density': 0.001,
            'glia_density': 0.001,
            'exc_density': 0.001
        },
        '3': {
            'cell_density': 0.0009,
            'glia_density': 0.002,
            'exc_density': 0.0008
        }
    }
    report = tested.DensityReport.from_dict(density_dictionary)
    nt.assert_dict_equal(report.to_dict(), density_dictionary)

def test_density_report_save_as():
    density_dictionary = {
        '1': {
            'cell_density': 0.001,
            'glia_density': 0.004,
            'exc_density': 0.004
        },
        '2': {
            'cell_density': 0.001,
            'glia_density': 0.001,
            'exc_density': 0.001
        },
        '3': {
            'cell_density': 0.0009,
            'glia_density': 0.002,
            'exc_density': 0.0008
        }
    }
    report = tested.DensityReport.from_dict(density_dictionary)
    with TemporaryDirectory() as tempdir:
        output_path = str(Path(tempdir, 'density_report.json'))
        report.save_as(output_path)
        saved_report = tested.DensityReport.load(output_path)
        nt.assert_dict_equal(saved_report.to_dict(), density_dictionary)


def test_density_report_from_files():
    shape = (4, 4, 4)
    raw = np.zeros(shape, dtype=np.int)
    raw[1:3, 1:3, 1:3] = 1
    pair = (0, 3)
    for t in itertools.product(pair, pair, pair):
        raw[t] = 2 # color each corner with the label 2
    raw[0, 1:3, 1:3] = 3
    voxel_dimensions = (2.0, 2.0, 2.0)
    annotation_voxeldata = voxcell.VoxelData(raw, voxel_dimensions)
    # Cell density
    cell_density_raw = np.zeros(shape, dtype=np.float)
    cell_density_raw[1:3, 1:3, 1:3] = 0.01
    cell_density_raw[2, 2, 2] = 0.02
    cell_density_raw[3, 3, 3] = 0.05
    cell_density_raw[0, 0, 0] = 0.025
    cell_density_raw[0, 1, 1] = 0.0125
    cell_density_raw[0, 1, 2] = 0.0125
    cell_density_voxeldata = voxcell.VoxelData(cell_density_raw, voxel_dimensions)
    # Glia density
    glia_density_raw = np.zeros(shape, dtype=np.float)
    glia_density_raw[1:3, 1:3, 1:3] = 0.03
    glia_density_raw[1, 2, 2] = 0.015
    glia_density_raw[2, 2, 1] = 0.015
    glia_density_raw[3, 0, 3] = 0.025
    glia_density_raw[0, 3, 0] = 0.025
    glia_density_raw[0, 0, 3] = 0.025
    glia_density_raw[0, 2, 1] = 0.0325
    glia_density_raw[0, 1, 2] = 0.0325
    glia_density_voxeldata = voxcell.VoxelData(glia_density_raw, voxel_dimensions)
    # Excitatory neurons density
    exc_density_raw = np.zeros(shape, dtype=np.float)
    exc_density_raw[1:3, 1:3, 1:3] = 0.04
    exc_density_raw[1, 1, 2] = 0.025
    exc_density_raw[2, 2, 1] = 0.015
    exc_density_raw[0, 0, 3] = 0.025
    exc_density_raw[0, 3, 0] = 0.025
    exc_density_raw[3, 0, 0] = 0.025
    exc_density_raw[0, 1, 1] = 0.0125
    exc_density_raw[0, 2, 2] = 0.0125
    exc_density_voxeldata = voxcell.VoxelData(exc_density_raw, voxel_dimensions)
    voxeldata_dict = {
        'cell_density.nrrd': cell_density_voxeldata,
        'glia_density.nrrd': glia_density_voxeldata,
        'exc_density.nrrd': exc_density_voxeldata
    }
    report = None
    with TemporaryDirectory() as tempdir:
        for filename, voxeldata in voxeldata_dict.items():
            output_path = Path(tempdir, filename)
            voxeldata.save_nrrd(str(output_path))
        leaf_ids = [1, 2, 3]
        filepaths = [Path(tempdir, filename).resolve() for filename in voxeldata_dict]
        report = tested.DensityReport.from_files(
            annotation_voxeldata, filepaths, leaf_ids)
    expected_report = {
        '1': {
            'cell_density': 0.00140625,
            'glia_density': 0.00328125,
            'exc_density': 0.004375
        },
        '2': {
            'cell_density': 0.001171875,
            'glia_density': 0.001171875,
            'exc_density': 0.001171875
        },
        '3': {
            'cell_density': 0.00078125,
            'glia_density': 0.00203125,
            'exc_density': 0.00078125
        }
    }
    for region_id in leaf_ids:
        region_id = str(region_id)
        actual_densities = report.to_dict()[region_id]
        expected_densities = expected_report[region_id]
        for density_type, actual_density in actual_densities.items():
            npt.assert_array_almost_equal(
                actual_density, expected_densities[density_type])


def test_histogram_to_dict():
    with nt.assert_raises(AtlasAnalysisError):
        # 3 values but only two bins
        tested.Histogram([1, 2, 2], [5, 10, 15], 10)

    with nt.assert_raises(AtlasAnalysisError):
        # 3 values, 3 bins, but the total count is 2 whereas the sum of bin counts
        # is 5
        tested.Histogram([1, 2, 2], [5, 10, 15, 20], 2)

def test_histogram_from_dict():
    dictionary = {
        'total': 3,
        '0': 0,
        '10': 2,
        '100': 1
    }
    actual_report = tested.Histogram.from_dict(dictionary)
    nt.assert_dict_equal(actual_report.to_dict(), dictionary)

    with nt.assert_raises(AtlasAnalysisError):
        # Missing 'total' field
        dictionary = {
            'total_count': 3,
            '0': 0,
            '10': 2,
            '100': 1
        }
        tested.Histogram.from_dict(dictionary)

    with nt.assert_raises(AtlasAnalysisError):
        # Only one bin edge
        dictionary = {
            'count': 3,
            '10': 2,
        }
        tested.Histogram.from_dict(dictionary)

    with nt.assert_raises(AtlasAnalysisError):
        # Some unexpected non-integer key
        dictionary = {
            'count': 3,
            '0': 0,
            '10': 2,
            'unexpected_key': 12.131415
        }
        tested.Histogram.from_dict(dictionary)


def test_connectivity_report_from_dict():
    dictionary = {
        'is_connected': False,
        'connected_component_histogram': {
            'total': 3,
            '0': 0,
            '10': 2,
            '100': 1,
            '1000': 0,
            '10000': 0,
            '100000': 0,
            '1000000': 0,
            '10000000': 0,
            '100000000': 0
        }
    }
    actual_report = tested.ConnectivityReport.from_dict(dictionary)
    nt.assert_dict_equal(actual_report.to_dict(), dictionary)

    with nt.assert_raises(AtlasAnalysisError):
        # Missing key 'connected_component_histogram'
        dictionary = {
            'is_connected': False
        }
        tested.ConnectivityReport.from_dict(dictionary)

    with nt.assert_raises(AtlasAnalysisError):
        # Connected is True but there are 3 connected components
        dictionary = {
            'is_connected': True,
            'connected_component_histogram': {
                'total': 3,
                '0': 0,
                '10': 2,
                '100': 1,
                '1000': 0,
                '10000': 0,
                '100000': 0,
                '1000000': 0,
                '10000000': 0,
                '100000000': 0
            }
        }
        tested.ConnectivityReport.from_dict(dictionary)

    with nt.assert_raises(AtlasAnalysisError):
        # has_cavities is False but there are 8 cavity voxels
        dictionary = {
            'has_cavities': False,
            'voxel_count': 8,
            'proportion': 8.0 / 116.0
        }
        tested.ConnectivityReport.from_dict(dictionary)

def test_connectivity_report_from_raw():
    raw = np.zeros([6] * 3)
    raw[:, 3, :] = 1
    raw[2, 5, 1] = 1
    raw[4:6, 0, 4:6] = 1
    expected = {
        'is_connected': False,
        'connected_component_histogram': {
            'total': 3,
            '0': 0,
            '10': 2,
            '100': 1,
            '1000': 0,
            '10000': 0,
            '100000': 0,
            '1000000': 0,
            '10000000': 0,
            '100000000': 0
        }
    }
    actual_report = tested.ConnectivityReport.from_raw(raw)
    nt.assert_dict_equal(actual_report.to_dict(), expected)
    raw[:, 3, :] = 0
    raw[2, 5, 1] = 0
    expected = {
        'is_connected': True,
        'connected_component_histogram': {
            'total': 1,
            '0': 0,
            '10': 1,
            '100': 0,
            '1000': 0,
            '10000': 0,
            '100000': 0,
            '1000000': 0,
            '10000000': 0,
            '100000000': 0
        }
    }
    actual_report = tested.ConnectivityReport.from_raw(raw)
    nt.assert_dict_equal(actual_report.to_dict(), expected)


def test_connectivity_report_save_as():
    connectivity_dictionary = {
        'is_connected': False,
        'connected_component_histogram': {
            'total': 3,
            '0': 0,
            '10': 2,
            '100': 1,
            '1000': 0,
            '10000': 0,
            '100000': 0,
            '1000000': 0,
            '10000000': 0,
            '100000000': 0
        }
    }
    report = tested.ConnectivityReport.from_dict(connectivity_dictionary)
    with TemporaryDirectory() as tempdir:
        output_path = str(Path(tempdir, 'connectivity_report.json'))
        report.save_as(output_path)
        saved_report = tested.ConnectivityReport.load(output_path)
        nt.assert_dict_equal(saved_report.to_dict(), connectivity_dictionary)


def test_cavity_report_from_raw():
    raw = np.zeros([6] * 3)
    raw[1:6, 1:6, 1:6] = 1
    raw[2, 5, 1] = 0
    raw[2:4, 2:4, 2:4] = 0
    expected = {
        'has_cavities': True,
        'voxel_count': 8,
        'proportion': 8.0 / 116.0
    }
    actual_report = tested.CavityReport.from_raw(raw)
    nt.assert_dict_equal(actual_report.to_dict(), expected)
    raw[2, 5, 1] = 1
    raw[2:4, 2:4, 2:4] = 1
    expected = {
        'has_cavities': False,
        'voxel_count': 0,
        'proportion': 0.0
    }
    actual_report = tested.CavityReport.from_raw(raw)
    nt.assert_dict_equal(actual_report.to_dict(), expected)

def test_cavity_report_from_dict():
    dictionary = {
        'has_cavities': True,
        'voxel_count': 8,
        'proportion': 8.0 / 116.0
    }
    actual_report = tested.CavityReport.from_dict(dictionary)
    nt.assert_dict_equal(actual_report.to_dict(), dictionary)

    with nt.assert_raises(AtlasAnalysisError):
        # Missing key 'has_cavities'
        dictionary = {
            'voxel_count': 8,
            'proportion': 8.0 / 116.0
        }
        tested.CavityReport.from_dict(dictionary)

    with nt.assert_raises(AtlasAnalysisError):
        # Missing key 'proportion'
        dictionary = {
            'has_cavities': True,
            'voxel_count': 8,
        }
        tested.CavityReport.from_dict(dictionary)

    with nt.assert_raises(AtlasAnalysisError):
        # has_cavities is False but there are 8 cavity voxels
        dictionary = {
            'has_cavities': False,
            'voxel_count': 8,
            'proportion': 8.0 / 116.0
        }
        tested.CavityReport.from_dict(dictionary)

def test_header_report_from_voxel_data():
    raw = np.zeros((3, 4, 5))
    voxel_dimensions = (10.0, 11.0, 12.0)
    offset = np.array((10.1, 100.2, 1000.3), dtype=np.float32)
    voxel_data = voxcell.VoxelData(raw, voxel_dimensions, offset)
    header = tested.HeaderReport.from_voxel_data(voxel_data).to_dict()
    expected_header = {
        'sizes': [3, 4, 5],
        'space_dimension': 3,
        'space_directions': [[10.0, 0.0, 0.0], [0.0, 11.0, 0.0], [0.0, 0.0, 12.0]],
        'space_origin': voxel_data.offset.tolist()
    }
    nt.assert_equal(isinstance(header['space_directions'], list), True)
    nt.assert_equal(isinstance(header['space_origin'], list), True)
    nt.assert_dict_equal(header, expected_header)

def test_header_report_from_dict():
    dictionary = {
        'sizes': [6, 6, 6],
        'space_dimension': 3,
        'space_directions': [[10, 0, 0], [0, 11, 0], [0, 0, 12]],
        'space_origin': [10.1, 100.2, 1000.3]
    }
    nt.assert_dict_equal(tested.HeaderReport.from_dict(dictionary).to_dict(), dictionary)

    with nt.assert_raises(AtlasAnalysisError):
        # Missing key 'sizes'
        dictionary = {
            'space_dimension': 3,
            'space_directions': [[10, 0, 0], [0, 11, 0], [0, 0, 12]],
            'space_origin': [10.1, 100.2, 1000.3]
        }
        tested.HeaderReport.from_dict(dictionary)

    with nt.assert_raises(AtlasAnalysisError):
        # Missing key 'space_dimension'
        dictionary = {
            'sizes': [6, 6, 6],
            'space_directions': [[10, 0, 0], [0, 11, 0], [0, 0, 12]],
            'space_origin': [10.1, 100.2, 1000.3]
        }
        tested.HeaderReport.from_dict(dictionary)

    with nt.assert_raises(AtlasAnalysisError):
        # Missing key 'space_directions'
        dictionary = {
            'sizes': [6, 6, 6],
            'space_dimension': 3,
            'space_origin': [10.1, 100.2, 1000.3]
        }
        tested.HeaderReport.from_dict(dictionary)

    with nt.assert_raises(AtlasAnalysisError):
        # Missing key 'space_origin'
        dictionary = {
            'sizes': [6, 6, 6],
            'space_dimension': 3,
            'space_directions': [[10, 0, 0], [0, 11, 0], [0, 0, 12]],
        }
        tested.HeaderReport.from_dict(dictionary)


def test_region_voxel_count_report_from_dict():
    dictionary = {
        'voxel_count': 73,
        'proportion': 73.0 / 85.0,
        'connectivity':{
            'is_connected': False,
            'connected_component_histogram':{
                'total': 3,
                '0': 0,
                '10': 1,
                '100': 2,
                '1000': 0,
                '10000': 0,
                '100000': 0,
                '1000000': 0,
                '10000000': 0,
                '100000000': 0
            }
        },
        'cavities':{
            'has_cavities': True,
            'voxel_count': 1,
            'proportion': 1.0 / 73.0
        }
    }
    nt.assert_dict_equal(tested.RegionVoxelCountReport.from_dict(dictionary).to_dict(), dictionary)

    with nt.assert_raises(AtlasAnalysisError):
        # Missing key 'proportion'
        dictionary = {
            'voxel_count': 73
        }
        tested.RegionVoxelCountReport.from_dict(dictionary)

    with nt.assert_raises(AtlasAnalysisError):
        # Missing key 'voxel_count'
        dictionary = {
            'proportion': 73.0 / 85.0
        }
        tested.RegionVoxelCountReport.from_dict(dictionary)


def test_header_report_save_as():
    header_dictionary = {
        'sizes': [6, 6, 6],
        'space_dimension': 3,
        'space_directions': [[10.0, 0.0, 0.0], [0.0, 11.0, 0.0], [0.0, 0.0, 12.0]],
        'space_origin': [10.1, 100.2, 1000.3]
    }
    report = tested.HeaderReport.from_dict(header_dictionary)
    with TemporaryDirectory() as tempdir:
        output_path = str(Path(tempdir, 'header_report.json'))
        report.save_as(output_path)
        saved_report = tested.HeaderReport.load(output_path)
        nt.assert_dict_equal(saved_report.to_dict(), header_dictionary)


def test_voxel_count_report_no_options():
    raw = np.zeros([6] * 3, dtype=np.int32)
    raw[0, :, :] = 1
    raw[2, 2, 1] = 1
    raw[4:6, 4:6, 4:6] = 1
    raw[4:6, 4:6, 0] = 2
    raw[0:3, 4:6, 0] = 2
    raw[0, 0, 5] = 2
    raw[1, 0, 0] = 3
    voxel_dimensions = (12.0, 10.0, 15.0)
    offset = np.array((1.1, 20.2, 300.3), dtype=np.float32)
    voxel_data = voxcell.VoxelData(raw, voxel_dimensions, offset)
    actual_report = tested.VoxelCountReport.from_voxel_data(voxel_data)
    expected_header = {
        'sizes': [6, 6, 6],
        'space_dimension': 3,
        'space_directions': [[12.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 15.0]],
        'space_origin': list(voxel_data.offset) # [1.1, 20.2, 300.3] with roundoff errors
    }
    expected = {
        'header': expected_header,
        'voxel_count': 216,
        'non_zero_voxel_count': 54,
        'proportion': 54.0 / 216.0,
        'number_of_regions': 3,
        'region_list': ['1', '2', '3'],
        'region_counts': {
            '1': {
                'voxel_count': 42,
                'proportion': 42.0 / 54.0
            },
            '2': {
                'voxel_count': 11,
                'proportion': 11.0 / 54.0
            },
            '3': {
                'voxel_count': 1,
                'proportion': 1.0 / 54.0
            }
        }
    }
    nt.assert_dict_equal(actual_report.to_dict(), expected)

def test_voxel_count_report_with_options():
    raw = np.zeros([7] * 3, dtype=np.int16)
    raw[0, :, :] = 1
    raw[2, 2, 1] = 1
    raw[4:7, 4:7, 4:7] = 1
    raw[4:6, 4:6, 0] = 2
    raw[0:3, 4:6, 0] = 2
    raw[0, 0, 5] = 2
    raw[1, 0, 0] = 3
    raw[5, 5, 5] = 0
    voxel_dimensions = (12.0, 10.0, 15.0)
    offset = np.array((1.1, 20.2, 300.3), dtype=np.float32)
    voxel_data = voxcell.VoxelData(raw, voxel_dimensions, offset)
    actual_report = tested.VoxelCountReport.from_voxel_data(
        voxel_data, connectivity_is_required=True, cavities_are_required=True)
    expected_header = {
        'sizes': [7, 7, 7],
        'space_dimension': 3,
        'space_directions': [[12.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 15.0]],
        'space_origin': list(voxel_data.offset) # [1.1, 20.2, 300.3] with roundoff errors
    }
    expected = {
        'header': expected_header,
        'voxel_count': 343,
        'non_zero_voxel_count': 85,
        'proportion': 85.0 / 343.0,
        'number_of_regions': 3,
        'region_list': ['1', '2', '3'],
        'connectivity': {
            'is_connected': False,
            'connected_component_histogram':{
                    'total': 4,
                    '0': 0,
                    '10': 2,
                    '100': 2,
                    '1000': 0,
                    '10000': 0,
                    '100000': 0,
                    '1000000': 0,
                    '10000000': 0,
                    '100000000': 0
            }
        },
        'cavities':{
            'has_cavities': True,
            'voxel_count': 1,
            'proportion': 1.0 / 85.0
        },
        'region_counts': {
            '1': {
                'voxel_count': 73,
                'proportion': 73.0 / 85.0,
                'connectivity':{
                    'is_connected': False,
                    'connected_component_histogram':{
                        'total': 3,
                        '0': 0,
                        '10': 1,
                        '100': 2,
                        '1000': 0,
                        '10000': 0,
                        '100000': 0,
                        '1000000': 0,
                        '10000000': 0,
                        '100000000': 0
                    }
                },
                'cavities':{
                    'has_cavities': True,
                    'voxel_count': 1,
                    'proportion': 1.0 / 73.0
                }
            },
            '2': {
                'voxel_count': 11,
                'proportion': 11.0 / 85.0,
                'connectivity':{
                    'is_connected': False,
                    'connected_component_histogram':{
                        'total': 3,
                        '0': 0,
                        '10': 3,
                        '100': 0,
                        '1000': 0,
                        '10000': 0,
                        '100000': 0,
                        '1000000': 0,
                        '10000000': 0,
                        '100000000': 0
                    }
                },
                'cavities':{
                    'has_cavities': False,
                    'voxel_count': 0,
                    'proportion': 0.0
                }
            },
            '3': {
                'voxel_count': 1,
                'proportion': 1.0 / 85.0,
                'connectivity':{
                    'is_connected': True,
                    'connected_component_histogram': {
                        'total': 1,
                        '0': 0,
                        '10': 1,
                        '100': 0,
                        '1000': 0,
                        '10000': 0,
                        '100000': 0,
                        '1000000': 0,
                        '10000000': 0,
                        '100000000': 0
                    }
                },
                'cavities':{
                    'has_cavities': False,
                    'voxel_count': 0,
                    'proportion': 0.0
                }
            }
        }
    }
    nt.assert_dict_equal(actual_report.to_dict(), expected)
    with TemporaryDirectory() as tempdir:
        output_path = str(Path(tempdir, 'voxel_count_report.json'))
        actual_report.save_as(output_path)
        saved_report = tested.VoxelCountReport.load(output_path)
        nt.assert_dict_equal(saved_report.to_dict(), expected)
