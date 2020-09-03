import difflib
from tempfile import TemporaryDirectory
from pathlib import Path
from functools import reduce

import numpy.testing as npt
import numpy as np
import nose.tools as nt

import atlas_analysis.vtk_utils as tested

import vtk

from tests.utils import path, load_nrrd


def assert_polydata_equals(p0, p1):
    """Simple equal for polydata ... """
    nt.assert_equal(p0.GetNumberOfPoints(), p1.GetNumberOfPoints())

    def _get_points(poly):
        res = sorted([poly.GetPoint(idx) for idx in range(0, poly.GetNumberOfPoints())])
        return res

    npt.assert_allclose(_get_points(p0), _get_points(p1))


def create_cube():
    P0 = [0.0, 0.0, 0.0]
    P1 = [1.0, 0.0, 0.0]
    P2 = [1.0, 1.0, 0.0]
    P3 = [0.0, 1.0, 0.0]
    P4 = [0.0, 0.0, 1.0]
    P5 = [1.0, 0.0, 1.0]
    P6 = [1.0, 1.0, 1.0]
    P7 = [0.0, 1.0, 1.0]

    # Create the points
    points = vtk.vtkPoints()
    points.InsertNextPoint(P0)
    points.InsertNextPoint(P1)
    points.InsertNextPoint(P2)
    points.InsertNextPoint(P3)
    points.InsertNextPoint(P4)
    points.InsertNextPoint(P5)
    points.InsertNextPoint(P6)
    points.InsertNextPoint(P7)

    # Create a hexahedron from the points
    hex = vtk.vtkHexahedron()
    hex.GetPointIds().SetId(0, 0)
    hex.GetPointIds().SetId(1, 1)
    hex.GetPointIds().SetId(2, 2)
    hex.GetPointIds().SetId(3, 3)
    hex.GetPointIds().SetId(4, 4)
    hex.GetPointIds().SetId(5, 5)
    hex.GetPointIds().SetId(6, 6)
    hex.GetPointIds().SetId(7, 7)

    # Add the hexahedron to a cell array
    hexs = vtk.vtkCellArray()
    hexs.InsertNextCell(hex)

    # Add the points and hexahedron to an unstructured grid
    uGrid = vtk.vtkUnstructuredGrid()
    uGrid.SetPoints(points)
    uGrid.InsertNextCell(hex.GetCellType(), hex.GetPointIds())
    return uGrid


def test_convert_points_to_vtk():
    points = np.array([[1., 2., 3.], [4., 5., 6.]])
    res = tested.convert_points_to_vtk(points)
    point0 = res.GetPoint(0)
    point1 = res.GetPoint(1)

    nt.assert_equal(points.shape[0], res.GetNumberOfPoints())
    npt.assert_allclose(points[0], point0)
    npt.assert_allclose(points[1], point1)

    expected = vtk.vtkPoints()
    expected.InsertNextPoint([1, 2, 3])
    expected.InsertNextPoint([4, 5, 6])

    res = tested.convert_points_to_vtk(expected)
    nt.assert_is(res, expected)


def test_unstructuredgrid_to_polydata():
    res = tested.unstructuredgrid_to_polydata(create_cube())
    tested.save_polydata_to_stl(res, '/tmp/cube.stl')
    cube = tested.load_stl(str(path('cube.stl')))
    assert_polydata_equals(res, cube)


def test_load_stl():
    res = tested.load_stl(str(path('cube.stl')))
    nt.assert_equal(res.GetNumberOfPoints(), 8)
    nt.assert_equal(res.GetNumberOfPolys(), 12)
    npt.assert_allclose(res.GetPoint(0), [0, 0, 0])
    npt.assert_allclose(res.GetPoint(1), [0.0, 0.0, 1.0])
    npt.assert_allclose(res.GetPoint(2), [0.0, 1.0, 0.0])
    npt.assert_allclose(res.GetPoint(3), [0.0, 1.0, 1.0])
    npt.assert_allclose(res.GetPoint(4), [1.0, 0.0, 0.0])
    npt.assert_allclose(res.GetPoint(5), [1.0, 1.0, 0.0])
    npt.assert_allclose(res.GetPoint(6), [1.0, 0.0, 1.0])
    npt.assert_allclose(res.GetPoint(7), [1.0, 1.0, 1.0])


def test_save_polydata_to_stl():
    with TemporaryDirectory() as directory:
        cube = tested.load_stl(str(path('cube.stl')))
        output_path = str(Path(directory, 'tested_cube.stl'))
        tested.save_polydata_to_stl(cube, output_path)

        res = tested.load_stl(output_path)
        assert_polydata_equals(res, cube)


def test_save_unstructuredgrid_to_stl():
    cube = tested.unstructuredgrid_to_polydata(create_cube())
    with TemporaryDirectory() as directory:
        output_path = str(Path(directory, 'tested_cube.stl'))
        tested.save_unstructuredgrid_to_stl(cube, output_path)
        res = tested.load_stl(output_path)
        assert_polydata_equals(res, cube)


def test_voxeldata_to_vtkImageData():
    voxel_data = load_nrrd("data.nrrd")
    res = tested.voxeldata_to_vtkImageData(voxel_data)
    npt.assert_allclose(res.GetSpacing(), voxel_data.voxel_dimensions)
    npt.assert_allclose(res.GetOrigin(), voxel_data.offset)
    npt.assert_allclose(res.GetDimensions(), voxel_data.raw.shape)
    nt.eq_(res.GetNumberOfPoints(), reduce(lambda x, y: x * y, voxel_data.raw.shape))
    npt.assert_almost_equal(res.GetScalarComponentAsDouble(0, 0, 0, 0), voxel_data.raw[0, 0, 0])
    npt.assert_almost_equal(res.GetScalarComponentAsDouble(0, 1, 0, 0), voxel_data.raw[0, 1, 0])
    npt.assert_almost_equal(res.GetScalarComponentAsDouble(0, 2, 0, 0), voxel_data.raw[0, 2, 0])


def test_create_cutter_from_stl():
    # not sure how to test more than just checking that the thing does not raise
    tested.create_cutter_from_stl(str(path('cube.stl')))


def test_update_vtk_plane():
    plane = vtk.vtkPlane()
    tested.update_vtk_plane(plane, [12, 12, 12], [0, 1, 0])
    npt.assert_almost_equal(plane.GetOrigin(), [12, 12, 12])
    npt.assert_almost_equal(plane.GetNormal(), [0, 1, 0])


def test_vtk_spline():
    points = np.array(
        [[0., 0, 0], [0.1, 0, 0], [0.2, 0, 0], [0.3, 0, 0], [0.4, 0, 0], [0.5, 0, 0], [0.6, 0, 0],
         [0.7, 0, 0], [0.8, 0, 0], [0.9, 0, 0], [1, 0, 0]])
    res = tested.create_vtk_spline(points)
    p = [0, 0, 0]
    d = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    u = [0., 0., 0.]
    res.Evaluate(u, p, d)
    npt.assert_allclose(p, [0, 0, 0])
    steps = np.linspace(0, 1, 50, endpoint=True)
    for step in steps:
        # It is strange how the spline evaluation behaves sometimes ...
        # The step should exactly equals to the u[0] but this does not happen in practice
        u[0] = step
        res.Evaluate(u, p, d)
        nt.ok_(abs(u[0] - step) < 0.02)

