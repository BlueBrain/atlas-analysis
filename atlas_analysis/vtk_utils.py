""" Utilities for vtk """
# pylint: disable=no-name-in-module
import vtk

from vtk.util import numpy_support  # pylint: disable=import-error


def convert_points_to_vtk(points):
    """ Convert points from numpy array to vtk points.

    Args:
        points (np.array([[x1,y1,z1], ..., [x2,y2,z2]])): the points to convert.

    Returns:
        Points using the vtk format (a vtkPoints object)
    """
    if isinstance(points, vtk.vtkPoints):
        return points
    vtk_points = vtk.vtkPoints()
    for point in points:
        vtk_points.InsertNextPoint(point[0], point[1], point[2])
    return vtk_points


def unstructuredgrid_to_polydata(unstructured_grid):
    """ Convert an unstructured grid to a polydata object """
    geom_filter = vtk.vtkGeometryFilter()
    geom_filter.SetInputData(unstructured_grid)
    geom_filter.Update()
    return geom_filter.GetOutput()


def load_stl(input_file):
    """ Function to load a vtk polydata object from a stl file

    Args:
        input_file: the path to the .stl file (str).

    Returns:
        A vtkUnstructuredGridReader already updated.
    """
    reader = vtk.vtkSTLReader()
    reader.SetFileName(input_file)
    reader.Update()
    return reader.GetOutput()


def save_polydata_to_stl(polydata, output_path):
    """ Save a polydata to the stl format (vtk).

    Args:
        unstructured_grid: a vtkUnstructuredGrid object.
        output_path: the ouput name of the file. You need to add the .stl suffix.

    Returns:
        The output file name.
    """
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(polydata)
    writer.Write()
    return output_path


def save_unstructuredgrid_to_stl(unstructured_grid, output_path):
    """ Save an unstructured grid to the stl format (vtk).

    Args:
        unstructured_grid: a vtkUnstructuredGrid object.
        output_path: the ouput name of the file. You need to add the .stl suffix.

    Returns:
        The output file name.
    """
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(unstructuredgrid_to_polydata(unstructured_grid))
    writer.Write()
    return output_path


def voxeldata_to_vtkImageData(voxel_data):
    """ Convert a VoxelData object into a vtkImageData object """
    image_data = vtk.vtkImageData()
    array_data = numpy_support.numpy_to_vtk(voxel_data.raw.transpose(2, 1, 0).flatten(),
                                            deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)

    image_data.SetDimensions(voxel_data.raw.shape)
    image_data.SetSpacing(voxel_data.voxel_dimensions)
    image_data.SetOrigin(voxel_data.offset)
    image_data.GetPointData().SetScalars(array_data)
    return image_data


def create_cutter_from_stl(stl_path):
    """ Create a vtkCutter from a stl unstructured_grid file """
    polydata = load_stl(stl_path)
    cutter = vtk.vtkCutter()
    cutter.SetInputData(polydata)
    return cutter


def update_vtk_plane(plane, point, normal):
    """ Update a vtkPlane with a loc and a normal

    Args:
        plane: a vtkPlane
        point: a position in 3d [x, y, z]
        normal: A 3d vector normal to the plane (normalized or not)
    """
    plane.SetNormal(normal[0], normal[1], normal[2])
    plane.SetOrigin(point[0], point[1], point[2])


def create_vtk_spline(points):
    """ Create a vtkParametricSpline using points as inputs"""
    spline = vtk.vtkParametricSpline()
    spline.SetPoints(convert_points_to_vtk(points))
    spline.ParameterizeByLengthOn()
    spline.ClosedOff()
    return spline


def alpha_hull(points, alpha=16, tol=32):
    """ Convert points cloud to a surface mesh using alpha hull algorithm.

    https://en.wikipedia.org/wiki/Alpha_shape

    Args:
        points: the numpy point cloud
        alpha: alpha used in the alpha hull algorithm
        tol: tolerance for the alpha hull algorithm

    Returns:
        an vtkUnstructuredGrid object representing the contour
    """
    poly_points = vtk.vtkPolyData()
    poly_points.SetPoints(convert_points_to_vtk(points))

    contour = vtk.vtkDelaunay3D()
    contour.SetInputData(poly_points)

    contour.SetTolerance(tol)
    contour.SetAlpha(alpha)
    contour.Update()
    return contour.GetOutput()


def marching_cubes(image_data, iso_value=0.5):
    """ Convert vtkImageData to a surface mesh using marching cubes algorithm.

    https://en.wikipedia.org/wiki/Marching_cubes

    Args:
        image_data: a vtkImageData
        iso_value: iso value used in the marching cubes algorithm

    Returns:
        an vtkPolyData object representing the contour
    """
    contour = vtk.vtkMarchingCubes()
    contour.SetInputData(image_data)
    contour.ComputeNormalsOn()
    contour.SetValue(0, iso_value)
    contour.Update()
    return contour.GetOutput()
