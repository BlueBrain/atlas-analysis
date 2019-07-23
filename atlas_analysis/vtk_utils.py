""" Utilities for vtk """

# pylint: disable=no-name-in-module
from vtk import vtkPoints, vtkUnstructuredGridReader, vtkCutter, vtkParametricSpline


def convert_points_to_vtk(points):
    """ Convert points from numpy array to vtk points.

    Args:
        points (np.array([[x1,y1,z1], ..., [x2,y2,z2]])): the points to convert.

    Returns:
        Points using the vtk format (a vtkPoints object)
    """
    vtk_points = vtkPoints()
    for point in points:
        vtk_points.InsertNextPoint(point[0], point[1], point[2])
    return vtk_points


def load_unstructured_grid(input_file):
    """ Function to load a vtk unstructured grid object

    Args:
        input_file: the path to the .vtu file (str).

    Returns:
        A vtkUnstructuredGridReader already updated.
    """
    reader = vtkUnstructuredGridReader()
    reader.SetFileName(input_file)
    reader.Update()
    return reader


def create_cutter_from_vtu(upper_filename):
    """ Create a vtkCutter from a vtu unstructured_grid file """
    reader = load_unstructured_grid(upper_filename)
    cutter = vtkCutter()
    cutter.SetInputData(reader.GetOutput())
    return cutter


def update_vtk_plane(plane, loc, normal):
    """ Update a vtkPlane with a loc and a normal

    Args:
        plane: a vtkPlane
        loc: a position in 3d
        normal: the normmal vector describing the plane orientation
    """
    plane.SetNormal(normal[0], normal[1], normal[2])
    plane.SetOrigin(loc[0], loc[1], loc[2])


def create_vtk_spline(points):
    """ Create a vtkParametricSpline using points as inputs"""
    spline = vtkParametricSpline()
    spline.SetPoints(convert_points_to_vtk(points))
    spline.ParameterizeByLengthOn()
    return spline
