""" Visualization module to represent vector fields in 3d with vtk. This is convenient to
check large orientation dataset (up to 500000 orientations).
"""
from os.path import exists

# pylint: disable=no-name-in-module
from vtk import (vtkPolyData, vtkPolyDataMapper, vtkActor,
                 vtkPoints, vtkCellArray, vtkLine, vtkRenderer,
                 vtkRenderWindow, vtkRenderWindowInteractor,
                 vtkDataSetMapper)
import numpy as np
from pyquaternion import Quaternion

import voxcell

from atlas_analysis.vtk_utils import load_stl
from atlas_analysis.utils import ensure_list


def _line_actor(points, lines):
    """ Create a line actor from points and lines """
    lines_polyData = vtkPolyData()
    lines_polyData.SetPoints(points)
    lines_polyData.SetLines(lines)
    mapper = vtkPolyDataMapper()
    mapper.SetInputData(lines_polyData)
    mapper.Update()

    line_actor = vtkActor()
    line_actor.SetMapper(mapper)
    return line_actor


def _create_vector_actor(locs, rots, axis, size_multiplier=200.):
    """ Create 3d vectors using rot and loc and create actor used for the display.

    Args :
        locs: list of positions for the vector origins (array([[x, y, z], [x2, y2, z2], ...)).
        rots Quaternion: the quaternion that defines the orientation field.
        axis: the axis to represent (0,0,1) for long, (0,1,0) for radial, (1,0,0) for transverse.
        size_multiplier: the vector norm in the scene (float).

    Returns :
        the vtk object for rendering vectors (vtkActor).
    """
    points = vtkPoints()
    lines = vtkCellArray()
    ids = 0
    for loc, rot in zip(locs, rots):
        loc = np.array(loc)
        dest = loc + size_multiplier * np.array(rot.rotate(axis))
        points.InsertNextPoint(loc)
        points.InsertNextPoint(dest)
        line = vtkLine()
        line.GetPointIds().SetId(0, ids)
        line.GetPointIds().SetId(1, ids + 1)
        lines.InsertNextCell(line)
        ids += 2
    return _line_actor(points, lines)


def render(orientation_file=None, rad=False, long=False, trans=False,
           stl_files=None, orientation_sampling=40000):
    """ The global vtk renderer

    Args:
        orientation_file: an nrrd orientation file
        rad: in case of a provided orientation file display the rad orientation.
        long: in case of a provided orientation file display the long orientation.
        trans: in case of a provided orientation file display the trans orientation.
        stl_files: a list of stl files that you want to display
        orientation_sampling: the number of unique orientation you want to display
    """
    #  Create graphics renderer
    global_renderer = vtkRenderer()
    window_renderer = vtkRenderWindow()
    window_renderer.AddRenderer(global_renderer)
    iren = vtkRenderWindowInteractor()
    iren.SetRenderWindow(window_renderer)

    # Axes representation
    if orientation_file:
        if not exists(orientation_file):
            raise FileExistsError('{} does not exists'.format(orientation_file))
        qf = voxcell.OrientationField.load_nrrd(orientation_file)

        idx = np.array(np.nonzero(qf.raw)).T
        sample_idx = np.random.choice(len(idx), orientation_sampling, replace=False)
        idx = idx[sample_idx, :3]
        locs = qf.indices_to_positions(idx)
        rots = np.array([Quaternion(qf.raw[tuple(i)]) for i in idx])

        if long:
            lines_actor_long = _create_vector_actor(locs, rots, (0, 0, 1), size_multiplier=200)
            lines_actor_long.GetProperty().SetColor(0, 0, 1)
            lines_actor_long.GetProperty().SetLineWidth(2)
            global_renderer.AddActor(lines_actor_long)

        if rad:
            lines_actor_rad = _create_vector_actor(locs, rots, (0, 1, 0), size_multiplier=200)
            lines_actor_rad.GetProperty().SetColor(0, 1, 0)
            lines_actor_rad.GetProperty().SetLineWidth(2)
            global_renderer.AddActor(lines_actor_rad)

        if trans:
            lines_actor_trans = _create_vector_actor(locs, rots, (1, 0, 0), size_multiplier=200)
            lines_actor_trans.GetProperty().SetColor(1, 0, 0)
            lines_actor_trans.GetProperty().SetLineWidth(2)
            global_renderer.AddActor(lines_actor_trans)

    #  Display stl files
    if stl_files:
        stl_files = ensure_list(stl_files)
        for stl in stl_files:
            if not exists(stl):
                raise FileExistsError('{} does not exists'.format(stl))
            mesh_data = load_stl(stl)
            mapper = vtkDataSetMapper()
            mapper.SetInputData(mesh_data)
            triangulation = vtkActor()
            triangulation.SetMapper(mapper)
            triangulation.GetProperty().SetColor(1, 0, 0)
            triangulation.GetProperty().SetOpacity(0.3)
            global_renderer.AddActor(triangulation)

    #  TODO: re add the preprocess part when the plane part is included in the lib

    # Add the actors to the renderer, set the background and size
    global_renderer.SetBackground(1, 1, 1)
    window_renderer.SetSize(750, 750)
    window_renderer.Render()

    cam1 = global_renderer.GetActiveCamera()
    cam1.Zoom(1.5)

    iren.Initialize()
    window_renderer.Render()
    iren.Start()
