""" Compute longitudinal and radial orientations, the longitudinala, radial and
transverse (l,r,t) coordinates and all placement hint atlases needed for the
circuit building.

Longitudinal orientations:
    The algorithm relies on manually created planes that follow the atlas
    orientation. We use a weighted combination of the plane's quaternions to
    compute the voxel longitudinal orientation.

Radial orientations:
    Radial orientations of neurons is defined as the vector field we would
    obtain by stretching the upper shell to the lower shell. To achieve this, we
    use the upper and lower shell of the atlas and the planes cited above.
    We find the longitudinal plane that includes the voxel and we slice the
    upper and lower shell accordingly. We then have planar 3d points
    corresponding to the upper and lower shell cuts. We use these points to
    create two 3d spline functions that fit the shell's slice and goes from 0 to
    1. The radial orientation is then defined by the vector that includes the
    voxel and link two similar values for the upper and lower splines.  Ex:
    vector that is composed of the 0.3 value of the upper shell, the 0.3 value
    of the lower shell and the voxel center.

Final orientations:
    It is a combination of the longitudinal and transverse quaternions that
    gives:
        q.rotate([0,0,1]) == longitudinal orient
        and
        q.rotate([0,1,0]) == radial orient.

Coordinates:
    Coordinates are the longitudinal, transverse and radial coordinates. This
    set of coordinates gives a natural coordinate system for atlas.

Longitudinal Coordinate:
    The longitudinal coordinate is computed using the planes. The 0 coordinate
    corresponds to the first blender plane and 1 to the last one. The idea is to
    find the longitudinal plane that cut the voxel, and to compute longitudinal
    values.

Transverse and Radial Coordinates:
    We use the output vector of the transverse orientation. The transverse
    coordinate corresponds to the spline value discussed before (0.3 in the
    example). The radial value is the ratio norm(upper, point)/norm(upper,
    lower) using the same vector.

Placement Hints atlases:
    These atlas are used for the circuit building. The y correspond to the
    height of the voxel from the upper shell to lower shell. This is free for us
    here and this part is done during the radial coordinate computation. The
    layers files correspond to the boundaries of layers projected on the radial
    axis.

Performance:
    It is very time consuming to cut shells, compute splines etc for each voxel
    so we pre-compute few thousands of splines and use these approximate splines
    to compute the radial and transverse coordinates. A kdtree is used to
    retrieve the closest plane quickly.
"""
import sys
import os
import logging
from collections import namedtuple

import vtk
from vtk.util.numpy_support import vtk_to_numpy  # pylint: disable=import-error, no-name-in-module
from pathos.multiprocessing import ProcessingPool

import numpy as np
from scipy.spatial import cKDTree  # pylint: disable=no-name-in-module
from pyquaternion import Quaternion

import voxcell

from atlas_analysis.atlas import sample_positions_from_voxeldata
from atlas_analysis.maths import normalize_vector
from atlas_analysis.vtk_utils import (
    create_vtk_spline,
    update_vtk_plane,
    create_cutter_from_stl,
)
from atlas_analysis.utils import pairwise, save_raw
from atlas_analysis.planes.maths import distances_to_planes
from atlas_analysis.planes.planes import (
    load_planes_centerline,
    add_interpolated_planes,
)
from atlas_analysis.constants import X, Y, Z, YVECTOR, ZVECTOR

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
L = logging.getLogger(__name__)

Layer = namedtuple('Layer', ['name', 'ratio', 'raw'])


def _cut_shell(origin, quat, vtk_plane, vtk_cutter, radius=2500.0):
    """Cut a shell mesh using a vtkCutter according to a loc and a quaternion.

    Args:
        origin(np.array): a position in 3d that defines the center of the plane (array([x1 ,y1, z1])
        quat(Quaternion): a Quaternion defining the orientation of the plane
        vtk_plane(vtkPlane): a vtkPlane that will be update using loc and quat
        vtk_cutter(vtkCutter): a vtkCutter that includes the mesh to cut
        radius(float): a l2 distance from loc to points to filter points (float)

    Returns:
        np.array: all positions of the mesh slice.

    Notes:
        the distance radius is used to prevent planes to cut shells far from the loc. This can
        happens in case of banana shaped volumes.
    """
    update_vtk_plane(vtk_plane, origin, quat.rotate(ZVECTOR))
    vtk_cutter.SetCutFunction(vtk_plane)
    vtk_cutter.Update()

    pos = vtk_to_numpy(vtk_cutter.GetOutput().GetPoints().GetData())
    dists = np.linalg.norm(pos - origin, axis=1)

    # for banana type of atlases a plane can cut the volume multiple times we keep only the < radius
    # points
    pos = pos[dists < radius]
    return pos


def _plane_basis_projection(points, loc, basis):
    """Projection of points within a plane.

    Args:
        points(np.array): the point cloud (np.array([[x1,y1,z1], ..., [x2,y2,z2]])) points
        must be planar.
        loc(np.array): the plane center in 3d space (np.array([x,y,z])). Arbitrary defines
        the origin of the new basis.
        basis(np.array): a 3x3 array containing the orthogonal basis composed of the plane normal,
        the vector linking the mean of the upper shell points and the mean of the lower points and
        the cross product of the two.

    Returns:
        np.array: (np.array([[x'1,y'1], ..., [x'2,y'2]])) the newly projected points on the plane.

    Notes:
        The idea of this basis change is to move from (x, y, z) to (x',y',z'). The original
        basis being the voxel basis and the "prime" basis being an orthonormal basis with the base
        vectors (nx', ny') included in the cut plane and nz' the normal of the plane. In this use
        case the new basis is the combination of the plan normal, the mean upper to lower vector
        and the cross product of these two vectors.
        By definition, points and loc are within the same plane. Therefore, the z' component,
        which is the normal to the plane, has to be very close to zero. So we move from (x, y, z)
        to (x',y', 0) and then to (x',y').
    """

    shifted_points = points - loc
    coords = []
    for pp in shifted_points:
        v = np.linalg.solve(basis, pp)
        coords.append(v[:2])
    return coords


def _clean_cut_array(points, loc, basis, distance_increment=5, is_upper=True):
    """Clean the plane cut array.

    Args:
        points(np.array): the point cloud (np.array([[x1,y1,z1], ..., [x2,y2,z2]]))
        points must be planar.
        loc(np.array): the plane center location (np.array([x,y,z])). Arbitrary defines the origin
        of the new basis.
        distance_increment (int/float): initial distance to merge two (or more) points. Default
        value is 5 which is smaller than the typical voxel edge length.
        basis(np.array): a 3x3 array containing the orthogonal basis composed of the plane normal
        (zaxis), the vector linking the mean of the upper shell points and the mean of
        the lower points (y axis) and the cross product of the two (x axis).
        is_upper(bool): defines if the array represents an upper or lower cut.
    Returns:
        np.array: A sample of points (np.array([[x1,y1,z1], ..., [x2,y2,z2]])) corresponding
        to the concave hull of points in the projected plan of the plane.

    Notes:
        Depending on the orientation of the cutting plane and shell topology, the shell cut can be
        more than one voxel thick. That means some voxels share the same x coordinate and this
        messes up the spline process. The purpose of this phase is to create a 2d concave hull of
        the points in the plane coordinates (basis) and to return points that compose this hull.
        This is super custom, poorly optimized and a library must do this way better than me
        but I did not find it (alpha hull added with circum gave poor results).
    """
    # pylint: disable=too-many-locals
    # points projected on the plane coordinates
    coords = np.array(_plane_basis_projection(points, loc, basis))

    tree = cKDTree(coords)  # pylint: disable=not-callable

    to_keep, to_skip = [], set()

    for i, point in enumerate(coords):
        if i in to_skip:
            continue

        to_skip.add(i)
        to_keep.append(i)

        if len(to_skip) == len(coords):
            break

        local_distance = 0
        not_in_skip_mask = []
        while len(not_in_skip_mask) == 0:
            local_distance += distance_increment

            # neighbors is a list of coords' indexes
            neighbors = tree.query_ball_point(point, local_distance)
            indices = set(neighbors) - to_skip
            not_in_skip_mask = np.fromiter(indices, int, len(indices))

        ys = coords[not_in_skip_mask, Y]

        # keep the highest or lowest neighbor --> highest or lowest y' value
        best = np.argmin(ys) if is_upper else np.argmax(ys)

        # retrieve initial idx
        best_arg = int(not_in_skip_mask[best])

        # skip all points before this one
        left = list(range(i, best_arg))
        to_skip.update(left)

    return points[to_keep]


def _global_sampling(upper_spline, lower_spline, nb_steps):
    """Returns the grid sampling of the orientation vectors.

    Args:
        upper_spline(vtk.vtkParametricSpline): the upper shell spline
        lower_spline(vtk.vtkParametricSpline): the lower shell spline
        nb_steps(int): the lower shell spline

    Returns:
        np.array: a multidimensional numpy array containing for each sampled point:
        [x, y, z, r, t, orient_x, orient_y, orient_z]
    """
    # pylint: disable=too-many-locals
    steps = np.linspace(0, 1, nb_steps, endpoint=True)
    sampling_upper = np.zeros((nb_steps, 3), dtype=float)
    sampling_lower = np.zeros((nb_steps, 3), dtype=float)
    ptu = [0, 0, 0]
    ptl = [0, 0, 0]
    d = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    u = [0.0, 0.0, 0.0]
    pos_all = []
    for i, step in enumerate(steps):
        u[0] = step
        upper_spline.Evaluate(u, ptu, d)
        lower_spline.Evaluate(u, ptl, d)
        sampling_upper[i] = ptu
        sampling_lower[i] = ptl
        xs = np.linspace(ptu[X], ptl[X], nb_steps, endpoint=True)
        ys = np.linspace(ptu[Y], ptl[Y], nb_steps, endpoint=True)
        zs = np.linspace(ptu[Z], ptl[Z], nb_steps, endpoint=True)
        rs = np.linspace(0, 1, nb_steps, endpoint=True)
        ts = np.full((nb_steps,), step)
        vect = np.asarray(ptl) - np.asarray(ptu)
        vects = np.full((nb_steps, vect.shape[0]), vect)
        pos = np.column_stack((xs, ys, zs, rs, ts, vects[:, 0], vects[:, 1], vects[:, 2]))
        pos_all.append(pos)
    return np.vstack(pos_all).astype(np.float32)


def _create_spline_indexing(planes, upper_cutter, lower_cutter, nb_spline_steps):
    """Function that creates splines to describe the upper and lower shells and the indexing.

    Args:
        planes(list): the list of Plane objects
        upper_cutter(vtkCutter): a vtkCutter that includes the upper_mesh to cut
        lower_cutter(vtkCutter): a vtkCutter that includes the lower_mesh to cut
        nb_spline_steps(int): the number of points after sampling for splines

    returns:
        np.array: (np.array([[x1,y1,z1, r1, t1, orient_x1, orient_y1, orient_z1],
        ..., [x2,y2,z2, r2, t2, orient_x2, orient_y2, orient_z2]])) pos of all sampled points

    Notes :
        The algorithm cuts the upper and lower shells according to planes.
        The clean_cut_array function is used to clean the positions from mesh's slices.
        Points from the slices are used to create parametric splines.
        Radial vectors are created using these splines.
        These vectors are sampled and an indexer is created to map voxel idx with splines ids
        Splines are kept in a map with the spline ids as index.
    """
    # pylint: disable=too-many-locals
    sampled_points = []
    nb_unused_planes = 0
    vtk_plane = vtk.vtkPlane()  # pylint: disable=no-member
    nb_points = len(planes) * nb_spline_steps * nb_spline_steps

    L.info('%i planes are used for indexing', len(planes))
    L.info('%i steps are used for indexing', nb_spline_steps)
    L.info('Ends up in %i indexed points', nb_points)

    for plane in planes:

        loc, rot = plane.point, plane.get_quaternion(ZVECTOR)

        array_upper = _cut_shell(loc, rot, vtk_plane, upper_cutter)
        array_lower = _cut_shell(loc, rot, vtk_plane, lower_cutter)

        if len(array_upper) < 3 or len(array_lower) < 3:
            nb_unused_planes += 1
            continue

        # this is a work in process so I keep it outside a function atm
        # =========
        mean_upper = array_upper.mean(axis=0)
        mean_lower = array_lower.mean(axis=0)

        # mean upper to lower shell vector
        up_down_mean_vector = mean_lower - mean_upper

        plane_normal = vtk_plane.GetNormal()

        # define a vector `prod_vect` such that
        # (`plane_normal`, `up_down_mean_vector`, `prod_vect`) is a direct orthogonal basis.
        # `up_down_mean_vector`, `prod_vect` are included in the plane.
        prod_vect = np.cross(plane_normal, up_down_mean_vector)

        # Sorting the cut arrays using the dot product between the mean upper and the points
        # the points are sorted from left to right inside the plane.
        uu = array_upper - mean_upper
        vu = np.dot(uu, prod_vect)
        array_upper_pre_clean = array_upper[vu.argsort()]

        dd = array_lower - mean_lower
        vd = np.dot(dd, prod_vect)
        array_lower_pre_clean = array_lower[vd.argsort()]

        # new basis using the plane normal, the up_down_mean_vector and their cross product
        basis = np.column_stack((normalize_vector(prod_vect), normalize_vector(up_down_mean_vector),
                                normalize_vector(plane_normal)))
        # =========

        # used to smooth the shells points before creating the spline
        cleaned_upper = _clean_cut_array(array_upper_pre_clean, loc, basis, is_upper=True)
        cleaned_lower = _clean_cut_array(array_lower_pre_clean, loc, basis, is_upper=False)

        upper_spline = create_vtk_spline(cleaned_upper)
        lower_spline = create_vtk_spline(cleaned_lower)

        coordinates = _global_sampling(upper_spline, lower_spline, nb_spline_steps)
        sampled_points.append(coordinates)

    sampled_points = np.concatenate(sampled_points, axis=0).astype(np.float32)
    L.debug('%i nb planes that do not intersect with shell', nb_unused_planes)
    return sampled_points


def _longitudinal_coordinate_orientation(point, planes):
    """Compute the longitudinal orient and coordinate of a point according to planes orientations.

    Args:
        point(np.array): you want the 'longitudinal quaternion' of this point (np.array([x,y,z]))
        planes(list): the planes you use as reference for the longitudinal orientation

    Returns:
        tuple: the quaternion corresponding to the longitudinal direction for this point and the
        longitudinal coordinate
    """
    distances = distances_to_planes(point, planes)
    side = distances < 0
    idx = np.argmin(side) if side[0] else np.argmax(side)
    if idx == 0 or idx == len(planes):
        return -1, -1

    p0 = planes[idx - 1]
    p1 = planes[idx]

    d0 = abs(distances[idx - 1])
    d1 = abs(distances[idx])

    t = d0 / (d0 + d1)

    long = (idx - 1 + t) / len(planes)

    q0 = p0.get_quaternion(ZVECTOR)
    q1 = p1.get_quaternion(ZVECTOR)

    quat = Quaternion.slerp(q0, q1, amount=t)

    return quat, long


def _get_coordinates(point, kdtree, sampled_points, neighbor_count=25):
    """Return the transverse splines corresponding to a position.

    Args:
        point(np.array): a position in 3 space (np.array([x,y,z]))
        kdtree(kdtree): the kdtree that includes all sampled points
        sampled_points(np.array): the map that gives map[spline_idx] = [upper_spline, lower_spline]
        neighbor_count(int) number of neighbors to use for the interpolations of coordinates

    Returns:
        np.array: Weighted coordinates of points.

    Notes :
        The function use a kdtree to query the closest sampled point in the tree.
    """
    dists, neigh = kdtree.query(point, k=neighbor_count)
    inv_dist = 1 / dists
    weights = inv_dist / np.sum(inv_dist)
    res = sampled_points[neigh] * weights[:, np.newaxis]
    return res.sum(axis=0)


def _combine_orientations(long, radial):
    """Combine longitudinal and transverse orientations in one quaternion.

    Args:
        long: the longitudinal orientation vector
        radial: radial orientation vector pl - pu

    Returns:
        Quaternion: the combined quaternion

    Notes:
        We need to combine 2 orientations from different source: long and radial.
    """
    unit_long = normalize_vector(long)
    unit_radial = normalize_vector(radial)

    z_rot_axis = np.cross(ZVECTOR, unit_long)
    z_costheta = np.dot(ZVECTOR, unit_long)
    qz = Quaternion(axis=z_rot_axis, angle=np.arccos(z_costheta)).unit
    if not np.allclose(qz.rotate(ZVECTOR), unit_long):
        qz = Quaternion(axis=z_rot_axis, angle=-np.arccos(z_costheta)).unit
    temp_y_axis = qz.rotate(YVECTOR)
    y_rot_axis = np.cross(temp_y_axis, unit_radial)
    y_costheta = np.dot(temp_y_axis, unit_radial)
    qy = Quaternion(axis=y_rot_axis, angle=-np.arccos(y_costheta)).unit
    if not np.allclose(qy.rotate(temp_y_axis), unit_radial):
        qy = Quaternion(axis=y_rot_axis, angle=np.arccos(y_costheta)).unit
    return qy * qz


def _closest_point_on_plane(point, loc, rot):
    """ Projection of point within the plane defined by loc and rot

    Args:
        point(np.array): the point you want to project on the plane
        loc(np.array): center of the plane
        rot(Quaternion): the quaternion containing the plane's orientation.

    Returns:
        np.array: the closest projected point on plane. array = [x, y, z]
    """
    point = np.array(point)
    vect = np.array(point) - np.array(loc)
    normal = np.array(rot.rotate(ZVECTOR))
    normal = normalize_vector(normal)
    dist = np.dot(vect, normal)
    return point - dist * normal


def _corrected_radial_vect(point, radial_vect, rot):
    """Correct the radial vector by projecting the vector on the point plane.

    Args:
        point(np.array): a position in 3d space (np.array([x,y,z]))
        radial_vect(np.array): the radial vector to correct
        rot(np.array): the quaternion containing the longitudinal orientation

    Returns:
        np.array: the projected and normalized vector.

    Notes:
         np.array: Even if the difference between the input and output is small, this
         correction is needed for the quaternion combination. This phase prevents
         quaternions to be chaotic sometimes.
    """
    # 50 is random. Could be anything. It is here just to push the vector away a little bit and
    # reduce a possible angle error
    depth_tmp = point + normalize_vector(radial_vect) * 50
    depth_tmp = _closest_point_on_plane(depth_tmp, point, rot)
    return depth_tmp - point


def _get_quaternion_t_r_h_layers(point_rot_long, tree, sampled_points, layer_ratios):
    """Compute the global quaternion for point and values relative to radial and trans coordinates

    Args:
        point_rot_long(tuple): tuple containing the point location, the quaternion and the
        longitudinal value of the point.
        tree(cKDTree): scipy kdtree: the tree that includes all sampled points
        sampled_points(np.array): the map that gives map[spline_idx] = [upper_spline, lower_spline]
        layer_ratios(the layer ratios): the ratio of the total height for all layers

    Returns:
        tuple: the combined quaternion and the t and r coordinates, the height from the
        upper shell, the layer limits.

    Notes:
        We need to pass this point_rot_long and deserialize it afterwards due to the multiprocessing
    """
    # pylint: disable=too-many-locals
    point = point_rot_long[0]
    rot = point_rot_long[1]
    long = point_rot_long[2]
    coords = _get_coordinates(
        point, tree, sampled_points
    )  # [x,y,z,r,t,vectx,vecty,vectz]
    # transverse vector
    r = coords[3]
    t = coords[4]
    radial_vect = coords[-3:]
    v_len = np.linalg.norm(radial_vect)
    h = r * v_len
    layers_lengths = np.cumsum(layer_ratios * v_len)
    # working with floats ...
    layers_lengths[-1] = v_len
    layer_limits = np.array(list(pairwise([0] + list(layers_lengths))))

    # radial = corrected_radial_vect(radial_vect, point, rot)
    long_orient = np.array(rot.rotate(ZVECTOR))

    # combine both
    quat = _combine_orientations(long_orient, radial_vect)
    return point, quat, long, t, r, h, layer_limits


def _initialize_raw(brain_regions, add_dim, dtype=np.float32, value=-1):
    """Initialize a np array that will contain atlas raw.

    Args:
        brain_regions(VoxelData): the ref atlas
        add_dim(int): the shape of the added dimension
        dtype(np.dtype): the type you want to store in the array
        value(float/nan/int): the initialization value for the array

    Returns:
        np.array: an array with dim = brain_regions.shape + (add_dim,) and fulfilled with value
    """
    if add_dim > 0:
        shape = brain_regions.shape + (add_dim,)
    else:
        shape = brain_regions.shape
    layer_array = np.empty(shape, dtype=dtype)
    layer_array.fill(value)
    return layer_array


def _fill_atlases(
    points,
    planes,
    tree,
    brain_regions,
    sampled_points,
    new_brain_regions,
    orients,
    coordinates,
    heights,
    layers,
    layer_ids=None,
):
    """Fills orientations for all point in points.

    Args:
        points: positions in 3d (np.array([[x1,y1,z1], ..., [x2,y2,z2]]))
        planes: list of Plane objects
        tree (scipy kdtree): the tree that includes all sampled points
        brain_regions : the atlas containing the brain region
        voxel_spline_indexing:(np.array([[x1,y1,z1,spline_idx1], ..., [x2,y2,z2,spline_idx1]]))
        spline_map: map[spline_idx] = [upper_spline, lower_spline]
        sampled_points: all sampled points
        new_brain_regions: the new brain regions atlas containing layer information
        orients: atlas containing the orientation field
        coordinates: atlas containing the natural coordinates
        heights: atlas containing the y information
        layers: the list of Layer object from upper shell to lower shell
        layer_ids: (Optional) list of integers identifying the layers, e.g., the AIBS structure
            ids. If specified, it should have the same length as `layers`.
            Defaults to None. In this case, the identifier assigned to a layer
            is its index augmented by one.
    """
    # pylint: disable=too-many-locals
    # Hack to avoid partial for Pool.map
    def local_get_longitudinal_quaternion(point):
        return _longitudinal_coordinate_orientation(point, planes)

    L.info("Start Longitudinal orientation computing")
    rots_longs = ProcessingPool().map(local_get_longitudinal_quaternion, points)
    rots = []
    longs = []
    new_points = []
    for point, (rot, long) in zip(points, rots_longs):
        if rot == -1:
            continue
        rots.append(rot)
        longs.append(long)
        new_points.append(point)

    L.info("Longitudinal orientation is done")

    layer_ratios = np.array([layer.ratio for layer in layers])
    layer_ratio_cum = np.cumsum(layer_ratios)
    layer_ratio_cum[-1] = 1.0
    layer_ratio_limits = np.array(list(pairwise([0.0] + list(layer_ratio_cum))))

    L.debug('limits of layer ratios %s', layer_ratio_limits)

    point_rot_longs = list(zip(new_points, rots, longs))

    def local_get_quaternion_t_r_h_layers(point_rot_long):
        return _get_quaternion_t_r_h_layers(
            point_rot_long, tree, sampled_points, layer_ratios
        )

    L.info("Start transverse/radial orientation computing")
    t = ProcessingPool().map(local_get_quaternion_t_r_h_layers, point_rot_longs)

    L.info("Transverse/radial orientation is done")
    for loc, quat, l, t, r, h, layer_limits in t:
        layer_idx = np.argmax(
            ((r >= layer_ratio_limits[:, 0]) & (r <= layer_ratio_limits[:, 1]))
        )

        idx = brain_regions.positions_to_indices(loc)
        idx = tuple(idx) + (Ellipsis,)

        orients[idx] = list(quat.elements)
        coordinates[idx] = [l, t, r]
        # if we still have some problems move to use h rounded to unit for the layer
        heights[idx] = h
        for ll, layer in enumerate(layers):
            layer.raw[idx] = list(layer_limits[ll])
        new_brain_regions[idx] = (
            layer_idx + 1 if layer_ids is None else layer_ids[layer_idx]
        )


def _create_layers(sizes, names, brain_regions, value=np.nan):
    """ Creates a layer list using names and sizes as inputs

    Args:
        sizes(list): sizes of layers ([size1, size2, ...])
        names(list): anmes of layers ([name1, name2, ...])
        brain_regions(VoxelData): default atlas
        value(int/float/nan): the default value for the layer data

    Returns:
        list: A list of Layer object containing a name, a ratio and a dataset
    """
    layers = []
    tot_size = sum(sizes)
    for i, name in enumerate(names):
        layers.append(
            Layer(
                name,
                sizes[i] / tot_size,
                _initialize_raw(brain_regions, 2, value=value),
            )
        )
    return layers


def creates(
    brain_regions_path,
    plane_centerline_path,
    nb_interplane,
    radial_transverse_sampling,
    sizes,
    names,
    upper_file,
    lower_file,
    sampling,
    output_dir,
    seed=42,
    layer_ids=None,
):
    """Creates the coordinate system and the atlases.

    Args:
        brain_regions_path(str): the brain region atlas path.
        plane_centerline_path(str): the plane centerline file path.
        nb_interplane(int): the number of planes you want to add between two consecutive planes.
        radial_transverse_sampling(int): the number of points you want to use to sample the
        tranverse and radial coordinates.
        sizes(list): list of floats containing the sizes of each layer.
        names(list): the name of each layer.
        upper_file(str): the path to the upper shell .stl file.
        lower_file(str): the path to the lower shell .stl file.
        sampling(int): number of voxels to sample inside the brain region atlas.
        output_dir(str): path to the directory that will contain the created atlases.
        seed(int): the value of the pseudo-random generator.
        layer_ids(list): layer_ids: (Optional) list of integers identifying the
            layers, e.g., the AIBS structure ids. If specified, it should have the same length as
            `names`. Defaults to None. In this case, the identifier assigned to a layer
            is its index augmented by one.
    """
    # pylint: disable=too-many-locals
    np.random.seed(seed)  # seed for sampling

    nb_points = sampling
    planes = load_planes_centerline(plane_centerline_path)['planes']
    brain_regions_path = voxcell.VoxelData.load_nrrd(brain_regions_path)

    orients = _initialize_raw(brain_regions_path, 4, value=0)
    coordinates = _initialize_raw(brain_regions_path, 3)
    heights = _initialize_raw(brain_regions_path, 0, value=np.nan)
    layers = _create_layers(sizes, names, brain_regions_path, value=np.nan)
    new_brain_region_raw = _initialize_raw(
        brain_regions_path, 0, dtype=np.int32, value=0
    )

    locs = sample_positions_from_voxeldata(brain_regions_path, voxel_count=nb_points)
    L.info('Running %i points', len(locs))

    upper_cutter = create_cutter_from_stl(upper_file)
    lower_cutter = create_cutter_from_stl(lower_file)
    planes_interp = add_interpolated_planes(planes, nb_interplane)

    sampled_points = _create_spline_indexing(
        planes_interp, upper_cutter, lower_cutter, radial_transverse_sampling
    )
    L.info('Point indexing done')
    L.info(sampled_points.shape)
    tree = cKDTree(sampled_points[:, :3])  # pylint: disable=not-callable
    L.info('Tree done')
    _fill_atlases(
        locs,
        planes,
        tree,
        brain_regions_path,
        sampled_points,
        new_brain_region_raw,
        orients,
        coordinates,
        heights,
        layers,
        layer_ids,
    )

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    save_raw(
        os.path.join(output_dir, 'orientation.nrrd'),
        orients,
        brain_regions_path,
        is_orientation=True,
    )
    save_raw(
        os.path.join(output_dir, 'coordinates.nrrd'), coordinates, brain_regions_path
    )
    save_raw(os.path.join(output_dir, '[PH]y.nrrd'), heights, brain_regions_path)
    for layer in layers:
        file_name = '[PH]' + layer.name + '.nrrd'
        save_raw(os.path.join(output_dir, file_name), layer.raw, brain_regions_path)
    save_raw(
        os.path.join(output_dir, 'brain_regions.nrrd'),
        new_brain_region_raw,
        brain_regions_path,
    )
