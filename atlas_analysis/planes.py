""" Module used to create planes according to a user defined main axis """
from itertools import combinations
import collections

import six
import numpy as np
import networkx
from pyquaternion import Quaternion
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree  # pylint: disable=no-name-in-module
from scipy.spatial.distance import cdist
from geomdl import BSpline
from geomdl import utilities

import voxcell
from plotly_helper.helper import PlotlyHelperPlane, plot_fig
from plotly_helper.object_creator import scatter, scatter_line

from atlas_analysis.vtk_utils import create_vtk_spline
from atlas_analysis.utils import pairwise, ensure_list
from atlas_analysis.maths import normalize_vectors, get_normal
from atlas_analysis.atlas import indices_to_voxel_centers
from atlas_analysis.constants import XYZ, ZVECTOR, X, Y, Z, QUAT
from atlas_analysis.exceptions import AtlasAnalysisError


def get_plane_quaternion(plane):
    """Create the plane quaternion and force the normal to follow the longitudinal axis.

    Args:
        plane: a plane using the format [x,y,z,a,b,c,d]

    Returns:
        a unitary quaternion from the [a,b,c,d] elements

    Notes:
        to prevent some weird flipping quaternions, we force the scalar part to be positive
    """
    plane = np.asarray(plane)
    plane = np.sign(plane[3]) * plane[QUAT]
    return Quaternion(plane).unit


def split_plane_elements(plane):
    """Split a plane into a position and a quaternion.

    Args:
        plane: a plane using the format [x,y,z,a,b,c,d]

    Returns:
        a 2-tuple with the first element being a list containing the position [x, y, z] and the
        second element an unitary quaternion from the [a,b,c,d] elements
    """
    return plane[XYZ], get_plane_quaternion(plane)


def add_interpolated_planes(planes, inter_plane_count):
    """Densify a plane list adding inter_plane_count between pairs of consecutive planes.

    Args:
        planes: the list of planes using the format [x,y,z,a,b,c,d]
        inter_plane_count: number of planes to add between two consecutive planes (int)

    Returns:
        a np.array using the format [x,y,z,a,b,c,d] with all planes
    """
    if inter_plane_count <= 0:
        return np.array(planes)

    new_planes = []
    first = 0

    # iterate over pairs of planes (plane --> (plane1, plane2), (plane2, plane3), ...)
    for plane1, plane2 in pairwise(planes):
        locs = np.vstack([np.linspace(plane1[axis], plane2[axis],
                                      inter_plane_count + 2, endpoint=True)
                          for axis in (X, Y, Z)]).T

        qs = Quaternion.intermediates(get_plane_quaternion(plane1),
                                      get_plane_quaternion(plane2),
                                      inter_plane_count, include_endpoints=True)
        qs = [q.elements for q in qs]
        new_planes.append(np.hstack([locs, qs])[first:])
        first = 1

    return np.vstack(new_planes)


def get_normals(planes):
    """Returns normal vector for each plane.

    Args:
        planes: the list of planes using the format [x,y,z,a,b,c,d]

    Returns:
        the normal vectors for all planes (np.array([x,y,z]))
    """
    planes = ensure_list(planes)
    return np.array([get_normal(get_plane_quaternion(plane)) for plane in planes])


def distances_to_planes(point, planes):
    """Returns the signed distance of a point to a list of planes.

    Args:
        point: the point [x, y, z]
        planes: the list of planes using the format [x,y,z,a,b,c,d]

    Returns:
        signed distance (np.array([d1, ..., d2])) to all plane
    """
    normals = get_normals(planes)
    point = np.asarray(point)
    planes = np.asarray(planes)
    xyzs = planes[:, XYZ]
    return np.einsum('ij,ij->i', normals, point - xyzs) / np.linalg.norm(normals, axis=1)


def save_planes_centerline(filepath, planes, centerline):
    """Save a file containing the centerline and the planes.

    Args:
        filepath: name of the output path
        planes: the planes you want to save array(N, 7)
        centerline: the centerline corresponding to the planes array(M, 3)

    Notes:
        The file is a npz file used for the relative coordinates. It contains all the planes
        and the centerline points.
    """
    planes = np.asarray(planes)
    centerline = np.asarray(centerline)
    np.savez(filepath, planes=planes, centerline=centerline)
    return filepath


def load_planes_centerline(filepath, name=None):
    """Loads the centerline and planes file you want to access.

    Args:
        filepath: the path to the npz file
        name: the data name you want to load ('planes'|'centerline') str or list

    Returns:
        return a tuple of arrays if name is None or a list, return an array if name is a str
    """
    res = np.load(filepath)
    if not name:
        return res['planes'], res['centerline']
    if isinstance(name, collections.Iterable) and not isinstance(name, six.string_types):
        missing = set(name) - {'planes', 'centerline'}
        if missing:
            raise AtlasAnalysisError('Cannot retrieve {}'.format(missing))
        return (res[n] for n in name)
    if name not in res:
        raise AtlasAnalysisError('Cannot retrieve {}'.format(name))
    return res[name]


def _distance_transform(voxeldata):
    """Compute a 3d euclidean distance transform.

    Args:
        voxeldata: a VoxelData object containing the volume

    Returns:
        an array with the same dimensions as voxeldata input containing the distance transform

    Notes:
        https://en.wikipedia.org/wiki/Distance_transform
    """
    # 0 outside volume and 1 inside
    dist = np.zeros_like(voxeldata.raw)
    dist[np.nonzero(voxeldata.raw)] = 1
    dist = distance_transform_edt(dist)
    return dist


def _chain(to_evaluate, proposal, chain_length, dist, downhill, nrun, starting_points):
    """Create a chain to evaluate the local maximum of the to_evaluate function"""
    chain = []
    # start from one side then the other
    init = starting_points[nrun % len(starting_points)]
    current_idx = init
    current_val = dist[tuple(current_idx)]
    # distance transform is 0 outside the volume so the user can provide a voxel with
    # distance_transform(voxel) == 0. So need to epsilon this for the first step.
    current_val = current_val if current_val != 0 else 0.00001
    # launch the chain
    for _ in range(chain_length):
        test_idx = proposal(current_idx)
        test_value = to_evaluate(tuple(test_idx))

        # will always go uphill if proposed but sometimes downhill if ratio is not too small
        if test_value / current_val >= downhill:
            current_val = test_value
            current_idx = test_idx
            chain.append(current_idx)
    return chain


def _explore_ridge(dist, starting_points, downhill=0.95, chain_length=100000,
                   chain_count=5, sampling=10, proposal_step=3):
    """Detect points inside the valley of the 3d distance transform between two points.

    The goal is to find a sample of points that represent the volume's centerline that goes from
    starting_points[0] to starting_points[1]. We choose the centerline definition as: where the
    distance transform is locally maximal inside the volume. We are dealing with non-convex volumes
    with complex shapes and there is no easy way to detect the correct local max of the
    distance transform. So the idea is to propagate a stochastic chain that will follow the
    ridge and go close to the starting points and hopefully converge at some point in the volume.

    Args:
        dist: the distance transform of the voxeldata
        starting_points: the starting points of the chain in the indices representation
        downhill: the ratio of at the step i+1: distance_transform(i + 1)/distance_transform(i)
        that authorizes the chain to go downhill
        chain_length: the length of each chain (including the not selected proposals)
        sampling: sampling at the end of the chain creation
        chain_count: the number of chains that you want to run for each starting points
        proposal_step: the allowed mean distance in the indices representation for the gaussian
        proposal function

    Returns:
        an [N x 3] array of point's indexes.
    """
    # pylint: disable=too-many-locals
    dim = len(dist.shape)
    downhill = min(1., downhill)

    def _proposal(iidx):
        """The proposal for the next step of chain in the indices representation.

        The proposal is a normal law centered on the current point index. The standard deviation
        is proposal_step. Retries if outside the voxeldata shape.

        Args:
            iidx: the current index

        Returns:
            the proposed future index.
        """
        next_idx = np.asarray([-1] * dim)
        while np.any(~((next_idx > 0) & (next_idx < dist.shape))):
            next_idx = (iidx + np.random.normal(0, proposal_step, size=3)).astype(int)
        return next_idx

    def _dist_evaluation(index):
        """Evaluation of the distance transform."""
        return dist[index]

    all_chains = []
    # launch multiple chains to scan more efficiently the distance transform ridges
    for nrun in range(chain_count * len(starting_points)):
        chain = _chain(_dist_evaluation, _proposal, chain_length,
                       dist, downhill, nrun, starting_points)
        all_chains.extend(chain[::sampling])
    return np.array(all_chains)


def _clusterize_cloud(cloud, max_length=25):
    """ Clusterize all points that are close enough. Returns the mean of each cluster.

    Args:
        cloud: a cloud of points positions
        max_length: the distance from a point to define neighbors and to create a cluster

    Returns:
        [N x 3] array positions of mean clusters positions.

    Notes:
         there is no overlap between clusters. If neighbors are [1,2,3], [3,4,5]
         and [4,5] then the clusters will be [1,2,3] and [4,5]. This implies that points
         are in one cluster only. It implies that the algorithm is not stable if we start from a
         different point.
     """
    clustered = []
    cloud = np.asarray(cloud)
    tree = cKDTree(cloud)
    to_skip = set()
    neighbors_groups = tree.query_ball_point(cloud, max_length)
    for neighbors in neighbors_groups:
        if not set(neighbors).intersection(to_skip):
            neighbor_positions = cloud[np.array(neighbors)]
            clustered.append(np.mean(neighbor_positions, axis=0))
        to_skip.update(neighbors)
    return clustered


def _create_graph(cloud, link_distance=500):  # pylint: disable=too-many-locals
    """ Create a graph from a could of point's positions

    Args:
        cloud: the could of point's positions. [N*3] array.
        link_distance: the distance from which you consider 2 points as connected neighbors

    Returns:
        a networkx graph object containing all points from cloud.

    Notes:
        Nodes are considered as neighbors if the distance between them is below link_distance.
        The graph must be composed of one connected component at the end of the process. So if the
        graph is composed of multiple connected components, they are linked altogether recursively
        by finding the 2 closest points for each pair of connected component and a edge is created
        between these points. This process is repeated until only one connected component remains.
    """
    graph = networkx.Graph()
    cloud = np.asarray(cloud)

    tree = cKDTree(cloud)

    for node_id, point in enumerate(cloud):
        indices = tree.query_ball_point(point, link_distance)
        # need to add node in the graph in case it has no neighbors
        graph.add_node(node_id)
        # first index is always self
        for idx in indices[1:]:
            graph.add_edge(node_id, idx)

    # want a unique connected component. Recursively link the closest connected components
    while True:
        connected_comp = list(networkx.connected_components(graph))
        if len(connected_comp) == 1:
            break
        distance_min = np.inf
        closest_node_1 = None
        closest_node_2 = None
        # find the min of distance matrices between all connected component element pairs
        for connected_comp_1, connected_comp_2 in combinations(connected_comp, 2):
            tmp_a1, tmp_a2 = np.array(list(connected_comp_1)), np.array(list(connected_comp_2))
            euclidean_dists = cdist(cloud[tmp_a1], cloud[tmp_a2], 'euclidean')
            # idx of the distance matrix min between c1 and c2
            tmp_d_idx = np.unravel_index(np.argmin(euclidean_dists), euclidean_dists.shape)
            if euclidean_dists[tmp_d_idx] < distance_min:
                distance_min = euclidean_dists[tmp_d_idx]
                closest_node_1 = tmp_a1[tmp_d_idx[0]]
                closest_node_2 = tmp_a2[tmp_d_idx[1]]
        # link the 2 closest connected components using the 2 closest points of these components
        graph.add_edge(closest_node_1, closest_node_2)
    return graph


def _create_centerline(voxeldata, starting_points, link_distance=500, downhill=0.95,
                       chain_length=100000, chain_count=5, sampling=10):
    """ Create the centerline using a distance transform

    Args:
        voxeldata: the considered volume
        starting_points: the starting points of the chain in the indices representation.
            Only two points are allowed for the moment. The choice of the order between
            first and second starting points will define the direction of plane orientations
        link_distance: the distance from which you consider 2 points as connected neighbors
        downhill: the ratio of loc/proposal that authorize the chain to go downhill
        chain_length: the length of each chain (including the not selected proposals)
        chain_count: the number of chain that you want to run for each starting points
        sampling: sampling at the end of the chain creation

    Returns:
        a list of points corresponding to the shortest path between both starting points
    """
    if len(starting_points) != 2:
        raise AtlasAnalysisError("Only 2 starting points are allowed for the moment")

    starting_points = np.asarray(starting_points)
    dist = _distance_transform(voxeldata)
    ridge_pos = indices_to_voxel_centers(voxeldata, _explore_ridge(dist, starting_points,
                                                                   downhill=downhill,
                                                                   chain_length=chain_length,
                                                                   chain_count=chain_count,
                                                                   sampling=sampling))
    clustered = _clusterize_cloud(ridge_pos, np.max(voxeldata.voxel_dimensions))
    clustered = np.concatenate([[indices_to_voxel_centers(voxeldata, starting_points[0])],
                                clustered,
                                [indices_to_voxel_centers(voxeldata, starting_points[1])]])

    graph = _create_graph(clustered, link_distance=link_distance)

    def _weight(node_id1, node_id2, _):
        """ weight function for the dijkstra. Must keep the _ in signature """
        return np.linalg.norm(clustered[node_id1] - clustered[node_id2])

    A = networkx.nx.dijkstra_path(graph, 0, len(clustered) - 1, weight=_weight)
    return clustered[np.array(A)]


def _split_path(path, point_count):
    """Splits the path in equal length segments.

    Args:
        path(list/np.array): a list of 3d points
        point_count: the number of points to sample the path

    Returns:
        np.array: with equidistant points along the path
    """
    path = np.asarray(path)
    # distances between point i and i + 1
    distances = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum_distances = np.cumsum([0] + distances.tolist())
    path_distances = np.linspace(0, cum_distances[-1], point_count, endpoint=True)[1:-1]
    # binning of path distance on cum_distance
    indices = np.digitize(path_distances, cum_distances) - 1
    # compute the remaining distance ratios on the segment [i, i+1] for all path distances
    ratios = (path_distances - cum_distances[indices]) / distances[indices]
    points = np.zeros((point_count, 3), dtype=np.float32)
    points[0] = path[0]
    points[-1] = path[-1]
    points[1:-1] = path[indices] + np.multiply((path[indices + 1] - path[indices]).T, ratios).T
    return points


def _smoothing(path, ctrl_point_count=10):
    """ Use a bezier curve to smooth a path defined by multiple 3d points"""
    curve = BSpline.Curve()
    curve.degree = 5
    curve.ctrlpts = _split_path(path, ctrl_point_count).tolist()
    # Auto-generate knot vector
    curve.knotvector = utilities.generate_knot_vector(curve.degree, curve.ctrlpts_size)
    curve.delta = 0.01

    step_count = 100
    steps = np.linspace(0, 1, step_count, endpoint=True)
    return np.asarray(curve.evaluate_list(steps))


def _create_planes(centerline, plane_count=25, delta=0.001):  # pylint: disable=too-many-locals
    """ Returns the plane quaternions and positions

    We need to recreate a spline on top of the bezier curve to have a proper parametric function
    with the correct distance between planes. To create the planes we use the tangent to the
    centerline and use this as a Z (longitudinal) orientation and compute the quaternion.

    Args:
        centerline: a ndarray(N, 3) representing the centerline
        plane_count: the number of planes to return
        delta: the parametric delta to define the tangent

    Returns:
        a list of plane_count x [x, y, z, a, b, c, d]
    """
    spline = create_vtk_spline(centerline)
    sampling_up = np.zeros((plane_count, 3), dtype=np.float)
    sampling_down = np.zeros((plane_count, 3), dtype=np.float)
    steps = np.linspace(0, 1, plane_count, endpoint=True)
    ptu = [0, 0, 0]
    ptd = [0, 0, 0]
    d = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    ups = np.zeros((len(steps), 3))
    ups[:, 0] = steps + delta
    ups[ups[:, 0] > 1, 0] = 1
    downs = np.zeros((len(steps), 3))
    downs[:, 0] = steps
    downs[downs[:, 0] + delta > 1, 0] -= delta
    for i in range(len(steps)):
        spline.Evaluate(downs[i], ptd, d)
        sampling_down[i] = ptd
        spline.Evaluate(ups[i], ptu, d)
        sampling_up[i] = ptu

    unit_longs = normalize_vectors(sampling_up - sampling_down)
    planes = []
    for unit_long, pos in zip(unit_longs, sampling_up):
        if np.allclose(unit_long, ZVECTOR):  # pragma: no cover
            qz = Quaternion(axis=ZVECTOR, angle=0.).unit
        else:
            z_rot_axis = np.cross(ZVECTOR, unit_long)
            z_costheta = np.dot(ZVECTOR, unit_long)
            qz = Quaternion(axis=z_rot_axis, angle=np.arccos(z_costheta)).unit

        planes.append(pos.tolist() + qz.elements.tolist())
    return planes


def create_centerline_planes(nrrd_path, output_path, starting_points,
                             downhill=0.95, chain_length=100000, chain_count=5, sampling=10,
                             link_distance=500, ctrl_point_count=10, plane_count=25, seed=41):
    """Create a centerline and create planes from a volume

    Args:
        nrrd_path: the considered volume path
        starting_points: the starting points of the chain in the indices representation.
            Only two points are allowed for the moment. The choose of the order between
            first and second starting points will define the sense of plane orientations
        output_path: the output for the results (a npz file) with the planes and the centerline
        downhill: the ratio of loc/proposal that authorize the chain to go downhill
        chain_length: the length of each chain (including the not selected proposals)
        sampling: sampling at the end of the chain creation
        chain_count: the number of chain that you want to run for each starting points
        link_distance: the distance from which you consider 2 points as connected neighbors
        ctrl_point_count: the number of control points for the bezier smoothing
        plane_count: the number of planes to return
        seed: the seed for the pseudo random generator

    Notes:
        The algorithm will do the following steps (the related input variables are put
        between parenthesis):
          - compute the volume distance transform (nrrd_path)
          - use multiple stochastic chains to sample the distance transform distribution ridge
          ie: local minimums along the volume.
            - user defines two entry points for the volume. ie: the centerline will go trough these
              points. (starting_points)
            - stochastic chains of a certain length (chain_length) are launched from these points.
              They will climb up the distance transform distribution with a high probability.
              With a small probability they will go downhill. This allows the chains to escape from
              local maximums 'hills' (downhill)
            - the volume can have multiple ridges and having multiple chains reduce the risk of
              missing some of them. (chain_count)
            - we sample the chain to reduce the autocorrelation of the chains (sampling)
          - the amount of points produced by the chains where the distance transform is high can
            be important so we do a clustering of points merging all the points that are in
            the range of voxel dimension.
          - we create a graph from these remaining points
            - nodes are point locations from the chains/clustering part
            - edges are created between close enough points (link_distance)
            - we need to find a path between the two starting points, this will define the
              centerline but it appears that the graph is not always connected. So we iteratively
              connect the graph by connecting the closest connected components first.
          - we do a shortest path research in the graph between the two starting points using a
            weighted Dijkstra. This creates the centerline.
          - the centerline usually jitter a lot and needs to be smoothed.
            - we use a bezier curve to do the smoothing with multiple control
              points (ctrl_point_count)
            - control points are taken from the shortest path points.
          - we create a parametric spline function on top of the bezier curve
          - derivative of this curve corresponds to the plane orientations (plane_count)
            - we take two points that are close on the spline. The vector between these points is
              define the normal of the local perpendicular plane.
          - save the results (output_path)
    """
    np.random.seed(seed)
    voxeldata = voxcell.VoxelData.load_nrrd(nrrd_path)
    centerline = _create_centerline(voxeldata, starting_points, link_distance=link_distance,
                                    downhill=downhill, chain_length=chain_length,
                                    chain_count=chain_count, sampling=sampling)
    centerline = _smoothing(centerline, ctrl_point_count=ctrl_point_count)
    planes = _create_planes(centerline, plane_count=plane_count)
    save_planes_centerline(output_path, planes, centerline)


def draw(nrrd_path, preprocess_path, name):  # pragma: no cover
    """ Draw to check quality of the whole process using plotly """

    def _create_plane(pos, quat, color='blue', size_multiplier=2500, opacity=0.7):
        """ Create a 3d plane using a center position and a quaternion for orientation

        Args :
            pos: x,y,z position of the plane's center (array([x,y,z]))
            quat: quaternion representing the orientations (Quaternion)
            color: color of plane (plain text red|purple|blue)
            size_multiplier: plane size in space coordinates (float)
            opacity: set the opacity value (float)

        Returns :
            A square surface to the plotly format
        """
        import plotly.graph_objs as go

        def _get_displaced_pos(axis):
            """ Compute the shifted position wrt the quaternion and axis """
            return pos + size_multiplier * np.array(quat.rotate(axis))

        positif_x = _get_displaced_pos((1, 0, 0))
        positif_y = _get_displaced_pos((0, 1, 0))
        negatif_x = _get_displaced_pos((-1, 0, 0))
        negatif_y = _get_displaced_pos((0, -1, 0))

        x = [[positif_x[0], positif_y[0]], [negatif_y[0], negatif_x[0]]]
        y = [[positif_x[1], positif_y[1]], [negatif_y[1], negatif_x[1]]]
        z = [[positif_x[2], positif_y[2]], [negatif_y[2], negatif_x[2]]]

        return go.Surface(
            z=z,
            x=x,
            y=y,
            showscale=False,
            surfacecolor=[color, color],
            opacity=opacity,
        )

    voxeldata = voxcell.VoxelData.load_nrrd(nrrd_path)

    planes, centerline = load_planes_centerline(preprocess_path, ['planes', 'centerline'])
    dist = _distance_transform(voxeldata)
    hull_idx = np.array(np.where((dist <= 1) & (dist > 0))).T
    hull_pos = indices_to_voxel_centers(voxeldata, hull_idx)
    hull_pos = hull_pos[np.random.choice(hull_pos.shape[0], min(10000, len(hull_pos)),
                                         replace=False), :]

    ph = PlotlyHelperPlane('plane creation', '3d')
    ph.add_data({name: scatter(hull_pos, name=name, color='red', width=2)})
    vis_planes = [_create_plane(*split_plane_elements(plane)) for plane in planes]

    ph.add_data({'planes': vis_planes})
    ph.add_data({'centerline': scatter_line(centerline, name='centerline', color='green', width=8)})
    plot_fig(ph.get_fig(), "plane_creation")
