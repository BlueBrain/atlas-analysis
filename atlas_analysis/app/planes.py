""" cli module to create plane from an atlas """
import logging

import click

import atlas_analysis.planes as planes
from atlas_analysis.app.utils import split_str, log_args, set_verbose, FILE_TYPE

L = logging.getLogger("planes")


@click.group()
@click.option('-v', '--verbose', count=True)
def app(verbose):
    """Run the different plane creation CLI """
    set_verbose(L, verbose)


@app.command()
@click.argument('nrrd_path', type=FILE_TYPE)
@click.option('-o', '--output', type=str, help='Output file containing centerline/planes',
              required=True)
@click.option('-s', '--start', type=str, help='The starting voxel (use itk-snap to define it)',
              required=True)
@click.option('-e', '--end', type=str, help='The ending voxel (use itk-snap to define it)',
              required=True)
@click.option('-d', '--downhill', type=float,
              help='The authorized ratio for downhill exploration of the distance transform',
              default=0.95)
@click.option('-c', '--chain_length', type=int,
              help='The number of points used for each chain to sample the distance transform',
              default=100000)
@click.option('-n', '--chain_count', type=int,
              help='The number of chains for the distance transform plateau search',
              default=5)
@click.option('-cs', '--sampling', type=int,
              help='The sampling used on the chain for the distance transform plateau search',
              default=10)
@click.option('-l', '--link_distance', type=float,
              help='The distance where all points are linked for the graph creation',
              default=500.)
@click.option('-p', '--plane_count', type=int,
              help='The number of wanted planes',
              default=25)
@click.option('--seed', type=int,
              help='The pseudo random generator seed',
              default=42)
@log_args(L)
def create(nrrd_path, output, start, end, downhill, chain_length, chain_count, sampling,
           link_distance, plane_count, seed):
    """ Run the plane creation for a given atlas (brain_region nrrd file).

    This function creates the centerline of the input volume together with
    a series of planes orthogonal to this curve. The number of planes to create
    is specified with the plane_count argument. The output of the function is an .npz file
    bearing the specified name. This file can be loaded subsequently with numpy.load().

    Notes:\n
        The algorithm will do the following steps
    (the related input variables are put between parentheses):\n
          - Compute the distance transform of the input volume (nrrd_path).\n
          - Use multiple stochastic chains to sample the distance transform distribution ridge,
    i.e., local maxima along the volume.\n
            - The user defines two entry points for the volume, i.e., the centerline
    will go through these points (starting_points).\n
            - Stochastic chains of a certain length (chain_length) are launched from these points.
    They will climb up the distance transform distribution with a high probability.
    With a small probability they will go downhill. This allows the chains to escape from
    local maxima 'hills' (downhill).\n
            - The volume can have multiple ridges and having multiple chains reduce the risk of
    missing some of them (chain_count).\n
          - We sample the chain to reduce the autocorrelation of the chains (sampling).\n
          - The amount of points produced by the chains where the distance transform is high can
    be important so we do a clustering of points merging all the points that are in
    the range of the voxel dimension.\n
          - We create a graph from these remaining points:\n
            - Nodes are point locations from the chains/clustering part.\n
            - Edges are created between close enough points (link_distance).\n
          - We need to find a path between the two starting points. This will define the
    centerline but it appears that the graph is not always connected. So we iteratively
    connect the graph by connecting the closest connected components first.\n
          - We do a shortest path search in the graph between the two starting points using a
    weighted Dijkstra. This creates the centerline.\n
          - The centerline usually jitters a lot and needs to be smoothed.\n
            - We use a Bézier curve to do the smoothing with multiple control
    points (ctrl_point_count).\n
            - Control points are taken from the shortest path points.\n
          - We create a parametric spline function on top of the Bézier curve.\n
          - The derivative of this curve corresponds to the plane normal (plane_count).\n
            - We take two points that are close on the spline. The vector between these points
    defines the normal of the local perpendicular plane.\n
          - Save the results (output_path).\n
    """
    bounds = [split_str(start, int), split_str(end, int)]
    planes.create_centerline_planes(nrrd_path, output, bounds,
                                    downhill=downhill, chain_length=chain_length,
                                    chain_count=chain_count,
                                    sampling=sampling, link_distance=link_distance,
                                    plane_count=plane_count, seed=seed)


@app.command()
@click.argument('nrrd_path', type=FILE_TYPE)
@click.argument('preprocess_path', type=FILE_TYPE)
@click.option('-n', '--name', type=str, help='The data set name', required=True)
def draw(nrrd_path, preprocess_path, name):
    """ Draw the result of the plane creation on top of the atlas.

      Usage: atlas-analysis planes draw [OPTIONS] NRRD_PATH PREPROCESS_PATH\n
      Draw the centerline and its orthogonal planes on top of the
      corresponding volumetric file.
      NRRD_PATH is the path to the volumetric file used to generate the
      centerline and its orthogonal planes.
      PREPROCESS_PATH is the path to the .npz file containing the centerline
      and the orthogonal planes data.
    """
    planes.draw(nrrd_path, preprocess_path, name)
