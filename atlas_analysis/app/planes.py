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
@click.option('-o', '--output', type=str, help='Output file containing centerline/planes files',
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
    """ Run the plane creation for a given atlas (brain_region nrrd file)

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
    """ Draw the result of the plane creation on top of the atlas"""
    planes.draw(nrrd_path, preprocess_path, name)
