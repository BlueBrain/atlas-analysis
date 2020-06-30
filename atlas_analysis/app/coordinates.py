"""Cli module to create relative coordinates atlases."""
import sys
import logging

import click

import atlas_analysis.coordinates as coordinates
from atlas_analysis.vtk_visualization import render
from atlas_analysis.app.utils import split_str, log_args, set_verbose, FILE_TYPE


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
L = logging.getLogger("Coordinates")


@click.group()
@click.option('-v', '--verbose', count=True)
def app(verbose):
    """Creation of the relative coordinates system."""
    set_verbose(L, verbose)


@app.command()
@click.argument('brain_regions_path', type=FILE_TYPE)
@click.argument('plane_centerline_path', type=FILE_TYPE)
@click.option(
    '--nb_interplane',
    type=int,
    default=10,
    help='Number of planes to add between preprocess planes.',
)
@click.option(
    '--radial_transverse_sampling',
    type=int,
    default=150,
    help='The number of points you want to use to sample the tranverse and '
    'radial coordinates.',
)
@click.option(
    '--sizes', type=str, help='Layer sizes separated with a coma', required=True
)
@click.option(
    '--names', type=str, help='Layer names separated with a coma', required=True
)
@click.option(
    '--upper_file', type=FILE_TYPE, help='Upper shell file (.stl)', required=True
)
@click.option(
    '--lower_file', type=FILE_TYPE, help='Lower shell file (.stl)', required=True
)
@click.option(
    '--sampling', type=int, default=-1, help='Number of computed voxel (-1 = all)'
)
@click.option('--output_dir', type=str, required=True)
@click.option(
    '--layer_ids',
    type=str,
    help='Output layer ids separated with a coma.'
    'If not specified, each layer will be labeled with its index augmented by one.',
    default=None,
)
@log_args(L)
def create(
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
    layer_ids,
):
    """ Create all atlases for circuit building """
    sizes = split_str(sizes, int)
    names = split_str(names, str)
    layer_ids = split_str(layer_ids, int)
    coordinates.creates(
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
        layer_ids=layer_ids,
    )


@app.command()
@click.argument('input_path', type=FILE_TYPE)
@click.option('--rad', is_flag=True, help='Plot the radial orientation')
@click.option('--long', is_flag=True, help='Plot the longitudinal orientation')
@click.option('--trans', is_flag=True, help='Plot the transverse orientation')
@click.option(
    '-u',
    '--upper_shell',
    type=FILE_TYPE,
    help='The upper shell file (.stl)',
    default=None,
)
@click.option(
    '-l',
    '--lower_shell',
    type=FILE_TYPE,
    help='The lower shell file (.stl)',
    default=None,
)
@click.option(
    '-c',
    '--centerline_file',
    type=FILE_TYPE,
    help='The centerline_file file (.npz)',
    default=None,
)
@click.option('-s', '--sampling', type=int, help='Sampling', default=40000)
def draw(
    input_path, rad, long, trans, upper_shell, lower_shell, centerline_file, sampling
):
    """ Draw the orientation file using opengl"""
    render(
        orientation_file=input_path,
        rad=rad,
        long=long,
        trans=trans,
        stl_files=[upper_shell, lower_shell],
        centerline_file=centerline_file,
        orientation_sampling=sampling,
    )
