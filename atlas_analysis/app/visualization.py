""" cli module with visualization tools for volumetric files """
import logging
import click

import voxcell
from atlas_analysis import visualization
from atlas_analysis.app.utils import log_args, set_verbose, FILE_TYPE

L = logging.getLogger("Visualization")


@click.group()
@click.option('-v', '--verbose', count=True)
def app(verbose):
    """Run the different visualization CLI """
    set_verbose(L, verbose)


@app.command()
@click.argument('input_path', type=FILE_TYPE)
@click.option(
    '-o',
    '--output_path',
    type=str,
    help='Output file path of the flatmap image (HTML file)',
    required=True,
)
@click.option(
    '-r',
    '--resolution',
    type=int,
    help='Integer value defining the resolution used to downscale the flatmap image.'
    ' The resolution is the number of pixels along the width of the displayed'
    ' image (x-axis).'
    ' The number of pixels along the height (y-axis) is obtained by applying'
    ' the aspect ratio of the original image.'
    ' Defaults to None, which means that the full resolution of the flatmap'
    ' image will be used.',
    default=None,
)
@click.option(
    '--show', is_flag=True, help='Show the flatmap image in the browser.',
)
@log_args(L)
def flatmap_image(input_path, output_path, resolution, show):
    """Visualize the flatmap image.

    The output file is an HTML file.

    The browser will open the image if requested, see --show option.
     The browser offers 2D navigation and screenshots.

    Note: The input flatmap array is required to be an integer array of shape (W, H, D, 2).
     Negative 2D coordinates are assigned to voxels which mustn\'t or cannot be mapped to the
     plane.
    """

    voxel_data = voxcell.VoxelData.load_nrrd(input_path)
    figure = visualization.flatmap_image_figure(voxel_data, resolution)
    figure.write_html(output_path)
    if show:
        figure.show()


@app.command()
@click.argument('input_path', type=FILE_TYPE)
@click.option(
    '-r',
    '--resolution',
    type=int,
    help='Integer value defining the resolution used to downscale the flatmap image.'
    ' The resolution is the number of pixels along the width of the displayed'
    ' image (x-axis).'
    ' The number of pixels along the height (y-axis) is obtained by applying'
    ' the aspect ratio of the original image.'
    ' Defaults to None, which means that the full resolution of the flatmap'
    ' image will be used.',
    default=None,
)
@click.option(
    '-o',
    '--output_path',
    type=str,
    help='Output file path of the flatmap image (HTML file)',
    required=True,
)
@click.option(
    '--show', is_flag=True, help='Show the flatmap histogram in the browser.',
)
@log_args(L)
def flatmap_histogram(input_path, output_path, resolution, show):
    """Visualize the histogram of voxel volumes lying over pixels.

    The output file is an HTML file.

    The browser will open the image if requested by, see --show option.
     The browser offers 3D navigation and screenshots.

    Note: The input flatmap array is required to be an integer array of shape (W, H, D, 2).
     Negative 2D coordinates are assigned to voxels which mustn\'t or cannot be mapped to the
     plane.
    """

    voxel_data = voxcell.VoxelData.load_nrrd(input_path)
    figure = visualization.flatmap_volume_histogram(voxel_data, resolution)
    figure.write_html(output_path)
    if show:
        figure.show()
