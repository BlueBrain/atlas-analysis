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
@log_args(L)
def flatmap_image(input_path, resolution):
    """Visualize the flatmap image.
    """

    voxel_data = voxcell.VoxelData.load_nrrd(input_path)
    visualization.flatmap_image_figure(voxel_data, resolution).show()


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
@log_args(L)
def flatmap_histogram(input_path, resolution):
    """Visualize the histogram of voxel volumes lying other pixels.
    """

    voxel_data = voxcell.VoxelData.load_nrrd(input_path)
    visualization.flatmap_volume_histogram(voxel_data, resolution).show()
