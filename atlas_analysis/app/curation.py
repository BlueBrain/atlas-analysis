""" cli module with simple curation operations """
import os
from pathlib import Path
import logging
import click
import voxcell
from atlas_analysis.curation import remove_connected_components as rm_components
from atlas_analysis.curation import create_aabbs, clip_region
from atlas_analysis.app.utils import log_args, set_verbose

L = logging.getLogger("Curation")


@click.group()
@click.option('-v', '--verbose', count=True)
def app(verbose):
    """Run the different curation CLI """
    set_verbose(L, verbose)


@app.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('-o', '--output_path', type=str, help='Output nrrd file name', required=True)
@click.option(
    '-s', '--threshold_size', type=int,
    help='Number of voxels below which a connected component is removed', required=True
)
@click.option(
    '-c', '--connectivity', type=int,
    help='Integer value defining connected voxels.'
    ' Two voxels are connected if their squared distance is not more than connectivity.'
    ' Defaults to 1, i.e., two voxels are connected if they share a common face.'
)
@log_args(L)
def remove_connected_components(input_path, output_path, threshold_size, connectivity=1):
    """ Remove the connected components whose sizes are less than or equal to
        a specified threshold, the size being the number of voxels.
        The connectivity is an optional parameter which defines what connected
        voxels are. By default, two voxels are connected if they share a common face.
    """
    voxeldata = voxcell.VoxelData.load_nrrd(input_path)
    voxeldata = rm_components(voxeldata, threshold_size, connectivity)
    voxeldata.save_nrrd(output_path)


@app.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('-o', '--output_dir',
              type=str,
              help='Output directory name where the region files will be saved. '
              'It will be created if it doesn\'t exist.', required=True)
@log_args(L)
def split_regions(input_path, output_dir):
    """ Split an nrrd image file into different region images.
        A region file is generated for each non-zero voxel value, a.k.a label.
        Each region is cropped to its smallest enclosing bounding box and is saved under the form of
        an nrrd file.
    """
    voxeldata = voxcell.VoxelData.load_nrrd(input_path)
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir()
    bounding_boxes = create_aabbs(voxeldata)
    for label, box in bounding_boxes.items():
        region = clip_region(label, voxeldata, box)
        output_path = os.path.join(output_dir, '{}.nrrd'.format(label))
        region.save_nrrd(output_path)
