""" cli module with simple curation operations """
import os
from pathlib import Path
import logging
import click
import voxcell
from atlas_analysis.curation import remove_connected_components as rm_components
from atlas_analysis.curation import create_aabbs, clip_region
from atlas_analysis.curation import median_filter as median_smoothing
from atlas_analysis.curation import assign_to_closest_region as reassign_to_closest_region
from atlas_analysis.app.utils import log_args, set_verbose, FILE_TYPE
from atlas_analysis.curation import merge as merge_region

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


@app.command()
@click.argument('input_path', type=FILE_TYPE)
@click.option('-o', '--output_path', type=str, help='Output nrrd file name', required=True)
@click.option(
    '-f', '--filter_size', type=int,
    help='edge size of the box used for filtering the input image', required=True
)
@click.option(
    '-c',
    '--closing_size',
    type=int,
    help='edge size of the box used to dilate the input image'
    ' before filtering and to erode it afterwards.',
    required=True)
@log_args(L)
def median_filter(input_path, output_path, filter_size, closing_size):
    """ Smooth the input image by applying a median filter of the specified filter size.

        This size, given in terms of voxels, is the edge length of the cube inside
        which the median is computed.
        (See https://en.wikipedia.org/wiki/Median_filter for the definition of the median filter.)
        A dilation is performed before the application of the median filter and an erosion
        is performed afterwards. Both operations use a box whose edge length is given by the
        specified closing size.
        This combination, which is a morphological closing
        with a filter in the middle, has proved useful to fill holes in shapes with
        large openings.
        See https://en.wikipedia.org/wiki/Mathematical_morphology
        for definitions.
        The output is saved to the the specified output path.
        Note: this function does not preserve the volume and is likely to expand it.
    """
    voxeldata = voxcell.VoxelData.load_nrrd(input_path)
    voxeldata = median_smoothing(voxeldata, filter_size, closing_size)
    voxeldata.save_nrrd(output_path)


@app.command()
@click.argument(
    'input_dir',
    type=click.Path(
        exists=True,
        readable=True,
        dir_okay=True,
        resolve_path=True))
@click.option('-o', '--output_path', type=str, help='Output nrrd file name', required=True)
@click.option('-m', '--master_path', type=str, help='Name of the original nrrd file'
              ' that was used to generate the region files of the input directory', required=True)
@click.option('-l', '--overlap_label', type=int, help='Special value used to label'
              ' the voxels which lie in the overlap of several regions', required=True)
@log_args(L)
def merge(input_dir, output_path, master_path, overlap_label):
    """ Merge the content of all the nrrd files located in the input directory.

        The output is a single nrrd volumetric file whose dimensions are those of the original
        atlas file used to extract the regions.
        Overlapping voxels are assigned the specified overlap label.
        This means that if two non-void voxels coming from two different regions occupy
        the same location in the original atlas space, then the corresponding output
        voxel will be labelled with the special overlap label.
    """

    output_voxeldata = voxcell.VoxelData.load_nrrd(master_path)
    output_voxeldata.raw[:] = 0
    filepaths = [Path.resolve(f) for f in Path(input_dir).glob('*.nrrd')]
    for filepath in filepaths:
        region_voxeldata = voxcell.VoxelData.load_nrrd(filepath)
        merge_region(region_voxeldata, output_voxeldata, overlap_label)
    output_voxeldata.save_nrrd(output_path)


@app.command()
@click.argument('input_path', type=FILE_TYPE)
@click.option('-o', '--output_path', type=str, help='Output nrrd file name', required=True)
@click.option(
    '-l', '--label', type=int,
    help='label of the voxels to be re-assigned to their closest regions', required=True
)
@log_args(L)
def assign_to_closest_region(input_path, output_path, label):
    """ Assign each voxel with the specified label to its closest region.

        For each voxel of the input volumetric image bearing the specified label,
        the algorithm selects one of the closest voxels with a different but non-zero label.
        After assignment, the region identified by the specified label is
        entirely distributed accross the other regions of the input.
    Args:
          voxeldata(VoxelData): the VoxelData object whose voxels are
          going to be re-assigned to their closest region.
          label(int): the label of the region to be redistributed.
    """
    voxeldata = voxcell.VoxelData.load_nrrd(input_path)
    reassign_to_closest_region(voxeldata, label)
    voxeldata.save_nrrd(output_path)
