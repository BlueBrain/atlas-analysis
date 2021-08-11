""" cli module with simple curation operations """
from pathlib import Path
import logging
import click
import voxcell
from atlas_analysis import curation
from atlas_analysis.curation import ALGORITHMS, NEAREST_NEIGHBOR_INTERPOLATION
from atlas_analysis.app.utils import log_args, set_verbose, FILE_TYPE

L = logging.getLogger("Curation")


@click.group()
@click.option('-v', '--verbose', count=True)
def app(verbose):
    """Run the different curation CLI """
    set_verbose(L, verbose)


@app.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option(
    '-o', '--output_path', type=str, help='Output nrrd file name', required=True
)
@click.option(
    '-s',
    '--threshold_size',
    type=int,
    help='Number of voxels below which a connected component is removed',
    required=True,
)
@click.option(
    '-c',
    '--connectivity',
    type=int,
    help='Integer value defining connected voxels.'
    ' Two voxels are connected if their squared distance is not more than connectivity.'
    ' Defaults to 1, i.e., two voxels are connected if they share a common face.',
    default=1,
)
@log_args(L)
def remove_connected_components(input_path, output_path, threshold_size, connectivity):
    """ Remove the connected components whose sizes are less than or equal to

    a specified threshold, the size being the number of voxels.
    The connectivity is an optional parameter which defines what connected
    voxels are. By default, two voxels are connected if they share a common face.
    """
    voxeldata = voxcell.VoxelData.load_nrrd(input_path)
    curation.remove_connected_components(voxeldata, threshold_size, connectivity)
    voxeldata.save_nrrd(output_path)


@app.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option(
    '-o',
    '--output_dir',
    type=str,
    help='Output directory name where the region files will be saved. '
    'It will be created if it doesn\'t exist.',
    required=True,
)
@log_args(L)
def split_regions(input_path, output_dir):
    """ Split an nrrd image file into different region images.

    A region file is generated for each non-zero voxel value, a.k.a label.
    Each region is cropped to its smallest enclosing bounding box and is saved under the form of
    an nrrd file.
    """
    voxeldata = voxcell.VoxelData.load_nrrd(input_path)
    curation.split_into_region_files(voxeldata, output_dir)


@app.command()
@click.argument('input_path', type=FILE_TYPE)
@click.option(
    '-o',
    '--output_dir',
    type=str,
    help='Output directory name where the connected component files will be saved. '
    'It will be created if it doesn\'t exist.',
    required=True,
)
@click.option(
    '-u',
    '--use_component_label',
    is_flag=True,
    help='If specified, voxels are labeled with the integer '
    'identifier of their component. Otherwise they keep their original labels.',
)
@log_args(L)
def split_components(input_path, output_dir, use_component_label):
    """  Split the input into different nrrd files, one for each connected component.

    A file is generated for each connected component of the input, the background excepted.
    Each component is cropped to its smallest enclosing bounding box and is saved under the form of
    an nrrd file in the specified output directory. The name of a connected component file is given
    by the integer identifier of the connected component (returned by scipy.ndimage.label)
    followed by \'.nrrd\'.
    If use_component_label is True, the voxels are labeled with the integer identifier
    of their component. Otherwise, they keep their original labels.
    """
    voxeldata = voxcell.VoxelData.load_nrrd(input_path)
    curation.split_into_connected_component_files(voxeldata, output_dir, use_component_label)


@app.command()
@click.argument('input_path', type=FILE_TYPE)
@click.option(
    '-o', '--output_path', type=str, help='Output nrrd file name', required=True
)
@click.option(
    '-f',
    '--filter_size',
    type=int,
    help='edge size of the box used for filtering the input image',
    required=True,
)
@click.option(
    '-c',
    '--closing_size',
    type=int,
    help='edge size of the box used to dilate the input image'
    ' before filtering and to erode it afterwards.',
    required=True,
)
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
    See https://en.wikipedia.org/wiki/Mathematical_morphology for definitions.
    The output is saved to the the specified output path.
    Note: this function does not preserve the volume and is likely to expand it.
    """
    voxeldata = voxcell.VoxelData.load_nrrd(input_path)
    voxeldata = curation.median_filter(voxeldata, filter_size, closing_size)
    voxeldata.save_nrrd(output_path)


@app.command()
@click.argument(
    'input_dir',
    type=click.Path(exists=True, readable=True, dir_okay=True, resolve_path=True),
)
@click.option(
    '-o', '--output_path', type=str, help='Output nrrd file name', required=True
)
@click.option(
    '-m',
    '--master_path',
    type=str,
    help='Name of the original nrrd file'
    ' that was used to generate the region files of the input directory',
    required=True,
)
@click.option(
    '-l',
    '--overlap_label',
    type=int,
    help='Special value used to label'
    ' the voxels which lie in the overlap of several regions',
    required=True,
)
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
    voxeldata = voxcell.VoxelData.load_nrrd(master_path)
    curation.merge_regions(input_dir, voxeldata, overlap_label)
    voxeldata.save_nrrd(output_path)


@app.command()
@click.argument('input_path', type=FILE_TYPE)
@click.option(
    '-o', '--output_path', type=str, help='Output nrrd file name', required=True
)
@click.option(
    '-l',
    '--label',
    type=int,
    help='label of the voxels to be re-assigned to their closest regions',
    required=True,
)
@click.option(
    '-a',
    '--algorithm',
    type=click.Choice(ALGORITHMS),
    help='string parameter to specify which algorithm is used',
    default=NEAREST_NEIGHBOR_INTERPOLATION,
)
@log_args(L)
def assign_to_closest_region(input_path, output_path, label, algorithm):
    """ Assign each voxel with the specified label to its closest region.

    For each voxel of the input volumetric image bearing the specified label,
    the algorithm selects one of the closest voxels with a different but non-zero label.
    After assignment, the region identified by the specified label is
    entirely distributed accross the other regions of the input.
    """
    voxeldata = voxcell.VoxelData.load_nrrd(input_path)
    curation.assign_to_closest_region(voxeldata, label, algorithm)
    voxeldata.save_nrrd(output_path)


@app.command()
@click.argument('input_path', type=FILE_TYPE)
@click.option(
    '-o',
    '--output_dir',
    type=str,
    help='Output directory name where the smoothed region files will be saved. '
    'It will be created if it doesn\'t exist.',
    required=True,
)
@click.option(
    '-s',
    '--threshold_size',
    type=int,
    help='Number of voxels below which a connected component is removed',
    default=1000,
)
@click.option(
    '-f',
    '--filter_size',
    type=int,
    help='edge size of the box used for filtering the input image',
    default=8,
)
@click.option(
    '-c',
    '--closing_size',
    type=int,
    help='edge size of the box used to dilate the input image'
    ' before filtering and to erode it afterwards.',
    default=14,
)
@log_args(L)
def smooth(input_path, output_dir, threshold_size, filter_size, closing_size):
    """ Smooth each individual region of the input volumetric file and merge.

    * Split the input file into different region files, each clipped to its minimal
    axis-aligned bounding box\n
    * Remove the small connected components of each region\n
    * Smooth each region using a median filter intertwined with a morphological closing\n
    * Merge all region files into the output file\n
    * Assign the voxels of overlapping regions to their closest regions\n
    * Save the smoothed volume into the specified output directory\n
    \n
    Note: the dimensions of output nrrd file will be different from those of the input file.
    This is so because the input image is first cropped to
    its smallest enclosing axis-aligned box for performance reasons
    and subsequently enlarged to take into account volume expansion caused by smoothing.
    """

    voxeldata = voxcell.VoxelData.load_nrrd(input_path)
    curation.smooth(voxeldata, output_dir, threshold_size, filter_size, closing_size)
    input_filename = Path(input_path).name
    output_path = Path(output_dir, input_filename)
    voxeldata.save_nrrd(str(output_path))


@app.command()
@click.argument('input_path', type=FILE_TYPE)
@click.option(
    '-o', '--output_path', type=str, help='Output nrrd file name', required=True
)
@log_args(L)
def fill_cavities(input_path, output_path):
    """ Fill the cavities of an nrrd image file.

    A cavity is a hole nested inside a thick 3D volume.
    Cavities are filled in by assigning the non-zero
    labels of the closest neighboring voxels.
    """
    voxeldata = voxcell.VoxelData.load_nrrd(input_path)
    curation.fill_cavities(voxeldata)
    voxeldata.save_nrrd(output_path)


@app.command()
@click.argument('input_path', type=FILE_TYPE)
@click.option(
    '-o', '--output_path', type=str, help='Output nrrd file name', required=True
)
@click.option(
    '-m',
    '--margin',
    type=int,
    help='Number of voxels used to pad the input volume in each dimension. Defaults to 5',
    default=5,
)
@log_args(L)
def add_margin(input_path, output_path, margin):
    """ Add margin around the input volume.

    Usage: atlas-analysis curation [OPTIONS] INPUT_PATH\n
    This function creates a margin of zero-valued voxels around the input volume and saves
    the modified volume to the nrrd file with path INPUT_PATH.
    The margin thickness, expressed in terms of voxels, is controlled by the margin option.
    Its default value is 5 voxels.
    Each dimension of the input array will be incremented by 2 * margin.
    The VoxelData offset is changed accordingly.
    """
    voxeldata = voxcell.VoxelData.load_nrrd(input_path)
    voxeldata = curation.add_margin(voxeldata, margin)
    voxeldata.save_nrrd(output_path)


@app.command()
@click.argument('input_path', type=FILE_TYPE)
@click.option(
    '-o',
    '--output_path',
    type=click.Path(),
    help='Output nrrd file name',
    required=True,
)
@click.option(
    '-m',
    '--margin',
    type=int,
    help='Number of voxels used to pad the output volume in each dimension. Defaults to 0',
    default=0,
)
@log_args(L)
def crop(input_path, output_path, margin):
    """ Crop the input volume using its smallest enclosing box.

    Usage: atlas-analysis curation [OPTIONS] INPUT_PATH\n
    This functions crops the input volume using its smallest enclosed Axis Aligned Bounding Box.
    Optionally, it creates a margin of zero-valued voxels around the input volume and saves
    the modified volume to the nrrd file with path INPUT_PATH.
    The margin thickness, expressed in terms of voxels, is controlled by the margin option.
    Its default value is 0 voxels.
    Each dimension of the input array will be incremented by 2 * margin.
    The VoxelData offset is changed accordingly.
    """
    voxeldata = voxcell.VoxelData.load_nrrd(input_path)
    curation.crop(voxeldata)
    voxeldata = curation.add_margin(voxeldata, margin)
    voxeldata.save_nrrd(output_path)
