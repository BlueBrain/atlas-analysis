""" cli module with simple reporting operators """
import logging
from pathlib import Path

import click

import voxcell
from atlas_analysis import reporting
from atlas_analysis.utils import string_to_type_converter
from atlas_analysis.app.utils import log_args, set_verbose, FILE_TYPE

L = logging.getLogger("Reporting")


@click.group()
@click.option('-v', '--verbose', count=True)
def app(verbose):
    """Run the different reporting CLI """
    set_verbose(L, verbose)


@app.command()
@click.argument('input_path', type=FILE_TYPE)
@click.option(
    '-i',
    '--input_density_dir',
    type=click.Path(exists=True, readable=True, dir_okay=True, resolve_path=True),
)
@click.option(
    '-h', 'hierarchy_path', type=FILE_TYPE, help='The hierachy file path', required=True
)
@click.option('-o', '--output_path', type=str, help='Output file name', required=True)
@click.option(
    '-f', '--field_name', type=str, help='The field name to check', required=True
)
@click.option('-va', '--value', type=str, help='The value to match', required=True)
@click.option(
    '-t',
    '--value_type',
    type=click.Choice(['int', 'float', 'str', 'bool']),
    help='The type for value to match',
    required=True,
)
@log_args(L)
def write_density_report(
    input_path,
    input_density_dir,
    hierarchy_path,
    output_path,
    field_name,
    value,
    value_type,
):
    """ Write a report file containing the average densities of the specified regions.

    Finds in the hierarchy file the IDs of all leaf regions
    matching the given attribute defined by the specified field name, value and value type.
    Creates subsequently a report file in json format that contains the average
    density for each leaf region and for each density file located in
    the input directory. A density file corresponds to a density type.
    For instance, we have density files of the form
    cell_density.nrrd, exc_density.nrrd (excitatory neurons) or glia_density.nrrd.
    The sections of the report are based on file names.

    Densities are expressed in (um)^{-3}.

    Note: input_path is the path to some nrrd annotation file whose dimensions and resolution
    should match those of each density file located in input_density_dir.
    """

    value = string_to_type_converter(value_type)(
        value
    )  # Converting value into the correct type
    rmap = voxcell.RegionMap.load_json(hierarchy_path)
    leaf_ids = []
    for identifier in rmap.find(value, field_name, with_descendants=True):
        labels = rmap.find(identifier, 'id', with_descendants=True)
        if len(labels) == 1:  # A label with no descendant. This is a leaf.
            leaf_ids.append(identifier)
    voxel_data = voxcell.VoxelData.load_nrrd(input_path)
    filepaths = [Path.resolve(f) for f in Path(input_density_dir).glob('*.nrrd')]
    report = reporting.DensityReport.from_files(voxel_data, filepaths, leaf_ids)
    report.save_as(output_path)


@app.command()
@click.argument('input_path', type=FILE_TYPE)
@click.option('-o', '--output_path', type=str, help='Output file name', required=True)
@click.option(
    '-c',
    '--connectivity',
    is_flag=True,
    help='Enable or disable connectivity reporting.'
    ' If enabled, the report indicates if a region is connected or not'
    ' and displays the count of connected components of less than'
    '1\'000\'000 voxels.'
    ' Defaults to False, i.e, no connectivity reporting.',
    default=False,
)
@click.option(
    '-ca',
    '--cavity',
    is_flag=True,
    help='Enable or disable cavity reporting.'
    ' Cavities are holes nested in the thick parts of a volume.'
    ' If enabled, the report will contain the voxel counts of cavities per region.'
    ' Defaults to False, i.e, no cavity reporting.',
    default=False,
)
@log_args(L)
def write_voxel_count_report(input_path, output_path, connectivity, cavity):
    """ Write a report file containing voxel counts per region.

    Write a report in json format including:
    - the number of regions.
    - the sorted list of regions where regions are represented by their voxel integer label.
    - the number of voxels per region.
    If the connectivity option is set to True, the report indicates whether regions are connected
    or not. The report contains then the total numbers of connected components with
    size <= 100\'000\'000 voxels together with an histogram of the connected the components counts
    with respect to size. The bins of the histogram are of the form [0, 10), ...,
    [10**N, 10**(N + 1)) for N = 7.
    The report indicates the number of connected components whose sizes lie in a bin
    of the above form.

    If the cavity option is set to True, the report includes the voxel count of cavities for
    the whole file and the break-down for each region as well.
    A cavity is a hole nested in a thick part of a volume.
    """

    voxel_data = voxcell.VoxelData.load_nrrd(input_path)
    report = reporting.VoxelCountReport.from_voxel_data(
        voxel_data, connectivity_is_required=connectivity, cavities_are_required=cavity
    )
    report.save_as(output_path)
