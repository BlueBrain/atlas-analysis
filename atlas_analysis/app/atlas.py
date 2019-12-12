""" cli module with simple atlases operators """
import sys
import logging
from pathlib import Path
import json

import numpy as np
import click

import voxcell

from atlas_analysis import atlas
from atlas_analysis.utils import add_suffix, string_to_type_converter
from atlas_analysis.app.utils import split_str, log_args, load_nrrds, set_verbose, FILE_TYPE

L = logging.getLogger("Atlas")


CASTING_STRATEGY_HELP = 'Defines the casting strategy when combining different atlases'


@click.group()
@click.option('-v', '--verbose', count=True)
def app(verbose):
    """Run the different atlas CLI """
    set_verbose(L, verbose)


@app.command()
@click.argument('input_path', nargs=-1, type=FILE_TYPE)
@click.option('-o', '--output_path', type=str, help='Output nrrd file name', required=True)
@click.option('-t', '--new_type', type=str, help='The new nrrd file type (numpy types, ex:int32)',
              required=True)
@log_args(L)
def cast_atlas(input_path, output_path, new_type):
    """ Safely cast an atlas into a new data type """
    voxeldata = voxcell.VoxelData.load_nrrd(input_path)
    voxeldata = atlas.safe_cast_atlas(voxeldata, new_type)
    voxeldata.save_nrrd(output_path)


@app.command()
@click.argument('input_paths', nargs=-1, type=FILE_TYPE)
@log_args(L)
def check_atlas_properties(input_paths):
    """ Find if dtype, shape, voxel dimension or offset are different """
    atlases = load_nrrds(input_paths)
    are_ok = True
    if not atlas.compare_all(atlases, lambda x: x.raw.dtype, comp=np.equal):
        are_ok = False
        L.warning('dtypes are different %r', list(map(lambda x: x.raw.dtype, atlases)))
    if not atlas.compare_all(atlases, lambda x: x.raw.shape, comp=np.allclose):
        are_ok = False
        L.warning('Shapes are different %r', list(map(lambda x: x.raw.shape, atlases)))
    if not atlas.compare_all(atlases, lambda x: x.voxel_dimensions, comp=np.allclose):
        are_ok = False
        L.warning('Voxel_dimensions are different %r', list(map(lambda x: x.voxel_dimensions,
                                                                atlases)))
    if not atlas.compare_all(atlases, lambda x: x.offset, comp=np.allclose):
        are_ok = False
        L.warning('Offsets are different %r', list(map(lambda x: x.offset, atlases)))
    if are_ok:
        L.info('%r are compatible', ":".join(list(input_paths)))
    sys.exit(not are_ok)


@app.command()
@click.argument('input_paths', nargs=-1, type=FILE_TYPE)
@click.option('-o', '--output_dir', type=str, help='Output directory name', required=True)
@click.option('-s', '--suffix', type=str,
              help='The suffix added to the filepath in the output directory', default='')
@click.option('-f', '--force', is_flag=True, default=False,
              help='Force the suffix and can override original files if needed')
@click.option('-c', '--cast', type=click.Choice(['safe', 'minimal', 'strict']),
              help=CASTING_STRATEGY_HELP, default='strict')
@log_args(L)
def homogenize_dtypes(input_paths, output_dir, suffix, force, cast):
    """ Homegenize multiple atlases dtypes into a safe dtype """
    atlases = load_nrrds(input_paths)
    atlases = atlas.homogenize_atlas_types(atlases, cast=cast)
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir()

    for input_path, voxel_data in zip(input_paths, atlases):
        file_name = Path(input_path).name
        output_path = str(Path(output_dir, add_suffix(file_name, suffix, force=force)))
        voxel_data.save_nrrd(output_path)


@app.command()
@click.argument('input_path', type=FILE_TYPE)
@click.option('-h', 'hierarchy_path', type=FILE_TYPE, help='The hierachy file path',
              required=True)
@click.option('-o', '--output_path', type=str, help='Output nrrd file name', required=True)
@click.option('-f', '--field_name', type=str, help='The field name to check', required=True)
@click.option('-va', '--value', type=str, help='The value to match', required=True)
@click.option('-t', '--value_type', type=click.Choice(['int', 'float', 'str', 'bool']),
              help='The type for value to match', required=True)
@click.option('-d', '--descendants', is_flag=True, default=False, help='Retrieve also the children')
@click.option('-l', '--label', type=int, default=None,
              help='New single label value for the full selected atlas')
@log_args(L)
def extract(input_path, hierarchy_path, output_path, field_name, value, value_type, descendants,
            label):
    """ Does a voxel extraction using a hierarchy file.

    Finds IDs of the regions matching a given attribute defined by the field_name,
    the value and value type in the hierarchy file. Then creates a new .nrrd file that will
    contain only the selected areas.
    """
    # need to convert the value into the correct type. Complex to infer a correct type here.
    value = string_to_type_converter(value_type)(value)
    rmap = voxcell.RegionMap.load_json(hierarchy_path)
    ids = list(rmap.find(value, field_name, with_descendants=descendants))
    L.info('split_atlas: selected ids: %r', sorted(ids))
    voxel_data = voxcell.VoxelData.load_nrrd(input_path)
    voxel_data = atlas.extract_labels(voxel_data, ids, new_label=label)
    voxel_data.save_nrrd(output_path)


@app.command()
@click.argument('input_path', type=FILE_TYPE)
@click.option('-o', '--output_path', type=str, help='Output nrrd file name', required=True)
@click.option('-i', '--ids', type=str, help='Ids to extract', required=True)
@click.option('-l', '--label', type=int, default=None,
              help='New single label value for the full selected atlas')
@log_args(L)
def simple_extract(input_path, output_path, ids, label):
    """ Extract voxels with labels contain into ids and create a new nrrd file """
    ids = split_str(ids, int)
    voxeldata = voxcell.VoxelData.load_nrrd(input_path)
    voxeldata = atlas.extract_labels(voxeldata, ids, new_label=label)
    voxeldata.save_nrrd(output_path)


@app.command()
@click.argument('input_paths', type=FILE_TYPE, nargs=-1)
@click.option('-o', '--output_path', type=str, help='Output nrrd file name', required=True)
@click.option('-l', '--label', type=int, default=None, help='New value for the selected labels')
@click.option('-c', '--cast', type=click.Choice(['safe', 'minimal', 'strict']),
              help=CASTING_STRATEGY_HELP, default='strict')
@log_args(L)
def regroup(input_paths, output_path, label, cast):
    """ Regroup multiple atlases in one"""
    atlases = load_nrrds(input_paths)
    voxeldata = atlas.regroup_atlases(atlases, new_label=label, cast=cast)
    voxeldata.save_nrrd(output_path)


@app.command()
@click.argument('input_paths', type=FILE_TYPE, nargs=-1)
@click.option('-o', '--output_path', type=str, help='Output nrrd file name', required=True)
@click.option('-l', '--label', type=int, help='Value set to the overlapping voxels',
              required=True)
@click.option('-c', '--cast', type=click.Choice(['safe', 'minimal', 'strict']),
              help=CASTING_STRATEGY_HELP, default='strict')
@log_args(L)
def logical_and(input_paths, output_path, label, cast):
    """ Compute the logical_and between atlases """
    atlases = load_nrrds(input_paths)
    voxeldata = atlas.logical_and(atlases, label, cast=cast)
    voxeldata.save_nrrd(output_path)


@app.command()
@click.argument('input_path', type=FILE_TYPE, nargs=1)
@click.option('-m', '--mask_path', type=FILE_TYPE, help='Mask nrrd file name', required=True)
@click.option('-o', '--output_path', type=str, help='Output nrrd file name', required=True)
@click.option('-l', '--label', type=int, default=None, help='New value for the selected labels')
@click.option('-mo', '--masked_off', is_flag=True, default=False,
              help='Define the mask as off (keep data outside the mask)')
@log_args(L)
def mask(input_path, cropping_path, output_path, label, masked_off):
    """Crop an atlas using a nrrd mask """
    input_data, mask_data = load_nrrds([input_path, cropping_path])
    voxeldata = atlas.voxel_mask(input_data, mask_data, masked_off=masked_off)
    if label is not None:
        voxeldata = atlas.reset_all_values(voxeldata, label)
    voxeldata.save_nrrd(output_path)


@app.command()
@click.argument('input_path', type=FILE_TYPE)
@click.option('-o', '--output_path', type=str,
              help='The output raw filepath', default=None)
@click.option('-e', '--encoding', type=click.Choice(atlas.VALID_ENCODING_NRRD),
              help='The new encoding', required=True)
@click.option('-s', '--suffix', type=str,
              help='The suffix added to the filepath', default='_new_encoding')
@log_args(L)
def change_nrrd_encoding(input_path, output_path, encoding, suffix):
    """ Create a new version of a nrrd file with a new encoding """
    output = atlas.change_encoding(input_path, output=output_path, encoding=encoding, suffix=suffix)
    L.info('create_raw:File created here: %s', output)


@app.command()
@click.argument('input_path', type=FILE_TYPE)
@click.option(
    '-i', '--input_density_dir',
    type=click.Path(exists=True, readable=True, dir_okay=True, resolve_path=True),
)
@click.option('-h', 'hierarchy_path', type=FILE_TYPE, help='The hierachy file path',
              required=True)
@click.option('-o', '--output_path', type=str, help='Output file name', required=True)
@click.option('-f', '--field_name', type=str, help='The field name to check', required=True)
@click.option('-va', '--value', type=str, help='The value to match', required=True)
@click.option('-t', '--value_type', type=click.Choice(['int', 'float', 'str', 'bool']),
              help='The type for value to match', required=True)
@log_args(L)
def write_density_report(
    input_path, input_density_dir, hierarchy_path,
    output_path, field_name, value, value_type
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

    value = string_to_type_converter(value_type)(value)  # Converting value into the correct type
    rmap = voxcell.RegionMap.load_json(hierarchy_path)
    leaf_ids = []
    for identifier in rmap.find(value, field_name, with_descendants=True):
        labels = rmap.find(identifier, 'id', with_descendants=True)
        if len(labels) == 1:  # A label with no descendant. This is a leaf.
            leaf_ids.append(identifier)
    voxel_data = voxcell.VoxelData.load_nrrd(input_path)
    filepaths = [Path.resolve(f) for f in Path(input_density_dir).glob('*.nrrd')]
    report_dict = atlas.compute_density_report(voxel_data, leaf_ids, filepaths)
    with open(output_path, 'w') as report_file:
        report_file.write(json.dumps(report_dict))
