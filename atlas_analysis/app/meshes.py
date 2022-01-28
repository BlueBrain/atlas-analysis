""" cli module to deal with meshes """
import sys
import logging
from pathlib import Path

import click
import voxcell

from atlas_analysis import meshes
from atlas_analysis.app.utils import log_args, split_str, set_verbose, FILE_TYPE
from atlas_analysis.vtk_visualization import render
from atlas_analysis.vtk_utils import save_unstructuredgrid_to_stl, save_polydata_to_stl
from atlas_analysis.exceptions import AtlasAnalysisError

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
L = logging.getLogger("Meshes")


@click.group()
@click.option('-v', '--verbose', count=True)
def app(verbose):
    """Run the different meshes CLI """
    set_verbose(L, verbose)


@app.command()
@click.argument('input_path', type=FILE_TYPE)
@click.option('-n', '--names', type=str, help='Names of the output files', required=True)
@click.option('-i', '--ids', type=str, help='Ids to extract', required=True)
@click.option('-a', '--algorithm', type=click.Choice(meshes.ALGORITHMS),
              default=meshes.MARCHING_CUBES, help='The algorithm to use for the mesh creation')
@click.option('-o', '--output_dir', type=click.Path(dir_okay=True),
              help='The output directory that will contain the extracted meshes', default=None)
@log_args(L)
def create(input_path, names, ids, algorithm, output_dir):
    """ Create a stl files from a nrrd file """
    names = split_str(names, str)
    ids = split_str(ids, int)
    if len(names) != len(ids):
        raise AtlasAnalysisError(
            'names (-n) and ids (-i) arguments must have the same number of elements')
    parameters = dict(zip(names, ids))

    output_dir_path = Path(output_dir)
    if not output_dir_path.exists():
        output_dir_path.mkdir()

    voxel_data = voxcell.VoxelData.load_nrrd(input_path)
    L.info('Use %s algorithm', algorithm)
    res = meshes.create_meshes(voxel_data, parameters, algorithm=algorithm)

    def _save(mesh, name_, save_fun):
        if Path(name_).suffix != '.stl':
            name_ = name_ + '.stl'
        path = str(Path(output_dir, name_).absolute())
        save_fun(mesh, path)
        L.info('File %s has been created', path)

    for mesh_name, mesh in res.items():
        if algorithm == meshes.MARCHING_CUBES:
            _save(mesh, mesh_name, save_polydata_to_stl)
        elif algorithm == meshes.ALPHA_HULL:
            _save(mesh, mesh_name, save_unstructuredgrid_to_stl)
        else:
            raise AtlasAnalysisError(f'{algorithm} unsupported mesh algorithm')


@app.command()
@click.argument('files', type=FILE_TYPE, nargs=-1)
def draw(files):
    """Draw the different files in the same scene """
    render(stl_files=files)
