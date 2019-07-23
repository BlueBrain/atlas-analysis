import requests
import zipfile
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
import nrrd
from voxcell import VoxelData
import atlas_analysis.atlas as atlas

DIRECTORY_PATH = Path('/tmp/Hippocampus')


def retrieve_ascoli_data(output_path):
    local_zip_path = Path(DIRECTORY_PATH, 'Hippocampus.zip')
    local_path = Path(DIRECTORY_PATH, 'hippocampus-voxeldb.txt')
    url = 'http://cng.gmu.edu/hippocampus3d/Hippo3DData/VoxelDB/Hippocampus-VoxelDB.zip'
    ascoli_column_header = 'X Y Z L T D B La Type\n'

    if not DIRECTORY_PATH.exists():
        DIRECTORY_PATH.mkdir()

    r = requests.get(url)
    with open(local_zip_path, 'wb') as original_fd:
        original_fd.write(r.content)

    with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
        zip_ref.extractall(DIRECTORY_PATH)

    with open(local_path, "r") as fd:
        contents = fd.readlines()
    contents.insert(0, ascoli_column_header)
    with open(local_path, "w") as fd:
        contents = "".join(contents)
        fd.writelines(contents)

    df = pd.read_csv(local_path, sep='\s+')

    raw = np.zeros((500, 500, 500), dtype=np.int32)
    xs = np.floor(df['X'].to_numpy() / 16.).astype(np.int32)
    ys = np.floor(df['Y'].to_numpy() / 16.).astype(np.int32)
    zs = np.floor(df['Z'].to_numpy() / 16.).astype(np.int32)
    raw[xs, ys, zs] = df['Type'].to_numpy()

    default_nrrd_header = {'space': 'left-posterior-superior',
                           'space directions': np.array([[16., 0., 0.],
                                                         [0., 16., 0.],
                                                         [0., 0., 16.]]),
                           'space origin': np.array([0, 0, 0])}

    nrrd.write(str(output_path), raw, options=default_nrrd_header)


def extract(input_path, to_extract, regrouped_name):
    v = VoxelData.load_nrrd(str(input_path))
    atlases = dict()

    def _save(catlas, cname):
        cpath = "{}.nrrd".format(Path(DIRECTORY_PATH, cname))
        catlas.save_nrrd(cpath)
        atlases[cname] = cpath

    filtered_atlases = []
    for i, (name, labels) in enumerate(to_extract.items(), 1):
        voxeldata = atlas.reset_all_values(atlas.extract_labels(v, labels), i)
        filtered_atlases.append(voxeldata)
        _save(voxeldata, name)

    regrouped = atlas.regroup_atlases(filtered_atlases)

    # should not be possible but keep it for the example
    overlap = atlas.logical_and(filtered_atlases, 1)
    assert np.count_nonzero(overlap.raw) == 0

    _save(regrouped, regrouped_name)
    return atlases


def compute_ratios(filepath, labels):
    voxeldata = VoxelData.load_nrrd(filepath)
    tot_volume = voxeldata.volume(list(labels.values()))
    res = {}
    for name, label in labels.items():
        res.update({name: voxeldata.volume(label) / tot_volume})
    return res


def main():
    nrrd_path = Path(DIRECTORY_PATH, 'Hippocampus.nrrd')
    # see http://krasnow1.gmu.edu/cn3/hippocampus3d/VOXELDB-README.html
    labels = OrderedDict([('CA3a', [21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91, 98]),
                          ('CA3b', [20, 27, 34, 41, 48, 55, 62, 69, 76, 83, 90, 97]),
                          ('CA3c', [19, 26, 33, 40, 47, 54, 61, 68, 75, 82, 89, 96])]
                         )

    retrieve_ascoli_data(nrrd_path)
    atlases = extract(nrrd_path, labels, 'CA3')
    ratios = compute_ratios(atlases['CA3'], {'CA3a': 1, 'CA3b': 2, 'CA3c': 3})
    print(ratios)


if __name__ == '__main__':
    main()
