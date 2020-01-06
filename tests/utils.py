from pathlib import Path

import numpy as np
import voxcell

DATA_PATH = Path(Path(__file__).parent, 'data')


def path(file):
    return Path(DATA_PATH, file)


def load_nrrd(name):
    return voxcell.VoxelData.load_nrrd(path(name))


def load_orientation(name):
    return voxcell.OrientationField.load_nrrd(path(name))


def load_nrrds(file_paths):
    return list(map(load_nrrd, map(path, file_paths)))


def create_rectangular_shape(length, width):
    raw = np.zeros((length, width, width))
    raw[1:-1, 1:-1, 1:-1] = 12
    return voxcell.VoxelData(raw, (10, 10, 10), (0, 0, 0))
