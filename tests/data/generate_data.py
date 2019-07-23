import nrrd
import numpy as np


default_header = {'space': 'left-posterior-superior',
          'space directions': np.array([[10., 0., 0.],
                                     [0., 10., 0.],
                                     [0., 0., 10.]]),
          'space origin': np.array([-1, -1, -1])}


t = np.zeros((3, 3, 3), dtype=np.uint8)
t[:, 0, 0] = 1
t[:, 1, 0] = 2
nrrd.write("1.nrrd", t, options=default_header)


t = np.zeros((3, 3, 3), dtype=np.uint8)
t[0, 0, 0] = 1
t[0, 1, 0] = 2
t[0, 2, 0] = 3
nrrd.write("data.nrrd", t, options=default_header)


t2 = np.zeros((3, 3, 3), dtype=np.uint8)
t2[0:2, :, 0] = 1
nrrd.write("2.nrrd", t2, options=default_header)

# ================================

t = np.zeros((4, 3, 3), dtype=np.uint8)
t[0, 0, 0] = 1
nrrd.write("1_shape.nrrd", t, options=default_header)


t = np.zeros((3, 3, 3), dtype=np.uint16)
t[0, 0, 0] = 1
nrrd.write("1_type.nrrd", t, options=default_header)


header = {'space': 'left-posterior-superior',
          'space directions': np.array([[11., 0., 0.],
                                     [0., 10., 0.],
                                     [0., 0., 10.]]),
          'space origin': np.array([-1, -1, -1])}


t = np.zeros((3, 3, 3), dtype=np.uint8)
t[0, 0, 0] = 1
nrrd.write("1_voxel_dimensions.nrrd", t, options=header)

header = {'space': 'left-posterior-superior',
          'space directions': np.array([[10., 0., 0.],
                                     [0., 10., 0.],
                                     [0., 0., 10.]]),
          'space origin': np.array([-2, -1, -1])}


t = np.zeros((3, 3, 3), dtype=np.uint8)
t[0, 0, 0] = 1
nrrd.write("1_offset.nrrd", t, options=header)

# ===================================================
default_header = {'space': 'left-posterior-superior',
          'space directions': np.array([[10., 0., 0.],
                                     [0., 10., 0.],
                                     [0., 0., 10.]]),
          'space origin': np.array([-1, -1, -1])}


t = np.zeros((1, 1, 1), dtype=np.int8)
t[0, 0, 0] = -1
nrrd.write("negative_int8.nrrd", t, options=default_header)

t = np.zeros((2, 1, 1), dtype=np.int8)
t[0, 0, 0] = -1
t[1, 0, 0] = 1
nrrd.write("negative_positive_int8.nrrd", t, options=default_header)

t = np.zeros((1, 1, 1), dtype=np.uint8)
t[0, 0, 0] = 250
nrrd.write("large_uint8.nrrd", t, options=default_header)

t = np.zeros((1, 1, 1), dtype=np.uint32)
t[0, 0, 0] = 2
nrrd.write("small_uint32.nrrd", t, options=default_header)

t = np.zeros((1, 1, 1), dtype=np.int32)
t[0, 0, 0] = 20000
nrrd.write("large_int32.nrrd", t, options=default_header)

# ===================================================
default_header = {'space': 'left-posterior-superior',
          'space directions': np.array([[10., 0., 0.],
                                     [0., 10., 0.],
                                     [0., 0., 10.]]),
          'space origin': np.array([-1, -1, -1])}


t = np.zeros((4, 1, 1, 1), dtype=np.float32)
t[0, 0, 0, 0] = 0.1
t[1, 0, 0, 0] = 0.2
t[2, 0, 0, 0] = 0.3
t[3, 0, 0, 0] = 0.4
nrrd.write("orientation.nrrd", t, options=default_header)