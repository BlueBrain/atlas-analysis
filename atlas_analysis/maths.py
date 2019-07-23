""" Maths utilities """
import numpy as np
from atlas_analysis.constants import ZVECTOR


def normalize_vectors(vectors):
    """ Normalize a bunch of vectors along axis==1
    Args:
        vectors: vectors with shape
         [[x1, y1, z1],
         [x2, y2, z2],
         [x3, y3, z3],
         ...
        ]
    """
    return vectors / np.sqrt(np.einsum('...i,...i', vectors, vectors)).reshape(-1, 1)


def normalize_vector(vector):
    """ Return a normalize vector (a bit faster for one vector than normalize_vectors) """
    return vector / np.linalg.norm(vector)


def get_middle(limits):
    """ Returns the mean of a 2-tuple. [a, b] --> (a+b)/2 """
    return (limits[0] + limits[1]) * 0.50


def get_normal(rot):
    """Returns the normal of the oriented plane obtained using the quaternion rot """
    return rot.rotate(ZVECTOR)
