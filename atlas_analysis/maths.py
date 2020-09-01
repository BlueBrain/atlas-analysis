""" Maths utilities """
import numpy as np


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
