""" Maths utilities for plane creation

The functions of this module are internal functions taylored for the planes module.

# TODO(Luc): make an actual Plane class with equation coefficients accessor.

The planes created by the planes module can be represented under two different output formats:

- the so-called quaterionic format [x, y, z, a, b, c, d]
- the equation format [x, y, z, A, B, C, D]

The quaternionic format is also the internal format used by atlas-analysis.
All function in this module operate only on the planes with the quaternionic format.
The equation format is only a user-friendly alternative ouput format, derived from the
internal format by means of conversion function defined in this file.

These formats are defined in the planes.create_centerline_planes docstring. We
recall here there meanings.

For both output formats the float vector (x, y, z) represents the 3D coordinates of a
distinguished point of the plane P. This point is an anchor that can be used as the origin
of a 2D orthonormal frame of plane for visualization purposes. Usually, the point (x, y, z) is
the intersection point of P with a computed centerline.

With the format [x, y, z, a, b, c, d] of a plane P, the (a, b, c, d)-part is a
unit quaternion q complying with the convention 'w, x, y, z'. The quaternion q =
w + x * i + y * j + z * k maps ZVECTOR = (0, 0, 1) to a unit vector orthogonal to P,
 i.e., qkq^{-1} = n_x * i + n_y * j + n_z * k where (n_x, n_y, n_z) is a normal unit vector of P.
 See https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation.

With the format [x, y, z, A, B, C, D] of a plane P, the float vector [A, B, C,
D] holds the coefficients of an equation representing P, that is, P is the locus
of the points (X, Y, Z) satisfying the equation A * X + B * Y + C * Z = D.
The normal vector (A, B, C) is assumed to be normalized.
"""
import numpy as np
from pyquaternion import Quaternion
from atlas_analysis.utils import ensure_list
from atlas_analysis.constants import QUAT, XYZ, ZVECTOR
from atlas_analysis.maths import normalize_vector

EQUATION = QUAT


def get_normal(quaternion):
    """Returns the normal of the oriented plane obtained using the quaternion rot

    Args:
        quaternion(Quaternion): quaternion whose rotation turns (0, 0, 1)
            in the queried 'normal' vector.

    Returns:
        numpy.ndarray of shape (3,), i.e., a 3D vector.
    """

    return quaternion.rotate(ZVECTOR)


def get_plane_quaternion(plane):
    """Create the plane quaternion and force the normal to follow the longitudinal axis.

    Args:
        plane: a plane using the format [x, y, z, a, b, c, d] of the create_centerline_planes
             output.

    Returns:
        a unitary Quaternion from the [w, x ,y ,z] elements [a, b, c, d] followed by normalization.
    """
    return Quaternion(plane[QUAT]).unit


def split_plane_elements(plane):
    """Split a plane into a position and a quaternion.

    Args:
        plane: a plane under the quaternionic format [x, y, z, a, b, c, d]

    Returns:
        a 2-tuple with
            - the first element being a list containing the position [x, y, z] of distinguished
             point in the plane
            - the second element is a 4D vector [a, b, c, d] holding the w, x, y, z coordinates of
             unitary quaternion.
    """

    return plane[XYZ], plane[QUAT]


def get_normals(planes):
    """Returns normal vector for each plane.

    Args:
        planes: the list of planes using the quaternionic format [x, y, z, a, b, c, d]

    Returns: the normal vectors for all planes (np.array([x,y,z]))
    """
    planes = ensure_list(planes)
    return np.array([get_normal(get_plane_quaternion(plane)) for plane in planes])


def distances_to_planes(point, planes):
    """Returns the signed distance of a point to a list of planes.

    Args:
        point: the point [x, y, z] to compute the distances to.
        planes: the list of planes under the quaternionic format
            [x, y, z, a, b, c, d].

    Returns: signed distance (np.array([d1, ..., d2])) to all planes.
    """
    normals = get_normals(planes)
    point = np.asarray(point)
    planes = np.asarray(planes)
    xyzs = planes[:, XYZ]
    return np.einsum('ij,ij->i', normals, point - xyzs) / np.linalg.norm(
        normals, axis=1
    )


def quaternion_format_to_equation(planes):
    """Turns planes under quaternionic format to plane equation format.

    Args:
        planes(numpy.ndarray): float array of shape (N, 7), more specifically a sequence of N
             planes with quaternionic format [x, y, z, a, b, c, d].

    Returns:
        numpy.ndarray of shape (N, 7), i.e., a sequence of N planes under the equation format
            [x, y, z, A, B, C, D] representing each input plane as the locus of points (X, Y, Z)
             satisfying the equation A * X + B * Y + C * Z = D. The first three coordinates
            (x, y, z) coincide with the input ones.
    """
    output_planes = []
    for plane in planes:
        point, quaternion = split_plane_elements(plane)
        quaternion = Quaternion(quaternion)
        normal = normalize_vector(get_normal(quaternion))
        scalar = np.dot(normal, point)
        output_planes.append(np.concatenate([point, normal, [scalar]]))

    return np.array(output_planes)


def quaternion_from_vectors(s, t):
    '''
    Returns the quaternion (s cross t) + (s dot t + |s||t|).

    This quaternion q maps s to t, i.e., qsq^{-1} = t.

    Note: The input vectors are required to have the same norm, i.e.,
        |s| = |t|.

    Args:
        s(numpy.ndarray): numeric array of shape (3,).
        t(numpy.ndarray): numeric array of shape (3,).

    Returns:
        Numeric array of shape (4,).
        This data is interpreted as a quaternion. A quaternion is a 4D
        vector [w, x, y, z] where [x, y, z] is the imaginary part.
    '''
    w = np.dot(s, t) + np.linalg.norm(s) * np.linalg.norm(t)
    return np.hstack([w, np.cross(s, t)])


# Perform the reverse operation of quaternion_format_to_equation for reading .npz plane
# output and for validation
def equation_format_to_quaternion(planes):
    """Turns planes under plane equation format to quaternionic format.

    Args:
        planes(numpy.ndarray): float array of shape (N, 7), more specifically a sequence of N
             planes with plane equation format [x, y, z, A, B, C, D].

    Returns:
        numpy.ndarray of shape (N, 7), i.e., a sequence of N planes under the quaternionic format
            [x, y, z, a, b, c, d] where [a, b, c, d] is a unitary quaternion
             q = a + b * i + c * j + d * k which maps ZVECTOR = (0, 0, 1) to (A, B, C), i.e.,
             qkq^{-1} = A * i + B * j + C * k and such that a > 0.0.
             The first three coordinates (x, y, z) coincide with the input ones.
    """
    output_planes = []
    for plane in planes:
        point, equation = plane[XYZ], plane[EQUATION]
        quaternion = quaternion_from_vectors(ZVECTOR, normalize_vector(equation[XYZ]))
        quaternion = normalize_vector(quaternion)
        output_planes.append(np.concatenate([point, quaternion]))

    return np.array(output_planes)
