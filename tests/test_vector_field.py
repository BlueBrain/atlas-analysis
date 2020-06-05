import numpy.testing as npt
import numpy as np
import itertools

import nose.tools as nt

from atlas_analysis.exceptions import AtlasAnalysisError
from atlas_analysis import vector_field as tested


def test_interpolate():
    expected = np.zeros((5, 5, 5, 3), dtype=float)
    # Compute a radial field centered at (3, 3, 3).
    for index in itertools.product(range(5), range(5), range(5)):
        expected[index, :] = np.array(index) - 3
    actual = expected.copy()
    unknown_values_mask = np.zeros_like(expected, dtype=bool)
    unknown_values_mask[1, 1, 1] = True
    unknown_values_mask[3, 3, 3] = True
    unknown_values_mask[1, 2, 3] = True
    unknown_values_mask[3, 2, 3] = True
    actual[unknown_values_mask] = np.nan
    # Known values are specified.
    tested.interpolate(actual, unknown_values_mask, ~unknown_values_mask)
    npt.assert_array_almost_equal(actual, expected)
    # Unknown and known values overlap.
    actual = expected.copy()
    actual[unknown_values_mask] = np.nan
    tested.interpolate(actual, unknown_values_mask)
    known_values_mask = ~unknown_values_mask
    known_values_mask[1, 1, 1] = True
    tested.interpolate(actual, unknown_values_mask, known_values_mask)
    npt.assert_array_almost_equal(actual, expected)
    # Known values are not specified.
    actual = expected.copy()
    actual[unknown_values_mask] = np.nan
    tested.interpolate(actual, unknown_values_mask)
    npt.assert_array_almost_equal(actual, expected)
    # Using a different interpolator.
    actual = expected.copy()
    unknown_values_mask[0, 0, 0] = True
    actual[unknown_values_mask] = np.nan
    tested.interpolate(actual, unknown_values_mask, interpolator='nearest-neighbour')
    assert np.all(
        ~np.isnan(actual)
    )  # No more NaNs, i.e., all values have been interpolated.
    npt.assert_allclose(
        actual, expected, atol=1.0
    )  # Accuracy is low for this use case.


@nt.raises(AtlasAnalysisError)
def test_interpolate_raises():
    tested.interpolate(np.array([]), np.array([]), interpolator='cubic')
