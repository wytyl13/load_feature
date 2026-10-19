import numpy as np
import pytest

from workstreams_load_characteristic_model_v2.shape_transform import (
    WINDOW_BOUNDS, smooth_window_boundaries, window_shares,
    apply_sequential_shape_transform, apply_peak_valley_target,
)
from workstreams_load_characteristic_model_v2.shape_parameters import extrapolated_increment
from workstreams_load_characteristic_model_v2.metrics import WINDOWS


def test_new_windows_cover_each_hour_once_and_match_statistics():
    assert WINDOW_BOUNDS == ((0, 6), (6, 12), (12, 17), (17, 24))
    assert [h for hours in WINDOWS.values() for h in hours] == list(range(24))
    assert [len(hours) for hours in WINDOWS.values()] == [6, 6, 5, 7]


def test_smoothing_reduces_artificial_step_preserves_energy_and_distant_ratios():
    shape = np.r_[np.ones(6), np.ones(6)*3, np.ones(5)*2, np.ones(7)]
    shape /= shape.sum()
    smooth = smooth_window_boundaries(shape)
    assert abs(smooth[6]-smooth[5]) < abs(shape[6]-shape[5])
    np.testing.assert_allclose(window_shares(smooth), window_shares(shape), atol=1e-12)
    assert smooth.sum() == pytest.approx(1)
    assert smooth.min() >= 0
    for a, b in ((1, 3), (8, 9), (13, 14), (19, 22)):
        assert smooth[a]/smooth[b] == pytest.approx(shape[a]/shape[b])


def test_final_calibration_uses_smoothed_curve_and_preserves_anchor():
    base = np.arange(1, 25, dtype=float)
    base /= base.sum()
    result = apply_sequential_shape_transform(base, np.array([.2,.3,.2,.3]), .6)
    expected = apply_peak_valley_target(smooth_window_boundaries(result.proportional_shape), .6)
    np.testing.assert_allclose(result.final_shape, expected.final_shape)
    anchor = apply_sequential_shape_transform(base, window_shares(base), 23/24)
    np.testing.assert_allclose(anchor.final_shape, base, atol=1e-12)


def test_signed_window_trends_equal_requested_exponential_formula():
    beta = np.array([-.01, .005, .002, .003])
    actual = np.array([extrapolated_increment(b,2030,8) for b in beta])
    np.testing.assert_allclose(actual, beta*8*(1-np.exp(-7/8)))
    assert actual.sum() == pytest.approx(0, abs=1e-12)
