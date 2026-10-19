from __future__ import annotations

import numpy as np
import pytest

from workstreams_load_characteristic_model_v2.shape_transform import (
    apply_peak_valley_target,
    apply_sequential_shape_transform,
    peak_valley_rate,
    window_shares,
)


def _shape() -> np.ndarray:
    values = np.asarray(
        [
            0.80,
            0.75,
            0.72,
            0.70,
            0.74,
            0.82,
            0.95,
            1.05,
            1.12,
            1.08,
            1.00,
            0.96,
            0.93,
            0.91,
            0.94,
            1.02,
            1.14,
            1.25,
            1.35,
            1.42,
            1.38,
            1.28,
            1.10,
            0.92,
        ],
        dtype=float,
    )
    return values / values.sum()


def _assert_valid_target(shape: np.ndarray, target: float) -> None:
    result = apply_peak_valley_target(shape, target)
    assert np.isfinite(result.final_shape).all()
    assert result.final_shape.min() >= 0.0
    assert result.final_shape.sum() == pytest.approx(1.0, abs=1e-12)
    assert peak_valley_rate(result.final_shape) == pytest.approx(target, abs=1e-10)


def test_current_target_is_identity() -> None:
    shape = _shape()
    current = peak_valley_rate(shape)
    result = apply_peak_valley_target(shape, current)
    assert result.lambda_value == 1.0
    assert np.array_equal(result.final_shape, shape)
    assert result.target_applied == pytest.approx(current)


def test_lower_target_flattens_and_higher_target_steepens() -> None:
    shape = _shape()
    current = peak_valley_rate(shape)
    flatter = apply_peak_valley_target(shape, current / 2.0)
    steeper = apply_peak_valley_target(shape, (1.0 + current) / 2.0)
    assert 0.0 <= flatter.lambda_value < 1.0
    assert steeper.lambda_value > 1.0
    _assert_valid_target(shape, current / 2.0)
    _assert_valid_target(shape, (1.0 + current) / 2.0)


def test_zero_target_is_flat_and_one_target_reaches_zero() -> None:
    shape = _shape()
    flat = apply_peak_valley_target(shape, 0.0)
    full = apply_peak_valley_target(shape, 1.0)
    assert np.allclose(flat.final_shape, np.full(24, 1.0 / 24.0), atol=1e-12)
    assert flat.lambda_value == 0.0
    assert full.final_shape.min() == pytest.approx(0.0, abs=1e-12)
    assert peak_valley_rate(full.final_shape) == pytest.approx(1.0, abs=1e-10)


@pytest.mark.parametrize(
    ("requested", "limited"),
    [(-0.2, 0.0), (1.2, 1.0)],
)
def test_requested_target_is_range_limited(requested: float, limited: float) -> None:
    result = apply_peak_valley_target(_shape(), requested)
    assert result.target_requested == requested
    assert result.target_range_limited == limited
    assert result.target_applied == pytest.approx(limited)
    assert result.target_clipped is True
    assert peak_valley_rate(result.final_shape) == pytest.approx(limited, abs=1e-10)


def test_flat_intermediate_curve_has_explicit_reachability_status() -> None:
    flat = np.full(24, 1.0 / 24.0)
    zero = apply_peak_valley_target(flat, 0.0)
    tiny_positive = apply_peak_valley_target(flat, 1e-12)
    unreachable = apply_peak_valley_target(flat, 0.3)
    assert np.array_equal(zero.final_shape, flat)
    assert zero.target_applied == 0.0
    assert zero.adjustment_status == "flat_intermediate_zero_target"
    assert tiny_positive.target_applied == 0.0
    assert tiny_positive.adjustment_status == "flat_intermediate_target_unreachable"
    assert np.array_equal(unreachable.final_shape, flat)
    assert unreachable.target_applied == 0.0
    assert unreachable.adjustment_status == "flat_intermediate_target_unreachable"


def test_near_flat_but_nonconstant_curve_can_reach_positive_target() -> None:
    near_flat = np.ones(24, dtype=float)
    near_flat[0] += 1e-11
    near_flat[1] -= 1e-11

    result = apply_peak_valley_target(near_flat, 0.3)

    assert result.adjustment_status == "mean_centered_scaling_applied"
    assert peak_valley_rate(result.final_shape) == pytest.approx(0.3, abs=1e-10)


@pytest.mark.parametrize(
    "bad_shape",
    [
        np.ones(23),
        np.r_[np.ones(23), np.nan],
        np.r_[np.ones(23), -1.0],
    ],
)
def test_invalid_intermediate_shapes_are_rejected(bad_shape: np.ndarray) -> None:
    with pytest.raises(ValueError):
        apply_peak_valley_target(bad_shape, 0.2)


@pytest.mark.parametrize("bad_target", [np.nan, np.inf, -np.inf])
def test_non_finite_targets_are_rejected(bad_target: float) -> None:
    with pytest.raises(ValueError):
        apply_peak_valley_target(_shape(), bad_target)


def test_random_nonflat_shapes_hit_all_targets() -> None:
    rng = np.random.default_rng(20260903)
    for _ in range(50):
        shape = rng.lognormal(mean=0.0, sigma=0.8, size=24)
        shape /= shape.sum()
        for target in (0.0, 0.1, 0.35, 0.7, 0.95, 1.0):
            _assert_valid_target(shape, target)


def test_sequential_transform_preserves_intermediate_target_only() -> None:
    base = _shape()
    target_shares = np.asarray([0.22, 0.16, 0.30, 0.32])
    result = apply_sequential_shape_transform(base, target_shares, 0.65)
    assert np.allclose(result.intermediate_window_shares, target_shares, atol=1e-12)
    assert np.allclose(window_shares(result.intermediate_shape), target_shares, atol=1e-12)
    assert peak_valley_rate(result.final_shape) == pytest.approx(0.65, abs=1e-10)
    assert np.allclose(
        result.final_window_share_deviation,
        result.final_window_shares - result.intermediate_window_shares,
        atol=1e-12,
    )
    assert not np.allclose(result.final_window_shares, target_shares, atol=1e-8)


def test_disabled_peak_valley_gate_returns_four_window_shape_unchanged() -> None:
    base = _shape()
    target_shares = np.asarray([0.22, 0.16, 0.30, 0.32])

    result = apply_sequential_shape_transform(
        base,
        target_shares,
        0.65,
        peak_valley_evolution_enabled=False,
        peak_valley_gate_reason="disabled_flat_curve",
    )

    assert np.array_equal(result.final_shape, result.intermediate_shape)
    assert np.allclose(result.final_window_shares, target_shares, atol=1e-12)
    assert np.allclose(result.final_window_share_deviation, 0.0, atol=1e-12)
    assert result.peak_valley.lambda_value == 1.0
    assert result.peak_valley.target_requested == pytest.approx(0.65)
    assert result.peak_valley.target_range_limited == pytest.approx(0.65)
    assert result.peak_valley.target_applied == pytest.approx(
        result.peak_valley.rate_before
    )
    assert result.peak_valley.adjustment_status == "skipped_flat_curve"
    assert peak_valley_rate(result.final_shape) == pytest.approx(
        result.peak_valley.rate_before
    )


def test_disabled_gate_requires_disabled_reason_code() -> None:
    with pytest.raises(ValueError, match="disabled_"):
        apply_sequential_shape_transform(
            _shape(),
            np.asarray([0.22, 0.16, 0.30, 0.32]),
            0.65,
            peak_valley_evolution_enabled=False,
            peak_valley_gate_reason="enabled",
        )
