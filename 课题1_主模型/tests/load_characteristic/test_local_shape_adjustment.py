from __future__ import annotations

import numpy as np
from pytest import approx, raises

from workstreams_load_characteristic_model_v2.local_shape_adjustment import (
    CONTROL_RATIO,
    adjust_control_period,
    adjust_local_shape,
    assess_flatness,
    circular_runs,
    evening_peak_hour,
    identify_control_periods,
    peak_valley_rate,
    relocate_evening_peak,
)


def _normalized(values: np.ndarray | list[float]) -> np.ndarray:
    curve = np.asarray(values, dtype=float)
    return curve / curve.sum()


def _control_pair(*, evening: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """D: 0.60 -> 0.50 and effective duration: 3 -> 5 hours."""

    baseline = np.full(24, 1.4, dtype=float)
    target = np.full(24, 1.5, dtype=float)
    baseline[0] = 1.0
    target[0] = 1.0
    start = 17 if evening else 8
    baseline[start : start + 5] = [1.8, 2.2, 2.5, 2.2, 1.8]
    target[start : start + 5] = [1.71, 1.8, 2.0, 1.8, 1.71]
    return _normalized(baseline), _normalized(target)


def test_flatness_gate_uses_strict_boundaries() -> None:
    mean = 1.0 / 24.0

    lf_boundary = np.full(24, (1.0 - 1.25 * mean) / 23.0)
    lf_boundary[0] = 1.25 * mean
    lf_diagnostics = assess_flatness(lf_boundary)
    assert lf_diagnostics.load_factor == approx(0.8)
    assert lf_diagnostics.coefficient_of_variation < 0.15
    assert not lf_diagnostics.is_flat

    cv_boundary = np.r_[np.full(12, 1.15 * mean), np.full(12, 0.85 * mean)]
    cv_diagnostics = assess_flatness(cv_boundary)
    assert cv_diagnostics.load_factor > 0.8
    assert cv_diagnostics.coefficient_of_variation == approx(0.15)
    assert not cv_diagnostics.is_flat

    flat = np.r_[np.full(12, 1.14 * mean), np.full(12, 0.86 * mean)]
    assert assess_flatness(flat).is_flat


def test_flat_gate_skips_both_local_changes() -> None:
    baseline = np.ones(24, dtype=float) / 24.0
    target = np.arange(1.0, 25.0)
    target = _normalized(target)

    result = adjust_local_shape(
        baseline,
        target,
        evening_triggered=True,
        evening_target_hour=22,
    )

    assert result.diagnostics.status == "flat_gate_skipped"
    assert result.diagnostics.control.status == "flat_gate_skipped"
    assert result.diagnostics.evening.status == "flat_gate_skipped"
    np.testing.assert_array_equal(result.curve, baseline)


def test_circular_control_runs_keep_only_runs_of_at_least_three_hours() -> None:
    mask = np.zeros(24, dtype=bool)
    mask[[23, 0, 1, 5, 6, 10, 11, 12]] = True
    assert circular_runs(mask) == ((5, 6), (10, 11, 12), (23, 0, 1))

    curve = np.full(24, 0.4, dtype=float)
    curve[mask] = 1.0
    periods = identify_control_periods(curve)
    assert periods.raw_runs == ((5, 6), (10, 11, 12), (23, 0, 1))
    assert periods.short_runs == ((5, 6),)
    assert periods.valid_runs == ((10, 11, 12), (23, 0, 1))
    assert periods.hours == (0, 1, 10, 11, 12, 23)
    assert periods.duration == 6


def test_control_threshold_includes_exact_85_percent_values() -> None:
    curve = np.full(24, 0.2, dtype=float)
    curve[4:7] = [CONTROL_RATIO, 1.0, CONTROL_RATIO]
    periods = identify_control_periods(curve)
    assert periods.hours == (4, 5, 6)


def test_control_trigger_includes_exact_6_5_percent_boundary() -> None:
    baseline = np.full(24, 0.8, dtype=float)
    baseline[0] = 0.6
    baseline[9:12] = [0.86, 1.0, 0.86]
    baseline = _normalized(baseline)

    target = np.full(24, 0.8, dtype=float)
    target[0] = 0.535
    target[8:13] = [0.86, 0.9, 1.0, 0.9, 0.86]
    target = _normalized(target)

    result = adjust_control_period(baseline, target, apply_flat_gate=False)
    assert result.diagnostics.absolute_peak_valley_change == approx(0.065)
    assert result.diagnostics.absolute_duration_change == 2
    assert result.diagnostics.peak_valley_triggered
    assert result.diagnostics.duration_triggered
    assert result.diagnostics.triggered
    assert result.diagnostics.applied


def test_control_linear_stretch_commits_only_after_all_invariants_pass() -> None:
    baseline, target = _control_pair()
    result = adjust_control_period(baseline, target)

    assert result.diagnostics.status == "applied"
    assert result.diagnostics.applied
    assert result.curve.sum() == approx(1.0)
    assert peak_valley_rate(result.curve) == approx(peak_valley_rate(target))
    assert identify_control_periods(result.curve).hours == (8, 9, 10, 11, 12)
    assert identify_control_periods(np.round(result.curve, 5)).hours == (
        8,
        9,
        10,
        11,
        12,
    )
    assert all(passed for _, passed in result.diagnostics.checks)

    outside = np.asarray([*range(8), *range(13, 24)])
    ratios = result.curve[outside] / baseline[outside]
    np.testing.assert_allclose(ratios, np.full_like(ratios, ratios[0]))


def test_failed_control_transaction_returns_unchanged_input() -> None:
    baseline, _ = _control_pair()
    all_day_target = np.ones(24, dtype=float) / 24.0

    result = adjust_control_period(
        baseline, all_day_target, apply_flat_gate=False
    )

    assert result.diagnostics.triggered
    assert result.diagnostics.status == "inapplicable_raw_full_day"
    assert not result.diagnostics.applied
    np.testing.assert_array_equal(result.curve, baseline)


def test_adjacent_unique_evening_peak_is_swapped_exactly_and_conserves_energy() -> None:
    baseline = np.ones(24, dtype=float)
    baseline[17:24] = [1.5, 3.0, 2.5, 2.0, 1.8, 1.6, 1.4]
    baseline = _normalized(baseline)

    result = relocate_evening_peak(baseline, 19)

    assert result.diagnostics.status == "applied_adjacent_swap"
    assert result.diagnostics.method == "adjacent_swap"
    assert evening_peak_hour(result.curve) == 19
    assert result.curve[18] == baseline[19]
    assert result.curve[19] == baseline[18]
    np.testing.assert_array_equal(result.curve[:17], baseline[:17])
    assert result.curve.sum() == approx(baseline.sum())
    assert result.curve[17:24].sum() == approx(baseline[17:24].sum())
    assert all(passed for _, passed in result.diagnostics.checks)


def test_plateau_uses_peak_packet_instead_of_adjacent_swap() -> None:
    baseline = np.full(24, 0.04, dtype=float)
    baseline[16:24] = [0.045, 0.050, 0.060, 0.060, 0.052, 0.048, 0.046, 0.045]
    baseline = _normalized(baseline)

    result = relocate_evening_peak(baseline, 19)

    assert result.diagnostics.applied
    assert result.diagnostics.method == "peak_packet"
    assert result.diagnostics.plateau_hours == (18, 19)
    assert result.diagnostics.status.startswith("applied_peak_packet_")
    assert evening_peak_hour(result.curve) == 19
    np.testing.assert_array_equal(result.curve[:17], baseline[:17])
    assert result.curve.sum() == approx(1.0)


def test_peak_packet_can_move_to_nonadjacent_absolute_hour() -> None:
    baseline = np.full(24, 0.04, dtype=float)
    baseline[14:24] = [
        0.046,
        0.047,
        0.045,
        0.060,
        0.054,
        0.050,
        0.048,
        0.047,
        0.046,
        0.045,
    ]
    baseline = _normalized(baseline)

    result = relocate_evening_peak(baseline, 22)

    assert result.diagnostics.status == "applied_peak_packet_partial_boundary"
    assert result.diagnostics.source_hour == 17
    assert result.diagnostics.target_hour == 22
    assert result.diagnostics.shift_hours == 5
    assert result.diagnostics.left_anchor == 16
    assert result.diagnostics.right_anchor == 23
    assert evening_peak_hour(result.curve) == 22
    assert result.curve.sum() == approx(baseline.sum())
    assert result.curve[17:24].sum() == approx(baseline[17:24].sum())
    np.testing.assert_array_equal(result.curve[:17], baseline[:17])
    assert all(passed for _, passed in result.diagnostics.checks)


def test_joint_control_constraint_blocks_adjacent_move_outside_target_set() -> None:
    baseline = np.ones(24, dtype=float)
    baseline[17:24] = [1.5, 3.0, 2.5, 2.0, 1.8, 1.6, 1.4]
    baseline = _normalized(baseline)

    result = relocate_evening_peak(
        baseline,
        19,
        enforce_control=True,
        control_target_hours=(18, 20, 21),
    )

    assert result.diagnostics.status == "joint_control_constraint_failed"
    assert result.diagnostics.method == "adjacent_swap"
    assert not result.diagnostics.applied
    np.testing.assert_array_equal(result.curve, baseline)


def test_combined_transaction_rechecks_control_invariants_after_evening_swap() -> None:
    baseline, target = _control_pair(evening=True)

    result = adjust_local_shape(
        baseline,
        target,
        evening_triggered=True,
        evening_target_hour=20,
    )

    assert result.diagnostics.status == "control_and_evening_applied"
    assert result.diagnostics.control.applied
    assert result.diagnostics.evening.status == "applied_adjacent_swap"
    assert result.diagnostics.evening.joint_control_constraint
    assert all(passed for _, passed in result.diagnostics.evening.checks)
    assert peak_valley_rate(result.curve) == approx(peak_valley_rate(target))
    assert identify_control_periods(result.curve).hours == (17, 18, 19, 20, 21)
    assert identify_control_periods(np.round(result.curve, 5)).hours == (
        17,
        18,
        19,
        20,
        21,
    )
    assert evening_peak_hour(result.curve) == 20


def test_2023_curve_is_a_target_not_a_whole_curve_replacement() -> None:
    baseline, target = _control_pair()
    result = adjust_local_shape(baseline, target)

    assert result.diagnostics.control.applied
    assert not np.allclose(result.curve, target)
    assert not np.allclose(result.curve, baseline)
    assert result.curve.sum() == approx(1.0)


def test_malformed_curve_raises_instead_of_silently_falling_back() -> None:
    with raises(ValueError, match="exactly 24"):
        adjust_local_shape(np.ones(23) / 23.0)
    malformed = np.ones(24) / 24.0
    malformed[3] = np.nan
    with raises(ValueError, match="finite"):
        relocate_evening_peak(malformed, 20)
    with raises(ValueError, match="sum of 1"):
        adjust_control_period(np.ones(24), np.ones(24))
