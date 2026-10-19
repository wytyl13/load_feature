from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from workstreams_load_characteristic_model_v2.quality import SHAPE_COLS
from workstreams_load_characteristic_model_v2.shape_parameters import (
    classify_peak_valley_delta,
    identify_shape_parameters,
)
from workstreams_load_characteristic_model_v2.shape_transform import (
    apply_proportional_window_shares,
    peak_valley_rate,
    window_shares,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "insufficient_history"),
        (0.064999, "below_lower_threshold"),
        (0.065, "within_gate_band"),
        (0.10, "within_gate_band"),
        (0.25, "within_gate_band"),
        (0.250001, "above_upper_threshold"),
    ],
)
def test_peak_valley_delta_band(value: float | None, expected: str) -> None:
    assert classify_peak_valley_delta(value) == expected


def _shape_with_shares(shares: np.ndarray) -> np.ndarray:
    base = np.arange(1, 25, dtype=float)
    base /= base.sum()
    return apply_proportional_window_shares(base, shares)


def _baselines(shape_2022: np.ndarray, shape_2023: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "department": "第二产业",
                "industry": "测试行业",
                "year": year,
                "season": "夏季",
                "final_day_type": "工作日",
                "day_group": "工作日",
                "is_future_baseline": year == 2023,
                **dict(zip(SHAPE_COLS, shape, strict=True)),
            }
            for year, shape in ((2022, shape_2022), (2023, shape_2023))
        ]
    )


def _quality(*, trend_eligible: bool = True) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "industry": "测试行业",
                "year": year,
                "season": "夏季",
                "final_day_type": "工作日",
                "trend_eligible": trend_eligible,
            }
            for year in (2022, 2023)
        ]
    )


def test_window_migration_uses_2022_and_2023_medoid_shares() -> None:
    shares_2022 = np.array([0.30, 0.20, 0.25, 0.25])
    shares_2023 = np.array([0.28, 0.21, 0.27, 0.24])
    history = identify_shape_parameters(
        _baselines(_shape_with_shares(shares_2022), _shape_with_shares(shares_2023)),
        _quality(),
    )
    row = history.iloc[0]
    assert np.allclose(row.b_observed, [-0.02, 0.01, 0.02, -0.01])
    assert row.beta_a == pytest.approx(0.03)
    assert np.allclose(row.direction_v, [-2 / 3, 1 / 3, 2 / 3, -1 / 3])
    assert np.allclose(row.shares_2022, window_shares(_shape_with_shares(shares_2022)))


def test_flat_curve_closes_peak_valley_gate_but_keeps_history_auditable() -> None:
    flat = np.ones(24, dtype=float) / 24
    nonflat = flat.copy()
    nonflat[0] *= 0.9
    nonflat[1] *= 1.1
    nonflat /= nonflat.sum()
    history = identify_shape_parameters(_baselines(nonflat, flat), _quality())
    row = history.iloc[0]
    assert bool(row.flat_curve)
    assert row.peak_valley_delta_band == "within_gate_band"
    assert not bool(row.peak_valley_evolution_enabled)
    assert row.peak_valley_gate_reason == "disabled_flat_curve"


def _shape_with_peak_valley_rate(rate: float) -> np.ndarray:
    values = np.linspace(1.0 - rate, 1.0, 24, dtype=float)
    return values / values.sum()


def _two_level_shape(low: float) -> np.ndarray:
    values = np.asarray([1.0] * 12 + [low] * 12, dtype=float)
    return values / values.sum()


@pytest.mark.parametrize(
    ("low_2022", "low_2023", "expected_delta"),
    [
        (0.60, 0.535, 0.065),
        (0.96, 0.71, 0.25),
    ],
)
def test_curve_derived_closed_delta_boundaries_are_snapped_and_enabled(
    low_2022: float,
    low_2023: float,
    expected_delta: float,
) -> None:
    history = identify_shape_parameters(
        _baselines(_two_level_shape(low_2022), _two_level_shape(low_2023)),
        _quality(),
    )
    row = history.iloc[0]

    assert not bool(row.flat_curve)
    assert row.peak_valley_delta_abs == expected_delta
    assert abs(row.peak_valley_delta_signed) == expected_delta
    assert row.peak_valley_delta_band == "within_gate_band"
    assert bool(row.peak_valley_evolution_enabled)
    assert row.peak_valley_gate_reason == "enabled"


@pytest.mark.parametrize(
    ("delta_abs", "expected_band", "enabled", "reason"),
    [
        (0.01, "below_lower_threshold", False, "disabled_delta_below_lower_bound"),
        (0.10, "within_gate_band", True, "enabled"),
        (0.30, "above_upper_threshold", False, "disabled_delta_above_upper_bound"),
    ],
)
def test_nonflat_history_applies_peak_valley_delta_gate(
    delta_abs: float,
    expected_band: str,
    enabled: bool,
    reason: str,
) -> None:
    shape_2023 = _shape_with_peak_valley_rate(0.40)
    shape_2022 = _shape_with_peak_valley_rate(0.40 - delta_abs)

    history = identify_shape_parameters(
        _baselines(shape_2022, shape_2023),
        _quality(),
    )
    row = history.iloc[0]

    assert not bool(row.flat_curve)
    assert row.peak_valley_delta_abs == pytest.approx(delta_abs)
    assert row.peak_valley_delta_band == expected_band
    assert bool(row.peak_valley_evolution_enabled) is enabled
    assert row.peak_valley_gate_reason == reason


def test_shape_history_records_signed_peak_valley_change() -> None:
    shape_2022 = _shape_with_shares(np.array([0.30, 0.20, 0.25, 0.25]))
    shape_2023 = _shape_with_shares(np.array([0.22, 0.18, 0.30, 0.30]))
    history = identify_shape_parameters(
        _baselines(shape_2022, shape_2023),
        _quality(),
    )
    row = history.iloc[0]
    expected_2022 = peak_valley_rate(shape_2022)
    expected_2023 = peak_valley_rate(shape_2023)
    assert row.peak_valley_rate_2022 == pytest.approx(expected_2022)
    assert row.peak_valley_rate_2023 == pytest.approx(expected_2023)
    assert row.peak_valley_delta_signed == pytest.approx(expected_2023 - expected_2022)
    assert row.peak_valley_delta_abs == pytest.approx(abs(expected_2023 - expected_2022))


def test_insufficient_history_contract_zeroes_both_shape_increments() -> None:
    shape_2022 = _shape_with_shares(np.array([0.30, 0.20, 0.25, 0.25]))
    shape_2023 = _shape_with_shares(np.array([0.22, 0.18, 0.30, 0.30]))

    history = identify_shape_parameters(
        _baselines(shape_2022, shape_2023),
        _quality(trend_eligible=False),
    )

    row = history.iloc[0]
    assert row.trend_status == "insufficient_history_no_extrapolation"
    assert row.beta_a == 0.0
    assert np.array_equal(row.direction_v, np.zeros(4))
    assert row.peak_valley_delta_signed == 0.0
    assert row.peak_valley_delta_abs == 0.0
    assert row.peak_valley_delta_band == "insufficient_history"
    assert not bool(row.peak_valley_evolution_enabled)
    assert row.peak_valley_gate_reason == "disabled_insufficient_history"
