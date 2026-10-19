from __future__ import annotations

import numpy as np
import pytest

from workstreams_load_characteristic_model_v2.shape_parameters import (
    estimate_peak_valley_change,
    evaluate_peak_valley_evolution_gate,
    forecast_peak_valley_target,
    is_flat_curve,
)
from workstreams_load_characteristic_model_v2.shape_transform import (
    diagnose_shape,
    peak_valley_rate,
)


def _curve(low: float, high: float) -> np.ndarray:
    values = np.linspace(low, high, 24, dtype=float)
    return values / values.sum()


def test_estimate_uses_signed_2022_to_2023_medoid_change() -> None:
    curve_2022 = _curve(0.8, 1.2)
    curve_2023 = _curve(0.6, 1.4)
    estimate = estimate_peak_valley_change(curve_2022, curve_2023)
    expected_2022 = peak_valley_rate(curve_2022)
    expected_2023 = peak_valley_rate(curve_2023)
    assert estimate.rate_2022 == pytest.approx(expected_2022)
    assert estimate.rate_2023 == pytest.approx(expected_2023)
    assert estimate.delta_signed == pytest.approx(expected_2023 - expected_2022)
    assert estimate.delta_abs == pytest.approx(abs(expected_2023 - expected_2022))
    assert estimate.diagnostic_band == "within_gate_band"


@pytest.mark.parametrize(
    ("load_factor", "cv", "expected"),
    [
        (0.800001, 0.149999, True),
        (0.80, 0.149999, False),
        (0.800001, 0.15, False),
        (0.80, 0.15, False),
    ],
)
def test_flat_gate_uses_strict_thresholds(
    load_factor: float,
    cv: float,
    expected: bool,
) -> None:
    assert is_flat_curve(load_factor, cv) is expected


def test_flat_gate_snaps_curve_derived_load_factor_boundary() -> None:
    lower_level = (24 * 0.80 - 1.0) / 23
    curve = 10.0 * np.asarray([1.0] + [lower_level] * 23, dtype=float)
    diagnostic = diagnose_shape(curve)

    assert diagnostic["load_factor"] > 0.80  # binary tail above exact boundary
    assert diagnostic["cv"] < 0.15
    assert is_flat_curve(diagnostic["load_factor"], diagnostic["cv"]) is False


@pytest.mark.parametrize("scale", [1.0, 10.0])
def test_flat_gate_snaps_curve_derived_cv_boundary(scale: float) -> None:
    curve = scale * np.asarray([1.15] * 12 + [0.85] * 12, dtype=float)
    diagnostic = diagnose_shape(curve)

    assert diagnostic["load_factor"] > 0.80
    assert diagnostic["cv"] == pytest.approx(0.15, abs=1e-15)
    assert is_flat_curve(diagnostic["load_factor"], diagnostic["cv"]) is False


@pytest.mark.parametrize(
    ("trend_eligible", "flat_curve", "delta_abs", "enabled", "reason"),
    [
        (False, False, None, False, "disabled_insufficient_history"),
        (True, True, 0.10, False, "disabled_flat_curve"),
        (True, False, 0.064999, False, "disabled_delta_below_lower_bound"),
        (True, False, 0.065, True, "enabled"),
        (True, False, 0.10, True, "enabled"),
        (True, False, 0.25, True, "enabled"),
        (True, False, 0.250001, False, "disabled_delta_above_upper_bound"),
    ],
)
def test_peak_valley_gate_is_history_nonflat_and_closed_delta_band(
    trend_eligible: bool,
    flat_curve: bool,
    delta_abs: float | None,
    enabled: bool,
    reason: str,
) -> None:
    actual = evaluate_peak_valley_evolution_gate(
        trend_eligible=trend_eligible,
        flat_curve=flat_curve,
        delta_abs=delta_abs,
    )
    assert actual == (enabled, reason)


def test_anchor_and_logit_damping_match_independent_odds_calculation() -> None:
    anchor = forecast_peak_valley_target(rate_2023=.4, delta_signed=.2,
                                        year=2023, phi=.8, trend_eligible=True)
    future = forecast_peak_valley_target(rate_2023=.4, delta_signed=.2,
                                        year=2030, phi=.8, trend_eligible=True)
    cumulative = sum(.8 ** k for k in range(1, 8))
    odds = (.4/.6) * ((.4/.6)/(.2/.8)) ** cumulative
    assert anchor.cumulative_damping == 0.0
    assert anchor.requested == .4
    assert future.cumulative_damping == pytest.approx(3.1611392)
    assert future.requested == pytest.approx(odds/(1+odds))


def test_school_case_matches_full_precision_ppt_method() -> None:
    result = forecast_peak_valley_target(rate_2023=.47403813998552075,
        delta_signed=.08688711761212531, year=2030, phi=.8, trend_eligible=True)
    assert result.requested == pytest.approx(.7348577076736251, abs=1e-14)


@pytest.mark.parametrize("rate,delta", [(0., -.2), (1., .2), (.2, .2), (.8, -.2)])
def test_exact_endpoint_rates_are_finite_and_bounded(rate, delta) -> None:
    result = forecast_peak_valley_target(rate_2023=rate, delta_signed=delta,
        year=2060, phi=.8, trend_eligible=True)
    assert np.isfinite(result.requested) and 0 <= result.requested <= 1


@pytest.mark.parametrize("year,phi,delta", [(2023,.8,.1), (2030,0.,.1), (2030,.8,0.)])
def test_anchor_zero_damping_and_no_change_preserve_baseline(year,phi,delta):
    assert forecast_peak_valley_target(rate_2023=.4,delta_signed=delta,
        year=year,phi=phi,trend_eligible=True).requested == .4


@pytest.mark.parametrize("delta", [0.15, -0.15])
def test_target_moves_monotonically_in_signed_direction(delta: float) -> None:
    targets = [
        forecast_peak_valley_target(
            rate_2023=0.5,
            delta_signed=delta,
            year=year,
            phi=0.8,
            trend_eligible=True,
        )
        for year in (2023, 2025, 2030, 2040, 2060)
    ]
    etas = [item.cumulative_damping for item in targets]
    requested = [item.requested for item in targets]
    assert etas == sorted(etas)
    if delta > 0:
        assert requested == sorted(requested)
    else:
        assert requested == sorted(requested, reverse=True)


def test_theoretical_target_formula_remains_independent_of_application_gate() -> None:
    small = forecast_peak_valley_target(
        rate_2023=0.4,
        delta_signed=0.01,
        year=2030,
        phi=0.8,
        trend_eligible=True,
    )
    large = forecast_peak_valley_target(
        rate_2023=0.4,
        delta_signed=0.30,
        year=2030,
        phi=0.8,
        trend_eligible=True,
    )
    assert small.requested > 0.4
    assert large.requested > small.requested


def test_ineligible_history_freezes_target_at_2023_rate() -> None:
    target = forecast_peak_valley_target(
        rate_2023=0.4,
        delta_signed=0.3,
        year=2060,
        phi=0.8,
        trend_eligible=False,
    )
    assert target.requested == 0.4
    assert target.range_limited == 0.4
    assert target.trend_status == "insufficient_history_no_extrapolation"


@pytest.mark.parametrize("phi", [-.1, 1., 6, np.nan, np.inf])
def test_invalid_phi_is_rejected(phi: float) -> None:
    with pytest.raises(ValueError):
        forecast_peak_valley_target(rate_2023=.4, delta_signed=.1,
            year=2030, phi=phi, trend_eligible=True)


@pytest.mark.parametrize("rate,delta", [(.95,1.),(.05,-1.)])
def test_impossible_historical_rate_is_rejected(rate,delta):
    with pytest.raises(ValueError):
        forecast_peak_valley_target(rate_2023=rate,delta_signed=delta,
            year=2030,phi=.8,trend_eligible=True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rate_2023": np.nan, "delta_signed": 0.1, "year": 2030},
        {"rate_2023": 0.4, "delta_signed": np.inf, "year": 2030},
        {"rate_2023": 0.4, "delta_signed": 0.1, "year": 2022},
        {"rate_2023": 0.4, "delta_signed": 0.1, "year": 2030.5},
    ],
)
def test_invalid_forecast_inputs_are_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        forecast_peak_valley_target(
            **kwargs,
            phi=0.8,
            trend_eligible=True,
        )
