from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from workstreams_load_characteristic_model_v2.config import PEAK_VALLEY_GATE_POLICY
from workstreams_load_characteristic_model_v2.parameter_pipeline import (
    build_future_parameter_table,
)
from workstreams_load_characteristic_model_v2.quality import SHAPE_COLS
from workstreams_load_characteristic_model_v2.shape_parameters import (
    constrain_window_migration,
    extrapolated_increment,
)
from workstreams_load_characteristic_model_v2.shape_transform import (
    WINDOW_BOUNDS,
    apply_proportional_window_shares,
    diagnose_shape,
    peak_valley_rate,
    window_shares,
)


def _base() -> np.ndarray:
    values = np.arange(1, 25, dtype=float)
    return values / values.sum()


def test_proportional_transform_hits_shares_and_preserves_internal_ratios() -> None:
    base = _base()
    target = np.array([0.25, 0.18, 0.30, 0.27])
    future = apply_proportional_window_shares(base, target)
    assert future.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.all(future >= 0)
    assert np.allclose(window_shares(future), target, atol=1e-12)
    for start, end in WINDOW_BOUNDS:
        assert np.allclose(
            future[start:end] / future[start:end].sum(),
            base[start:end] / base[start:end].sum(),
            atol=1e-12,
        )


def test_a_applied_is_non_decreasing_and_plateaus_at_a_max() -> None:
    base = np.array([0.25, 0.25, 0.25, 0.25])
    direction = np.array([-1.0, 0.5, 0.25, 0.25])
    applied: list[float] = []
    for year in (2025, 2030, 2040, 2060):
        requested = extrapolated_increment(0.05, year, 8)
        a_max, value, _ = constrain_window_migration(base, direction, requested)
        applied.append(value)
        assert a_max == pytest.approx(0.24)
    assert applied == sorted(applied)
    assert applied[-1] <= 0.24 + 1e-12


def test_diagnostic_sets_do_not_change_the_curve(monkeypatch: pytest.MonkeyPatch) -> None:
    base = _base()
    target = np.array([0.20, 0.20, 0.35, 0.25])
    expected = apply_proportional_window_shares(base, target)
    monkeypatch.setattr(
        "workstreams_load_characteristic_model_v2.shape_transform.diagnose_shape",
        lambda _: {
            "evening_peak_hour": 23,
            "raw_85pct_hours": (11, 12, 13),
            "effective_control_hours": (11, 12, 13),
        },
    )
    actual = apply_proportional_window_shares(base, target)
    assert np.allclose(actual, expected)


def test_diagnose_shape_uses_circular_three_hour_control_runs() -> None:
    shape = np.ones(24)
    shape[[23, 0, 1]] = 2.0
    diag = diagnose_shape(shape)
    assert diag["raw_85pct_hours"] == (0, 1, 23)
    assert diag["effective_control_hours"] == (0, 1, 23)


def test_future_table_uses_2023_shape_and_emits_applied_curve() -> None:
    base = _base()
    baseline = pd.DataFrame(
        [
            {
                "department": "第二产业",
                "industry": "测试行业",
                "year": 2023,
                "season": "夏季",
                "final_day_type": "工作日",
                "day_group": "工作日",
                "is_future_baseline": True,
                **dict(zip(SHAPE_COLS, base, strict=True)),
            }
        ]
    )
    shares = window_shares(base)
    direction = np.array([-1.0, 0.5, 0.25, 0.25])
    history = pd.DataFrame(
        [
            {
                "industry": "测试行业",
                "season": "夏季",
                "final_day_type": "工作日",
                "trend_eligible": True,
                "trend_status": "paired_2022_2023_industry_medoids",
                "shares_2023": shares,
                "beta_a": 0.02,
                "direction_v": direction,
                "peak_valley_rate_2023": peak_valley_rate(base),
                "peak_valley_delta_signed": 0.10,
                "peak_valley_delta_abs": 0.10,
                "peak_valley_delta_band": "within_gate_band",
                "flat_curve": False,
                "flat_borderline": False,
                "peak_valley_evolution_enabled": True,
                "peak_valley_gate_reason": "enabled",
                "peak_valley_gate_policy": PEAK_VALLEY_GATE_POLICY,
            }
        ]
    )
    future = build_future_parameter_table(baseline, history, [2030])
    assert len(future) == 3
    formal = future.loc[future.H.eq(8) & future.phi.eq(0.8)].iloc[0]
    curve = formal[SHAPE_COLS].to_numpy(float)
    assert formal.base_source_year == 2023
    assert np.allclose([
        formal.four_window_share_intermediate_night,
        formal.four_window_share_intermediate_morning,
        formal.four_window_share_intermediate_midday,
        formal.four_window_share_intermediate_evening,
    ], [
        formal.applied_share_night,
        formal.applied_share_morning,
        formal.applied_share_midday,
        formal.applied_share_evening,
    ])
    assert peak_valley_rate(curve) == pytest.approx(
        formal.peak_valley_target_applied,
        abs=1e-10,
    )
    assert np.allclose(
        window_shares(curve),
        [
            formal.four_window_share_final_night,
            formal.four_window_share_final_morning,
            formal.four_window_share_final_midday,
            formal.four_window_share_final_evening,
        ],
        atol=1e-12,
    )
    assert (
        formal.shape_extrapolation_method
        == "four_window_then_local_smoothing_then_gated_peak_valley_mean_scaling"
    )
    assert (
        formal.four_window_final_share_policy
        == "exact_if_peak_valley_skipped_diagnostic_if_applied"
    )
    assert not any(
        name in future.columns
        for name in ("anchor_repair_applied", "repair_l2_distance", "rho", "gamma", "delta")
    )


def test_insufficient_history_freezes_both_stages_at_2023_baseline() -> None:
    base = _base()
    baseline = pd.DataFrame(
        [
            {
                "department": "第二产业",
                "industry": "历史不足行业",
                "year": 2023,
                "season": "夏季",
                "final_day_type": "工作日",
                "day_group": "工作日",
                "is_future_baseline": True,
                **dict(zip(SHAPE_COLS, base, strict=True)),
            }
        ]
    )
    history = pd.DataFrame(
        [
            {
                "industry": "历史不足行业",
                "season": "夏季",
                "final_day_type": "工作日",
                "trend_eligible": False,
                "trend_status": "insufficient_history_no_extrapolation",
                "shares_2023": window_shares(base),
                # Nonzero source values prove eligibility, not their magnitude,
                # controls the mandatory freeze of both stages.
                "beta_a": 0.02,
                "direction_v": np.array([-1.0, 0.5, 0.25, 0.25]),
                "peak_valley_rate_2023": peak_valley_rate(base),
                "peak_valley_delta_signed": 0.20,
                "peak_valley_delta_abs": 0.0,
                "peak_valley_delta_band": "insufficient_history",
                "flat_curve": False,
                "flat_borderline": False,
                "peak_valley_evolution_enabled": False,
                "peak_valley_gate_reason": "disabled_insufficient_history",
                "peak_valley_gate_policy": PEAK_VALLEY_GATE_POLICY,
            }
        ]
    )

    future = build_future_parameter_table(baseline, history, [2060])

    assert len(future) == 3
    assert np.allclose(future["a_applied"], 0.0, atol=0.0, rtol=0.0)
    assert np.allclose(
        future["peak_valley_target_requested"],
        peak_valley_rate(base),
        atol=1e-12,
        rtol=0.0,
    )
    assert set(future["trend_status"]) == {"insufficient_history_no_extrapolation"}
    assert np.allclose(
        future[SHAPE_COLS].to_numpy(float),
        np.tile(base, (len(future), 1)),
        atol=1e-12,
        rtol=0.0,
    )
