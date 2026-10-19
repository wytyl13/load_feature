from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from workstreams_load_characteristic_model_v2.annual_assembly import (
    _validated_correction_shapes,
    _validated_future_parameters,
    assemble_province_year,
)
from workstreams_load_characteristic_model_v2.quality import SHAPE_COLS
from workstreams_load_characteristic_model_v2.shape_transform import (
    apply_sequential_shape_transform,
    peak_valley_rate,
    window_shares,
)


SEASONS = ("春季", "夏季", "秋季", "冬季")
DAY_TYPES = ("工作日", "普通周末", "法定节假日")


def _inputs(year: int = 2030) -> tuple[pd.DataFrame, ...]:
    industries = [f"行业{index + 1:02d}" for index in range(24)]
    total = 1000.0
    industry_energy = 25.0
    energy = pd.DataFrame(
        [
            {
                "method": "M1",
                "planning_scenario": "基准",
                "year": year,
                "annual_energy_quantile": "P50",
                "annual_energy_basis": "P50_natural_total",
                "natural_total_energy_MWh": total,
                "component": industry,
                "industry": industry,
                "component_type": "传统细分行业",
                "electricity_share": industry_energy / total,
                "annual_energy_MWh": industry_energy,
            }
            for industry in industries
        ]
        + [
            {
                "method": "M1",
                "planning_scenario": "基准",
                "year": year,
                "annual_energy_quantile": "P50",
                "annual_energy_basis": "P50_natural_total",
                "natural_total_energy_MWh": total,
                "component": "修正组成",
                "industry": "修正组成",
                "component_type": "修正组成",
                "electricity_share": 0.4,
                "annual_energy_MWh": 400.0,
            }
        ]
    )

    workday = np.arange(1.0, 25.0)
    workday /= workday.sum()
    nonwork = workday[::-1].copy()
    baseline_rows: list[dict[str, object]] = []
    future_rows: list[dict[str, object]] = []
    for industry in industries:
        for season in SEASONS:
            for day_type in DAY_TYPES:
                base = workday if day_type == "工作日" else nonwork
                target_shares = np.asarray([0.18, 0.17, 0.30, 0.35])
                transformed = apply_sequential_shape_transform(
                    base,
                    target_shares,
                    0.65,
                )
                future = transformed.final_shape
                baseline_rows.append(
                    {
                        "industry": industry,
                        "year": 2023,
                        "season": season,
                        "final_day_type": day_type,
                        "trend_eligible": True,
                        **dict(zip(SHAPE_COLS, base, strict=True)),
                    }
                )
                future_rows.append(
                    {
                        "industry": industry,
                        "season": season,
                        "final_day_type": day_type,
                        "year": year,
                        "H": 8,
                        "phi": 0.8,
                        "peak_valley_phi": 0.8,
                        "formal_H8": True,
                        "formal_phi08": True,
                        "base_source_year": 2023,
                        "trend_eligible": True,
                        "flat_curve": False,
                        "flat_borderline": False,
                        "peak_valley_delta_abs": 0.10,
                        "peak_valley_delta_band": "within_gate_band",
                        "peak_valley_evolution_enabled": True,
                        "peak_valley_gate_reason": "enabled",
                        "peak_valley_gate_policy": (
                            "trend_eligible_and_nonflat_and_abs_delta_within_closed_interval_0.065_0.25"
                        ),
                        "applied_share_night": target_shares[0],
                        "applied_share_morning": target_shares[1],
                        "applied_share_midday": target_shares[2],
                        "applied_share_evening": target_shares[3],
                        "peak_valley_target_requested": 0.65,
                        "peak_valley_target_range_limited": 0.65,
                        "peak_valley_target_applied": 0.65,
                        "peak_valley_target_clipped": False,
                        "peak_valley_adjustment_status": (
                            transformed.peak_valley.adjustment_status
                        ),
                        "peak_valley_rate_after_four_window": (
                            transformed.peak_valley.rate_before
                        ),
                        "peak_valley_lambda": transformed.peak_valley.lambda_value,
                        "peak_valley_rate_final": peak_valley_rate(future),
                        **{
                            f"four_window_share_intermediate_{name}": value
                            for name, value in zip(
                                ("night", "morning", "midday", "evening"),
                                transformed.intermediate_window_shares,
                                strict=True,
                            )
                        },
                        **{
                            f"four_window_share_final_{name}": value
                            for name, value in zip(
                                ("night", "morning", "midday", "evening"),
                                transformed.final_window_shares,
                                strict=True,
                            )
                        },
                        **{
                            f"four_window_share_deviation_{name}": value
                            for name, value in zip(
                                ("night", "morning", "midday", "evening"),
                                transformed.final_window_share_deviation,
                                strict=True,
                            )
                        },
                        "shape_extrapolation_method": (
                            "four_window_then_local_smoothing_then_gated_peak_valley_mean_scaling"
                        ),
                        "four_window_final_share_policy": (
                            "exact_if_peak_valley_skipped_diagnostic_if_applied"
                        ),
                        "peak_valley_extrapolation_enabled": True,
                        "late_peak_control_policy": "diagnostic_only",
                        "peak_damping_enabled": False,
                        "anchor_repair_enabled": False,
                        "effective_control_set_changed_from_2023": True,
                        **dict(zip(SHAPE_COLS, future, strict=True)),
                    }
                )
    factors = pd.DataFrame(
        [
            {
                "season": season,
                "final_day_type": day_type,
                "factor": factor,
                "source_year": 2023,
                "factor_scope": "province_shared_all_25_components",
            }
            for season in SEASONS
            for day_type, factor in zip(DAY_TYPES, (1.0, 0.8, 0.7), strict=True)
        ]
    )
    envelope = pd.DataFrame({"month": range(1, 13), "factor_p50": 1.0})
    correction = pd.DataFrame(
        [
            {
                "season": season,
                "day_group": {"工作日": "工作日", "普通周末": "周末", "法定节假日": "节假日"}[day_type],
                "hour": hour,
                "correction_normalized_shape": value,
            }
            for season in SEASONS
            for day_type in DAY_TYPES
            for hour, value in enumerate(nonwork)
        ]
    )
    return (
        energy,
        pd.DataFrame(baseline_rows),
        factors,
        pd.DataFrame(future_rows),
        envelope,
        correction,
        workday,
    )


def test_annual_curve_and_all_components_close_to_p50_natural_total() -> None:
    energy, baseline, factors, future, envelope, correction, _ = _inputs()

    hourly, checks = assemble_province_year(
        2030,
        "基准",
        energy,
        baseline,
        factors,
        future,
        envelope,
        correction,
        method="M1",
        include_daily_detail=True,
    )

    assert hourly.load_MW.sum() == pytest.approx(1000.0, rel=1e-10, abs=1e-6)
    component_annual = checks.query("closure_scope == 'component_annual'")
    assert component_annual.actual_energy_MWh.sum() == pytest.approx(
        1000.0, rel=1e-10, abs=1e-6
    )
    assert checks.passed.all()
    assert len(hourly) == 8760
    assert hourly.timestamp.is_unique and hourly.timestamp.is_monotonic_increasing
    assert set(hourly.day_type_factor_scope) == {
        "province_shared_all_25_components"
    }
    assert set(hourly.shape_extrapolation_method) == {
        "four_window_then_local_smoothing_then_gated_peak_valley_mean_scaling"
    }
    assert hourly.peak_valley_extrapolation_enabled.all()
    assert set(hourly.four_window_final_share_policy) == {
        "exact_if_peak_valley_skipped_diagnostic_if_applied"
    }
    assert set(hourly.phi) == {0.8}

    sample = future.iloc[0]
    final_shares = window_shares(sample[SHAPE_COLS].to_numpy(float))
    intermediate_shares = sample[
        [
            "four_window_share_intermediate_night",
            "four_window_share_intermediate_morning",
            "four_window_share_intermediate_midday",
            "four_window_share_intermediate_evening",
        ]
    ].to_numpy(float)
    assert peak_valley_rate(sample[SHAPE_COLS].to_numpy(float)) == pytest.approx(
        0.65,
        abs=1e-10,
    )
    assert not np.allclose(final_shares, intermediate_shares, atol=1e-8)

    component_days = checks.query("closure_scope == 'component_day'").copy()
    component_days["daily_fraction"] = (
        component_days.target_energy_MWh
        / component_days.groupby(["component", "month"]).target_energy_MWh.transform("sum")
    )
    fractions = component_days.pivot(index="date", columns="component", values="daily_fraction")
    assert np.allclose(fractions.to_numpy(), fractions.iloc[:, [0]].to_numpy())


def test_control_set_change_is_diagnostic_and_does_not_fail_assembly() -> None:
    energy, baseline, factors, future, envelope, correction, _ = _inputs()
    assert future.effective_control_set_changed_from_2023.any()

    hourly, checks = assemble_province_year(
        2030, "基准", energy, baseline, factors, future, envelope, correction
    )

    assert len(hourly) == 8760
    assert checks.passed.all()


def test_future_parameter_gate_uses_strict_string_boolean_and_identity_stage() -> None:
    _, _, _, future, _, _, _ = _inputs()
    row_index = future.index[0]
    future["peak_valley_evolution_enabled"] = future[
        "peak_valley_evolution_enabled"
    ].astype(object)
    future.at[row_index, "peak_valley_evolution_enabled"] = "False"
    future.at[row_index, "peak_valley_gate_reason"] = (
        "disabled_delta_below_lower_bound"
    )
    future.at[row_index, "peak_valley_delta_abs"] = 0.01
    future.at[row_index, "peak_valley_delta_band"] = "below_lower_threshold"
    future.at[row_index, "peak_valley_adjustment_status"] = (
        "skipped_delta_below_lower_bound"
    )
    actual_rate = peak_valley_rate(future.loc[row_index, SHAPE_COLS].to_numpy(float))
    future.at[row_index, "peak_valley_rate_after_four_window"] = actual_rate
    future.at[row_index, "peak_valley_target_applied"] = actual_rate
    future.at[row_index, "peak_valley_rate_final"] = actual_rate
    future.at[row_index, "peak_valley_lambda"] = 1.0
    for name in ("night", "morning", "midday", "evening"):
        final_share = future.at[row_index, f"four_window_share_final_{name}"]
        future.at[row_index, f"applied_share_{name}"] = final_share
        future.at[row_index, f"four_window_share_intermediate_{name}"] = final_share
        future.at[row_index, f"four_window_share_deviation_{name}"] = 0.0

    records = _validated_future_parameters(
        future,
        [f"行业{index + 1:02d}" for index in range(24)],
        year=2030,
        horizon=8,
        peak_valley_phi=0.8,
        tolerance=1e-10,
    )

    assert len(records) == 24 * len(SEASONS) * len(DAY_TYPES)


def test_future_parameter_gate_rejects_closed_stage_with_nonidentity_lambda() -> None:
    _, _, _, future, _, _, _ = _inputs()
    row_index = future.index[0]
    future.at[row_index, "peak_valley_evolution_enabled"] = False
    future.at[row_index, "peak_valley_gate_reason"] = (
        "disabled_delta_below_lower_bound"
    )
    future.at[row_index, "peak_valley_delta_abs"] = 0.01
    future.at[row_index, "peak_valley_delta_band"] = "below_lower_threshold"
    # Deliberately retain the active-stage lambda and final-share deviations.

    with pytest.raises(ValueError, match="仅保留四时段结果"):
        _validated_future_parameters(
            future,
            [f"行业{index + 1:02d}" for index in range(24)],
            year=2030,
            horizon=8,
            peak_valley_phi=0.8,
            tolerance=1e-10,
        )


def test_future_parameter_gate_rejects_delta_band_inconsistent_with_delta() -> None:
    _, _, _, future, _, _, _ = _inputs()
    future["peak_valley_delta_band"] = "above_upper_threshold"

    with pytest.raises(ValueError, match="peak_valley_delta_band"):
        _validated_future_parameters(
            future,
            [f"行业{index + 1:02d}" for index in range(24)],
            year=2030,
            horizon=8,
            peak_valley_phi=0.8,
            tolerance=1e-10,
        )


def test_future_parameter_gate_rejects_evidence_drift_outside_formal_slice() -> None:
    _, _, _, future, _, _, _ = _inputs()
    conflicting = future.iloc[[0]].copy()
    conflicting["H"] = 5
    conflicting["phi"] = 5
    conflicting["peak_valley_phi"] = 5
    conflicting["formal_H8"] = False
    conflicting["formal_phi08"] = False
    conflicting["peak_valley_delta_abs"] = 0.01
    conflicting["peak_valley_delta_band"] = "below_lower_threshold"
    conflicting["peak_valley_evolution_enabled"] = False
    conflicting["peak_valley_gate_reason"] = "disabled_delta_below_lower_bound"
    future = pd.concat([future, conflicting], ignore_index=True)

    with pytest.raises(ValueError, match="跨year/H/phi必须恒定"):
        _validated_future_parameters(
            future,
            [f"行业{index + 1:02d}" for index in range(24)],
            year=2030,
            horizon=8,
            peak_valley_phi=0.8,
            tolerance=1e-10,
        )


def test_missing_nonwork_shape_falls_back_to_same_season_2023_workday() -> None:
    energy, baseline, factors, future, envelope, correction, _ = _inputs()
    missing = (
        baseline.industry.eq("行业01")
        & baseline.season.eq("春季")
        & baseline.final_day_type.eq("普通周末")
    )
    baseline = baseline.loc[~missing]
    future_missing = (
        future.industry.eq("行业01")
        & future.season.eq("春季")
        & future.final_day_type.eq("普通周末")
    )
    future = future.loc[~future_missing]

    _, checks = assemble_province_year(
        2030,
        "基准",
        energy,
        baseline,
        factors,
        future,
        envelope,
        correction,
        include_daily_detail=True,
    )
    usage = checks.query(
        "closure_scope == 'component_day' and industry == '行业01' and "
        "calendar_day_type == '普通周末' and shape_fallback_used"
    )

    assert not usage.empty
    assert set(usage.shape_source_day_type) == {"工作日"}
    assert not usage.trend_eligible.any()


def test_missing_same_season_2023_workday_baseline_blocks() -> None:
    energy, baseline, factors, future, envelope, correction, _ = _inputs()
    baseline = baseline.loc[
        ~(
            baseline.industry.eq("行业01")
            & baseline.season.eq("春季")
            & baseline.final_day_type.eq("工作日")
        )
    ]

    with pytest.raises(RuntimeError, match="2023.*工作日基准.*缺失"):
        assemble_province_year(
            2030, "基准", energy, baseline, factors, future, envelope, correction
        )


def test_leap_year_assembly_has_8784_continuous_hours() -> None:
    energy, baseline, factors, future, envelope, correction, _ = _inputs(2028)

    hourly, checks = assemble_province_year(
        2028, "基准", energy, baseline, factors, future, envelope, correction
    )

    assert len(hourly) == 8784
    assert hourly.timestamp.is_unique and hourly.timestamp.is_monotonic_increasing
    assert hourly.timestamp.iloc[0] == pd.Timestamp("2028-01-01 00:00:00")
    assert hourly.timestamp.iloc[-1] == pd.Timestamp("2028-12-31 23:00:00")
    assert checks.passed.all()


def test_correction_shape_tiny_negative_tail_is_cleaned_for_assembly() -> None:
    _, _, _, _, _, correction, _ = _inputs()
    first = (
        correction.season.eq("春季")
        & correction.day_group.eq("工作日")
        & correction.hour.eq(0)
    )
    second = (
        correction.season.eq("春季")
        & correction.day_group.eq("工作日")
        & correction.hour.eq(1)
    )
    original = float(correction.loc[first, "correction_normalized_shape"].iloc[0])
    correction.loc[first, "correction_normalized_shape"] = -5e-11
    correction.loc[second, "correction_normalized_shape"] += original + 5e-11

    shapes = _validated_correction_shapes(correction, tolerance=1e-10)
    cleaned = shapes[("春季", "工作日")]

    assert cleaned.min() >= 0.0
    assert cleaned.sum() == pytest.approx(1.0, abs=1e-12)
