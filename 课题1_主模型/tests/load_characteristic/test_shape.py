from __future__ import annotations

import numpy as np
import pandas as pd
from pytest import approx, raises

from workstreams_load_characteristic_model_v2.baseline_shapes import (
    build_baseline_shapes,
    estimate_nonwork_factors,
    weighted_medoid,
)
from workstreams_load_characteristic_model_v2.parameter_pipeline import (
    build_future_parameter_table,
    build_industry_cell_judgement,
    identify_industry_parameters,
)
from workstreams_load_characteristic_model_v2.quality import SHAPE_COLS
from workstreams_load_characteristic_model_v2.representative_selection import (
    build_industry_parameter_trace,
    rank_representative_users,
)
from workstreams_load_characteristic_model_v2.shape_parameters import (
    constrain_window_migration,
    estimate_window_migration,
    extrapolated_increment,
    forecast_parameter_trajectory,
    identify_shape_parameters,
    judge_control_period_change,
    judge_evening_peak_position,
    shape_for_trajectory_row,
)
from workstreams_load_characteristic_model_v2.shape_transform import (
    apply_window_shares,
    window_shares,
)
from workstreams_load_characteristic_model_v2 import shape_transform


SEASONS = ("春季", "夏季", "秋季", "冬季")
DAY_GROUPS = ("工作日", "非工作日")


def _base_shape() -> np.ndarray:
    values = np.asarray(
        [
            0.72, 0.68, 0.64, 0.62, 0.64, 0.70, 0.88, 1.08,
            1.18, 1.12, 1.00, 0.94, 0.90, 0.92, 0.98, 1.06,
            1.18, 1.34, 1.52, 1.72, 2.00, 1.78, 1.42, 1.04,
        ],
        dtype=float,
    )
    return values / values.sum()


def _quality_rows(industry: str, user: str, paired_workdays: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for year in (2022, 2023):
        for season in SEASONS:
            rows.append(
                {
                    "industry": industry,
                    "user_id": user,
                    "year": year,
                    "season": season,
                    "day_type": "工作日",
                    "expected_days": 60,
                    "valid_days": paired_workdays,
                    "sample_energy": 100.0,
                }
            )
    return rows


def _month_days(year: int, month: int, count: int, peak_hour: int) -> pd.DataFrame:
    rows = []
    start = pd.Timestamp(year=year, month=month, day=1)
    for offset in range(count):
        curve = np.full(24, 1.0, dtype=float)
        curve[17:24] = np.asarray([1.0, 1.1, 1.3, 1.5, 1.3, 1.1, 1.0])
        curve[peak_hour] = 3.0
        curve /= curve.sum()
        rows.append(
            {
                "year": year,
                "month": month,
                "season": "夏季",
                "date": start + pd.Timedelta(days=offset),
                **dict(zip(SHAPE_COLS, curve)),
            }
        )
    return pd.DataFrame(rows)


def _industry_records(industries: list[tuple[str, str]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    base = _base_shape()
    for industry_index, (industry, broad_sector) in enumerate(industries):
        for year in (2022, 2023):
            for season_index, season in enumerate(SEASONS):
                for day_index, day_group in enumerate(DAY_GROUPS):
                    curve = np.roll(base, (industry_index + season_index + day_index) % 3)
                    curve = curve / curve.sum()
                    rows.append(
                        {
                            "department": broad_sector,
                            "industry": industry,
                            "user": f"{industry}-用户1",
                            "year": year,
                            "season": season,
                            "day_group": day_group,
                            "sample_energy": 100.0 + industry_index,
                            **dict(zip(SHAPE_COLS, curve)),
                        }
                    )
    return pd.DataFrame(rows)


def _cell_judgement(industries: list[tuple[str, str]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for industry_index, (industry, broad_sector) in enumerate(industries):
        for season in SEASONS:
            for day_group in DAY_GROUPS:
                changed = industry_index == 0 and season == "春季" and day_group == "工作日"
                sufficient = not (
                    industry_index == len(industries) - 1
                    and season == "冬季"
                    and day_group == "非工作日"
                )
                rows.append(
                    {
                        "industry": industry,
                        "broad_sector": broad_sector,
                        "season": season,
                        "day_group": day_group,
                        "sample_sufficient": sufficient,
                        "trend_eligible": sufficient,
                        "trend_status": (
                            "short_history_exploratory"
                            if sufficient
                            else "insufficient_history_no_extrapolation"
                        ),
                        "judgement_status": (
                            "evaluable" if sufficient else "judgement_not_evaluable"
                        ),
                        "evening_judge_enabled": broad_sector in {"第三产业", "居民生活"},
                        "changed": changed,
                        "source_year": 2023 if (changed or not sufficient) else 2022,
                        "reason": "判据触发" if changed else "两条判据均未触发",
                    }
                )
    return pd.DataFrame(rows)


def test_representative_ranking_prefers_complete_paired_history() -> None:
    rows = _quality_rows("行业甲", "用户A", 22) + _quality_rows("行业甲", "用户B", 18)
    result = rank_representative_users(pd.DataFrame(rows))
    assert result.selected.loc[0, "user_id"] == "用户A"
    assert bool(result.selected.loc[0, "trend_eligible"])


def test_industry_parameter_trace_keeps_every_industry_as_its_own_source() -> None:
    candidates = pd.DataFrame(
        [
            {"industry": "农业", "broad_sector": "第一产业", "parameter_valid": True},
            {"industry": "化工", "broad_sector": "第二产业", "parameter_valid": True},
            {"industry": "学校", "broad_sector": "第三产业", "parameter_valid": False},
        ]
    )
    energy = pd.DataFrame(
        {"industry": ["农业", "化工", "学校"], "official_energy_2025": [10.0, 100.0, 20.0]}
    )
    result = build_industry_parameter_trace(candidates, energy).trace
    assert result.industry.tolist() == ["农业", "化工", "学校"]
    assert result.uses_own_parameters.tolist() == [True, True, True]
    assert result.parameter_status.tolist() == [
        "own_industry_estimated",
        "own_industry_estimated",
        "own_industry_frozen_zero",
    ]
    assert "run_group" not in result.columns
    assert "parameter_group" not in result.columns
    assert not any(column.startswith("selected_for_") for column in result.columns)


def test_industry_parameter_trace_rejects_missing_official_energy() -> None:
    candidates = pd.DataFrame(
        [{"industry": "缺电量行业", "broad_sector": "第二产业", "parameter_valid": True}]
    )
    energy = pd.DataFrame(columns=["industry", "official_energy_2025"])
    with raises(ValueError, match="missing official 2025 energy"):
        build_industry_parameter_trace(candidates, energy)


def test_weighted_medoid_is_an_observed_curve() -> None:
    curves = np.asarray(
        [[1.0] + [0.0] * 23, [0.0, 1.0] + [0.0] * 22, [0.0, 1.0] + [0.0] * 22]
    )
    curve, selected, _ = weighted_medoid(curves, np.asarray([1.0, 2.0, 1.0]))
    assert selected == 1
    np.testing.assert_allclose(curve, curves[1])


def test_window_migration_uses_each_industrys_own_paired_curves() -> None:
    base = _base_shape()
    target_shares = np.asarray([0.18, 0.16, 0.30, 0.36])
    target = apply_window_shares(base, target_shares)
    estimate = estimate_window_migration(base, target)
    assert estimate.beta_a_observed == approx(
        0.5 * np.abs(target_shares - window_shares(base)).sum()
    )
    assert estimate.direction_v.sum() == approx(0.0, abs=1e-12)
    np.testing.assert_allclose(
        estimate.shares_2022 + estimate.beta_a_observed * estimate.direction_v,
        estimate.shares_2023,
        atol=1e-12,
    )


def test_forecast_trajectory_contains_only_window_migration_fields() -> None:
    base = _base_shape()
    target = apply_window_shares(base, np.asarray([0.18, 0.16, 0.30, 0.36]))
    estimate = identify_shape_parameters(base, target, trend_eligible=True)
    trajectory = forecast_parameter_trajectory(
        estimate, [2025, 2030], base_window_shares=window_shares(base)
    )
    assert len(trajectory) == 6
    prohibited = ("gamma", "delta", "evening_shift")
    assert not any(any(token in column.lower() for token in prohibited) for column in trajectory.columns)
    assert trajectory.a_applied.gt(0).all()
    shape = shape_for_trajectory_row(base, trajectory.loc[trajectory.H.eq(8)].iloc[-1])
    assert shape.sum() == approx(1.0, abs=1e-12)
    assert np.all(shape >= 0)


def test_single_year_or_ineligible_parameter_history_is_frozen() -> None:
    base = _base_shape()
    single_year = identify_shape_parameters(base, None, trend_eligible=False)
    assert not single_year.trend_eligible
    assert single_year.beta_a == 0.0
    assert single_year.trend_status == "insufficient_history_no_extrapolation"


def test_untriggered_but_sufficient_industry_cell_is_not_frozen() -> None:
    industries = [("农业", "第一产业"), ("化工", "第二产业")]
    records = _industry_records(industries)
    checked = build_industry_cell_judgement(
        _cell_judgement(industries), expected_industries=[x[0] for x in industries]
    )
    baseline, _ = build_baseline_shapes(records, checked)

    untriggered = baseline.loc[
        baseline.industry.eq("农业")
        & baseline.season.eq("夏季")
        & baseline.day_group.eq("工作日")
    ].iloc[0]
    assert int(untriggered.anchor_year) == 2022
    assert not bool(untriggered.trend_frozen)

    insufficient = baseline.loc[
        baseline.industry.eq("化工")
        & baseline.season.eq("冬季")
        & baseline.day_group.eq("非工作日")
    ].iloc[0]
    assert bool(insufficient.trend_frozen)
    assert int(insufficient.anchor_year) == 2023


def test_24_industries_produce_exactly_192_baseline_cells() -> None:
    industries = []
    for index in range(24):
        sector = (
            "第一产业" if index < 4 else
            "第二产业" if index < 16 else
            "第三产业" if index < 23 else
            "居民生活"
        )
        industries.append((f"行业{index + 1:02d}", sector))
    records = _industry_records(industries)
    judgement = build_industry_cell_judgement(
        _cell_judgement(industries), expected_industries=[x[0] for x in industries]
    )
    baseline, diagnostics = build_baseline_shapes(records, judgement)
    assert len(baseline) == len(diagnostics) == 192
    assert not baseline.duplicated(["industry", "season", "day_group"]).any()
    assert set(baseline.groupby("industry").size()) == {8}
    assert "run_group" not in baseline.columns
    assert "parameter_group" not in baseline.columns


def test_each_industry_future_table_uses_its_own_parameter_estimate() -> None:
    base = _base_shape()
    target_a = apply_window_shares(base, np.asarray([0.18, 0.16, 0.30, 0.36]))
    target_b = apply_window_shares(base, np.asarray([0.24, 0.19, 0.25, 0.32]))
    industries = ("行业甲", "行业乙")
    estimates = {
        ("行业甲", "春季", "工作日"): identify_shape_parameters(
            base, target_a, trend_eligible=True
        ),
        ("行业乙", "春季", "工作日"): identify_shape_parameters(
            base, target_b, trend_eligible=True
        ),
    }
    baseline = pd.DataFrame(
        [
            {
                "industry": industry,
                "broad_sector": "第二产业",
                "season": "春季",
                "day_group": "工作日",
                "anchor_year": 2022,
                "trend_frozen": False,
                "judgement_changed": False,
                **dict(zip(SHAPE_COLS, base)),
            }
            for industry in industries
        ]
    )
    table = build_future_parameter_table(baseline, estimates, [2030])
    formal = table.loc[table.H.eq(8)].set_index("industry")
    assert formal.loc["行业甲", "a_applied"] != approx(formal.loc["行业乙", "a_applied"])
    assert formal.loc["行业甲", "base_anchor_year"] == 2022
    assert not bool(formal.loc["行业甲", "cell_trend_frozen"])
    assert not bool(formal.loc["行业甲", "judgement_changed"])
    assert not any("source" in column.lower() for column in table.columns)
    assert "run_group" not in table.columns
    assert "parameter_group" not in table.columns


def test_only_insufficient_cell_is_zeroed_in_future_table() -> None:
    base = _base_shape()
    target = apply_window_shares(base, np.asarray([0.18, 0.16, 0.30, 0.36]))
    estimate = identify_shape_parameters(base, target, trend_eligible=True)
    baseline = pd.DataFrame(
        [
            {
                "industry": "行业甲",
                "broad_sector": "第二产业",
                "season": season,
                "day_group": "工作日",
                "anchor_year": 2022,
                "trend_frozen": frozen,
                "judgement_changed": False,
                **dict(zip(SHAPE_COLS, base)),
            }
            for season, frozen in (("春季", False), ("夏季", True))
        ]
    )
    table = build_future_parameter_table(
        baseline,
        {
            ("行业甲", "春季", "工作日"): estimate,
            ("行业甲", "夏季", "工作日"): identify_shape_parameters(
                base, None, trend_eligible=False
            ),
        },
        [2030],
    )
    formal = table.loc[table.H.eq(8)].set_index("season")
    assert formal.loc["春季", "a_applied"] > 0
    assert formal.loc["夏季", "a_applied"] == 0
    assert formal.loc["夏季", "trend_status"] == "insufficient_history_no_extrapolation"


def test_same_industry_cells_use_their_own_cell_parameter_estimates() -> None:
    base = _base_shape()
    spring_target = apply_window_shares(base, np.asarray([0.18, 0.16, 0.30, 0.36]))
    summer_target = apply_window_shares(base, np.asarray([0.24, 0.19, 0.25, 0.32]))
    baseline = pd.DataFrame(
        [
            {
                "industry": "行业甲",
                "broad_sector": "第二产业",
                "season": season,
                "day_group": "工作日",
                "anchor_year": 2022,
                "trend_frozen": False,
                "judgement_changed": False,
                **dict(zip(SHAPE_COLS, base)),
            }
            for season in ("春季", "夏季")
        ]
    )
    estimates = {
        ("行业甲", "春季", "工作日"): identify_shape_parameters(
            base, spring_target, trend_eligible=True
        ),
        ("行业甲", "夏季", "工作日"): identify_shape_parameters(
            base, summer_target, trend_eligible=True
        ),
    }
    table = build_future_parameter_table(baseline, estimates, [2030])
    formal = table.loc[table.H.eq(8)].set_index("season")
    assert formal.loc["春季", "a_applied"] != approx(formal.loc["夏季", "a_applied"])


def test_flat_future_cell_keeps_theoretical_target_but_skips_second_stage() -> None:
    base = np.ones(24, dtype=float) / 24.0
    policy = (
        "trend_eligible_and_nonflat_and_abs_delta_within_closed_interval_0.065_0.25"
    )
    baseline = pd.DataFrame(
        [
            {
                "department": "第二产业",
                "industry": "平坦测试行业",
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
                "industry": "平坦测试行业",
                "season": "夏季",
                "final_day_type": "工作日",
                "trend_eligible": True,
                "trend_status": "paired_2022_2023_industry_medoids",
                "shares_2023": window_shares(base),
                "beta_a": 0.02,
                "direction_v": np.array([-1.0, 0.5, 0.25, 0.25]),
                "peak_valley_rate_2023": 0.0,
                "peak_valley_delta_signed": -0.10,
                "flat_curve": True,
                "flat_borderline": False,
                "peak_valley_delta_abs": 0.10,
                "peak_valley_delta_band": "within_gate_band",
                "peak_valley_evolution_enabled": False,
                "peak_valley_gate_reason": "disabled_flat_curve",
                "peak_valley_gate_policy": policy,
            }
        ]
    )

    future = build_future_parameter_table(baseline, history, [2030])
    row = future.loc[
        future.H.eq(8) & future.phi.eq(0.8) & future.year.eq(2030)
    ].iloc[0]

    assert bool(row.trend_eligible)
    assert bool(row.flat_curve)
    assert not bool(row.flat_borderline)
    assert row.peak_valley_delta_abs == approx(0.10)
    assert row.peak_valley_delta_band == "within_gate_band"
    assert not bool(row.peak_valley_evolution_enabled)
    assert row.peak_valley_gate_reason == "disabled_flat_curve"
    assert row.peak_valley_gate_policy == policy
    assert 0.0 <= row.peak_valley_target_requested < 1e-12
    assert row.peak_valley_target_range_limited == approx(0.0)
    assert row.peak_valley_target_applied == approx(
        row.peak_valley_rate_after_four_window
    )
    assert row.peak_valley_rate_final == approx(
        row.peak_valley_rate_after_four_window
    )
    assert row.peak_valley_target_applied != approx(
        row.peak_valley_target_range_limited
    )
    assert row.peak_valley_lambda == approx(1.0)
    assert row.peak_valley_adjustment_status == "skipped_flat_curve"
    for name in ("night", "morning", "midday", "evening"):
        assert row[f"four_window_share_final_{name}"] == approx(
            row[f"four_window_share_intermediate_{name}"]
        )
        assert row[f"four_window_share_deviation_{name}"] == approx(0.0)


def test_future_table_requires_history_gate_policy_without_defaulting() -> None:
    base = np.ones(24, dtype=float) / 24.0
    baseline = pd.DataFrame(
        [
            {
                "industry": "门控合同测试行业",
                "season": "夏季",
                "final_day_type": "工作日",
                **dict(zip(SHAPE_COLS, base, strict=True)),
            }
        ]
    )
    history = pd.DataFrame(
        [
            {
                "industry": "门控合同测试行业",
                "season": "夏季",
                "final_day_type": "工作日",
                "trend_eligible": True,
                "beta_a": 0.0,
                "direction_v": np.zeros(4),
                "peak_valley_rate_2023": 0.0,
                "peak_valley_delta_signed": 0.10,
                "flat_curve": False,
                "flat_borderline": False,
                "peak_valley_delta_abs": 0.10,
                "peak_valley_delta_band": "within_gate_band",
                "peak_valley_evolution_enabled": True,
                "peak_valley_gate_reason": "enabled",
            }
        ]
    )

    with raises(ValueError, match="peak_valley_gate_policy"):
        build_future_parameter_table(baseline, history, [2030])


def test_evening_peak_position_is_binary_judgement_only() -> None:
    moved = pd.concat(
        [_month_days(2022, 7, 10, 19), _month_days(2023, 7, 10, 21)], ignore_index=True
    )
    judgement, detail = judge_evening_peak_position(moved, industry="学校")
    assert judgement.enabled
    assert judgement.triggered_seasons == frozenset({"夏季"})
    july = detail.loc[detail.month.eq(7)].iloc[0]
    assert bool(july.month_triggered)
    assert july.ref_peak_hour == 19
    assert july.compare_typical_peak_hour == 21
    assert july.moved_share == approx(1.0)
    assert bool(july.peak_position_consistent)
    assert not any(
        token in column.lower()
        for column in detail.columns
        for token in ("delta", "shift", "centroid")
    )

    disabled, empty = judge_evening_peak_position(moved, industry="餐饮业")
    assert not disabled.enabled
    assert empty.empty


def test_control_period_change_judgement() -> None:
    base = _base_shape()
    unchanged = judge_control_period_change(base, base.copy())
    assert not unchanged["control_triggered"]
    boosted = base.copy()
    boosted[int(np.argmax(base))] *= 1.5
    triggered = judge_control_period_change(base, boosted)
    assert triggered["control_triggered"]
    assert triggered["control_change_rate"] > 0.10


def test_identify_industry_parameters_emits_eight_industry_cells_without_gamma_or_delta() -> None:
    industry = "化工"
    base = _base_shape()
    target = apply_window_shares(base, np.asarray([0.18, 0.16, 0.30, 0.36]))
    rows: list[dict[str, object]] = []
    season_month = {"春季": 3, "夏季": 6, "秋季": 9, "冬季": 12}
    day_offset = 0
    for year, curve in ((2022, base), (2023, target)):
        for season in SEASONS:
            for day_group, count in (("工作日", 20), ("非工作日", 8)):
                for _ in range(count):
                    rows.append(
                        {
                            "industry": industry,
                            "user_key": f"{industry}-用户1",
                            "user": f"{industry}-用户1",
                            "year": year,
                            "month": season_month[season],
                            "season": season,
                            "day_group": day_group,
                            "date": pd.Timestamp(year=year, month=season_month[season], day=1)
                            + pd.Timedelta(days=day_offset % 20),
                            "sample_energy": 100.0,
                            **dict(zip(SHAPE_COLS, curve)),
                        }
                    )
                    day_offset += 1
    mapping = pd.DataFrame([{"industry": industry, "broad_sector": "第二产业"}])
    energy = pd.DataFrame([{"industry": industry, "official_energy_2025": 1000.0}])
    representative, parameter_trace, history, evening, cells, estimates = identify_industry_parameters(
        pd.DataFrame(rows), mapping, energy
    )
    assert len(representative.selected) == 1
    assert len(parameter_trace.trace) == 1
    assert len(history) == 8
    assert evening.empty
    assert len(cells) == 8
    assert set(estimates) == {
        (industry, season, day_group)
        for season in SEASONS
        for day_group in DAY_GROUPS
    }
    prohibited = ("gamma", "delta", "run_group", "parameter_group")
    for frame in (parameter_trace.trace, history, cells):
        assert not any(any(token in column.lower() for token in prohibited) for column in frame.columns)


def test_nonwork_factors_are_estimated_at_industry_grain() -> None:
    industries = [("农业", "第一产业"), ("化工", "第二产业")]
    records = _industry_records(industries)
    factors = estimate_nonwork_factors(records, expected_industries=[x[0] for x in industries])
    assert len(factors) == 8
    assert set(factors.groupby("industry").size()) == {4}
    assert "run_group" not in factors.columns


def test_window_constraint_preserves_nonnegativity_and_sum() -> None:
    shares = np.asarray([0.20, 0.20, 0.30, 0.30])
    direction = np.asarray([-1.0, 0.4, 0.3, 0.3])
    requested = extrapolated_increment(0.04, 2030, 8)
    maximum, used, hit = constrain_window_migration(shares, direction, requested)
    target = shares + used * direction
    assert target.sum() == approx(1.0, abs=1e-12)
    assert np.all(target >= 0.01 - 1e-12)
    assert 0 <= used <= maximum
    assert isinstance(hit, bool)


def test_peak_damping_api_is_retired_from_formal_transform_module() -> None:
    base = _base_shape()
    shares = window_shares(base)
    target = shares.copy()
    target[0] -= 0.04
    target[3] += 0.04
    proportional = apply_window_shares(base, target)
    np.testing.assert_allclose(window_shares(proportional), target, atol=1e-12)
    assert proportional.sum() == approx(1.0, abs=1e-12)
    assert not hasattr(shape_transform, "apply_window_shares_peak_damped")
    assert not hasattr(shape_transform, "PEAK_DAMP_FACTOR")
