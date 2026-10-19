"""25个电量组成年度装配的合成数据测试。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from workstreams_load_characteristic_model_v2.annual_assembly import assemble_province_year
from workstreams_load_characteristic_model_v2.calendar_utils import calendar_frame
from workstreams_load_characteristic_model_v2.monthly_envelope import target_month_shares
from workstreams_load_characteristic_model_v2.pipeline import (
    _build_annual_weight_ledger,
    _build_daily_energy_tables,
)
from workstreams_load_characteristic_model_v2.shape_transform import peak_valley_rate


SEASONS = ("春季", "夏季", "秋季", "冬季")
DAY_GROUPS = ("工作日", "非工作日")
SHAPE_COLS = tuple(f"p{hour:02d}" for hour in range(24))


def _inputs(year: int = 2030) -> tuple[pd.DataFrame, ...]:
    industries = [f"行业{index + 1:02d}" for index in range(24)]
    method = "方法A"
    planning_scenario = "基准"
    total_energy = 300.0
    industry_energy = 10.0
    correction_energy = total_energy - len(industries) * industry_energy
    energy_rows = [
        {
            "method": method,
            "planning_scenario": planning_scenario,
            "year": year,
            "component": industry,
            "industry": industry,
            "component_type": "传统细分行业",
            "annual_energy_quantile": "P50",
            "electricity_share": industry_energy / total_energy,
            "annual_energy_MWh": industry_energy,
        }
        for industry in industries
    ]
    energy_rows.append(
        {
            "method": method,
            "planning_scenario": planning_scenario,
            "year": year,
            "component": "修正组成",
            "industry": "修正组成",
            "component_type": "修正组成",
            "annual_energy_quantile": "P50",
            "electricity_share": correction_energy / total_energy,
            "annual_energy_MWh": correction_energy,
        }
    )

    base_shape = np.arange(1.0, 25.0)
    base_shape /= base_shape.sum()
    base_window_shares = (
        float(base_shape[0:6].sum()),
        float(base_shape[6:12].sum()),
        float(base_shape[12:17].sum()),
        float(base_shape[17:24].sum()),
    )
    baseline_rows: list[dict[str, object]] = []
    parameter_rows: list[dict[str, object]] = []
    nonwork_rows: list[dict[str, object]] = []
    for industry in industries:
        for season in SEASONS:
            nonwork_rows.append(
                {
                    "industry": industry,
                    "season": season,
                    "workday_factor": 1.0,
                    "nonwork_factor": 0.5,
                }
            )
            for day_group in DAY_GROUPS:
                baseline_rows.append(
                    {
                        "industry": industry,
                        "season": season,
                        "day_group": day_group,
                        **dict(zip(SHAPE_COLS, base_shape)),
                    }
                )
                parameter_rows.append(
                    {
                        "industry": industry,
                        "season": season,
                        "day_group": day_group,
                        "year": year,
                        "H": 8,
                        "phi": 0.8,
                        "peak_valley_phi": 0.8,
                        "formal_H8": True,
                        "formal_phi08": True,
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
                        "night_share": base_window_shares[0],
                        "morning_share": base_window_shares[1],
                        "midday_share": base_window_shares[2],
                        "evening_share": base_window_shares[3],
                        "applied_share_night": base_window_shares[0],
                        "applied_share_morning": base_window_shares[1],
                        "applied_share_midday": base_window_shares[2],
                        "applied_share_evening": base_window_shares[3],
                        "peak_valley_target_requested": peak_valley_rate(base_shape),
                        "peak_valley_target_range_limited": peak_valley_rate(base_shape),
                        "peak_valley_target_applied": peak_valley_rate(base_shape),
                        "peak_valley_target_clipped": False,
                        "peak_valley_adjustment_status": "target_already_met",
                        "peak_valley_rate_after_four_window": peak_valley_rate(base_shape),
                        "peak_valley_lambda": 1.0,
                        "peak_valley_rate_final": peak_valley_rate(base_shape),
                        **{
                            f"four_window_share_intermediate_{name}": value
                            for name, value in zip(
                                ("night", "morning", "midday", "evening"),
                                base_window_shares,
                                strict=True,
                            )
                        },
                        **{
                            f"four_window_share_final_{name}": value
                            for name, value in zip(
                                ("night", "morning", "midday", "evening"),
                                base_window_shares,
                                strict=True,
                            )
                        },
                        **{
                            f"four_window_share_deviation_{name}": 0.0
                            for name in ("night", "morning", "midday", "evening")
                        },
                        "shape_extrapolation_method": (
                            "four_window_then_local_smoothing_then_gated_peak_valley_mean_scaling"
                        ),
                        "four_window_final_share_policy": (
                            "exact_if_peak_valley_skipped_diagnostic_if_applied"
                        ),
                    }
                )

    correction_shape = base_shape[::-1].copy()
    correction_rows = [
        {
            "season": season,
            "day_group": day_group,
            "hour": hour,
            "correction_normalized_shape": value,
        }
        for season in SEASONS
        for day_group in DAY_GROUPS
        for hour, value in enumerate(correction_shape)
    ]
    envelope = pd.DataFrame(
        {
            "month": np.arange(1, 13),
            "factor_p50": np.ones(12),
        }
    )
    return (
        pd.DataFrame(energy_rows),
        pd.DataFrame(baseline_rows),
        pd.DataFrame(nonwork_rows),
        pd.DataFrame(parameter_rows),
        envelope,
        pd.DataFrame(correction_rows),
        base_shape,
        correction_shape,
    )


class AnnualAssemblyTests(unittest.TestCase):
    def test_25_components_close_by_month_and_year_without_legacy_group_columns(self):
        (
            energy,
            baseline,
            nonwork,
            parameters,
            envelope,
            correction,
            _,
            _,
        ) = _inputs()

        hourly, closure = assemble_province_year(
            2030,
            "基准",
            energy,
            baseline,
            nonwork,
            parameters,
            envelope,
            correction,
            method="方法A",
        )

        self.assertEqual(len(hourly), 8760)
        self.assertAlmostEqual(float(hourly.load_MW.sum()), 300.0, places=10)
        self.assertTrue((hourly.load_MW >= 0).all())
        self.assertFalse(
            {"run_group", "parameter_group"}.intersection(hourly.columns)
        )
        self.assertFalse(
            {"run_group", "parameter_group"}.intersection(closure.columns)
        )
        self.assertEqual(
            closure.closure_scope.value_counts().to_dict(),
            {
                "component_month": 25 * 12,
                "component_annual": 25,
                "province_month": 12,
                "province_annual": 1,
            },
        )
        self.assertTrue(closure.passed.all())
        self.assertLessEqual(float(closure.relative_error.max()), 1e-10)
        self.assertEqual(
            set(hourly.curve_energy_scope),
            {
                "natural_load_excluding_plant_aux_line_loss_pumped_storage_charging"
            },
        )
        self.assertEqual(set(hourly.annual_energy_basis), {"P50_natural_total"})
        self.assertEqual(
            set(hourly.trajectory_quantile_basis), {"annual_peak_bootstrap"}
        )
        self.assertEqual(set(hourly.annual_energy_quantile), {"P50"})
        self.assertTrue(hourly.annual_special_deduction_applied.all())
        self.assertTrue((~hourly.hourly_special_deduction_applied).all())
        self.assertTrue((~hourly.special_items_in_correction_component).all())

    def test_only_p50_natural_energy_is_used_when_all_quantiles_are_present(self):
        energy, baseline, nonwork, parameters, envelope, correction, _, _ = _inputs()
        quantile_frames = []
        for quantile, scale in (("P10", 0.8), ("P50", 1.0), ("P90", 1.2)):
            frame = energy.copy()
            frame["annual_energy_quantile"] = quantile
            frame["annual_energy_MWh"] = frame.annual_energy_MWh * scale
            quantile_frames.append(frame)

        hourly, closure = assemble_province_year(
            2030,
            "基准",
            pd.concat(quantile_frames, ignore_index=True),
            baseline,
            nonwork,
            parameters,
            envelope,
            correction,
            method="方法A",
        )

        self.assertAlmostEqual(float(hourly.load_MW.sum()), 300.0, places=10)
        self.assertEqual(set(hourly.annual_energy_quantile), {"P50"})
        self.assertEqual(set(closure.annual_energy_quantile), {"P50"})
        province_annual = closure.loc[
            closure.closure_scope.eq("province_annual"), "target_energy_MWh"
        ]
        self.assertEqual(province_annual.tolist(), [300.0])

    def test_opt_in_daily_detail_closes_all_25_components_for_each_day(self):
        energy, baseline, nonwork, parameters, envelope, correction, _, _ = _inputs()
        hourly, closure = assemble_province_year(
            2030,
            "基准",
            energy,
            baseline,
            nonwork,
            parameters,
            envelope,
            correction,
            method="方法A",
            include_daily_detail=True,
        )

        self.assertEqual(
            closure.closure_scope.value_counts().to_dict(),
            {
                "component_day": 25 * 365,
                "province_day": 365,
                "component_month": 25 * 12,
                "component_annual": 25,
                "province_month": 12,
                "province_annual": 1,
            },
        )
        component_days = closure.loc[closure.closure_scope.eq("component_day")]
        self.assertTrue(component_days.groupby("date").component.nunique().eq(25).all())
        component_sum = component_days.groupby("date").actual_energy_MWh.sum()
        province_sum = (
            hourly.assign(date=hourly.timestamp.dt.normalize())
            .groupby("date")
            .load_MW.sum()
        )
        np.testing.assert_allclose(component_sum, province_sum, rtol=0, atol=1e-10)

        component_month, component_daily, daily_checks = _build_daily_energy_tables(
            closure
        )
        ledger = _build_annual_weight_ledger(energy, closure)
        self.assertEqual(len(component_month), 25 * 12)
        self.assertEqual(len(component_daily), 25 * 365)
        self.assertEqual(len(daily_checks), 365)
        self.assertTrue(daily_checks.passed.all())
        self.assertTrue(daily_checks.component_count.eq(25).all())
        self.assertEqual(len(ledger), 25)
        self.assertTrue(ledger.assembly_passed.all())

    def test_shared_nonwork_factor_and_frozen_correction_shape_are_used(self):
        (
            energy,
            baseline,
            nonwork,
            parameters,
            envelope,
            correction,
            base_shape,
            correction_shape,
        ) = _inputs()
        hourly, _ = assemble_province_year(
            2030,
            "基准",
            energy,
            baseline,
            nonwork,
            parameters,
            envelope,
            correction,
            method="方法A",
        )

        month_share = float(
            target_month_shares(2030, envelope).set_index("month").loc[1, "month_share"]
        )
        january = calendar_frame(2030).loc[lambda frame: frame.month.eq(1)]
        workdays = int(january.day_group.eq("工作日").sum())
        nonworkdays = int(january.day_group.eq("非工作日").sum())
        traditional_month_energy = 240.0 * month_share
        correction_month_energy = 60.0 * month_share
        traditional_workday_energy = traditional_month_energy / (
            workdays + 0.5 * nonworkdays
        )
        correction_workday_energy = correction_month_energy / (
            workdays + 0.5 * nonworkdays
        )
        expected_workday = (
            traditional_workday_energy * base_shape
            + correction_workday_energy * correction_shape
        )
        workday = pd.Timestamp(
            january.loc[january.day_group.eq("工作日"), "date"].iloc[0]
        )
        nonworkday = pd.Timestamp(
            january.loc[january.day_group.eq("非工作日"), "date"].iloc[0]
        )
        actual_workday = hourly.loc[
            hourly.timestamp.dt.normalize().eq(workday), "load_MW"
        ].to_numpy(float)
        np.testing.assert_allclose(actual_workday, expected_workday, rtol=0, atol=1e-12)

        daily = hourly.assign(date=hourly.timestamp.dt.normalize()).groupby("date").load_MW.sum()
        expected_nonworkday_energy = 0.5 * (
            traditional_workday_energy + correction_workday_energy
        )
        self.assertAlmostEqual(
            float(daily.loc[workday]),
            traditional_workday_energy + correction_workday_energy,
            places=12,
        )
        self.assertAlmostEqual(
            float(daily.loc[nonworkday]), expected_nonworkday_energy, places=12
        )

    def test_missing_formal_industry_parameter_cell_hard_fails(self):
        energy, baseline, nonwork, parameters, envelope, correction, _, _ = _inputs()
        parameters = parameters.loc[
            ~(
                parameters.industry.eq("行业01")
                & parameters.season.eq("春季")
                & parameters.day_group.eq("工作日")
            )
        ]

        with self.assertRaisesRegex(ValueError, "未来形状参数.*行业01.*春季.*工作日"):
            assemble_province_year(
                2030,
                "基准",
                energy,
                baseline,
                nonwork,
                parameters,
                envelope,
                correction,
                method="方法A",
            )

    def test_negative_correction_shape_hard_fails(self):
        energy, baseline, nonwork, parameters, envelope, correction, _, _ = _inputs()
        target = (
            correction.season.eq("春季")
            & correction.day_group.eq("工作日")
            & correction.hour.eq(0)
        )
        correction.loc[target, "correction_normalized_shape"] = -0.01

        with self.assertRaisesRegex(ValueError, "修正组成.*负"):
            assemble_province_year(
                2030,
                "基准",
                energy,
                baseline,
                nonwork,
                parameters,
                envelope,
                correction,
                method="方法A",
            )


if __name__ == "__main__":
    unittest.main()
