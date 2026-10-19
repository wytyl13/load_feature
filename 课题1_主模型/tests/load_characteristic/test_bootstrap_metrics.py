from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from workstreams_load_characteristic_model_v2.metrics import calculate_load_metrics, typical_workdays
from workstreams_load_characteristic_model_v2.monthly_envelope import build_interday_residual_library
from workstreams_load_characteristic_model_v2.province_bootstrap import apply_daily_multipliers, bootstrap_daily_multipliers
from workstreams_load_characteristic_model_v2.representative_paths import build_scenario_features, select_representative_scenarios
from workstreams_load_characteristic_model_v2.quality import HOUR_COLS
from workstreams_load_characteristic_model_v2.pipeline import (
    _evaluate_all_bootstrap_scenarios,
    _materialize_bootstrap_daily_output,
)


class BootstrapTests(unittest.TestCase):
    @staticmethod
    def residuals() -> pd.DataFrame:
        rows = []
        for year in (2019, 2020, 2021):
            for stamp in pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D"):
                rows.append({"history_year": year, "month": stamp.month, "date": stamp, "residual": 0.08 * np.sin(stamp.dayofyear / 5)})
        return pd.DataFrame(rows)

    def test_reproducible_and_month_closed(self):
        dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")
        baseline = pd.Series(100.0 + np.arange(len(dates)) % 7, index=dates)
        first, checks = bootstrap_daily_multipliers(self.residuals(), baseline, 2025, n_scenarios=20)
        second, _ = bootstrap_daily_multipliers(self.residuals(), baseline, 2025, n_scenarios=20)
        np.testing.assert_allclose(first.daily_multiplier, second.daily_multiplier)
        self.assertLessEqual(float(checks.relative_error.max()), 1e-12)
        self.assertEqual(len(first), 20 * 365)

    def test_daily_output_defaults_to_three_real_representative_paths(self):
        dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")
        baseline = pd.Series(100.0, index=dates)
        multipliers, _ = bootstrap_daily_multipliers(
            self.residuals(), baseline, 2025, n_scenarios=5
        )
        selected = {"P10": 1, "P50": 2, "P90": 3}

        representative = _materialize_bootstrap_daily_output(
            multipliers, selected, materialize_all=False
        )
        self.assertEqual(len(representative), 3 * 365)
        self.assertEqual(set(representative.representative_track), set(selected))
        self.assertTrue(representative.is_representative.all())
        self.assertFalse(
            representative.duplicated(
                ["representative_track", "scenario_id", "date"]
            ).any()
        )

        full = _materialize_bootstrap_daily_output(
            multipliers, selected, materialize_all=True
        )
        self.assertEqual(len(full), 5 * 365)
        self.assertEqual(set(full.loc[full.is_representative, "scenario_id"]), {1, 2, 3})
        self.assertTrue(full.loc[full.scenario_id.eq(0), "representative_track"].eq("").all())

    def test_representatives_are_real_scenarios(self):
        dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")
        baseline = pd.Series(100.0, index=dates)
        peak = pd.Series(6.0, index=dates)
        multipliers, _ = bootstrap_daily_multipliers(self.residuals(), baseline, 2025, n_scenarios=20)
        monthly, annual = build_scenario_features(multipliers, baseline, peak)
        selected, ranking = select_representative_scenarios(monthly, annual)
        self.assertTrue(set(selected.values()).issubset(set(ranking.scenario_id)))

    def test_residual_baseline_separates_three_day_classes_with_recorded_fallback(self):
        rows = []
        for stamp in pd.date_range("2019-01-01", "2019-01-31", freq="D"):
            daily_energy = 2400.0 + (200.0 if stamp.weekday() >= 5 else 0.0)
            row = {"date": stamp}
            row.update({column: daily_energy / 24.0 for column in HOUR_COLS})
            rows.append(row)
        residuals, diagnostics = build_interday_residual_library(pd.DataFrame(rows))
        self.assertEqual(set(residuals.residual_day_class), {"工作日", "普通周末", "法定节假日"})
        holiday = residuals.loc[residuals.day_type == "法定节假日"].iloc[0]
        self.assertIn("非工作日合并", holiday.residual_baseline_source)
        self.assertTrue(bool(holiday.baseline_fallback))
        self.assertAlmostEqual(float(residuals.residual.mean()), 0.0, places=14)
        self.assertEqual(int(diagnostics.fallback_days.iloc[0]), 1)

    def test_apply_multiplier_rejects_duplicate_scenario_date(self):
        hourly = pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-01-01", "2025-01-01 23:00", freq="h"),
                "load_MW": 100.0,
            }
        )
        multipliers = pd.DataFrame(
            {
                "scenario_id": [0, 0],
                "date": [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-01")],
                "daily_multiplier": [1.0, 1.0],
            }
        )
        with self.assertRaisesRegex(ValueError, "必须唯一"):
            apply_daily_multipliers(hourly, multipliers)


class MetricsTests(unittest.TestCase):
    @staticmethod
    def hourly() -> pd.DataFrame:
        stamps = pd.date_range("2025-01-01", "2025-12-31 23:00", freq="h")
        load = 100.0 + 20.0 * np.sin(2 * np.pi * stamps.hour / 24) + 5.0 * np.cos(2 * np.pi * stamps.dayofyear / 365)
        return pd.DataFrame({"timestamp": stamps, "load_MW": load})

    def test_tmax_identity_and_typical_days(self):
        _, _, annual = calculate_load_metrics(self.hourly())
        row = annual.iloc[0]
        self.assertAlmostEqual(row.annual_load_factor, row.Tmax_h / row.year_hours, places=14)
        typical = typical_workdays(self.hourly())
        self.assertEqual(len(typical), 4 * 24)
        np.testing.assert_allclose(typical.groupby("season").normalized_hour_share.sum(), 1.0)

    def test_metrics_reject_non_continuous_hourly_index(self):
        hourly = self.hourly()
        hourly.loc[1, "timestamp"] = pd.Timestamp("2025-01-01 01:30")
        with self.assertRaisesRegex(ValueError, "严格连续整点"):
            calculate_load_metrics(hourly)

    def test_vectorized_bootstrap_metrics_match_expanded_hourly_path(self):
        hourly = self.hourly()
        dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")
        baseline_energy = hourly.assign(date=hourly.timestamp.dt.normalize()).groupby("date").load_MW.sum()
        residuals = BootstrapTests.residuals()
        multipliers, _ = bootstrap_daily_multipliers(residuals, baseline_energy, 2025, n_scenarios=10)
        annual_all, monthly_all = _evaluate_all_bootstrap_scenarios(hourly, multipliers, "基准", 2025)
        selected = multipliers.loc[multipliers.scenario_id == 0]
        expanded = apply_daily_multipliers(hourly, selected)
        _, monthly_exact, annual_exact = calculate_load_metrics(expanded[["timestamp", "load_MW"]])
        annual_vec = annual_all.loc[annual_all.bootstrap_scenario_id == 0].iloc[0]
        for column in ("annual_energy_MWh", "annual_max_MW", "annual_min_MW", "Tmax_h", "annual_load_factor", "mean_daily_load_factor"):
            self.assertAlmostEqual(float(annual_vec[column]), float(annual_exact.iloc[0][column]), places=9)
        monthly_vec = monthly_all.loc[monthly_all.bootstrap_scenario_id == 0].sort_values("month")
        monthly_exact = monthly_exact.sort_values("month")
        for column in ("monthly_energy_MWh", "monthly_average_MW", "monthly_max_MW", "peak_day_average_MW", "peak_day_min_MW", "peak_day_load_factor", "monthly_imbalance_rate"):
            np.testing.assert_allclose(monthly_vec[column], monthly_exact[column], rtol=0, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
