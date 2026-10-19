from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from workstreams_load_characteristic_model_v2.calendar_matched_bootstrap import (
    bootstrap_daily_multipliers_calendar_matched,
)
from workstreams_load_characteristic_model_v2.peak_month_representatives import (
    build_scenario_features_with_peak_month,
    peak_month_frequency,
    select_representative_scenarios_peak_banded,
)
from workstreams_load_characteristic_model_v2.representative_paths import FEATURES


class CalendarMatchedBootstrapTests(unittest.TestCase):
    @staticmethod
    def residuals() -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for history_year in (2019, 2020):
            for stamp in pd.date_range(
                f"{history_year}-01-01",
                f"{history_year}-12-31",
                freq="D",
            ):
                if stamp.day == 1:
                    day_class = "法定节假日"
                    residual = 2.0  # 用于检验冻结和日型匹配两种快速口径。
                elif stamp.weekday() >= 5:
                    day_class = "普通周末"
                    residual = 0.03 if history_year == 2019 else -0.03
                else:
                    day_class = "工作日"
                    residual = 0.06 if history_year == 2019 else -0.06
                rows.append(
                    {
                        "history_year": history_year,
                        "month": stamp.month,
                        "date": stamp,
                        "residual_day_class": day_class,
                        "residual": residual,
                    }
                )
        return pd.DataFrame(rows)

    def test_uses_one_balanced_source_year_per_scenario_and_matches_day_class(self):
        dates = pd.date_range("2030-01-01", "2030-12-31", freq="D")
        baseline = pd.Series(100.0 + (np.arange(len(dates)) % 5), index=dates)

        multipliers, checks = bootstrap_daily_multipliers_calendar_matched(
            self.residuals(),
            baseline,
            2030,
            n_scenarios=10,
        )

        source_years_per_scenario = multipliers.groupby(
            "scenario_id"
        ).source_history_year.nunique()
        self.assertTrue(source_years_per_scenario.eq(1).all())
        source_year_counts = (
            multipliers[["scenario_id", "source_history_year"]]
            .drop_duplicates()
            .source_history_year.value_counts()
        )
        self.assertEqual(source_year_counts.to_dict(), {2019: 5, 2020: 5})

        nonholiday = multipliers.loc[~multipliers.holiday_residual_frozen]
        self.assertTrue(
            nonholiday.target_day_class.eq(
                nonholiday.residual_source_day_class
            ).all()
        )
        self.assertLessEqual(float(checks.relative_error.max()), 1e-12)

    def test_holiday_residuals_are_frozen_at_zero_and_multiplier_one(self):
        dates = pd.date_range("2030-01-01", "2030-12-31", freq="D")
        baseline = pd.Series(100.0, index=dates)
        multipliers, _ = bootstrap_daily_multipliers_calendar_matched(
            self.residuals(),
            baseline,
            2030,
            n_scenarios=4,
        )

        holidays = multipliers.loc[
            multipliers.target_day_class.eq("法定节假日")
        ]
        self.assertFalse(holidays.empty)
        np.testing.assert_allclose(
            holidays.raw_residual, 0.0, rtol=0.0, atol=0.0
        )
        np.testing.assert_allclose(
            holidays.daily_multiplier, 1.0, rtol=0.0, atol=1e-15
        )
        self.assertTrue(holidays.holiday_residual_frozen.all())

    def test_holiday_matched_mode_never_maps_holiday_residual_to_workday(self):
        dates = pd.date_range("2030-01-01", "2030-12-31", freq="D")
        baseline = pd.Series(100.0, index=dates)
        multipliers, checks = bootstrap_daily_multipliers_calendar_matched(
            self.residuals(),
            baseline,
            2030,
            n_scenarios=4,
            freeze_holidays=False,
        )

        holidays = multipliers.loc[
            multipliers.target_day_class.eq("法定节假日")
        ]
        self.assertFalse(holidays.empty)
        self.assertFalse(holidays.holiday_residual_frozen.any())
        self.assertTrue(
            holidays.residual_source_day_class.eq("法定节假日").all()
        )
        self.assertTrue((holidays.raw_residual > 0).all())
        self.assertTrue((holidays.daily_multiplier != 1.0).any())
        self.assertLessEqual(float(checks.relative_error.max()), 1e-12)

    def test_incomplete_source_year_never_borrows_another_years_residuals(self):
        dates = pd.date_range("2030-01-01", "2030-12-31", freq="D")
        baseline = pd.Series(100.0, index=dates)
        residuals = self.residuals()

        missing_month = residuals.loc[
            ~(
                residuals.history_year.eq(2019)
                & residuals.month.eq(3)
                & residuals.residual_day_class.eq("工作日")
            )
        ].copy()
        multipliers, _ = bootstrap_daily_multipliers_calendar_matched(
            missing_month,
            baseline,
            2030,
            n_scenarios=2,
            freeze_holidays=False,
        )
        fallback = multipliers.loc[
            multipliers.source_history_year.eq(2019)
            & multipliers.month.eq(3)
            & multipliers.target_day_class.eq("工作日")
        ]
        self.assertFalse(fallback.empty)
        self.assertTrue(
            fallback.sampling_pool_source.eq(
                "same_history_year_day_class_all_months_fallback"
            ).all()
        )
        np.testing.assert_allclose(fallback.raw_residual, 0.06, rtol=0.0, atol=0.0)

        missing_month_in_every_year = residuals.loc[
            ~(
                residuals.month.eq(3)
                & residuals.residual_day_class.eq("工作日")
            )
        ].copy()
        all_fallback, _ = bootstrap_daily_multipliers_calendar_matched(
            missing_month_in_every_year,
            baseline,
            2030,
            n_scenarios=2,
            freeze_holidays=False,
        )
        march_workdays = all_fallback.loc[
            all_fallback.month.eq(3)
            & all_fallback.target_day_class.eq("工作日")
        ]
        self.assertTrue(
            march_workdays.sampling_pool_source.eq(
                "same_history_year_day_class_all_months_fallback"
            ).all()
        )
        expected_residual = march_workdays.source_history_year.map(
            {2019: 0.06, 2020: -0.06}
        )
        np.testing.assert_allclose(
            march_workdays.raw_residual,
            expected_residual,
            rtol=0.0,
            atol=0.0,
        )

        missing_year_class = residuals.loc[
            ~(
                residuals.history_year.eq(2019)
                & residuals.residual_day_class.eq("工作日")
            )
        ].copy()
        with self.assertRaisesRegex(
            ValueError,
            r"source_history_year=2019.*工作日.*同年残差",
        ):
            bootstrap_daily_multipliers_calendar_matched(
                missing_year_class,
                baseline,
                2030,
                n_scenarios=2,
                freeze_holidays=False,
            )


class RepresentativeSelectionTests(unittest.TestCase):
    @staticmethod
    def flat_monthly_features(n_scenarios: int) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for scenario_id in range(n_scenarios):
            for month in range(1, 13):
                row: dict[str, object] = {
                    "scenario_id": scenario_id,
                    "month": month,
                }
                row.update({feature: 0.0 for feature in FEATURES})
                rows.append(row)
        return pd.DataFrame(rows)

    def test_p50_is_selected_inside_45_to_55_percent_peak_band(self):
        n_scenarios = 20
        annual = pd.DataFrame(
            {
                "scenario_id": np.arange(n_scenarios),
                "annual_peak_MW": 100.0 + np.arange(n_scenarios),
            }
        )
        selected, ranking = select_representative_scenarios_peak_banded(
            self.flat_monthly_features(n_scenarios),
            annual,
        )
        selected_row = ranking.loc[
            ranking.scenario_id.eq(selected["P50"])
        ].iloc[0]
        self.assertGreaterEqual(
            float(selected_row.peak_empirical_quantile), 0.45
        )
        self.assertLessEqual(
            float(selected_row.peak_empirical_quantile), 0.55
        )
        self.assertNotEqual(selected["P50"], 0)

    def test_scenario_features_record_peak_month_and_frequency(self):
        dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")
        baseline_energy = pd.Series(100.0, index=dates)
        baseline_peak = pd.Series(10.0, index=dates)
        rows: list[dict[str, object]] = []
        for scenario_id, peak_date in (
            (0, pd.Timestamp("2025-01-10")),
            (1, pd.Timestamp("2025-08-10")),
            (2, pd.Timestamp("2025-08-20")),
        ):
            for stamp in dates:
                rows.append(
                    {
                        "scenario_id": scenario_id,
                        "date": stamp,
                        "daily_multiplier": (
                            2.0 if stamp == peak_date else 1.0
                        ),
                    }
                )
        multipliers = pd.DataFrame(rows)

        _, annual = build_scenario_features_with_peak_month(
            multipliers,
            baseline_energy,
            baseline_peak,
        )
        observed = annual.set_index("scenario_id").annual_peak_month.to_dict()
        self.assertEqual(observed, {0: 1, 1: 8, 2: 8})

        frequency = peak_month_frequency(annual)
        shares = frequency.set_index("peak_month").scenario_share.to_dict()
        self.assertAlmostEqual(shares[1], 1.0 / 3.0)
        self.assertAlmostEqual(shares[8], 2.0 / 3.0)
        self.assertAlmostEqual(float(frequency.scenario_share.sum()), 1.0)


if __name__ == "__main__":
    unittest.main()
