from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from workstreams_load_characteristic_model_v2 import MODEL_VERSION
from workstreams_load_characteristic_model_v2.annual_energy import (
    ANNUAL_ENERGY_METHOD,
    build_industry_annual_energy,
)
from workstreams_load_characteristic_model_v2.baseline_shapes import weighted_medoid
from workstreams_load_characteristic_model_v2.calendar_utils import calendar_frame, classify_date
from workstreams_load_characteristic_model_v2.monthly_envelope import build_monthly_envelope, target_month_shares
from workstreams_load_characteristic_model_v2.quality import HOUR_COLS, parse_user_sheet_name
from workstreams_load_characteristic_model_v2 import result_tables, shape_transform


class ActiveContractTests(unittest.TestCase):
    def test_major_version_matches_fine_industry_component_model(self):
        self.assertEqual(MODEL_VERSION, "3.5.0")

    def test_m1_is_the_primary_annual_energy_method(self):
        self.assertEqual(ANNUAL_ENERGY_METHOD, "M1")

    def test_shape_transform_public_api_has_no_retired_continuous_mechanisms(self):
        retired = ("gamma", "delta", "shift", "centroid")
        self.assertFalse(
            any(
                token in public_name.lower()
                for public_name in shape_transform.__all__
                for token in retired
            )
        )

    def test_monthly_peak_curve_replaces_load_duration_output(self):
        self.assertTrue(hasattr(result_tables, "build_annual_monthly_peak_curve"))
        self.assertFalse(hasattr(result_tables, "build_load_duration_curve"))


class CalendarTests(unittest.TestCase):
    def test_planning_leap_year_and_source(self):
        frame = calendar_frame(2028)
        self.assertEqual(len(frame), 366)
        self.assertEqual(set(frame.calendar_source), {"规划日历代理"})

    def test_official_adjusted_workday(self):
        detailed, group, source = classify_date(pd.Timestamp("2025-01-26"))
        self.assertEqual((detailed, group, source), ("调休工作日", "工作日", "已公布官方日历"))

    def test_sheet_name_trim(self):
        self.assertEqual(parse_user_sheet_name(" 第一产业-农业-用户1 "), ("第一产业", "农业", "用户1"))


class BaselineTests(unittest.TestCase):
    def test_weighted_medoid_is_observed_curve(self):
        curves = np.array([[1.0] + [0.0] * 23, [0.0, 1.0] + [0.0] * 22, [0.0, 1.0] + [0.0] * 22])
        curve, selected, _ = weighted_medoid(curves, np.array([1.0, 2.0, 1.0]))
        self.assertEqual(selected, 1)
        np.testing.assert_allclose(curve, curves[1])


class EnvelopeTests(unittest.TestCase):
    @staticmethod
    def synthetic_history() -> pd.DataFrame:
        rows = []
        for year in (2019, 2020, 2021):
            for stamp in pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D"):
                level = 100.0 * (1.0 + stamp.month / 100.0)
                rows.append({"date": stamp, **{column: level for column in HOUR_COLS}})
        return pd.DataFrame(rows)

    def test_month_shares_include_real_month_hours(self):
        _, factors = build_monthly_envelope(self.synthetic_history())
        shares = target_month_shares(2028, factors)
        self.assertAlmostEqual(float(shares.month_share.sum()), 1.0, places=14)
        feb_hours = int(shares.loc[shares.month == 2, "hours"].iloc[0])
        self.assertEqual(feb_hours, 29 * 24)


class AnnualAllocationTests(unittest.TestCase):
    def test_24_industries_plus_correction_close_without_sector_allocation(self):
        annual = pd.DataFrame([
            {
                "method": "方法A", "scenario": "基准", "year": 2025,
                "P10_natural_MWh": 80.0,
                "P50_natural_MWh": 100.0,
                "P90_natural_MWh": 120.0,
            }
        ])
        industries = pd.DataFrame(
            {
                "industry": [f"行业{i:02d}" for i in range(24)],
                "broad_sector": ["第一产业"] * 24,
                "is_traditional": True,
                "official_energy_2025": 1.0,
            }
        )
        evolution = pd.DataFrame(
            {
                "industry": industries.industry,
                "base_year": 2025,
                "target_year": 2040,
                "target_multiplier": 1.0,
                "rationale": "测试",
            }
        )
        result, closure = build_industry_annual_energy(
            annual,
            industries,
            official_natural_total_2025=100.0,
            evolution=evolution,
        )
        self.assertEqual(len(result), 25 * 3)
        self.assertTrue(
            np.isclose(
                result.groupby("annual_energy_quantile").electricity_share.sum(), 1.0
            ).all()
        )
        self.assertAlmostEqual(
            float(
                result.loc[result.annual_energy_quantile.eq("P50"), "annual_energy_MWh"].sum()
            ),
            100.0,
        )
        correction = result.loc[
            result.component_type.eq("修正组成")
            & result.annual_energy_quantile.eq("P50")
        ].iloc[0]
        self.assertAlmostEqual(float(correction.electricity_share), 0.76)
        self.assertLessEqual(float(closure.relative_error.max()), 1e-12)


if __name__ == "__main__":
    unittest.main()
