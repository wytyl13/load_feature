"""季节×日型第25个电量组成的单元测试（合成数据，不依赖真实输入）。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from workstreams_load_characteristic_model_v2.correction_curve import (
    build_season_daytype_correction_component,
    compose_with_fixed_correction_shape,
)
from workstreams_load_characteristic_model_v2.metrics import season_daytype_typical_shapes


SEASONS = ("春季", "夏季", "秋季", "冬季")
DAY_GROUPS = ("工作日", "非工作日")


def _component_inputs(
    *, industry_share_total: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    industry_shape = np.arange(1.0, 25.0)
    industry_shape /= industry_shape.sum()
    correction_shape = industry_shape[::-1].copy()
    province_shape = (
        industry_share_total * industry_shape
        + (1.0 - industry_share_total) * correction_shape
    )
    province_rows: list[dict[str, object]] = []
    industry_rows: list[dict[str, object]] = []
    for season in SEASONS:
        for day_group in DAY_GROUPS:
            for hour, value in enumerate(province_shape):
                province_rows.append(
                    {
                        "season": season,
                        "day_group": day_group,
                        "hour": hour,
                        "province_typical_share": value,
                    }
                )
            for industry_index in range(24):
                for hour, value in enumerate(industry_shape):
                    industry_rows.append(
                        {
                            "industry": f"行业{industry_index + 1:02d}",
                            "season": season,
                            "day_group": day_group,
                            "hour": hour,
                            "share": industry_share_total / 24.0,
                            "normalized_load": value,
                        }
                    )
    return (
        pd.DataFrame(province_rows),
        pd.DataFrame(industry_rows),
        industry_shape,
        correction_shape,
    )


class CorrectionComponentTests(unittest.TestCase):
    def test_recovers_25th_component_and_reconstructs_all_eight_cells(self):
        province, industries, _, expected_correction_shape = _component_inputs()

        result = build_season_daytype_correction_component(province, industries)

        self.assertEqual(len(result), 4 * 2 * 24)
        self.assertEqual(set(result.component), {"correction_component"})
        self.assertTrue(result.correction_scope.str.contains("三项排除电量不进入").all())
        self.assertEqual(
            set(result.historical_shape_scope),
            {"dispatch_proxy_not_hourly_deducted"},
        )
        self.assertEqual(
            set(result.correction_shape_status),
            {"proxy_derived_from_dispatch_shape"},
        )
        self.assertTrue((~result.special_items_in_correction_component).all())
        np.testing.assert_allclose(
            result.province_shape_proxy, result.province_typical_share
        )
        np.testing.assert_allclose(
            result.C24_contribution, result.industry_contribution_share
        )
        np.testing.assert_allclose(
            result.R_contribution, result.correction_contribution_share
        )
        np.testing.assert_allclose(
            result.correction_component_shape_proxy,
            result.correction_normalized_shape,
        )
        np.testing.assert_allclose(result.omega_R, result.correction_component_share)
        self.assertTrue(np.allclose(result.correction_component_share, 0.2))
        self.assertEqual(set(result.correction_day_energy_factor), {1.0})
        self.assertEqual(
            set(result.day_energy_factor_status),
            {"fixed_1_not_identifiable_from_normalized_shapes"},
        )
        for _, cell in result.groupby(["season", "day_group"]):
            np.testing.assert_allclose(
                cell.sort_values("hour").correction_normalized_shape,
                expected_correction_shape,
                rtol=0,
                atol=1e-12,
            )
            self.assertAlmostEqual(float(cell.correction_normalized_shape.sum()), 1.0, places=12)
            self.assertAlmostEqual(float(cell.reconstructed_share.sum()), 1.0, places=12)
            self.assertLessEqual(float(cell.reconstruction_max_abs_error.max()), 1e-12)

    def test_future_uses_fixed_shape_but_recomputes_residual_energy_share(self):
        province, baseline_industries, industry_shape, expected_correction_shape = _component_inputs()
        correction = build_season_daytype_correction_component(
            province, baseline_industries
        )
        future_industries = baseline_industries.copy()
        future_industries["share"] = 0.7 / 24.0

        result = compose_with_fixed_correction_shape(future_industries, correction)

        expected = 0.7 * industry_shape + 0.3 * expected_correction_shape
        for _, cell in result.groupby(["season", "day_group"]):
            cell = cell.sort_values("hour")
            self.assertTrue(np.allclose(cell.correction_component_share, 0.3))
            np.testing.assert_allclose(cell.correction_normalized_shape, expected_correction_shape)
            np.testing.assert_allclose(cell.total_normalized_load, expected)
            self.assertAlmostEqual(float(cell.total_normalized_load.sum()), 1.0, places=12)

    def test_negative_residual_hard_fails_with_cell_and_hour(self):
        province, industries, _, _ = _component_inputs()
        target = province.season.eq("春季") & province.day_group.eq("工作日")
        province.loc[target & province.hour.eq(0), "province_typical_share"] -= 0.017
        province.loc[target & province.hour.eq(1), "province_typical_share"] += 0.017

        with self.assertRaisesRegex(
            ValueError, "春季.*工作日.*hour=0|hour=0.*春季.*工作日"
        ):
            build_season_daytype_correction_component(province, industries)

    def test_nonpositive_residual_share_hard_fails(self):
        province, industries, _, _ = _component_inputs(industry_share_total=0.8)
        industries["share"] = 1.0 / 24.0

        with self.assertRaisesRegex(ValueError, "修正项电量份额.*正"):
            build_season_daytype_correction_component(province, industries)

    def test_fixed_correction_rejects_special_items_in_residual(self):
        province, industries, _, _ = _component_inputs()
        correction = build_season_daytype_correction_component(province, industries)
        correction["special_items_in_correction_component"] = True

        with self.assertRaisesRegex(ValueError, "厂用电.*线损.*抽蓄"):
            compose_with_fixed_correction_shape(industries, correction)


class TypicalShapeTests(unittest.TestCase):
    def test_hourly_wrapper_uses_arithmetic_mean_and_normalizes_each_cell(self):
        timestamps = pd.date_range("2025-01-01", "2025-12-31 23:00", freq="h")
        hourly = pd.DataFrame(
            {
                "timestamp": timestamps,
                "load_MW": 100.0 + timestamps.hour.to_numpy(float),
            }
        )

        result = season_daytype_typical_shapes(hourly)

        self.assertEqual(len(result), 4 * 2 * 24)
        expected = np.arange(100.0, 124.0)
        expected /= expected.sum()
        for _, cell in result.groupby(["season", "day_group"]):
            cell = cell.sort_values("hour")
            np.testing.assert_allclose(cell.typical_load_MW, np.arange(100.0, 124.0))
            np.testing.assert_allclose(cell.normalized_load, expected)
            self.assertGreater(int(cell.valid_days.min()), 0)


if __name__ == "__main__":
    unittest.main()
