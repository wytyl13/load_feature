"""自然负荷权重下的24行业组成形状测试。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from workstreams_load_characteristic_model_v2.component_shapes import (
    NATURAL_LOAD_SCOPE,
    build_base_year_industry_components,
    build_future_industry_components,
)


SEASONS = ("春季", "夏季", "秋季", "冬季")
DAY_GROUPS = ("工作日", "非工作日")
SHAPE_COLS = tuple(f"p{hour:02d}" for hour in range(24))


def _baseline_and_parameters(year: int = 2030) -> tuple[pd.DataFrame, pd.DataFrame]:
    shape = np.arange(1.0, 25.0)
    shape /= shape.sum()
    shares = (
        float(shape[:6].sum()),
        float(shape[6:12].sum()),
        float(shape[12:17].sum()),
        float(shape[17:].sum()),
    )
    baseline_rows: list[dict[str, object]] = []
    parameter_rows: list[dict[str, object]] = []
    for index in range(24):
        industry = f"行业{index + 1:02d}"
        for season in SEASONS:
            for day_group in DAY_GROUPS:
                baseline_rows.append(
                    {
                        "industry": industry,
                        "broad_sector": "第二产业",
                        "season": season,
                        "day_group": day_group,
                        **dict(zip(SHAPE_COLS, shape)),
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
                        "applied_share_night": shares[0],
                        "applied_share_morning": shares[1],
                        "applied_share_midday": shares[2],
                        "applied_share_evening": shares[3],
                        "peak_valley_target_range_limited": (
                            float(shape.max() - shape.min()) / float(shape.max())
                        ),
                        "flat_curve": False,
                        "flat_borderline": False,
                        "peak_valley_delta_abs": 0.10,
                        "peak_valley_delta_band": "within_gate_band",
                        "peak_valley_evolution_enabled": True,
                        "peak_valley_gate_reason": "enabled",
                        "peak_valley_gate_policy": (
                            "trend_eligible_and_nonflat_and_abs_delta_within_closed_"
                            "interval_0.065_0.25"
                        ),
                        "night_share": shares[0],
                        "morning_share": shares[1],
                        "midday_share": shares[2],
                        "evening_share": shares[3],
                    }
                )
    return pd.DataFrame(baseline_rows), pd.DataFrame(parameter_rows)


class ComponentShapeScopeTests(unittest.TestCase):
    def test_base_year_weights_use_official_natural_load_denominator(self):
        baseline, _ = _baseline_and_parameters()
        industry_energy = pd.DataFrame(
            {
                "industry": [f"行业{index + 1:02d}" for index in range(24)],
                "official_energy": [2.0] * 24,
            }
        )

        result = build_base_year_industry_components(
            baseline,
            industry_energy,
            official_natural_total_energy=80.0,
        )

        weights = result[["industry", "share"]].drop_duplicates("industry")
        self.assertTrue(np.allclose(weights.share, 2.0 / 80.0))
        self.assertAlmostEqual(float(weights.share.sum()), 0.6, places=12)
        self.assertEqual(set(result.weight_denominator_scope), {NATURAL_LOAD_SCOPE})

    def test_future_shapes_filter_to_p50_before_context_uniqueness_checks(self):
        baseline, parameters = _baseline_and_parameters()
        rows: list[dict[str, object]] = []
        for quantile, share in (("P10", 0.01), ("P50", 0.02), ("P90", 0.03)):
            for index in range(24):
                industry = f"行业{index + 1:02d}"
                rows.append(
                    {
                        "method": "M1",
                        "planning_scenario": "基准",
                        "year": 2030,
                        "industry": industry,
                        "component_type": "传统细分行业",
                        "broad_sector": "第二产业",
                        "annual_energy_quantile": quantile,
                        "electricity_share": share,
                    }
                )

        result = build_future_industry_components(
            baseline,
            parameters,
            pd.DataFrame(rows),
        )

        self.assertEqual(set(result.annual_energy_quantile), {"P50"})
        self.assertTrue(np.allclose(result.share, 0.02))
        self.assertEqual(set(result.weight_denominator_scope), {NATURAL_LOAD_SCOPE})
        self.assertEqual(set(result.H), {8})
        self.assertEqual(set(result.phi), {0.8})
        # The final contract always materializes the canonical three day types;
        # a legacy merged non-workday input is expanded to weekend + holiday.
        self.assertEqual(len(result), 24 * 4 * 3 * 24)
        self.assertFalse(
            result.duplicated(
                [
                    "method",
                    "planning_scenario",
                    "year",
                    "industry",
                    "season",
                    "day_group",
                    "hour",
                ]
            ).any()
        )

    def test_future_shapes_reject_parameters_without_second_stage_target(self):
        baseline, parameters = _baseline_and_parameters()
        parameters = parameters.drop(columns="peak_valley_target_range_limited")
        annual_rows = [
            {
                "method": "M1",
                "planning_scenario": "基准",
                "year": 2030,
                "industry": f"行业{index + 1:02d}",
                "component_type": "传统细分行业",
                "broad_sector": "第二产业",
                "annual_energy_quantile": "P50",
                "electricity_share": 1.0 / 24.0,
            }
            for index in range(24)
        ]

        with self.assertRaisesRegex(ValueError, "peak_valley_target_range_limited"):
            build_future_industry_components(
                baseline,
                parameters,
                pd.DataFrame(annual_rows),
            )

    def test_future_shapes_require_explicit_peak_valley_gate_fields(self):
        baseline, parameters = _baseline_and_parameters()
        parameters = parameters.drop(columns="peak_valley_evolution_enabled")
        annual_rows = [
            {
                "method": "M1",
                "planning_scenario": "基准",
                "year": 2030,
                "industry": f"行业{index + 1:02d}",
                "component_type": "传统细分行业",
                "broad_sector": "第二产业",
                "annual_energy_quantile": "P50",
                "electricity_share": 1.0 / 24.0,
            }
            for index in range(24)
        ]

        with self.assertRaisesRegex(ValueError, "peak_valley_evolution_enabled"):
            build_future_industry_components(
                baseline,
                parameters,
                pd.DataFrame(annual_rows),
            )


if __name__ == "__main__":
    unittest.main()
