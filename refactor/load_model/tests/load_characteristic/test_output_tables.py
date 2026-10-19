from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from workstreams_load_characteristic_model_v2.result_tables import (
    build_annual_monthly_peak_curve,
    build_industry_classification_table,
    build_target_month_energy_table,
)
class IndustryClassificationTableTests(unittest.TestCase):
    @staticmethod
    def mapping() -> pd.DataFrame:
        rows = []
        for index in range(25):
            is_traditional = index < 24
            rows.append(
                {
                    "industry": "充换电服务业" if index == 24 else f"行业{index + 1:02d}",
                    "broad_sector": "第三产业" if index == 24 else "第二产业",
                    "is_traditional": is_traditional,
                    "official_source_row": index + 1,
                    "official_label": f"官方行业{index + 1:02d}",
                    "notes": "",
                }
            )
        return pd.DataFrame(rows)

    def test_all_25_sample_industries_and_excluded_new_load_are_retained(self):
        mapping = self.mapping()
        energy = mapping[["industry"]].copy()
        energy["official_energy_2025"] = np.arange(1, 26, dtype=float)
        energy["official_energy_basis"] = "source_row_value"
        energy["unit"] = "万千瓦时"
        trace = pd.DataFrame(
            {
                "industry": mapping.loc[mapping.is_traditional, "industry"],
                "parameter_valid": True,
                "uses_own_parameters": True,
                "parameter_status": "own_industry_estimated",
            }
        )

        result = build_industry_classification_table(mapping, energy, trace)

        self.assertEqual(len(result), 25)
        self.assertIn("official_label", result)
        self.assertIn("unit", result)
        self.assertFalse(any(column.endswith(("_x", "_y")) for column in result.columns))
        charging = result.loc[result.industry == "充换电服务业"].iloc[0]
        self.assertFalse(bool(charging.is_traditional))
        self.assertFalse(bool(charging.uses_own_parameters))
        self.assertIn("后续新型负荷模块", charging.traditional_model_status)
        self.assertIn("修正组成", charging.traditional_model_status)
        self.assertEqual(
            charging.parameter_status,
            "future_new_load_module_not_shaped_now",
        )
        self.assertEqual(charging.unit, "万千瓦时")


class TargetMonthEnergyTableTests(unittest.TestCase):
    def test_month_targets_use_real_hours_and_close_to_each_annual_track(self):
        annual = pd.DataFrame(
            [
                {
                    "method": "方法A",
                    "scenario": "基准",
                    "year": 2025,
                    "P10_natural_MWh": 100.0,
                    "P50_natural_MWh": 120.0,
                    "P90_natural_MWh": 140.0,
                }
            ]
        )
        factors = pd.DataFrame(
            {
                "month": np.arange(1, 13),
                "factor_p50": np.linspace(0.85, 1.15, 12),
            }
        )

        result = build_target_month_energy_table(annual, factors)

        self.assertEqual(len(result), 12)
        self.assertEqual(int(result.hours.sum()), 8760)
        self.assertAlmostEqual(float(result.month_share.sum()), 1.0, places=14)
        self.assertEqual(set(result.planning_scenario), {"基准"})
        self.assertEqual(
            set(result.energy_scope),
            {"natural_load_excluding_plant_aux_line_loss_pumped_storage_charging"},
        )
        for track, expected in (("P10", 100.0), ("P50", 120.0), ("P90", 140.0)):
            self.assertAlmostEqual(
                float(result[f"{track}_target_month_energy_MWh"].sum()),
                expected,
                places=12,
            )


class AnnualMonthlyPeakCurveTests(unittest.TestCase):
    @staticmethod
    def inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
        deterministic_rows = []
        representative_rows = []
        for month in range(1, 13):
            deterministic_rows.append(
                {
                    "method": "方法A",
                    "planning_scenario": "基准",
                    "year": 2025,
                    "month": month,
                    "monthly_max_MW": 100.0 + month,
                    "monthly_peak_timestamp": pd.Timestamp(2025, month, 15, 18),
                }
            )
            for track, scenario_id, offset in (
                ("P10", 3, -10.0),
                ("P50", 7, 0.0),
                ("P90", 11, 10.0),
            ):
                representative_rows.append(
                    {
                        "method": "方法A",
                        "planning_scenario": "基准",
                        "year": 2025,
                        "month": month,
                        "monthly_max_MW": 100.0 + month + offset,
                        "monthly_peak_timestamp": pd.Timestamp(2025, month, 16, 19),
                        "representative_track": track,
                        "bootstrap_scenario_id": scenario_id,
                    }
                )
        return pd.DataFrame(deterministic_rows), pd.DataFrame(representative_rows)

    def test_builds_twelve_real_monthly_maxima_with_three_representative_tracks(self):
        deterministic, representative = self.inputs()

        result = build_annual_monthly_peak_curve(deterministic, representative)

        self.assertEqual(len(result), 12)
        self.assertEqual(result.month.tolist(), list(range(1, 13)))
        january = result.iloc[0]
        self.assertEqual(january.deterministic_max_MW, 101.0)
        self.assertEqual(
            [january.P10_real_max_MW, january.P50_real_max_MW, january.P90_real_max_MW],
            [91.0, 101.0, 111.0],
        )
        self.assertEqual(
            [january.P10_scenario_id, january.P50_scenario_id, january.P90_scenario_id],
            [3, 7, 11],
        )
        self.assertIn("真实代表情景", january.curve_definition)

    def test_missing_representative_track_hard_fails(self):
        deterministic, representative = self.inputs()
        representative = representative.loc[
            ~(
                representative.month.eq(12)
                & representative.representative_track.eq("P90")
            )
        ]

        with self.assertRaisesRegex(ValueError, "每个月.*P10/P50/P90"):
            build_annual_monthly_peak_curve(deterministic, representative)


class HourlyOutputTests(unittest.TestCase):
    def test_smoke_output_explicitly_falls_back_to_csv_gz_without_parquet_engine(self):
        # 延迟导入，避免结果表纯函数测试被尚在集成中的总流水线入口阻断。
        from workstreams_load_characteristic_model_v2.pipeline import _write_hourly_output

        hourly = pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-01-01", periods=3, freq="h"),
                "load_MW": [100.0, 101.0, 102.0],
            }
        )
        with TemporaryDirectory() as tmp, patch.object(
            pd.DataFrame,
            "to_parquet",
            side_effect=ImportError("no parquet engine"),
        ):
            path, output_format, warning = _write_hourly_output(hourly, Path(tmp))
            self.assertEqual(output_format, "csv.gz")
            self.assertTrue(path.name.endswith(".csv.gz"))
            self.assertTrue(path.exists())
            self.assertIn("正式全量必须安装Parquet引擎", str(warning))
            restored = pd.read_csv(path)
            self.assertEqual(len(restored), 3)

    def test_formal_hourly_output_hard_fails_without_parquet_engine(self):
        from workstreams_load_characteristic_model_v2.pipeline import _write_hourly_output

        hourly = pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-01-01", periods=3, freq="h"),
                "load_MW": [100.0, 101.0, 102.0],
            }
        )
        with TemporaryDirectory() as tmp, patch.object(
            pd.DataFrame,
            "to_parquet",
            side_effect=ImportError("no parquet engine"),
        ):
            with self.assertRaisesRegex(RuntimeError, "正式全量必须安装Parquet引擎"):
                _write_hourly_output(hourly, Path(tmp), require_parquet=True)
            self.assertFalse(any(Path(tmp).glob("*.csv.gz")))


if __name__ == "__main__":
    unittest.main()
