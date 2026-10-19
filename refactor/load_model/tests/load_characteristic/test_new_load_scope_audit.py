from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from workstreams_load_characteristic_model_v2.annual_energy import (
    build_industry_annual_energy,
    load_upstream_annual_energy,
    load_upstream_scope_bridge,
)


class NaturalLoadScopeTests(unittest.TestCase):
    @staticmethod
    def _payload() -> dict[str, object]:
        annual = [
            {
                "method": "M1",
                "scenario": scenario,
                "year": 2025,
                "P10": 100.0,
                "P50": 110.0,
                "P90": 120.0,
                "P10_natural_total": 80.0,
                "P50_natural_total": 90.0,
                "P90_natural_total": 100.0,
            }
            for scenario in ("低碳", "基准", "高电气化")
        ]
        bridge = []
        for row in annual:
            for quantile in ("P10", "P50", "P90"):
                full = float(row[quantile])
                natural = float(row[f"{quantile}_natural_total"])
                bridge.append(
                    {
                        "method": row["method"],
                        "scenario": row["scenario"],
                        "year": row["year"],
                        "quantile": quantile,
                        "full_society_electricity": full,
                        "special_deduction": full - natural,
                        "natural_load_electricity": natural,
                        "deduction_ratio": (full - natural) / full,
                    }
                )
        return {"annual_electricity": annual, "scope_bridge": bridge}

    def test_upstream_reader_uses_natural_total_without_second_deduction(self) -> None:
        payload = self._payload()
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "annual.json"
            source.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            frame = load_upstream_annual_energy(source, years=(2025,))
        self.assertEqual(set(frame.method), {"M1"})
        self.assertTrue((frame.P50_full_MWh == 110.0 * 1e5).all())
        self.assertTrue((frame.P50_natural_MWh == 90.0 * 1e5).all())
        self.assertTrue((frame.P50_special_deduction_MWh == 20.0 * 1e5).all())
        self.assertTrue((~frame.second_deduction_applied).all())
        self.assertNotIn("P50_total_MWh", frame.columns)

    def test_scope_bridge_is_unique_complete_and_natural_load_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "annual.json"
            source.write_text(
                json.dumps(self._payload(), ensure_ascii=False), encoding="utf-8"
            )
            bridge = load_upstream_scope_bridge(source, years=(2025,))
        self.assertEqual(len(bridge), 3 * 3)
        self.assertFalse(
            bridge.duplicated(["method", "planning_scenario", "year", "quantile"]).any()
        )
        self.assertTrue(bridge.annual_special_deduction_applied.all())
        self.assertTrue((~bridge.hourly_special_deduction_applied).all())
        self.assertTrue((~bridge.special_items_in_correction_component).all())
        self.assertTrue(
            (
                bridge.Q_full_MWh
                - bridge.D_MWh
                - bridge.Q_natural_MWh
            ).abs().le(1e-6).all()
        )

    def test_missing_natural_total_hard_fails(self) -> None:
        payload = self._payload()
        del payload["annual_electricity"][0]["P50_natural_total"]
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "annual.json"
            source.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "natural"):
                load_upstream_annual_energy(source, years=(2025,))

    def test_uncovered_and_new_load_energy_remains_in_correction_component(self) -> None:
        annual = pd.DataFrame(
            [
                {
                    "method": "方法A",
                    "scenario": "基准",
                    "year": 2025,
                    "P10_natural_MWh": 80.0,
                    "P50_natural_MWh": 90.0,
                    "P90_natural_MWh": 100.0,
                }
            ]
        )
        industries = pd.DataFrame(
            {
                "industry": [f"行业{i:02d}" for i in range(24)],
                "broad_sector": ["第二产业"] * 24,
                "is_traditional": True,
                "official_energy_2025": 2.0,
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
        components, _ = build_industry_annual_energy(
            annual,
            industries,
            official_natural_total_2025=90.0,
            evolution=evolution,
        )
        self.assertEqual(len(components), 25 * 3)
        self.assertTrue(
            components.groupby("annual_energy_quantile").component.nunique().eq(25).all()
        )
        correction = components.loc[
            components.component_type.eq("修正组成")
            & components.annual_energy_quantile.eq("P50")
        ].iloc[0]
        self.assertAlmostEqual(float(correction.electricity_share), 42.0 / 90.0)
        self.assertAlmostEqual(float(correction.annual_energy_MWh), 42.0)
        self.assertIn("新型负荷", str(correction.rationale))
        self.assertFalse(bool(correction.special_items_in_correction_component))
        self.assertFalse(bool(correction.second_deduction_applied))
        self.assertEqual(correction.natural_total_source_field, "P50_natural_total")


if __name__ == "__main__":
    unittest.main()
