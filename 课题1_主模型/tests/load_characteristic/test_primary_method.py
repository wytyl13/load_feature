from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from workstreams_load_characteristic_model_v2.pipeline import (
    _assert_method_consistency,
    run_model,
)
from workstreams_load_characteristic_model_v2.prepare_workbook_sources import (
    WorkbookSourceError,
    _validate_method_contract,
)


class PrimaryAnnualEnergyMethodTests(unittest.TestCase):
    def test_run_model_forwards_explicit_method(self) -> None:
        with TemporaryDirectory() as tmp, patch(
            "workstreams_load_characteristic_model_v2.pipeline.run_pipeline"
        ) as mocked:
            mocked.return_value = {
                "hourly_output": {"file": str(Path(tmp) / "hourly.parquet")},
                "metadata_file": str(Path(tmp) / "run_metadata.json"),
                "status_file": str(Path(tmp) / "运行状态.txt"),
            }
            run_model(
                years=(2030,),
                n_scenarios=2,
                output_dir=tmp,
                run_label="test",
                method="M1",
            )
        self.assertEqual(mocked.call_args.kwargs["method"], "M1")

    def test_workbook_contract_rejects_method_mismatch(self) -> None:
        metadata = {"method": "M1"}
        tables = {
            "annual_energy_interval": pd.DataFrame({"method": ["方法A"]}),
            "industry_inventory": pd.DataFrame({"industry": ["农业"]}),
        }
        with self.assertRaisesRegex(WorkbookSourceError, "M1"):
            _validate_method_contract(metadata, tables)

    def test_workbook_contract_accepts_m1_tables(self) -> None:
        metadata = {"method": "M1"}
        tables = {
            "annual_energy_interval": pd.DataFrame({"method": ["M1", "M1"]}),
            "industry_inventory": pd.DataFrame({"industry": ["农业"]}),
        }
        _validate_method_contract(metadata, tables)

    def test_formal_table_gate_rejects_non_primary_method(self) -> None:
        tables = {
            "annual_energy_interval": pd.DataFrame({"method": ["M1"]}),
            "representative_annual_metrics": pd.DataFrame(
                {"method": ["方法A"]}
            ),
        }
        with self.assertRaisesRegex(AssertionError, "representative_annual_metrics"):
            _assert_method_consistency(tables, "M1")


if __name__ == "__main__":
    unittest.main()
