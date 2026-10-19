from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from workstreams_load_characteristic_model_v2 import pipeline


def test_fixed_current_method_switches_are_self_consistent() -> None:
    pipeline._assert_current_method_contract()


def test_smoke_wrapper_uses_final_version_and_expected_scale(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_run_model(
        years: object,
        n_scenarios: int,
        output_dir: str | Path,
        **kwargs: object,
    ) -> dict[str, Path]:
        captured.update(
            years=tuple(years),
            n_scenarios=n_scenarios,
            output_dir=Path(output_dir),
            **kwargs,
        )
        return {"output_dir": Path(output_dir)}

    monkeypatch.setattr(pipeline, "run_model", fake_run_model)

    result = pipeline.run_smoke(output_dir=tmp_path)

    assert result == {"output_dir": tmp_path}
    assert captured["years"] == pipeline.SMOKE_YEARS == (2025, 2030)
    assert captured["n_scenarios"] == pipeline.SMOKE_BOOTSTRAP_SCENARIOS == 20
    assert captured["method"] == "M1"
    assert captured["require_parquet"] is False
    assert captured["mode"] == "smoke"
    assert str(captured["run_label"]).startswith("3.5.0")


def test_smoke_cli_dispatches_the_final_manifest_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}

    def fake_run_pipeline(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"status": "success", "model_version": "3.5.0"}

    monkeypatch.setattr(pipeline, "run_pipeline", fake_run_pipeline)

    assert pipeline.main(["--smoke", "--method", "M1", "--output-dir", str(tmp_path)]) == 0

    assert captured["years"] == pipeline.SMOKE_YEARS
    assert captured["bootstrap_scenarios"] == 20
    assert captured["method"] == "M1"
    assert captured["mode"] == "smoke"
    assert captured["require_parquet"] is False
    assert str(captured["run_label"]).startswith("3.5.0")
    assert json.loads(capsys.readouterr().out)["status"] == "success"


def test_pipeline_guard_rejects_retired_correction_table_alias() -> None:
    tables = {
        "shape_future": pd.DataFrame(),
        "season_daytype_correction_component": pd.DataFrame(),
    }

    with pytest.raises(AssertionError, match="season_daytype_correction_component"):
        pipeline._assert_no_legacy_outputs(tables)


def test_pipeline_rejects_non_m1_before_reading_or_writing(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="只允许年度电量方法M1"):
        pipeline.run_pipeline(
            years=(2025,),
            output_dir=tmp_path,
            bootstrap_scenarios=1,
            method="方法A",
        )

    assert not any(tmp_path.iterdir())
