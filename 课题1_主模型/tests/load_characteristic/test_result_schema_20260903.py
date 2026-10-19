from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from workstreams_load_characteristic_model_v2.prepare_workbook_sources import (
    TABLE_CONTRACTS as WORKBOOK_TABLE_CONTRACTS,
)
from workstreams_load_characteristic_model_v2.reporting_outputs import (
    REPORT_METHOD_STATEMENTS,
    load_reporting_contract,
)
from workstreams_load_characteristic_model_v2.result_tables import (
    FORBIDDEN_SHAPE_FUTURE_COLUMNS,
    REQUIRED_FORMAL_TABLES,
    RUN_MANIFEST_CONTRACT,
    save_result_tables,
    validate_result_tables_contract,
    validate_run_manifest,
)


def test_run_manifest_contract_includes_conditional_peak_valley_priority() -> None:
    assert RUN_MANIFEST_CONTRACT["peak_valley_target_priority"] == (
        "final_when_gate_enabled_otherwise_four_window_final"
    )


def _result_tables() -> dict[str, pd.DataFrame]:
    tables = {name: pd.DataFrame({"audit": []}) for name in REQUIRED_FORMAL_TABLES}
    tables["shape_future"] = pd.DataFrame(
        {
            "industry": ["测试行业"],
            "season": ["夏季"],
            "final_day_type": ["工作日"],
            "year": [2030],
            "H": [8],
            "phi": [0.8],
            "peak_valley_phi": [0.8],
            "base_source_year": [2023],
            "trend_eligible": [True],
            "trend_status": ["paired_2022_2023_industry_medoids"],
            "flat_curve": [False],
            "flat_borderline": [False],
            "peak_valley_delta_abs": [0.10],
            "peak_valley_delta_band": ["within_gate_band"],
            "peak_valley_evolution_enabled": [True],
            "peak_valley_gate_reason": ["enabled"],
            "peak_valley_gate_policy": [
                "trend_eligible_and_nonflat_and_abs_delta_within_closed_interval_0.065_0.25"
            ],
            "a_requested": [0.10],
            "a_max": [0.08],
            "a_applied": [0.08],
            "requested_share_night": [0.20],
            "requested_share_morning": [0.25],
            "requested_share_midday": [0.30],
            "requested_share_evening": [0.25],
            "applied_share_night": [0.21],
            "applied_share_morning": [0.24],
            "applied_share_midday": [0.30],
            "applied_share_evening": [0.25],
            "window_boundary_hit": [True],
            "peak_valley_cumulative_damping": [0.5],
            "peak_valley_target_requested": [0.25],
            "peak_valley_target_range_limited": [0.25],
            "peak_valley_target_applied": [0.25],
            "peak_valley_target_clipped": [False],
            "peak_valley_adjustment_status": ["mean_centered_scaling_applied"],
            "peak_valley_rate_after_four_window": [0.30],
            "peak_valley_lambda": [0.80],
            "peak_valley_rate_final": [0.25],
            "four_window_share_intermediate_night": [0.21],
            "four_window_share_intermediate_morning": [0.24],
            "four_window_share_intermediate_midday": [0.30],
            "four_window_share_intermediate_evening": [0.25],
            "four_window_share_final_night": [0.20],
            "four_window_share_final_morning": [0.24],
            "four_window_share_final_midday": [0.31],
            "four_window_share_final_evening": [0.25],
            "four_window_share_deviation_night": [-0.01],
            "four_window_share_deviation_morning": [0.0],
            "four_window_share_deviation_midday": [0.01],
            "four_window_share_deviation_evening": [0.0],
            "shape_extrapolation_method": [
                "four_window_then_local_smoothing_then_gated_peak_valley_mean_scaling"
            ],
            "four_window_final_share_policy": [
                "exact_if_peak_valley_skipped_diagnostic_if_applied"
            ],
            "peak_valley_extrapolation_enabled": [True],
            "future_peak_valley_rate": [0.25],
            "future_evening_peak_hour": [20],
            "future_raw_85pct_hours": ["[18,19,20]"],
            "future_effective_control_hours": ["[18,19,20]"],
        }
    )
    base_shape_row = tables["shape_future"].iloc[0].to_dict()
    tables["shape_future"] = pd.DataFrame(
        [
            {
                **base_shape_row,
                "H": horizon,
                "phi": peak_valley_phi,
                "peak_valley_phi": peak_valley_phi,
            }
            for horizon in (5, 8, 15)
            for peak_valley_phi in (0.8,)
        ]
    )
    tables["day_type_factors"] = pd.DataFrame(
        [
            {
                "season": season,
                "final_day_type": day_type,
                "factor": factor,
                "sample_days": 60,
                "holiday_merged_with_weekend": False,
                "source_year": 2023,
                "factor_scope": "province_shared_all_25_components",
                "fallback_status": "none",
            }
            for season in ("春季", "夏季", "秋季", "冬季")
            for day_type, factor in (
                ("工作日", 1.0),
                ("普通周末", 0.9),
                ("法定节假日", 0.8),
            )
        ]
    )
    return tables


def _run_manifest(hourly_file: Path | str = "hourly.parquet") -> dict[str, object]:
    return {
        "status": "success",
        "method": "M1",
        **RUN_MANIFEST_CONTRACT,
        "hourly_output": {"format": "parquet", "file": str(hourly_file)},
    }


def _make_gate_disabled_rows_valid(
    shape_future: pd.DataFrame,
    mask: pd.Series,
    *,
    enabled_value: object = False,
) -> None:
    if not isinstance(enabled_value, bool):
        shape_future["peak_valley_evolution_enabled"] = shape_future[
            "peak_valley_evolution_enabled"
        ].astype(object)
    shape_future.loc[mask, "peak_valley_evolution_enabled"] = enabled_value
    shape_future.loc[mask, "peak_valley_gate_reason"] = (
        "disabled_delta_below_lower_bound"
    )
    shape_future.loc[mask, "peak_valley_delta_abs"] = 0.01
    shape_future.loc[mask, "peak_valley_delta_band"] = "below_lower_threshold"
    shape_future.loc[mask, "peak_valley_adjustment_status"] = (
        "skipped_delta_below_lower_bound"
    )
    shape_future.loc[mask, "peak_valley_lambda"] = 1.0
    shape_future.loc[mask, "peak_valley_target_applied"] = shape_future.loc[
        mask, "peak_valley_rate_after_four_window"
    ].to_numpy()
    shape_future.loc[mask, "peak_valley_rate_final"] = shape_future.loc[
        mask, "peak_valley_rate_after_four_window"
    ].to_numpy()
    shape_future.loc[mask, "future_peak_valley_rate"] = shape_future.loc[
        mask, "peak_valley_rate_after_four_window"
    ].to_numpy()
    for name in ("night", "morning", "midday", "evening"):
        shape_future.loc[mask, f"four_window_share_final_{name}"] = (
            shape_future.loc[mask, f"four_window_share_intermediate_{name}"].to_numpy()
        )
        shape_future.loc[mask, f"four_window_share_deviation_{name}"] = 0.0


def test_shape_future_schema_has_no_legacy_or_anchor_repair_fields() -> None:
    tables = _result_tables()

    validate_result_tables_contract(tables)

    columns = set(tables["shape_future"].columns)
    assert columns.isdisjoint(FORBIDDEN_SHAPE_FUTURE_COLUMNS)
    assert {
        "a_requested",
        "a_max",
        "a_applied",
        "future_peak_valley_rate",
        "future_evening_peak_hour",
        "future_raw_85pct_hours",
        "future_effective_control_hours",
    } <= columns


def test_day_type_factor_schema_is_province_shared() -> None:
    table = _result_tables()["day_type_factors"]

    validate_result_tables_contract(_result_tables())

    assert "industry" not in table.columns
    assert set(table.factor_scope) == {"province_shared_all_25_components"}


@pytest.mark.parametrize(
    ("key", "bad_value"),
    [
        ("model_version", "3.2.0"),
        ("shape_extrapolation_method", "peak_valley"),
        ("late_peak_control_policy", "fixed_to_2023_anchor"),
        ("day_type_factor_scope", "industry_specific"),
        ("peak_valley_extrapolation_enabled", False),
        ("peak_valley_gate_policy", "always_enabled"),
        ("peak_valley_delta_lower_bound", 0.064),
        ("peak_valley_delta_upper_bound", 0.251),
        ("flat_load_factor_threshold", 0.79),
        ("flat_cv_threshold", 0.16),
        ("formal_horizon", 5),
        ("formal_peak_valley_phi", 5),
        ("peak_damping_enabled", True),
        ("anchor_repair_enabled", True),
    ],
)
def test_run_manifest_rejects_old_or_wrong_scope(
    key: str, bad_value: object
) -> None:
    manifest = _run_manifest()
    manifest[key] = bad_value

    with pytest.raises(RuntimeError, match="结果口径不符"):
        validate_run_manifest(manifest, require_parquet=True)


def test_formal_manifest_rejects_csv_fallback() -> None:
    manifest = _run_manifest("hourly.csv.gz")
    manifest["hourly_output"] = {
        "format": "csv.gz",
        "file": "hourly.csv.gz",
    }

    with pytest.raises(RuntimeError, match="Parquet"):
        validate_run_manifest(manifest, require_parquet=True)


def test_shape_future_requires_complete_h_by_h_pv_sensitivity_grid() -> None:
    tables = _result_tables()
    tables["shape_future"] = tables["shape_future"].iloc[:-1].copy()

    with pytest.raises(ValueError, match="敏感性H×phi覆盖不完整"):
        validate_result_tables_contract(tables)


def test_shape_future_rejects_peak_valley_adjustment_when_gate_is_disabled() -> None:
    tables = _result_tables()
    mask = pd.Series(True, index=tables["shape_future"].index)
    tables["shape_future"].loc[mask, "peak_valley_evolution_enabled"] = False
    tables["shape_future"].loc[mask, "peak_valley_gate_reason"] = (
        "disabled_delta_below_lower_bound"
    )
    tables["shape_future"].loc[mask, "peak_valley_delta_abs"] = 0.01
    tables["shape_future"].loc[mask, "peak_valley_delta_band"] = (
        "below_lower_threshold"
    )

    with pytest.raises(ValueError, match="仅保留四时段结果"):
        validate_result_tables_contract(tables)


def test_shape_future_accepts_valid_disabled_gate_rows_and_string_false() -> None:
    tables = _result_tables()
    shape_future = tables["shape_future"]
    mask = pd.Series(True, index=shape_future.index)
    _make_gate_disabled_rows_valid(shape_future, mask, enabled_value="False")

    validate_result_tables_contract(tables)


def test_shape_future_rejects_gate_evidence_that_varies_by_sensitivity() -> None:
    tables = _result_tables()
    shape_future = tables["shape_future"]
    mask = shape_future.H.eq(8) & shape_future.phi.eq(0.8)
    _make_gate_disabled_rows_valid(shape_future, mask)

    with pytest.raises(ValueError, match="跨year/H/phi必须恒定"):
        validate_result_tables_contract(tables)


def test_shape_future_rejects_delta_band_inconsistent_with_delta() -> None:
    tables = _result_tables()
    tables["shape_future"]["peak_valley_delta_band"] = "above_upper_threshold"

    with pytest.raises(ValueError, match="peak_valley_delta_band"):
        validate_result_tables_contract(tables)


def test_shape_future_rejects_inexact_gate_reason() -> None:
    tables = _result_tables()
    shape_future = tables["shape_future"]
    mask = shape_future.H.eq(8) & shape_future.phi.eq(0.8)
    _make_gate_disabled_rows_valid(shape_future, mask)
    shape_future.loc[mask, "peak_valley_gate_reason"] = "disabled_flat_curve"

    with pytest.raises(ValueError, match="门控结果或原因"):
        validate_result_tables_contract(tables)


def test_shape_future_rejects_unknown_boolean_text() -> None:
    tables = _result_tables()
    tables["shape_future"]["flat_curve"] = tables["shape_future"][
        "flat_curve"
    ].astype(object)
    tables["shape_future"].loc[0, "flat_curve"] = "not-a-boolean"

    with pytest.raises(ValueError, match="必须为可解析布尔值"):
        validate_result_tables_contract(tables)


def test_save_result_tables_stamps_3_4_1_manifest(tmp_path: Path) -> None:
    manifest = save_result_tables(_result_tables(), tmp_path / "数据表")

    assert set(manifest) >= set(REQUIRED_FORMAL_TABLES)
    assert {entry["schema_version"] for entry in manifest.values()} == {"3.5.0"}
    assert not {
        "representative_selection",
        "parameter_source_trace",
        "group_annual_energy",
        "load_duration",
    }.intersection(manifest)


def test_report_loader_blocks_old_manifest_before_building_artifacts(
    tmp_path: Path,
) -> None:
    root = tmp_path / "formal"
    table_dir = root / "数据表"
    table_dir.mkdir(parents=True)
    (root / "run_metadata.json").write_text(
        json.dumps({**_run_manifest(), "model_version": "3.4.0"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (table_dir / "tables_manifest.json").write_text(
        json.dumps({}, ensure_ascii=False), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="model_version"):
        load_reporting_contract(root)


def test_report_method_statements_cover_final_shape_and_energy_scope() -> None:
    joined = "".join(REPORT_METHOD_STATEMENTS)
    assert "同行业同日有效用户等权平均" in joined
    assert "2023行业medoid" in joined
    assert "四时段份额调整" in joined
    assert "同比例迁移" in joined
    assert "日负荷率大于0.80" in joined
    assert "变异系数小于0.15" in joined
    assert "6.5%—25%" in joined
    assert "门控未通过" in joined
    assert "等于四时段" in joined
    assert "门控通过时" in joined and "最终优先" in joined
    assert "只作诊断" in joined
    assert "全部25个组成" in joined
    assert "自然负荷口径" in joined
    assert "未逐时扣除" in joined


def test_workbook_sources_use_current_correction_shape_contract() -> None:
    assert "correction_component_shape" in WORKBOOK_TABLE_CONTRACTS
    assert "season_daytype_correction_component" not in WORKBOOK_TABLE_CONTRACTS
