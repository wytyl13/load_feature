from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from workstreams_load_characteristic_model_v2.formal_output_validation import (
    validate_formal_output,
)
from workstreams_load_characteristic_model_v2.reporting_outputs import (
    build_adaptive_typical_days,
    build_monthly_maximum_curves,
    build_typical_day_load_factors,
    extract_baseline_p50_workday_display,
    normalize_hourly_trajectories,
    select_m1_baseline_p50_annual_scale,
)
from workstreams_load_characteristic_model_v2.result_tables import (
    REQUIRED_FORMAL_TABLES,
    RUN_MANIFEST_CONTRACT,
)


YEAR = 2025
SCENARIO = "基准"
TRAJECTORY = "P50_real"
TARGET_MWH = 1_000_000.0


def _complete_hourly() -> pd.DataFrame:
    timestamps = pd.date_range(
        f"{YEAR}-01-01 00:00:00",
        f"{YEAR}-12-31 23:00:00",
        freq="h",
    )
    hour = timestamps.hour.to_numpy(float)
    month = timestamps.month.to_numpy(float)
    raw = 1.0 + 0.35 * np.sin(2 * np.pi * hour / 24.0) + month / 120.0
    values = raw * (TARGET_MWH / raw.sum())
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "year": YEAR,
            "planning_scenario": SCENARIO,
            "trajectory": TRAJECTORY,
            "value": values,
        }
    )


def _write_table(table_dir: Path, name: str, frame: pd.DataFrame) -> dict[str, object]:
    path = table_dir / f"{name}.csv"
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return {
        "file": path.name,
        "format": "csv",
        "schema_version": "3.5.0",
        "rows": len(frame),
        "columns": list(frame.columns),
    }


def _build_saved_result(root: Path) -> Path:
    table_dir = root / "数据表"
    hourly_dir = root / "逐时曲线"
    display_dir = root / f"展示_{YEAR}-{YEAR}_基准P50"
    formal_dir = root / f"正式调用_{YEAR}-{YEAR}"
    for directory in (table_dir, hourly_dir, display_dir, formal_dir):
        directory.mkdir(parents=True, exist_ok=True)

    source = _complete_hourly()
    hourly = normalize_hourly_trajectories(source).drop(
        columns="annual_peak_base_value"
    )
    hourly["method"] = "M1"
    hourly_path = hourly_dir / "province_hourly_normalized_trajectories.parquet"
    hourly.to_parquet(hourly_path, index=False)

    typical = build_adaptive_typical_days(source)
    typical["method"] = "M1"
    factors = build_typical_day_load_factors(typical)
    factors["method"] = "M1"
    monthly_peaks = build_monthly_maximum_curves(source).drop(
        columns="annual_peak_base_value"
    )
    monthly_peaks["method"] = "M1"

    annual_energy = pd.DataFrame(
        {
            "method": ["M1"],
            "scenario": [SCENARIO],
            "year": [YEAR],
            "P10_natural_MWh": [TARGET_MWH * 0.9],
            "P50_natural_MWh": [TARGET_MWH],
            "P90_natural_MWh": [TARGET_MWH * 1.1],
        }
    )
    annual_metrics = pd.DataFrame(
        {
            "method": ["M1"],
            "planning_scenario": [SCENARIO],
            "year": [YEAR],
            "trajectory": [TRAJECTORY],
            "annual_energy_MWh": [TARGET_MWH],
            "annual_energy_quantile": ["P50"],
            "Tmax_h": [5000.0],
        }
    )
    monthly_metrics = pd.DataFrame(
        {
            "method": "M1",
            "planning_scenario": SCENARIO,
            "year": YEAR,
            "trajectory": TRAJECTORY,
            "month": range(1, 13),
            "monthly_imbalance_rate": np.linspace(0.05, 0.12, 12),
        }
    )
    display_curves = extract_baseline_p50_workday_display(
        typical, start_year=YEAR, end_year=YEAR
    )
    display_factors = factors.loc[factors.final_day_type.eq("工作日")].copy()
    display_monthly = monthly_metrics.copy()
    display_annual = annual_metrics.copy()
    display_peaks = monthly_peaks.copy()
    display_scale = select_m1_baseline_p50_annual_scale(
        annual_energy, years=(YEAR,)
    )

    tables = {
        "annual_energy_interval": annual_energy,
        "trajectory_monthly_metrics": monthly_metrics,
        "trajectory_annual_metrics": annual_metrics,
        "adaptive_typical_day_curves": typical,
        "typical_day_load_factors": factors,
        "monthly_maximum_curves_pu": monthly_peaks,
        "display_workday_curves_2025_2045": display_curves,
        "display_typical_day_load_factors_2025_2045": display_factors,
        "display_monthly_metrics_2025_2045": display_monthly,
        "display_annual_metrics_2025_2045": display_annual,
        "display_monthly_peak_curves_2025_2045": display_peaks,
        "display_m1_p50_annual_scale_2025_2045": display_scale,
    }
    for name in REQUIRED_FORMAL_TABLES:
        tables.setdefault(name, pd.DataFrame({"audit": []}))
    tables["shape_future"] = pd.DataFrame(
        {
            "industry": ["测试行业"],
            "season": ["夏季"],
            "final_day_type": ["工作日"],
            "year": [YEAR],
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
            "a_requested": [0.1],
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
            "peak_valley_cumulative_damping": [0.2],
            "peak_valley_target_requested": [0.2],
            "peak_valley_target_range_limited": [0.2],
            "peak_valley_target_applied": [0.2],
            "peak_valley_target_clipped": [False],
            "peak_valley_adjustment_status": ["mean_centered_scaling_applied"],
            "peak_valley_rate_after_four_window": [0.3],
            "peak_valley_lambda": [0.7],
            "peak_valley_rate_final": [0.2],
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
            "future_peak_valley_rate": [0.2],
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
    manifest = {
        name: _write_table(table_dir, name, frame) for name, frame in tables.items()
    }
    (table_dir / "tables_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    display_files = {
        "四季工作日标幺曲线_长表.csv": display_curves,
        "四季工作日典型日日负荷率.csv": display_factors,
        "月度指标与月不均衡系数.csv": display_monthly,
        "年度指标与年利用小时数.csv": display_annual,
        "12个月月最大负荷标幺曲线.csv": display_peaks,
        "M1基准情景年度电量P50.csv": display_scale,
    }
    for file_name, frame in display_files.items():
        frame.to_csv(display_dir / file_name, index=False, encoding="utf-8-sig")

    formal_paths = {"normalized_hourly_trajectories": hourly_path}
    for name in (
        "annual_energy_interval",
        "trajectory_annual_metrics",
        "trajectory_monthly_metrics",
        "adaptive_typical_day_curves",
        "typical_day_load_factors",
        "monthly_maximum_curves_pu",
    ):
        formal_paths[name] = table_dir / str(manifest[name]["file"])
    pd.DataFrame(
        [
            {
                "result_name": name,
                "file": str(path.resolve()),
                "formal_year_filter": f"{YEAR}<=year<={YEAR}",
            }
            for name, path in formal_paths.items()
        ]
    ).to_csv(formal_dir / "调用索引.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "status": "success",
        "method": "M1",
        **RUN_MANIFEST_CONTRACT,
        "annual_energy_basis": "P50_natural_total",
        "years": [YEAR],
        "planning_scenarios": [SCENARIO],
        "bootstrap_scenarios_per_context": 1000,
        "hourly_output": {
            "file": str(hourly_path.resolve()),
            "format": "parquet",
        },
    }
    (root / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return root


def _validate_fixture(root: Path) -> dict[str, object]:
    return validate_formal_output(
        root,
        base_years=(YEAR,),
        study_years=(YEAR,),
        display_years=(YEAR,),
        scenarios=(SCENARIO,),
        trajectories=(TRAJECTORY,),
        check_component_curves=False,
    )


def test_validator_accepts_complete_saved_result_and_index_only_formal_access(
    tmp_path: Path,
) -> None:
    report = _validate_fixture(_build_saved_result(tmp_path / "formal_result"))

    assert report["passed"] is True
    assert report["summary"]["failed"] == 0
    formal = report["formal_call_access"]
    assert formal["status"] == "passed"
    assert formal["materialization_mode"] == "index_only_references_saved_full_tables"
    assert formal["large_tables_copied_by_validator"] is False
    assert formal["source_year_coverage"]["normalized_hourly_trajectories"] is True


def test_validator_rejects_trajectory_energy_that_does_not_close_to_m1_p50(
    tmp_path: Path,
) -> None:
    root = _build_saved_result(tmp_path / "bad_energy")
    path = root / "数据表" / "trajectory_annual_metrics.csv"
    metrics = pd.read_csv(path)
    metrics.loc[0, "annual_energy_MWh"] = TARGET_MWH + 10_000.0
    metrics.to_csv(path, index=False, encoding="utf-8-sig")

    report = _validate_fixture(root)

    assert report["passed"] is False
    check = next(
        item
        for item in report["checks"]
        if item["name"] == "trajectory_annual_energy_closure"
    )
    assert check["status"] == "failed"
    assert check["max_abs_energy_error_MWh"] == 10_000.0


def test_validator_rejects_m1_display_scale_from_non_p50_field(tmp_path: Path) -> None:
    root = _build_saved_result(tmp_path / "bad_scale")
    path = root / "数据表" / "display_m1_p50_annual_scale_2025_2045.csv"
    scale = pd.read_csv(path)
    scale.loc[0, "annual_scale_MWh"] = TARGET_MWH * 0.9
    scale.to_csv(path, index=False, encoding="utf-8-sig")

    report = _validate_fixture(root)

    assert report["passed"] is False
    check = next(
        item
        for item in report["checks"]
        if item["name"] == "display_contract_and_M1_P50_scale"
    )
    assert check["status"] == "failed"
    assert check["M1_baseline_scale_matches_upstream_P50"] is False
