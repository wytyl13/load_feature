"""24细分行业+修正组成的全省中长期负荷特性流水线。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from . import MODEL_VERSION
from .annual_assembly import assemble_province_year
from .annual_energy import (
    ANNUAL_ENERGY_METHOD,
    NATURAL_LOAD_SCOPE,
    SCENARIO_ORDER,
    build_industry_annual_energy,
    build_proxy_sector_weights,
    load_industry_weight_evolution,
    load_upstream_annual_energy,
    load_upstream_scope_bridge,
)
from .baseline_shapes import build_baseline_shapes
from .component_shapes import (
    build_base_year_industry_components,
    build_future_industry_components,
    build_proxy_sector_typical_shapes,
)
from .config import (
    ANCHOR_REPAIR_ENABLED,
    ANNUAL_ENERGY_UPSTREAM,
    DAY_TYPE_FACTOR_SCOPE,
    DEFAULT_LOAD_WORKBOOK,
    DEFAULT_OFFICIAL_WORKBOOK,
    FLAT_CV_THRESHOLD,
    FLAT_LOAD_FACTOR_THRESHOLD,
    FOUR_WINDOW_EXTRAPOLATION_ENABLED,
    FOUR_WINDOW_FINAL_SHARE_POLICY,
    FOUR_WINDOW_TRANSFORM,
    LATE_PEAK_CONTROL_POLICY,
    METHOD_REVISION,
    INDUSTRY_WEIGHT_EVOLUTION_CSV,
    OFFICIAL_2025_MAPPING_CSV,
    PEAK_DAMPING_ENABLED,
    PEAK_VALLEY_EVOLUTION_DELTA_LOWER_BOUND,
    PEAK_VALLEY_EVOLUTION_DELTA_UPPER_BOUND,
    PEAK_VALLEY_EXTRAPOLATION_ENABLED,
    PEAK_VALLEY_EXTRAPOLATION_METHOD,
    PEAK_VALLEY_FORMAL_PHI,
    PEAK_VALLEY_GATE_POLICY,
    PEAK_VALLEY_RATE_TOLERANCE,
    PEAK_VALLEY_SUPPORTED_PHIS,
    PEAK_VALLEY_TARGET_PRIORITY,
    PEAK_VALLEY_TRANSFORM,
    SAMPLE_INDUSTRY_MAPPING_CSV,
    SMOKE_OUTPUT_DIR,
    SHAPE_EXTRAPOLATION_METHOD,
)
from .correction_curve import (
    build_correction_artifacts,
    compose_with_fixed_correction_shape,
)
from .data_io import (
    build_input_manifest,
    load_and_validate_inputs,
    sample_industry_energy_year_frame,
)
from .metrics import (
    WINDOWS,
    calculate_load_metrics,
    historical_season_daytype_typical_shapes,
    season_daytype_typical_shapes,
    typical_workdays,
)
from .reporting_outputs import (
    build_adaptive_typical_days,
    build_monthly_maximum_curves,
    build_typical_day_load_factors,
    extract_baseline_p50_workday_display,
    normalize_hourly_trajectories,
    select_m1_baseline_p50_annual_scale,
)
from .monthly_envelope import (
    build_interday_residual_library,
    build_monthly_envelope,
    estimate_province_day_type_factors,
)
from .parameter_pipeline import (
    build_cell_quality,
    build_future_parameter_table,
)
from .province_bootstrap import (
    bootstrap_daily_multipliers,
)
from .quality import (
    HOUR_COLS,
    aggregate_industry_daily,
    read_system_history,
    read_user_daily,
)
from .representative_paths import (
    build_scenario_features,
    select_representative_scenarios,
)
from .shape_parameters import identify_shape_parameters
from .result_tables import (
    annual_indicator_quantiles,
    build_annual_monthly_peak_curve,
    build_industry_classification_table,
    build_target_month_energy_table,
    monthly_indicator_quantiles,
    save_result_tables,
    validate_run_manifest,
)


SMOKE_YEARS = (2025, 2030)
FORMAL_YEARS = tuple(range(2025, 2061))
SMOKE_BOOTSTRAP_SCENARIOS = 20
FORMAL_BOOTSTRAP_SCENARIOS = 1000
FORMAL_HORIZON = 8
FORMAL_PEAK_VALLEY_PHI = PEAK_VALLEY_FORMAL_PHI
REPRESENTATIVE_TRACKS = ("P10", "P50", "P90")
HISTORICAL_SHAPE_SCOPE = "dispatch_proxy_not_hourly_deducted"
CORRECTION_SHAPE_STATUS = "proxy_derived_from_dispatch_shape"
ANNUAL_ENERGY_BASIS = "P50_natural_total"
TRAJECTORY_QUANTILE_BASIS = "annual_peak_bootstrap"
FORBIDDEN_TABLE_NAMES = {
    "group_annual_energy",
    "load_duration",
    "parameter_source_trace",
    "representative_selection",
    "season_daytype_correction_component",
}
FORBIDDEN_SHAPE_COLUMNS = {
    "anchor_repair_applied",
    "application_run_group",
    "beta_delta_hours",
    "centroid",
    "delta_applied_hours",
    "delta_continuous_hours",
    "gamma",
    "log_gamma_increment",
    "parameter_group",
    "repair_l2_distance",
    "rho",
    "source_industry",
}


def _concat(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    usable = [frame for frame in frames if isinstance(frame, pd.DataFrame) and not frame.empty]
    return pd.concat(usable, ignore_index=True, sort=False) if usable else pd.DataFrame()


def _assert_current_method_contract() -> None:
    """Fail before reading data if any fixed 3.5.0 method switch drifted."""

    observed = {
        "model_version": MODEL_VERSION,
        "method_revision": METHOD_REVISION,
        "peak_valley_extrapolation_enabled": PEAK_VALLEY_EXTRAPOLATION_ENABLED,
        "four_window_extrapolation_enabled": FOUR_WINDOW_EXTRAPOLATION_ENABLED,
        "four_window_transform": FOUR_WINDOW_TRANSFORM,
        "peak_valley_extrapolation_method": PEAK_VALLEY_EXTRAPOLATION_METHOD,
        "peak_valley_transform": PEAK_VALLEY_TRANSFORM,
        "peak_valley_target_priority": PEAK_VALLEY_TARGET_PRIORITY,
        "peak_valley_supported_phis": PEAK_VALLEY_SUPPORTED_PHIS,
        "peak_valley_formal_phi": PEAK_VALLEY_FORMAL_PHI,
        "peak_valley_rate_tolerance": PEAK_VALLEY_RATE_TOLERANCE,
        "peak_valley_gate_policy": PEAK_VALLEY_GATE_POLICY,
        "peak_valley_delta_lower_bound": (
            PEAK_VALLEY_EVOLUTION_DELTA_LOWER_BOUND
        ),
        "peak_valley_delta_upper_bound": (
            PEAK_VALLEY_EVOLUTION_DELTA_UPPER_BOUND
        ),
        "flat_load_factor_threshold": FLAT_LOAD_FACTOR_THRESHOLD,
        "flat_cv_threshold": FLAT_CV_THRESHOLD,
        "shape_extrapolation_method": SHAPE_EXTRAPOLATION_METHOD,
        "four_window_final_share_policy": FOUR_WINDOW_FINAL_SHARE_POLICY,
        "peak_valley_target_priority": PEAK_VALLEY_TARGET_PRIORITY,
        "late_peak_control_policy": LATE_PEAK_CONTROL_POLICY,
        "peak_damping_enabled": PEAK_DAMPING_ENABLED,
        "anchor_repair_enabled": ANCHOR_REPAIR_ENABLED,
        "day_type_factor_scope": DAY_TYPE_FACTOR_SCOPE,
    }
    expected = {
        "model_version": "3.5.0",
        "method_revision": "2026-09-15-logit-damped-phi08",
        "peak_valley_extrapolation_enabled": True,
        "four_window_extrapolation_enabled": True,
        "four_window_transform": "proportional_within_window",
        "peak_valley_extrapolation_method": "logit_damped",
        "peak_valley_transform": "global_mean_centered_scaling",
        "peak_valley_target_priority": (
            "final_when_gate_enabled_otherwise_four_window_final"
        ),
        "peak_valley_supported_phis": (0.8,),
        "peak_valley_formal_phi": 0.8,
        "peak_valley_rate_tolerance": 1e-10,
        "peak_valley_gate_policy": (
            "trend_eligible_and_nonflat_and_abs_delta_within_closed_interval_0.065_0.25"
        ),
        "peak_valley_delta_lower_bound": 0.065,
        "peak_valley_delta_upper_bound": 0.25,
        "flat_load_factor_threshold": 0.80,
        "flat_cv_threshold": 0.15,
        "shape_extrapolation_method": (
            "four_window_then_local_smoothing_then_gated_peak_valley_mean_scaling"
        ),
        "four_window_final_share_policy": (
            "exact_if_peak_valley_skipped_diagnostic_if_applied"
        ),
        "late_peak_control_policy": "diagnostic_only",
        "peak_damping_enabled": False,
        "anchor_repair_enabled": False,
        "day_type_factor_scope": "province_shared_all_25_components",
    }
    if observed != expected:
        raise RuntimeError(f"3.5.0方法开关漂移：{observed!r}")


def _add_scope_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach the single natural-load scope contract to derived result tables."""

    result = frame.copy()
    result["curve_energy_scope"] = NATURAL_LOAD_SCOPE
    result["historical_shape_scope"] = HISTORICAL_SHAPE_SCOPE
    result["annual_special_deduction_applied"] = True
    result["hourly_special_deduction_applied"] = False
    result["special_items_in_correction_component"] = False
    result["correction_shape_status"] = CORRECTION_SHAPE_STATUS
    result["annual_energy_basis"] = ANNUAL_ENERGY_BASIS
    result["trajectory_quantile_basis"] = TRAJECTORY_QUANTILE_BASIS
    result["shape_extrapolation_method"] = SHAPE_EXTRAPOLATION_METHOD
    result["peak_valley_extrapolation_enabled"] = PEAK_VALLEY_EXTRAPOLATION_ENABLED
    result["four_window_final_share_policy"] = FOUR_WINDOW_FINAL_SHARE_POLICY
    result["late_peak_control_policy"] = LATE_PEAK_CONTROL_POLICY
    result["peak_damping_enabled"] = PEAK_DAMPING_ENABLED
    result["anchor_repair_enabled"] = ANCHOR_REPAIR_ENABLED
    return result


def _system_history_year_hourly(system_history: pd.DataFrame, year: int) -> pd.DataFrame:
    required = {"date", *HOUR_COLS}
    missing = required - set(system_history.columns)
    if missing:
        raise ValueError(f"全省历史负荷缺少字段：{sorted(missing)}")
    frame = system_history.copy()
    frame["date"] = pd.to_datetime(frame.date).dt.normalize()
    frame = frame.loc[frame.date.dt.year.eq(int(year))].sort_values("date").reset_index(drop=True)
    expected_dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    if not frame.date.equals(pd.Series(expected_dates, name="date")):
        raise ValueError(f"{year}年全省历史负荷日期不完整")
    values = frame[HOUR_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError(f"{year}年全省历史负荷含非有限值或负值")
    timestamps = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")
    return pd.DataFrame({"timestamp": timestamps, "load_MW": values.reshape(-1)})


def _validated_bootstrap_inputs(
    hourly_baseline: pd.DataFrame,
    multipliers: pd.DataFrame,
    year: int,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
    required_hourly = {"timestamp", "load_MW"}
    missing = required_hourly - set(hourly_baseline.columns)
    if missing:
        raise ValueError(f"Bootstrap逐时基线缺少字段：{sorted(missing)}")
    hourly = hourly_baseline[["timestamp", "load_MW"]].copy()
    hourly["timestamp"] = pd.to_datetime(hourly.timestamp)
    hourly = hourly.sort_values("timestamp").reset_index(drop=True)
    expected_timestamps = pd.date_range(
        f"{int(year)}-01-01", f"{int(year)}-12-31 23:00", freq="h"
    )
    if not hourly.timestamp.equals(pd.Series(expected_timestamps, name="timestamp")):
        raise ValueError(f"Bootstrap逐时基线必须完整覆盖{year}年连续整点")
    hourly["load_MW"] = pd.to_numeric(hourly.load_MW, errors="coerce")
    if (
        hourly.load_MW.isna().any()
        or not np.isfinite(hourly.load_MW.to_numpy(float)).all()
        or (hourly.load_MW < 0).any()
    ):
        raise ValueError("Bootstrap逐时基线必须为有限非负负荷")

    required_multipliers = {"scenario_id", "date", "daily_multiplier"}
    missing = required_multipliers - set(multipliers.columns)
    if missing:
        raise ValueError(f"Bootstrap日乘数缺少字段：{sorted(missing)}")
    daily = multipliers[list(required_multipliers)].copy()
    scenario_ids = pd.to_numeric(daily.scenario_id, errors="coerce")
    if (
        scenario_ids.isna().any()
        or not np.equal(scenario_ids, np.floor(scenario_ids)).all()
        or (scenario_ids < 0).any()
    ):
        raise ValueError("Bootstrap scenario_id必须为非负整数")
    daily["scenario_id"] = scenario_ids.astype(int)
    daily["date"] = pd.to_datetime(daily.date, errors="coerce").dt.normalize()
    daily["daily_multiplier"] = pd.to_numeric(
        daily.daily_multiplier, errors="coerce"
    )
    if daily.date.isna().any() or daily.duplicated(["scenario_id", "date"]).any():
        raise ValueError("Bootstrap日乘数的scenario_id/date必须完整且唯一")
    if (
        daily.daily_multiplier.isna().any()
        or not np.isfinite(daily.daily_multiplier.to_numpy(float)).all()
        or (daily.daily_multiplier <= 0).any()
    ):
        raise ValueError("Bootstrap日乘数必须为有限正数")
    expected_dates = pd.date_range(f"{int(year)}-01-01", f"{int(year)}-12-31", freq="D")
    pivot = daily.pivot(index="scenario_id", columns="date", values="daily_multiplier")
    unexpected_dates = pivot.columns.difference(expected_dates)
    if len(unexpected_dates):
        raise ValueError(f"Bootstrap日乘数包含目标年之外日期：{list(unexpected_dates[:3])}")
    pivot = pivot.reindex(columns=expected_dates)
    if pivot.empty or pivot.isna().any().any():
        raise ValueError("每个Bootstrap场景必须完整覆盖目标年每一天")
    values = hourly.load_MW.to_numpy(float).reshape(len(expected_dates), 24)
    return expected_timestamps, expected_dates, values, pivot


def _evaluate_all_bootstrap_scenarios(
    hourly_baseline: pd.DataFrame,
    multipliers: pd.DataFrame,
    planning_scenario: str,
    year: int,
    method: str = ANNUAL_ENERGY_METHOD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """逐场景计算精确年度/月度指标，不构造全场景逐时长表。"""

    timestamps, dates, base_values, multiplier_matrix = _validated_bootstrap_inputs(
        hourly_baseline, multipliers, year
    )
    base_daily_load_factor = base_values.mean(axis=1) / base_values.max(axis=1)
    month_by_day = dates.month.to_numpy()
    annual_rows: list[dict[str, object]] = []
    monthly_rows: list[dict[str, object]] = []
    for scenario_id, multiplier_row in multiplier_matrix.iterrows():
        daily_multiplier = multiplier_row.to_numpy(float)
        scenario_values = base_values * daily_multiplier[:, None]
        flattened = scenario_values.reshape(-1)
        annual_energy = float(flattened.sum())
        annual_max = float(flattened.max())
        annual_min = float(flattened.min())
        annual_peak_position = int(np.argmax(flattened))
        annual: dict[str, object] = {
            "method": str(method),
            "planning_scenario": str(planning_scenario),
            "year": int(year),
            "bootstrap_scenario_id": int(scenario_id),
            "annual_energy_MWh": annual_energy,
            "annual_max_MW": annual_max,
            "annual_peak_timestamp": timestamps[annual_peak_position],
            "annual_min_MW": annual_min,
            "peak_valley_difference_MW": annual_max - annual_min,
            "Tmax_h": annual_energy / annual_max,
            "annual_load_factor": annual_energy / (annual_max * len(flattened)),
            "year_hours": len(flattened),
            "mean_daily_load_factor": float(base_daily_load_factor.mean()),
            "result_type": "Bootstrap真实场景指标",
        }
        for label, hours in WINDOWS.items():
            annual[f"share_{label}"] = float(
                scenario_values[:, list(hours)].sum() / annual_energy
            )
        annual_rows.append(annual)

        for month in range(1, 13):
            day_positions = np.flatnonzero(month_by_day == month)
            month_values = scenario_values[day_positions]
            month_energy = float(month_values.sum())
            month_average = float(month_values.mean())
            flat_peak_position = int(np.argmax(month_values))
            peak_day_local, peak_hour = divmod(flat_peak_position, 24)
            peak_day_position = int(day_positions[peak_day_local])
            peak_day_values = scenario_values[peak_day_position]
            peak = float(month_values.reshape(-1)[flat_peak_position])
            peak_day_average = float(peak_day_values.mean())
            monthly_rows.append(
                {
                    "method": str(method),
                    "planning_scenario": str(planning_scenario),
                    "year": int(year),
                    "month": month,
                    "bootstrap_scenario_id": int(scenario_id),
                    "monthly_energy_MWh": month_energy,
                    "month_hours": int(month_values.size),
                    "monthly_average_MW": month_average,
                    "monthly_max_MW": peak,
                    "monthly_peak_timestamp": dates[peak_day_position]
                    + pd.Timedelta(hours=peak_hour),
                    "peak_day": dates[peak_day_position],
                    "peak_day_average_MW": peak_day_average,
                    "peak_day_min_MW": float(peak_day_values.min()),
                    "peak_day_load_factor": peak_day_average / peak,
                    "monthly_imbalance_rate": month_average / peak_day_average,
                    "result_type": "Bootstrap真实场景指标",
                }
            )
    return (
        _add_scope_metadata(pd.DataFrame(annual_rows)),
        _add_scope_metadata(pd.DataFrame(monthly_rows)),
    )


def _representative_curve_table(
    hourly_baseline: pd.DataFrame,
    multipliers: pd.DataFrame,
    selected: dict[str, int],
    *,
    method: str,
    planning_scenario: str,
    year: int,
) -> pd.DataFrame:
    baseline = hourly_baseline[["timestamp", "load_MW"]].copy()
    baseline["timestamp"] = pd.to_datetime(baseline.timestamp)
    baseline = baseline.sort_values("timestamp").reset_index(drop=True)
    baseline["date"] = baseline.timestamp.dt.normalize()
    result = baseline[["timestamp"]].copy()
    result["deterministic_baseline_MW"] = baseline.load_MW.to_numpy(float)
    for track in REPRESENTATIVE_TRACKS:
        scenario_id = int(selected[track])
        chosen = multipliers.loc[
            pd.to_numeric(multipliers.scenario_id).eq(scenario_id),
            ["date", "daily_multiplier"],
        ].copy()
        chosen["date"] = pd.to_datetime(chosen.date).dt.normalize()
        if chosen.date.duplicated().any():
            raise ValueError(f"代表场景日乘数日期重复：{track}/{scenario_id}")
        factor = baseline[["date"]].merge(
            chosen, on="date", how="left", validate="many_to_one"
        ).daily_multiplier
        if factor.isna().any():
            raise ValueError(f"代表场景日乘数日期不完整：{track}/{scenario_id}")
        result[f"{track}_real_MW"] = baseline.load_MW.to_numpy(float) * factor.to_numpy(float)
        result[f"{track}_scenario_id"] = scenario_id
    result["method"] = str(method)
    result["planning_scenario"] = str(planning_scenario)
    result["year"] = int(year)
    result["curve_energy_scope"] = NATURAL_LOAD_SCOPE
    result["historical_shape_scope"] = HISTORICAL_SHAPE_SCOPE
    result["annual_special_deduction_applied"] = True
    result["hourly_special_deduction_applied"] = False
    result["special_items_in_correction_component"] = False
    result["correction_shape_status"] = CORRECTION_SHAPE_STATUS
    result["annual_energy_basis"] = ANNUAL_ENERGY_BASIS
    result["trajectory_quantile_basis"] = TRAJECTORY_QUANTILE_BASIS
    result["new_load_independently_shaped"] = False
    result["new_load_in_correction_component"] = True
    result["external_new_load_overlay"] = False
    result["annual_energy_track"] = "P50_natural_total"
    result["H"] = FORMAL_HORIZON
    result["phi"] = FORMAL_PEAK_VALLEY_PHI
    result["component_count"] = 25
    return result


def _representative_curves_long(hourly_wide: pd.DataFrame) -> pd.DataFrame:
    """Convert one context's deterministic and three real tracks to long MW."""

    required = {
        "timestamp",
        "year",
        "planning_scenario",
        "deterministic_baseline_MW",
        "P10_real_MW",
        "P50_real_MW",
        "P90_real_MW",
    }
    missing = required - set(hourly_wide.columns)
    if missing:
        raise ValueError(f"代表逐时曲线缺少字段：{sorted(missing)}")
    columns = {
        "deterministic_baseline_MW": "deterministic_baseline",
        "P10_real_MW": "P10_real",
        "P50_real_MW": "P50_real",
        "P90_real_MW": "P90_real",
    }
    parts: list[pd.DataFrame] = []
    for value_column, trajectory in columns.items():
        part = hourly_wide[
            ["timestamp", "year", "planning_scenario", value_column]
        ].rename(columns={value_column: "value"})
        part["trajectory"] = trajectory
        parts.append(part)
    result = pd.concat(parts, ignore_index=True, sort=False)
    return result[
        ["timestamp", "year", "planning_scenario", "trajectory", "value"]
    ]


def _trajectory_daily_load_factors(hourly_long: pd.DataFrame) -> pd.DataFrame:
    """Calculate one daily load-factor series for every saved trajectory."""

    required = {"timestamp", "year", "planning_scenario", "trajectory", "value"}
    missing = required - set(hourly_long.columns)
    if missing:
        raise ValueError(f"轨迹逐日负荷率输入缺少字段：{sorted(missing)}")
    frame = hourly_long.copy()
    frame["timestamp"] = pd.to_datetime(frame.timestamp)
    frame["date"] = frame.timestamp.dt.normalize()
    keys = ["year", "planning_scenario", "trajectory", "date"]
    daily = frame.groupby(keys, as_index=False, sort=False).agg(
        daily_energy_MWh=("value", "sum"),
        daily_average_MW=("value", "mean"),
        daily_max_MW=("value", "max"),
        daily_min_MW=("value", "min"),
        hour_count=("value", "size"),
    )
    if not daily.hour_count.eq(24).all() or (daily.daily_max_MW <= 0).any():
        raise ValueError("每条轨迹的逐日负荷率必须基于完整24小时且日最大值为正")
    daily["daily_load_factor"] = daily.daily_average_MW / daily.daily_max_MW
    daily["metric_definition"] = "daily_average_MW/daily_max_MW"
    return daily


def _write_hourly_output(
    hourly: pd.DataFrame,
    output_dir: Path,
    stem: str = "province_hourly_baseline",
    *,
    require_parquet: bool = False,
) -> tuple[Path, str, str | None]:
    """优先写Parquet；冒烟可回退CSV.gz，正式模式失败即中止。"""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if not str(stem).strip() or Path(str(stem)).name != str(stem):
        raise ValueError("逐时结果stem必须是不含目录的非空文件名")
    parquet_path = directory / f"{stem}.parquet"
    temporary_path = directory / f"{stem}.tmp.parquet"
    warning: str | None = None
    try:
        hourly.to_parquet(temporary_path, index=False)
        temporary_path.replace(parquet_path)
        return parquet_path, "parquet", None
    except Exception as exc:
        if temporary_path.is_file():
            temporary_path.unlink()
        message = (
            "正式全量必须安装Parquet引擎并成功写出Parquet；"
            f"当前写出失败：{type(exc).__name__}: {exc}"
        )
        if require_parquet:
            raise RuntimeError(message) from exc
        warning = message + "；本次非正式运行回退为CSV.gz。"
        csv_path = directory / f"{stem}.csv.gz"
        hourly.to_csv(
            csv_path,
            index=False,
            encoding="utf-8",
            compression="gzip",
            date_format="%Y-%m-%d %H:%M:%S",
        )
        return csv_path, "csv.gz", warning


def _add_context(
    frame: pd.DataFrame,
    *,
    method: str,
    planning_scenario: str,
    year: int,
    result_type: str,
) -> pd.DataFrame:
    result = frame.copy()
    result["method"] = str(method)
    result["planning_scenario"] = str(planning_scenario)
    result["year"] = int(year)
    result["result_type"] = result_type
    return _add_scope_metadata(result)


def _selected_metric_rows(
    annual_metrics: pd.DataFrame,
    monthly_metrics: pd.DataFrame,
    selected: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    annual_rows: list[pd.DataFrame] = []
    monthly_rows: list[pd.DataFrame] = []
    for track in REPRESENTATIVE_TRACKS:
        scenario_id = int(selected[track])
        annual = annual_metrics.loc[
            annual_metrics.bootstrap_scenario_id.eq(scenario_id)
        ].copy()
        monthly = monthly_metrics.loc[
            monthly_metrics.bootstrap_scenario_id.eq(scenario_id)
        ].copy()
        if len(annual) != 1 or len(monthly) != 12:
            raise AssertionError(f"真实代表轨迹指标不完整：{track}/{scenario_id}")
        annual["representative_track"] = track
        monthly["representative_track"] = track
        annual["result_type"] = "真实Bootstrap代表情景"
        monthly["result_type"] = "真实Bootstrap代表情景"
        annual_rows.append(annual)
        monthly_rows.append(monthly)
    return _concat(annual_rows), _concat(monthly_rows)


def _materialize_bootstrap_daily_output(
    multipliers: pd.DataFrame,
    selected: dict[str, int],
    *,
    materialize_all: bool,
) -> pd.DataFrame:
    """默认仅保留三条真实代表轨迹；显式全量模式保留全部并标记代表。"""

    required = {"scenario_id", "date", "daily_multiplier"}
    missing = required - set(multipliers.columns)
    if missing:
        raise ValueError(f"Bootstrap日乘数缺少字段：{sorted(missing)}")
    if set(selected) != set(REPRESENTATIVE_TRACKS):
        raise ValueError("代表轨迹必须恰有P10/P50/P90")
    available = set(pd.to_numeric(multipliers.scenario_id, errors="raise").astype(int))
    absent = {
        track: int(scenario_id)
        for track, scenario_id in selected.items()
        if int(scenario_id) not in available
    }
    if absent:
        raise ValueError(f"代表轨迹scenario_id不在日乘数中：{absent}")

    if materialize_all:
        result = multipliers.copy()
        labels_by_scenario: dict[int, list[str]] = {}
        for track in REPRESENTATIVE_TRACKS:
            labels_by_scenario.setdefault(int(selected[track]), []).append(track)
        result["representative_track"] = (
            pd.to_numeric(result.scenario_id, errors="raise")
            .astype(int)
            .map({key: "|".join(value) for key, value in labels_by_scenario.items()})
            .fillna("")
        )
        result["is_representative"] = result.representative_track.ne("")
        result["materialization_scope"] = "all_bootstrap_scenarios"
        return result.reset_index(drop=True)

    parts: list[pd.DataFrame] = []
    scenario_ids = pd.to_numeric(multipliers.scenario_id, errors="raise").astype(int)
    for track in REPRESENTATIVE_TRACKS:
        scenario_id = int(selected[track])
        part = multipliers.loc[scenario_ids.eq(scenario_id)].copy()
        part["representative_track"] = track
        part["is_representative"] = True
        part["materialization_scope"] = "representative_scenarios_only"
        parts.append(part)
    result = pd.concat(parts, ignore_index=True, sort=False)
    expected_dates = multipliers.date.nunique()
    if len(result) != len(REPRESENTATIVE_TRACKS) * expected_dates:
        raise AssertionError("P10/P50/P90代表轨迹日乘数行数不完整")
    return result


def _assert_no_legacy_outputs(tables: dict[str, pd.DataFrame]) -> None:
    forbidden_names = FORBIDDEN_TABLE_NAMES.intersection(tables)
    if forbidden_names:
        raise AssertionError(f"结果表仍含已删除旧表：{sorted(forbidden_names)}")
    shape_future = tables.get("shape_future", pd.DataFrame())
    legacy = sorted(FORBIDDEN_SHAPE_COLUMNS.intersection(shape_future.columns))
    if legacy:
        raise AssertionError(f"结果表shape_future仍含旧字段：{legacy}")


def _assert_method_consistency(
    tables: dict[str, pd.DataFrame], expected_method: str
) -> None:
    """所有含method字段的正式表必须且只能使用本次选择的年度电量方法。"""

    expected = str(expected_method).strip()
    if not expected:
        raise AssertionError("年度电量方法不能为空")
    for name, frame in tables.items():
        if "method" not in frame.columns:
            continue
        observed = set(frame.method.dropna().astype(str).str.strip().unique())
        if observed != {expected}:
            raise AssertionError(
                f"结果表{name}的方法值{sorted(observed)}与本次主方法{expected}不一致"
            )


def _build_daily_energy_tables(
    assembly_closure: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """由逐组成闭合明细生成正式日/月电量表及逐日25组成合计校验。"""

    required = {
        "method",
        "planning_scenario",
        "year",
        "closure_scope",
        "component",
        "industry",
        "component_type",
        "month",
        "date",
        "target_energy_MWh",
        "actual_energy_MWh",
        "relative_error",
        "tolerance",
        "passed",
    }
    missing = required - set(assembly_closure.columns)
    if missing:
        raise ValueError(f"年度装配闭合表缺少字段：{sorted(missing)}")
    component_daily = assembly_closure.loc[
        assembly_closure.closure_scope.eq("component_day")
    ].copy()
    component_month = assembly_closure.loc[
        assembly_closure.closure_scope.eq("component_month")
    ].copy()
    province_daily = assembly_closure.loc[
        assembly_closure.closure_scope.eq("province_day")
    ].copy()
    if component_daily.empty or province_daily.empty:
        raise ValueError("流水线要求年度装配开启include_daily_detail=True")
    component_daily["date"] = pd.to_datetime(component_daily.date).dt.normalize()
    province_daily["date"] = pd.to_datetime(province_daily.date).dt.normalize()
    context = ["method", "planning_scenario", "year", "month", "date"]
    grouped = (
        component_daily.groupby(context, as_index=False, sort=False)
        .agg(
            component_count=("component", "nunique"),
            component_energy_sum_MWh=("actual_energy_MWh", "sum"),
            component_target_sum_MWh=("target_energy_MWh", "sum"),
        )
    )
    province = province_daily[
        [
            *context,
            "target_energy_MWh",
            "actual_energy_MWh",
            "tolerance",
        ]
    ].rename(
        columns={
            "target_energy_MWh": "province_target_energy_MWh",
            "actual_energy_MWh": "province_actual_energy_MWh",
        }
    )
    checks = grouped.merge(province, on=context, how="left", validate="one_to_one")
    if checks.isna().any().any() or not checks.component_count.eq(25).all():
        raise AssertionError("每个自然日必须恰有25个电量组成并有全省闭合记录")
    checks["absolute_error_MWh"] = (
        checks.component_energy_sum_MWh - checks.province_actual_energy_MWh
    ).abs()
    checks["relative_error"] = checks.absolute_error_MWh / checks.province_actual_energy_MWh
    checks["component_share_sum"] = (
        checks.component_energy_sum_MWh / checks.province_actual_energy_MWh
    )
    checks["passed"] = (
        checks.relative_error.le(checks.tolerance)
        & np.isclose(
            checks.component_share_sum.to_numpy(float),
            1.0,
            rtol=0.0,
            atol=1e-10,
        )
        & checks.component_energy_sum_MWh.ge(0)
    )
    if not checks.passed.all():
        bad = checks.loc[~checks.passed].iloc[0]
        raise AssertionError(
            f"25组成逐日合计未闭合：{bad.planning_scenario}/{bad.date}"
        )
    totals = checks.set_index(context).province_actual_energy_MWh
    component_daily = component_daily.merge(
        totals.rename("province_daily_energy_MWh").reset_index(),
        on=context,
        how="left",
        validate="many_to_one",
    )
    component_daily["daily_electricity_share"] = (
        component_daily.actual_energy_MWh / component_daily.province_daily_energy_MWh
    )
    share_closure = component_daily.groupby(context).daily_electricity_share.sum()
    if not np.isclose(share_closure.to_numpy(float), 1.0, rtol=0.0, atol=1e-10).all():
        raise AssertionError("25组成逐日电量份额之和不为1")
    return (
        component_month.reset_index(drop=True),
        component_daily.reset_index(drop=True),
        checks.reset_index(drop=True),
    )


def _build_annual_weight_ledger(
    component_energy: pd.DataFrame,
    assembly_closure: pd.DataFrame,
) -> pd.DataFrame:
    """为三分位年度组成账本附加P50逐时装配闭合证据。"""

    if "annual_energy_quantile" not in component_energy.columns:
        raise ValueError("年度组成账本缺少annual_energy_quantile")
    keys = [
        "method",
        "planning_scenario",
        "year",
        "component",
        "industry",
        "component_type",
    ]
    annual_checks = assembly_closure.loc[
        assembly_closure.closure_scope.eq("component_annual"),
        [*keys, "actual_energy_MWh", "relative_error", "passed"],
    ].rename(
        columns={
            "actual_energy_MWh": "assembled_annual_energy_MWh",
            "relative_error": "assembly_relative_error",
            "passed": "assembly_passed",
        }
    )
    p50 = component_energy.loc[
        component_energy.annual_energy_quantile.astype(str).eq("P50")
    ].merge(
        annual_checks, on=keys, how="left", validate="one_to_one"
    )
    if p50[["assembled_annual_energy_MWh", "assembly_passed"]].isna().any().any():
        raise AssertionError("P50年度25组成权重账本缺少装配闭合证据")
    p50["assembly_status"] = "assembled_to_8760_P50_natural_total"
    intervals = component_energy.loc[
        ~component_energy.annual_energy_quantile.astype(str).eq("P50")
    ].copy()
    intervals["assembled_annual_energy_MWh"] = np.nan
    intervals["assembly_relative_error"] = np.nan
    intervals["assembly_passed"] = pd.NA
    intervals["assembly_status"] = "interval_only_not_assembled_to_8760"
    ledger = pd.concat([p50, intervals], ignore_index=True, sort=False)
    return ledger.sort_values(
        ["method", "planning_scenario", "year", "annual_energy_quantile", "component"],
        kind="mergesort",
    ).reset_index(drop=True)


def _assert_natural_energy_targets(
    annual_energy: pd.DataFrame,
    deterministic_annual: pd.DataFrame,
    representative_annual: pd.DataFrame,
    annual_curves: pd.DataFrame,
) -> None:
    """Prove that every generated 8760 track closes to upstream P50 natural energy."""

    targets = annual_energy[
        ["method", "scenario", "year", "P50_natural_MWh"]
    ].rename(
        columns={
            "scenario": "planning_scenario",
            "P50_natural_MWh": "target_natural_energy_MWh",
        }
    )
    keys = ["method", "planning_scenario", "year"]

    def check(frame: pd.DataFrame, label: str) -> None:
        evidence = frame.merge(targets, on=keys, how="left", validate="many_to_one")
        if evidence.target_natural_energy_MWh.isna().any():
            raise AssertionError(f"{label}缺少P50自然负荷年度目标")
        if not np.isclose(
            pd.to_numeric(evidence.annual_energy_MWh),
            pd.to_numeric(evidence.target_natural_energy_MWh),
            rtol=1e-10,
            atol=1e-3,
        ).all():
            raise AssertionError(f"{label}未闭合到P50自然负荷年度电量")

    check(deterministic_annual, "确定性年度指标")
    check(representative_annual, "P10/P50/P90真实代表轨迹")

    curve_energy = annual_curves.groupby(keys, as_index=False).agg(
        deterministic_baseline_MWh=("deterministic_baseline_MW", "sum"),
        P10_real_MWh=("P10_real_MW", "sum"),
        P50_real_MWh=("P50_real_MW", "sum"),
        P90_real_MWh=("P90_real_MW", "sum"),
    )
    curve_energy = curve_energy.merge(
        targets, on=keys, how="left", validate="one_to_one"
    )
    for column in (
        "deterministic_baseline_MWh",
        "P10_real_MWh",
        "P50_real_MWh",
        "P90_real_MWh",
    ):
        if not np.isclose(
            curve_energy[column],
            curve_energy.target_natural_energy_MWh,
            rtol=1e-10,
            atol=1e-3,
        ).all():
            raise AssertionError(f"逐时曲线{column}未闭合到P50自然负荷年度电量")


def _official_balance_tables(balance: object) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.DataFrame(
        [
            {
                "unit": balance.unit,
                "total_label": balance.total_label,
                "total_value_2025": balance.total_value_2025,
                "mapped_value_2025": balance.mapped_value_2025,
                "absolute_tolerance": balance.absolute_tolerance,
                "relative_tolerance": balance.relative_tolerance,
                "absolute_error": balance.absolute_error,
                "relative_error": balance.relative_error,
                "passed": balance.passed,
                "scope_note": (
                    "官方毛电量平衡校验表；模型年度目标另经厂用电、线损、"
                    "抽蓄用电桥接为自然负荷"
                ),
            }
        ]
    )
    departments = pd.DataFrame([asdict(item) for item in balance.department_balances])
    return summary, departments


def _load_input_ledger(
    *,
    load_workbook: str | Path,
    official_workbook: str | Path,
    sample_mapping_path: str | Path,
    official_mapping_path: str | Path,
    annual_energy_upstream: str | Path,
    industry_weight_evolution_path: str | Path,
):
    """读取并校验六项输入，产出输入台账、官方平衡表与行业映射（§3.3 输入侧）。"""
    gate = load_and_validate_inputs(
        load_workbook=load_workbook,
        official_workbook=official_workbook,
        sample_mapping_path=sample_mapping_path,
        official_mapping_path=official_mapping_path,
    )
    additional_manifest = build_input_manifest(
        {
            "annual_energy_upstream": annual_energy_upstream,
            "industry_weight_evolution": industry_weight_evolution_path,
        }
    )
    input_manifest = pd.DataFrame(
        [item.as_dict() for item in (*gate["manifest"], *additional_manifest)]
    )
    sample_mapping = gate["sample_mapping"]
    sample_energy = gate["sample_industry_energy"]
    official_balance = gate["official_balance"]
    excluded_energy_ledger = gate["excluded_energy_ledger"].copy()
    official_balance_summary, official_department_balance = _official_balance_tables(
        official_balance
    )
    sample_catalog = pd.DataFrame([asdict(item) for item in gate["sample_catalog"]])
    official_mapping = gate["official_mapping"]
    return (
        input_manifest,
        sample_catalog,
        official_mapping,
        sample_mapping,
        sample_energy,
        official_balance_summary,
        official_department_balance,
        excluded_energy_ledger,
    )


def _build_history_shapes(
    *,
    load_workbook: str | Path,
    sample_mapping: pd.DataFrame,
    year_values: tuple[int, ...],
):
    """历史→行业日聚合→典型日→形状参数→月包络与残差库（§3.3 步骤1—4、7）。"""
    system_history = read_system_history(load_workbook)
    user_records, user_quality = read_user_daily(load_workbook, include_new_load=False)
    industry_daily, industry_aggregate_diagnostics = aggregate_industry_daily(
        user_records
    )
    cell_quality = build_cell_quality(industry_daily)
    baseline_shapes, baseline_diagnostics = build_baseline_shapes(
        industry_daily, cell_quality
    )
    shape_history = identify_shape_parameters(baseline_shapes, cell_quality)
    traditional_industries = tuple(
        sample_mapping.loc[sample_mapping.is_traditional.astype(bool), "industry"].astype(str)
    )
    if set(industry_daily.industry.astype(str).unique()) != set(traditional_industries):
        raise AssertionError("行业日聚合结果必须覆盖且只覆盖24个传统细分行业")
    day_type_factors = estimate_province_day_type_factors(system_history, source_year=2023)
    future_parameters = build_future_parameter_table(
        baseline_shapes, shape_history, year_values
    )
    envelope_history, envelope_factors = build_monthly_envelope(system_history)
    residual_library, residual_diagnostics = build_interday_residual_library(
        system_history
    )
    return (
        system_history,
        user_quality,
        industry_daily,
        industry_aggregate_diagnostics,
        cell_quality,
        baseline_shapes,
        baseline_diagnostics,
        shape_history,
        traditional_industries,
        day_type_factors,
        future_parameters,
        envelope_history,
        envelope_factors,
        residual_library,
        residual_diagnostics,
    )


def _decompose_annual_energy(
    *,
    annual_energy_upstream: str | Path,
    industry_weight_evolution_path: str | Path,
    year_values: tuple[int, ...],
    scenario_values: tuple[str, ...],
    method: str,
    sample_energy: pd.DataFrame,
    excluded_energy_ledger: pd.DataFrame,
    envelope_factors: pd.DataFrame,
):
    """上游年度电量→24行业+修正组成的年电量分解与闭合（§3.3 步骤5）。"""
    annual_energy = load_upstream_annual_energy(
        annual_energy_upstream, years=year_values, method=method
    )
    annual_energy = annual_energy.loc[
        annual_energy.scenario.astype(str).isin(scenario_values)
    ].reset_index(drop=True)
    scope_bridge = load_upstream_scope_bridge(
        annual_energy_upstream, years=year_values, method=method
    )
    scope_bridge = scope_bridge.loc[
        scope_bridge.planning_scenario.astype(str).isin(scenario_values)
    ].reset_index(drop=True)
    expected_bridge_rows = len(year_values) * len(scenario_values) * len(REPRESENTATIVE_TRACKS)
    if len(scope_bridge) != expected_bridge_rows or scope_bridge.duplicated(
        ["method", "planning_scenario", "year", "quantile"]
    ).any():
        raise AssertionError("自然负荷scope_bridge覆盖不完整或键不唯一")

    official_2025 = excluded_energy_ledger.loc[
        pd.to_numeric(excluded_energy_ledger.year).eq(2025)
    ]
    if len(official_2025) != 1:
        raise AssertionError("三项排除电量台账必须恰有一条2025记录")
    official_natural_2025_10k_kWh = float(
        official_2025.iloc[0].natural_load_10k_kWh
    )
    official_natural_2025_MWh = float(official_2025.iloc[0].natural_load_MWh)
    if 2025 in year_values:
        anchor = scope_bridge.loc[pd.to_numeric(scope_bridge.year).eq(2025)]
        if not np.isclose(
            anchor.Q_natural_MWh,
            official_natural_2025_MWh,
            rtol=1e-10,
            atol=1e-3,
        ).all():
            raise AssertionError("上游2025自然负荷与官方三项扣除台账不一致")
    evolution = load_industry_weight_evolution(industry_weight_evolution_path)
    component_energy, component_energy_closure = build_industry_annual_energy(
        annual_energy,
        sample_energy,
        official_natural_total_2025=official_natural_2025_10k_kWh,
        evolution=evolution,
    )
    proxy_sector_weights = build_proxy_sector_weights(component_energy)
    target_month_energy = build_target_month_energy_table(
        annual_energy, envelope_factors
    )
    return (
        annual_energy,
        scope_bridge,
        evolution,
        official_2025,
        component_energy,
        component_energy_closure,
        proxy_sector_weights,
        target_month_energy,
    )


def _build_baseline_and_correction(
    *,
    baseline_shapes: pd.DataFrame,
    system_history: pd.DataFrame,
    excluded_energy_ledger: pd.DataFrame,
    sample_mapping_path: str | Path,
    official_mapping_path: str | Path,
    official_workbook: str | Path,
    future_parameters: pd.DataFrame,
    component_energy: pd.DataFrame,
):
    """2023基准形状、修正组成与未来行业组成形状（§3.3 步骤6 及组成形状）。"""
    if "is_future_baseline" in baseline_shapes.columns:
        baseline_2023 = baseline_shapes.loc[
            baseline_shapes.is_future_baseline.astype(bool)
        ].copy()
    elif "year" in baseline_shapes.columns:
        baseline_2023 = baseline_shapes.loc[
            pd.to_numeric(baseline_shapes.year).eq(2023)
        ].copy()
    else:
        raise ValueError("行业基准形状缺少2023基准标识")
    if baseline_2023.empty:
        raise RuntimeError("2023行业聚合medoid基准为空")

    actual_2023 = _system_history_year_hourly(system_history, 2023)
    province_typical_2023 = historical_season_daytype_typical_shapes(actual_2023)
    province_typical_2023 = province_typical_2023.rename(
        columns={
            "day_type": "day_group",
            "normalized_load": "province_typical_share",
        }
    )
    province_typical_2023["day_group"] = province_typical_2023.day_group.replace(
        {"普通周末": "周末", "法定节假日": "节假日"}
    )
    official_2023 = excluded_energy_ledger.loc[
        pd.to_numeric(excluded_energy_ledger.year).eq(2023)
    ]
    if len(official_2023) != 1:
        raise AssertionError("三项排除电量台账必须恰有一条2023记录")
    industry_energy_2023 = sample_industry_energy_year_frame(
        2023,
        sample_mapping_path=sample_mapping_path,
        official_mapping_path=official_mapping_path,
        official_workbook=official_workbook,
    )
    industry_energy_2023 = industry_energy_2023.loc[
        industry_energy_2023.is_traditional.astype(bool)
    ].copy()
    base_year_industry_components = build_base_year_industry_components(
        baseline_2023,
        industry_energy_2023[["industry", "official_energy"]],
        float(official_2023.iloc[0].natural_load_10k_kWh),
    )
    correction_weights_2023 = industry_energy_2023[
        ["industry", "official_energy"]
    ].copy()
    correction_weights_2023["share"] = (
        correction_weights_2023.official_energy
        / float(official_2023.iloc[0].natural_load_10k_kWh)
    )
    correction_weights_2023["weight_year"] = 2023
    correction = build_correction_artifacts(
        province_typical_2023[
            ["season", "day_group", "hour", "province_typical_share"]
        ],
        baseline_2023,
        correction_weights_2023[["industry", "share", "weight_year"]],
    )

    future_industry_components = build_future_industry_components(
        baseline_2023,
        future_parameters,
        component_energy,
        horizon=FORMAL_HORIZON,
        peak_valley_phi=FORMAL_PEAK_VALLEY_PHI,
    )
    future_total_component_shapes = compose_with_fixed_correction_shape(
        future_industry_components,
        correction.component,
        context_columns=("method", "planning_scenario", "year"),
    )
    proxy_sector_typical_shapes = build_proxy_sector_typical_shapes(
        future_industry_components
    )
    return (
        baseline_2023,
        province_typical_2023,
        base_year_industry_components,
        correction,
        future_industry_components,
        future_total_component_shapes,
        proxy_sector_typical_shapes,
    )


def run_pipeline(
    *,
    years: Iterable[int],
    output_dir: str | Path,
    bootstrap_scenarios: int,
    planning_scenarios: Sequence[str] = SCENARIO_ORDER,
    method: str = ANNUAL_ENERGY_METHOD,
    load_workbook: str | Path = DEFAULT_LOAD_WORKBOOK,
    official_workbook: str | Path = DEFAULT_OFFICIAL_WORKBOOK,
    annual_energy_upstream: str | Path = ANNUAL_ENERGY_UPSTREAM,
    sample_mapping_path: str | Path = SAMPLE_INDUSTRY_MAPPING_CSV,
    official_mapping_path: str | Path = OFFICIAL_2025_MAPPING_CSV,
    industry_weight_evolution_path: str | Path = INDUSTRY_WEIGHT_EVOLUTION_CSV,
    require_parquet: bool = False,
    materialize_bootstrap_daily: bool = False,
    block_length: int = 7,
    mode: str = "custom",
    run_label: str | None = None,
) -> dict[str, Any]:
    """运行完整新口径链路；Bootstrap按情景/年分块，避免全量日乘数常驻内存。"""

    _assert_current_method_contract()
    output_path = Path(output_dir).resolve()
    if output_path.exists() and (not output_path.is_dir() or any(output_path.iterdir())):
        raise FileExistsError(f"输出目录必须为空或不存在，拒绝覆盖：{output_path}")
    if str(method).strip() != ANNUAL_ENERGY_METHOD:
        raise ValueError(
            f"3.5.0正式与冒烟链路只允许年度电量方法{ANNUAL_ENERGY_METHOD}"
        )
    year_values = tuple(sorted({int(value) for value in years}))
    scenario_values = tuple(str(value) for value in planning_scenarios)
    if not year_values or any(year < 2025 for year in year_values):
        raise ValueError("规划年份不能为空且不得早于2025")
    if not scenario_values or set(scenario_values).difference(SCENARIO_ORDER):
        raise ValueError(f"规划情景必须取自{SCENARIO_ORDER}")
    if len(scenario_values) != len(set(scenario_values)):
        raise ValueError("规划情景不得重复")
    if int(bootstrap_scenarios) <= 0 or int(block_length) <= 0:
        raise ValueError("Bootstrap场景数和块长必须为正整数")

    (
        input_manifest,
        sample_catalog,
        official_mapping,
        sample_mapping,
        sample_energy,
        official_balance_summary,
        official_department_balance,
        excluded_energy_ledger,
    ) = _load_input_ledger(
        load_workbook=load_workbook,
        official_workbook=official_workbook,
        sample_mapping_path=sample_mapping_path,
        official_mapping_path=official_mapping_path,
        annual_energy_upstream=annual_energy_upstream,
        industry_weight_evolution_path=industry_weight_evolution_path,
    )
    (
        system_history,
        user_quality,
        industry_daily,
        industry_aggregate_diagnostics,
        cell_quality,
        baseline_shapes,
        baseline_diagnostics,
        shape_history,
        traditional_industries,
        day_type_factors,
        future_parameters,
        envelope_history,
        envelope_factors,
        residual_library,
        residual_diagnostics,
    ) = _build_history_shapes(
        load_workbook=load_workbook,
        sample_mapping=sample_mapping,
        year_values=year_values,
    )

    (
        annual_energy,
        scope_bridge,
        evolution,
        official_2025,
        component_energy,
        component_energy_closure,
        proxy_sector_weights,
        target_month_energy,
    ) = _decompose_annual_energy(
        annual_energy_upstream=annual_energy_upstream,
        industry_weight_evolution_path=industry_weight_evolution_path,
        year_values=year_values,
        scenario_values=scenario_values,
        method=method,
        sample_energy=sample_energy,
        excluded_energy_ledger=excluded_energy_ledger,
        envelope_factors=envelope_factors,
    )

    (
        baseline_2023,
        province_typical_2023,
        base_year_industry_components,
        correction,
        future_industry_components,
        future_total_component_shapes,
        proxy_sector_typical_shapes,
    ) = _build_baseline_and_correction(
        baseline_shapes=baseline_shapes,
        system_history=system_history,
        excluded_energy_ledger=excluded_energy_ledger,
        sample_mapping_path=sample_mapping_path,
        official_mapping_path=official_mapping_path,
        official_workbook=official_workbook,
        future_parameters=future_parameters,
        component_energy=component_energy,
    )

    assembly_closure_frames: list[pd.DataFrame] = []
    deterministic_daily_frames: list[pd.DataFrame] = []
    deterministic_monthly_frames: list[pd.DataFrame] = []
    deterministic_annual_frames: list[pd.DataFrame] = []
    typical_workday_frames: list[pd.DataFrame] = []
    season_daytype_typical_frames: list[pd.DataFrame] = []
    bootstrap_daily_frames: list[pd.DataFrame] = []
    bootstrap_check_frames: list[pd.DataFrame] = []
    bootstrap_ranking_frames: list[pd.DataFrame] = []
    bootstrap_annual_frames: list[pd.DataFrame] = []
    bootstrap_monthly_frames: list[pd.DataFrame] = []
    representative_annual_frames: list[pd.DataFrame] = []
    representative_monthly_frames: list[pd.DataFrame] = []
    annual_curve_absolute_frames: list[pd.DataFrame] = []
    annual_curve_normalized_frames: list[pd.DataFrame] = []
    all_trajectory_daily_frames: list[pd.DataFrame] = []
    adaptive_typical_frames: list[pd.DataFrame] = []
    typical_load_factor_frames: list[pd.DataFrame] = []
    normalized_monthly_peak_frames: list[pd.DataFrame] = []

    for planning_scenario in scenario_values:
        for year in year_values:
            baseline, assembly_closure = assemble_province_year(
                year,
                planning_scenario,
                component_energy,
                baseline_2023,
                day_type_factors,
                future_parameters,
                envelope_factors,
                correction.component,
                method=method,
                horizon=FORMAL_HORIZON,
                peak_valley_phi=FORMAL_PEAK_VALLEY_PHI,
                include_daily_detail=True,
            )
            assembly_closure_frames.append(assembly_closure)
            daily, monthly, annual = calculate_load_metrics(
                baseline[["timestamp", "load_MW"]]
            )
            daily = _add_context(
                daily,
                method=method,
                planning_scenario=planning_scenario,
                year=year,
                result_type="确定性基线",
            )
            monthly = _add_context(
                monthly,
                method=method,
                planning_scenario=planning_scenario,
                year=year,
                result_type="确定性基线",
            )
            annual = _add_context(
                annual,
                method=method,
                planning_scenario=planning_scenario,
                year=year,
                result_type="确定性基线",
            )
            deterministic_daily_frames.append(daily)
            deterministic_monthly_frames.append(monthly)
            deterministic_annual_frames.append(annual)
            typical_workday_frames.append(
                _add_context(
                    typical_workdays(baseline[["timestamp", "load_MW"]]),
                    method=method,
                    planning_scenario=planning_scenario,
                    year=year,
                    result_type="确定性基线典型工作日",
                )
            )
            season_daytype_typical_frames.append(
                _add_context(
                    season_daytype_typical_shapes(
                        baseline[["timestamp", "load_MW"]]
                    ),
                    method=method,
                    planning_scenario=planning_scenario,
                    year=year,
                    result_type="确定性基线季节日型典型日",
                )
            )

            baseline_daily_energy = daily.set_index("date").daily_energy_MWh
            baseline_daily_peak = daily.set_index("date").daily_max_MW
            multipliers, bootstrap_checks = bootstrap_daily_multipliers(
                residual_library,
                baseline_daily_energy,
                year,
                n_scenarios=int(bootstrap_scenarios),
                block_length=int(block_length),
            )
            multipliers["method"] = str(method)
            multipliers["planning_scenario"] = str(planning_scenario)
            multipliers["year"] = int(year)
            bootstrap_checks["method"] = str(method)
            bootstrap_checks["planning_scenario"] = str(planning_scenario)
            bootstrap_checks["year"] = int(year)
            bootstrap_check_frames.append(bootstrap_checks)

            annual_all, monthly_all = _evaluate_all_bootstrap_scenarios(
                baseline,
                multipliers,
                planning_scenario,
                year,
                method=method,
            )
            bootstrap_annual_frames.append(annual_all)
            bootstrap_monthly_frames.append(monthly_all)
            monthly_features, annual_features = build_scenario_features(
                multipliers,
                baseline_daily_energy,
                baseline_daily_peak,
            )
            selected, ranking = select_representative_scenarios(
                monthly_features, annual_features
            )
            bootstrap_daily_frames.append(
                _materialize_bootstrap_daily_output(
                    multipliers,
                    selected,
                    materialize_all=materialize_bootstrap_daily,
                )
            )
            ranking["method"] = str(method)
            ranking["planning_scenario"] = str(planning_scenario)
            ranking["year"] = int(year)
            bootstrap_ranking_frames.append(ranking)
            representative_annual, representative_monthly = _selected_metric_rows(
                annual_all, monthly_all, selected
            )
            representative_annual_frames.append(representative_annual)
            representative_monthly_frames.append(representative_monthly)
            representative_wide = _representative_curve_table(
                baseline,
                multipliers,
                selected,
                method=method,
                planning_scenario=planning_scenario,
                year=year,
            )
            annual_curve_absolute_frames.append(representative_wide)
            trajectory_long = _representative_curves_long(representative_wide)
            normalized_hourly = normalize_hourly_trajectories(trajectory_long).drop(
                columns="annual_peak_base_value"
            )
            normalized_hourly["method"] = str(method)
            normalized_hourly["annual_energy_basis"] = ANNUAL_ENERGY_BASIS
            normalized_hourly["unit"] = "p.u."
            annual_curve_normalized_frames.append(normalized_hourly)
            all_trajectory_daily_frames.append(
                _add_scope_metadata(_trajectory_daily_load_factors(trajectory_long)).assign(
                    method=str(method)
                )
            )
            adaptive_typical = build_adaptive_typical_days(trajectory_long)
            adaptive_typical["method"] = str(method)
            adaptive_typical["unit"] = "p.u.; 24_hour_sum_equals_1"
            adaptive_typical_frames.append(adaptive_typical)
            typical_factors = build_typical_day_load_factors(adaptive_typical)
            typical_factors["method"] = str(method)
            typical_load_factor_frames.append(typical_factors)
            normalized_monthly = build_monthly_maximum_curves(trajectory_long).drop(
                columns="annual_peak_base_value"
            )
            normalized_monthly["method"] = str(method)
            normalized_monthly["unit"] = "p.u."
            normalized_monthly_peak_frames.append(normalized_monthly)
            del multipliers, monthly_features, annual_features, annual_all, monthly_all

    deterministic_daily = _concat(deterministic_daily_frames)
    deterministic_monthly = _concat(deterministic_monthly_frames)
    deterministic_annual = _concat(deterministic_annual_frames)
    bootstrap_annual_metrics = _concat(bootstrap_annual_frames)
    bootstrap_monthly_metrics = _concat(bootstrap_monthly_frames)
    representative_annual_metrics = _concat(representative_annual_frames)
    representative_monthly_metrics = _concat(representative_monthly_frames)
    annual_curves_absolute = _concat(annual_curve_absolute_frames)
    annual_curves = _concat(annual_curve_normalized_frames)
    all_trajectory_daily = _concat(all_trajectory_daily_frames)
    adaptive_typical_days = _concat(adaptive_typical_frames)
    typical_day_load_factors = _concat(typical_load_factor_frames)
    normalized_monthly_peak_curves = _concat(normalized_monthly_peak_frames)
    assembly_closure = _concat(assembly_closure_frames)
    _assert_natural_energy_targets(
        annual_energy,
        deterministic_annual,
        representative_annual_metrics,
        annual_curves_absolute,
    )
    component_month_energy, component_daily_energy, daily_energy_checks = (
        _build_daily_energy_tables(assembly_closure)
    )
    annual_weight_ledger = _build_annual_weight_ledger(
        component_energy, assembly_closure
    )
    indicator_quantiles = _add_scope_metadata(
        annual_indicator_quantiles(bootstrap_annual_metrics)
    )
    monthly_quantiles = _add_scope_metadata(
        monthly_indicator_quantiles(bootstrap_monthly_metrics)
    )
    deterministic_annual_tracks = deterministic_annual.copy()
    deterministic_annual_tracks["trajectory"] = "deterministic_baseline"
    representative_annual_tracks = representative_annual_metrics.copy()
    representative_annual_tracks["trajectory"] = (
        representative_annual_tracks.representative_track.astype(str) + "_real"
    )
    trajectory_annual_metrics = _concat(
        [deterministic_annual_tracks, representative_annual_tracks]
    )
    deterministic_monthly_tracks = deterministic_monthly.copy()
    deterministic_monthly_tracks["trajectory"] = "deterministic_baseline"
    representative_monthly_tracks = representative_monthly_metrics.copy()
    representative_monthly_tracks["trajectory"] = (
        representative_monthly_tracks.representative_track.astype(str) + "_real"
    )
    trajectory_monthly_metrics = _concat(
        [deterministic_monthly_tracks, representative_monthly_tracks]
    )

    display_start = max(2025, min(year_values))
    display_end = min(2045, max(year_values))
    study_start = min(2030, max(year_values))
    study_end = max(year_values)
    full_display_years = set(range(display_start, display_end + 1))
    if (
        str(method) == "M1"
        and full_display_years.issubset(year_values)
        and "基准" in scenario_values
    ):
        display_workday_curves = extract_baseline_p50_workday_display(
            adaptive_typical_days, start_year=display_start, end_year=display_end
        )
        display_typical_day_load_factors = typical_day_load_factors.loc[
            typical_day_load_factors.year.between(display_start, display_end)
            & typical_day_load_factors.planning_scenario.eq("基准")
            & typical_day_load_factors.trajectory.eq("P50_real")
            & typical_day_load_factors.final_day_type.eq("工作日")
        ].reset_index(drop=True)
        display_monthly_metrics = trajectory_monthly_metrics.loc[
            trajectory_monthly_metrics.year.between(display_start, display_end)
            & trajectory_monthly_metrics.planning_scenario.eq("基准")
            & trajectory_monthly_metrics.trajectory.eq("P50_real")
        ].reset_index(drop=True)
        display_annual_metrics = trajectory_annual_metrics.loc[
            trajectory_annual_metrics.year.between(display_start, display_end)
            & trajectory_annual_metrics.planning_scenario.eq("基准")
            & trajectory_annual_metrics.trajectory.eq("P50_real")
        ].reset_index(drop=True)
        display_monthly_peak_curves = normalized_monthly_peak_curves.loc[
            normalized_monthly_peak_curves.year.between(display_start, display_end)
            & normalized_monthly_peak_curves.planning_scenario.eq("基准")
            & normalized_monthly_peak_curves.trajectory.eq("P50_real")
        ].reset_index(drop=True)
        display_m1_p50_annual_scale = select_m1_baseline_p50_annual_scale(
            annual_energy, years=range(display_start, display_end + 1)
        )
    else:
        display_workday_curves = pd.DataFrame()
        display_typical_day_load_factors = pd.DataFrame()
        display_monthly_metrics = pd.DataFrame()
        display_annual_metrics = pd.DataFrame()
        display_monthly_peak_curves = pd.DataFrame()
        display_m1_p50_annual_scale = pd.DataFrame()
    baseline_industries = set(baseline_2023.industry.astype(str).unique())
    classification_trace = pd.DataFrame(
        [
            {
                "industry": industry,
                "parameter_valid": industry in baseline_industries,
                "uses_own_parameters": True,
                "parameter_status": (
                    "own_industry_2022_2023_medoids"
                    if industry in baseline_industries
                    else "baseline_missing_2023"
                ),
            }
            for industry in traditional_industries
        ]
    )
    classification = build_industry_classification_table(
        sample_mapping, sample_energy, classification_trace
    )
    industry_inventory = classification.loc[
        classification.is_traditional.astype(bool)
    ].copy()
    if len(industry_inventory) != 24 or industry_inventory.industry.duplicated().any():
        raise AssertionError("行业清单必须恰有24个唯一传统细分行业")
    industry_inventory["shape_diagnostics_enabled"] = True
    industry_inventory["inventory_role"] = "25组成模型的24行业基本层；四大类仅作派生"
    day_type_factors_output = day_type_factors.copy()
    correction_diagnostics_table = correction.component[
        [
            "season",
            "day_group",
            "industry_share_sum",
            "correction_component_share",
            "province_shape_sum",
            "industry_contribution_sum",
            "correction_contribution_sum",
            "correction_shape_sum",
            "correction_shape_min",
            "reconstruction_max_abs_error",
            "industry_count",
        ]
    ].drop_duplicates(["season", "day_group"])
    correction_diagnostics_table["nonnegative_passed"] = (
        correction_diagnostics_table.correction_shape_min.ge(-1e-10)
    )
    correction_diagnostics_table["normalization_passed"] = np.isclose(
        correction_diagnostics_table.correction_shape_sum.to_numpy(float),
        1.0,
        rtol=0.0,
        atol=1e-10,
    )
    correction_diagnostics_table["reconstruction_passed"] = (
        correction_diagnostics_table.reconstruction_max_abs_error.le(1e-10)
    )
    diagnostic_columns = [
        column
        for column in (
            "industry",
            "season",
            "final_day_type",
            "day_group",
            "year",
            "H",
            "phi",
            "peak_valley_phi",
            "base_source_year",
            "trend_eligible",
            "flat_curve",
            "flat_borderline",
            "peak_valley_delta_abs",
            "peak_valley_delta_band",
            "peak_valley_evolution_enabled",
            "peak_valley_gate_reason",
            "peak_valley_gate_policy",
            "peak_valley_cumulative_damping",
            "peak_valley_target_requested",
            "peak_valley_target_range_limited",
            "peak_valley_target_applied",
            "peak_valley_target_clipped",
            "peak_valley_adjustment_status",
            "peak_valley_rate_after_four_window",
            "peak_valley_lambda",
            "peak_valley_rate_final",
            "future_peak_valley_rate",
            "future_load_factor",
            "future_cv",
            "future_evening_peak_hour",
            "future_raw_85pct_hours",
            "future_effective_control_hours",
            "evening_peak_changed_from_2023",
            "raw_85pct_set_changed_from_2023",
            "effective_control_set_changed_from_2023",
        )
        if column in future_parameters.columns
    ]
    shape_diagnostics = future_parameters[diagnostic_columns].copy()

    tables: dict[str, pd.DataFrame] = {
        "input_manifest": input_manifest,
        "sample_catalog": sample_catalog,
        "official_mapping": official_mapping,
        "official_balance_summary": official_balance_summary,
        "official_department_balance": official_department_balance,
        "excluded_energy_ledger": excluded_energy_ledger,
        "scope_bridge": scope_bridge,
        "industry_classification": classification,
        "industry_inventory": industry_inventory,
        "user_quality": user_quality,
        "industry_daily_aggregate": industry_daily,
        "industry_aggregate_diagnostics": industry_aggregate_diagnostics,
        "cell_quality": cell_quality,
        "shape_history": shape_history,
        "shape_diagnostics": shape_diagnostics,
        "baseline_shapes": baseline_shapes,
        "baseline_diagnostics": baseline_diagnostics,
        "day_type_factors": day_type_factors_output,
        "shape_future": future_parameters,
        "industry_weight_evolution": evolution,
        "annual_energy_interval": annual_energy,
        "component_annual_energy": component_energy,
        "component_energy_closure": component_energy_closure,
        "annual_weight_ledger": annual_weight_ledger,
        "derived_sector_energy": proxy_sector_weights,
        "target_month_energy": target_month_energy,
        "monthly_envelope_history": envelope_history,
        "monthly_envelope_factors": envelope_factors,
        "residual_library": residual_library,
        "residual_diagnostics": residual_diagnostics,
        "province_typical_2023": province_typical_2023,
        "base_year_industry_components": base_year_industry_components,
        "correction_component_shape": correction.component,
        "correction_component_diagnostics": correction_diagnostics_table,
        "future_industry_components": future_industry_components,
        "future_total_component_shapes": future_total_component_shapes,
        "proxy_sector_typical_shapes": proxy_sector_typical_shapes,
        "annual_assembly_closure": assembly_closure,
        "component_month_energy": component_month_energy,
        "component_daily_energy": component_daily_energy,
        "daily_energy_checks": daily_energy_checks,
        "energy_checks": assembly_closure.loc[
            ~assembly_closure.closure_scope.eq("component_day")
        ].reset_index(drop=True),
        "daily_load_factor": deterministic_daily,
        "all_trajectory_daily_load_factor": all_trajectory_daily,
        "deterministic_monthly_metrics": deterministic_monthly,
        "deterministic_annual_metrics": deterministic_annual,
        "trajectory_monthly_metrics": trajectory_monthly_metrics,
        "trajectory_annual_metrics": trajectory_annual_metrics,
        "province_season_daytype_typical": _concat(season_daytype_typical_frames),
        "typical_workdays": _concat(typical_workday_frames),
        "adaptive_typical_day_curves": adaptive_typical_days,
        "typical_day_load_factors": typical_day_load_factors,
        "bootstrap_monthly_checks": _add_scope_metadata(
            _concat(bootstrap_check_frames)
        ),
        "bootstrap_scenario_ranking": _add_scope_metadata(
            _concat(bootstrap_ranking_frames)
        ),
        "bootstrap_scenario_annual_metrics": bootstrap_annual_metrics,
        "bootstrap_scenario_monthly_metrics": bootstrap_monthly_metrics,
        "indicator_quantiles": indicator_quantiles,
        "monthly_indicator_quantiles": monthly_quantiles,
        "representative_annual_metrics": representative_annual_metrics,
        "representative_monthly_metrics": representative_monthly_metrics,
        "monthly_maximum_curves_pu": normalized_monthly_peak_curves,
        "display_workday_curves_2025_2045": display_workday_curves,
        "display_typical_day_load_factors_2025_2045": display_typical_day_load_factors,
        "display_monthly_metrics_2025_2045": display_monthly_metrics,
        "display_annual_metrics_2025_2045": display_annual_metrics,
        "display_monthly_peak_curves_2025_2045": display_monthly_peak_curves,
        "display_m1_p50_annual_scale_2025_2045": display_m1_p50_annual_scale,
    }
    tables["bootstrap_daily_multipliers"] = _add_scope_metadata(
        _concat(bootstrap_daily_frames)
    )
    _assert_no_legacy_outputs(tables)
    _assert_method_consistency(tables, method)

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    # 正式运行先验证并写出强制Parquet；若引擎缺失，在任何CSV结果表落盘前失败。
    hourly_path, hourly_format, hourly_warning = _write_hourly_output(
        annual_curves,
        directory / "逐时曲线",
        stem="province_hourly_normalized_trajectories",
        require_parquet=require_parquet,
    )
    table_manifest = save_result_tables(tables, directory / "数据表")
    display_directory = directory / f"展示_{display_start}-{display_end}_基准P50"
    display_directory.mkdir(parents=True, exist_ok=True)
    display_tables = {
        "四季工作日标幺曲线_长表": display_workday_curves,
        "四季工作日典型日日负荷率": display_typical_day_load_factors,
        "月度指标与月不均衡系数": display_monthly_metrics,
        "年度指标与年利用小时数": display_annual_metrics,
        "12个月月最大负荷标幺曲线": display_monthly_peak_curves,
        "M1基准情景年度电量P50": display_m1_p50_annual_scale,
    }
    display_files: dict[str, str] = {}
    for display_name, display_frame in display_tables.items():
        display_path = display_directory / f"{display_name}.csv"
        display_frame.to_csv(
            display_path,
            index=False,
            encoding="utf-8-sig",
            date_format="%Y-%m-%d %H:%M:%S",
        )
        display_files[display_name] = display_path.relative_to(directory).as_posix()

    formal_directory = directory / f"正式调用_{study_start}-{study_end}"
    formal_directory.mkdir(parents=True, exist_ok=True)
    formal_index = pd.DataFrame(
        [
            {
                "result_name": "normalized_hourly_trajectories",
                "file": hourly_path.relative_to(directory).as_posix(),
                "formal_year_filter": f"{study_start}<=year<={study_end}",
                "curve_unit": "p.u.; annual maximum equals 1",
            },
            *[
                {
                    "result_name": name,
                    "file": (Path("数据表") / entry["file"]).as_posix(),
                    "formal_year_filter": f"{study_start}<=year<={study_end} when year exists",
                    "curve_unit": (
                        "p.u." if "curve" in name or "typical" in name else "scalar metric"
                    ),
                }
                for name, entry in table_manifest.items()
            ],
        ]
    )
    formal_index_path = formal_directory / "调用索引.csv"
    formal_index.to_csv(formal_index_path, index=False, encoding="utf-8-sig")
    metadata: dict[str, Any] = {
        "status": "success",
        "model_version": MODEL_VERSION,
        "method_revision": METHOD_REVISION,
        "mode": str(mode),
        "run_label": str(run_label or f"{mode}运行"),
        "method": str(method),
        "annual_energy_method": str(method),
        "industry_daily_aggregation_method": (
            "user_day_normalize_then_equal_weight"
        ),
        "future_shape_baseline": "2023_industry_aggregate_medoid",
        "peak_valley_extrapolation_method": PEAK_VALLEY_EXTRAPOLATION_METHOD,
        "shape_extrapolation_method": SHAPE_EXTRAPOLATION_METHOD,
        "four_window_final_share_policy": FOUR_WINDOW_FINAL_SHARE_POLICY,
        "peak_valley_target_priority": PEAK_VALLEY_TARGET_PRIORITY,
        "late_peak_control_policy": LATE_PEAK_CONTROL_POLICY,
        "day_type_factor_scope": DAY_TYPE_FACTOR_SCOPE,
        "peak_valley_extrapolation_enabled": bool(
            PEAK_VALLEY_EXTRAPOLATION_ENABLED
        ),
        "peak_valley_gate_policy": PEAK_VALLEY_GATE_POLICY,
        "peak_valley_delta_lower_bound": (
            PEAK_VALLEY_EVOLUTION_DELTA_LOWER_BOUND
        ),
        "peak_valley_delta_upper_bound": (
            PEAK_VALLEY_EVOLUTION_DELTA_UPPER_BOUND
        ),
        "flat_load_factor_threshold": FLAT_LOAD_FACTOR_THRESHOLD,
        "flat_cv_threshold": FLAT_CV_THRESHOLD,
        "peak_damping_enabled": bool(PEAK_DAMPING_ENABLED),
        "anchor_repair_enabled": bool(ANCHOR_REPAIR_ENABLED),
        "years": list(year_values),
        "four_window_hours": [[0, 5], [6, 11], [12, 16], [17, 23]],
        "boundary_smoothing_hours": [5, 6, 11, 12, 16, 17],
        "boundary_smoothing_method": "simultaneous_3h_mean_then_window_energy_correction",
        "planning_scenarios": list(scenario_values),
        "bootstrap_scenarios_per_context": int(bootstrap_scenarios),
        "block_length": int(block_length),
        "formal_horizon": FORMAL_HORIZON,
        "formal_peak_valley_phi": FORMAL_PEAK_VALLEY_PHI,
        "energy_scope": NATURAL_LOAD_SCOPE,
        "curve_energy_scope": NATURAL_LOAD_SCOPE,
        "historical_shape_scope": HISTORICAL_SHAPE_SCOPE,
        "annual_energy_basis": ANNUAL_ENERGY_BASIS,
        "trajectory_quantile_basis": TRAJECTORY_QUANTILE_BASIS,
        "annual_special_deduction_applied": True,
        "hourly_special_deduction_applied": False,
        "special_items_in_correction_component": False,
        "correction_shape_status": CORRECTION_SHAPE_STATUS,
        "future_special_item_split_available": False,
        "special_deduction_definition": (
            "plant_auxiliary_power+line_loss+pumped_storage_charging"
        ),
        "official_2025_deduction_ratio": float(
            official_2025.iloc[0].deduction_ratio
        ),
        "component_count": 25,
        "new_load_deducted": False,
        "new_load_independently_shaped": False,
        "new_load_in_correction_component": True,
        "external_new_load_overlay": False,
        "materialize_bootstrap_daily": bool(materialize_bootstrap_daily),
        "bootstrap_daily_output_scope": (
            "all_bootstrap_scenarios"
            if materialize_bootstrap_daily
            else "P10/P50/P90_representative_scenarios_only"
        ),
        "hourly_output": {
            "file": hourly_path.relative_to(directory).as_posix(),
            "format": hourly_format,
            "warning": hourly_warning,
            "unit": "p.u.",
            "normalization": "each year/scenario/trajectory divided by its own annual maximum",
        },
        "formal_call_index": formal_index_path.relative_to(directory).as_posix(),
        "formal_study_period": [study_start, study_end],
        "display_period": [display_start, display_end],
        "display_contract": "M1 baseline annual-energy P50 + P50_real bootstrap trajectory",
        "display_files": display_files,
        "table_manifest": table_manifest,
        "correction_diagnostics": correction.diagnostics,
    }
    metadata_path = directory / "run_metadata.json"
    run_manifest_path = directory / "run_manifest.json"
    method_manifest_path = directory / (
        "参数与校验_冒烟验证.json" if str(mode) == "smoke" else "参数与校验_正式运行.json"
    )
    status_path = directory / "运行状态.txt"
    metadata["metadata_file"] = metadata_path.relative_to(directory).as_posix()
    metadata["run_manifest_file"] = run_manifest_path.relative_to(directory).as_posix()
    metadata["method_manifest_file"] = method_manifest_path.relative_to(directory).as_posix()
    metadata["status_file"] = status_path.relative_to(directory).as_posix()
    validate_run_manifest(metadata, require_parquet=require_parquet)
    payload = json.dumps(metadata, ensure_ascii=False, indent=2)
    for target in (metadata_path, run_manifest_path, method_manifest_path):
        temporary = target.with_suffix(".tmp.json")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(target)
    temporary_status = directory / "运行状态.tmp.txt"
    temporary_status.write_text(
        "\n".join(
            [
                "运行状态：成功",
                f"运行标签：{metadata['run_label']}",
                f"年度电量主方法：{metadata['method']}（上游自然负荷P10/P50/P90；确定性8760采用P50自然负荷）",
                f"范围：{min(year_values)}—{max(year_values)}年；规划情景={','.join(scenario_values)}",
                f"Bootstrap：每个规划情景×年份{int(bootstrap_scenarios)}条场景",
                "组成口径：24个传统细分行业+1个修正组成；闭合到自然负荷电量",
                "三项扣除：厂用电、线损、抽蓄用电已在年度层扣除；未来按上游固定比例扣除且不重复扣减",
                "逐时口径：历史全省曲线仅作调度形状代理，不虚构逐时三项拆分或再次扣除",
                "修正组成：承接新型负荷和24行业未覆盖的自然负荷；三项排除电量不进入",
                "行业形状：用户日先归一化，同一行业同一自然日有效用户等权聚合；2023行业medoid为唯一未来基准",
                "形状外推：按0—5、6—11、12—16、17—23四时段目标同比例迁移，交界六小时局部三点平均并恢复各时段电量；仅在历史充分、2023基准非平坦且峰谷差率绝对变化量位于6.5%—25%闭区间时执行第二阶段",
                "形状优先级：门控未通过时四时段中间曲线即最终曲线；门控通过时峰谷差率最终优先，最终四时段份额只统计、不恢复；晚峰与控制时段只作诊断",
                "日型系数：由2023全省负荷识别并统一应用于全部25个组成",
                f"逐时结果：{hourly_path.name}（{hourly_format}；逐轨迹年最大值归一为1）",
                f"展示口径：M1基准情景年度电量P50 + Bootstrap P50_real；{display_start}—{display_end}四季工作日",
                f"Parquet要求：{'正式强制且已满足' if require_parquet else '非正式运行允许明确回退'}",
                f"数据表：{len(table_manifest)}张；不含代表用户、负荷持续曲线、旧Gamma/晚峰位移δ、rho或锚点修复字段",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    temporary_status.replace(status_path)
    return metadata


def run_model(
    years: Iterable[int],
    n_scenarios: int,
    output_dir: str | Path,
    *,
    run_label: str,
    method: str = ANNUAL_ENERGY_METHOD,
    materialize_bootstrap_daily: bool = False,
    require_parquet: bool = False,
    mode: str | None = None,
) -> dict[str, Path]:
    """兼容项目运行脚本的稳定包装器；计算口径固定为自然负荷25组成。"""

    metadata = run_pipeline(
        years=years,
        output_dir=output_dir,
        bootstrap_scenarios=n_scenarios,
        method=method,
        require_parquet=require_parquet,
        materialize_bootstrap_daily=materialize_bootstrap_daily,
        mode=mode or ("formal" if require_parquet else "custom"),
        run_label=run_label,
    )
    directory = Path(output_dir).resolve()
    return {
        "output_dir": directory,
        "tables_manifest": directory / "数据表" / "tables_manifest.json",
        "hourly_output": directory / metadata["hourly_output"]["file"],
        "metadata": directory / metadata["metadata_file"],
        "status": directory / metadata["status_file"],
    }


def run_smoke(
    years: Iterable[int] = SMOKE_YEARS,
    n_scenarios: int = SMOKE_BOOTSTRAP_SCENARIOS,
    output_dir: str | Path = SMOKE_OUTPUT_DIR,
    *,
    method: str = ANNUAL_ENERGY_METHOD,
    materialize_bootstrap_daily: bool = False,
) -> dict[str, Path]:
    """运行小规模结构与闭合验证；逐时文件可明确回退CSV.gz。"""

    return run_model(
        years,
        n_scenarios,
        output_dir,
        run_label="3.5.0自然负荷：24行业+修正组成冒烟验证",
        method=method,
        materialize_bootstrap_daily=materialize_bootstrap_daily,
        require_parquet=False,
        mode="smoke",
    )


def run_full_example(
    years: Iterable[int] = FORMAL_YEARS,
    n_scenarios: int = FORMAL_BOOTSTRAP_SCENARIOS,
    output_dir: str | Path | None = None,
    *,
    method: str = ANNUAL_ENERGY_METHOD,
    materialize_bootstrap_daily: bool = False,
) -> dict[str, Path]:
    """运行2025—2060一次性底座；正式调用期为2030—2060。"""

    if output_dir is None:
        raise ValueError("正式全量必须显式提供output_dir")
    return run_model(
        years,
        n_scenarios,
        output_dir,
        run_label=(
            f"2025—2060正式全量（2030—2060正式调用；年度电量{method}；"
            "24行业+修正组成）"
        ),
        method=method,
        materialize_bootstrap_daily=materialize_bootstrap_daily,
        require_parquet=True,
        mode="formal",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="2025/2030×三情景×20场景")
    mode.add_argument(
        "--formal",
        "--full-example",
        dest="formal",
        action="store_true",
        help="2025—2060×三情景×1000场景",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--method",
        choices=("M1",),
        default=ANNUAL_ENERGY_METHOD,
        help=f"上游年度电量方法（模型消费其自然负荷桥接），默认{ANNUAL_ENERGY_METHOD}",
    )
    parser.add_argument("--materialize-bootstrap-daily", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.formal and args.output_dir is None:
        raise SystemExit("正式运行必须显式提供 --output-dir")
    if args.smoke:
        output_dir = args.output_dir or SMOKE_OUTPUT_DIR
        metadata = run_pipeline(
            years=SMOKE_YEARS,
            output_dir=output_dir,
            bootstrap_scenarios=SMOKE_BOOTSTRAP_SCENARIOS,
            method=args.method,
            require_parquet=False,
            materialize_bootstrap_daily=args.materialize_bootstrap_daily,
            mode="smoke",
            run_label=(
                f"3.5.0自然负荷；年度电量{args.method}；"
                "24行业+修正组成冒烟验证"
            ),
        )
    else:
        metadata = run_pipeline(
            years=FORMAL_YEARS,
            output_dir=args.output_dir,
            bootstrap_scenarios=FORMAL_BOOTSTRAP_SCENARIOS,
            method=args.method,
            require_parquet=True,
            materialize_bootstrap_daily=args.materialize_bootstrap_daily,
            mode="formal",
            run_label=(
                f"2025—2060正式全量（2030—2060正式调用；年度电量{args.method}；"
                "24行业+修正组成）"
            ),
        )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FORMAL_BOOTSTRAP_SCENARIOS",
    "FORMAL_YEARS",
    "SMOKE_BOOTSTRAP_SCENARIOS",
    "SMOKE_YEARS",
    "_evaluate_all_bootstrap_scenarios",
    "_write_hourly_output",
    "main",
    "run_full_example",
    "run_model",
    "run_pipeline",
    "run_smoke",
]
