"""Independent, read-only validation for saved formal model results.

Run after the model has finished::

    python -m workstreams_load_characteristic_model_v2.formal_output_validation OUTPUT_DIR

The validator never launches the model and never copies large result tables.
It reads the saved files referenced by the manifest and reports whether the
2030--2060 formal-call directory is an index over directly filterable results.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .calendar_utils import calendar_frame
from .result_tables import (
    PROVINCE_SHARED_FACTOR_SCOPE,
    REQUIRED_FORMAL_TABLES,
    validate_result_tables_contract,
    validate_run_manifest,
    validate_tables_manifest_contract,
)

FORMAL_BASE_YEARS = tuple(range(2025, 2061))
FORMAL_STUDY_YEARS = tuple(range(2030, 2061))
DISPLAY_YEARS = tuple(range(2025, 2046))
PLANNING_SCENARIOS = ("低碳", "基准", "高电气化")
SAVED_TRAJECTORIES = (
    "deterministic_baseline",
    "P10_real",
    "P50_real",
    "P90_real",
)
SEASONS = ("春季", "夏季", "秋季", "冬季")
SHAPE_COLS = tuple(f"p{hour:02d}" for hour in range(24))


@dataclass
class _Collector:
    checks: list[dict[str, Any]] = field(default_factory=list)

    def passed(self, name: str, detail: str, **evidence: Any) -> None:
        self.checks.append(
            {"name": name, "status": "passed", "detail": detail, **evidence}
        )

    def failed(self, name: str, detail: str, **evidence: Any) -> None:
        self.checks.append(
            {"name": name, "status": "failed", "detail": detail, **evidence}
        )

    def warning(self, name: str, detail: str, **evidence: Any) -> None:
        self.checks.append(
            {"name": name, "status": "warning", "detail": detail, **evidence}
        )

    def run(self, name: str, function: Any) -> Any | None:
        try:
            result, detail, evidence = function()
            if result:
                self.passed(name, detail, **evidence)
            else:
                self.failed(name, detail, **evidence)
            return result
        except Exception as exc:  # collect all independent failures
            self.failed(name, f"{type(exc).__name__}: {exc}")
            return None


class _ResultStore:
    def __init__(self, output_dir: str | Path):
        self.root = Path(output_dir).resolve()
        self.table_dir = self.root / "数据表"
        self.metadata = self._load_json(self.root / "run_metadata.json")
        self.table_manifest = self._load_json(self.table_dir / "tables_manifest.json")
        self._cache: dict[str, pd.DataFrame] = {}

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(path)
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(value, dict):
            raise ValueError(f"JSON根节点必须是对象：{path}")
        return value

    def table_path(self, name: str) -> Path:
        entry = self.table_manifest.get(name)
        if isinstance(entry, Mapping) and isinstance(entry.get("file"), str):
            candidate = (self.table_dir / str(entry["file"])).resolve()
            if candidate.parent == self.table_dir.resolve() and candidate.is_file():
                return candidate
        raise FileNotFoundError(f"结果清单未指向有效表文件：{name}")

    @staticmethod
    def read_path(path: Path) -> pd.DataFrame:
        lower = path.name.lower()
        if lower.endswith(".parquet"):
            return pd.read_parquet(path)
        if lower.endswith(".csv") or lower.endswith(".csv.gz"):
            return pd.read_csv(path, low_memory=False)
        raise ValueError(f"不支持的结果格式：{path}")

    def table(self, name: str) -> pd.DataFrame:
        if name not in self._cache:
            self._cache[name] = self.read_path(self.table_path(name))
        return self._cache[name].copy()

    def hourly_path(self) -> Path:
        metadata_entry = self.metadata.get("hourly_output", {})
        recorded = metadata_entry.get("file") if isinstance(metadata_entry, Mapping) else None
        if isinstance(recorded, str):
            recorded_path = Path(recorded)
            for candidate in (self.root / recorded_path, self.root / "逐时曲线" / recorded_path.name):
                if candidate.is_file() and candidate.suffix.lower() == ".parquet":
                    return candidate.resolve()
        raise FileNotFoundError("正式逐时轨迹必须是运行清单指向的Parquet文件")

    def hourly(self) -> pd.DataFrame:
        return self.read_path(self.hourly_path())


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{label}缺少字段：{sorted(missing)}")


def _finite_nonnegative(frame: pd.DataFrame, column: str, label: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError(f"{label}.{column}含空值、非有限值或负值")
    return values


def _expected_product(
    frame: pd.DataFrame,
    columns: Sequence[str],
    expected_values: Sequence[Sequence[Any]],
    label: str,
) -> tuple[bool, str, dict[str, Any]]:
    _require_columns(frame, columns, label)
    if frame.duplicated(list(columns)).any():
        return False, f"{label}键重复", {"duplicate_rows": int(frame.duplicated(list(columns)).sum())}
    expected = pd.MultiIndex.from_product(expected_values, names=list(columns))
    actual = pd.MultiIndex.from_frame(frame[list(columns)])
    missing = expected.difference(actual)
    extra = actual.difference(expected)
    ok = len(missing) == 0 and len(extra) == 0 and len(frame) == len(expected)
    return (
        ok,
        f"{label}覆盖{'完整' if ok else '不完整'}",
        {
            "expected_rows": len(expected),
            "actual_rows": len(frame),
            "missing_examples": [list(value) for value in missing[:5]],
            "extra_examples": [list(value) for value in extra[:5]],
        },
    )


def _years_from(frame: pd.DataFrame) -> set[int]:
    _require_columns(frame, ["year"], "年份表")
    years = pd.to_numeric(frame.year, errors="coerce")
    if years.isna().any() or not np.equal(years, np.floor(years)).all():
        raise ValueError("year必须为整数")
    return set(years.astype(int))


def _validate_grouped_curve(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    value_column: str,
    closure: str,
    tolerance: float,
    hour_column: str | None = "hour",
) -> tuple[bool, str, dict[str, Any]]:
    required = [*group_columns, value_column]
    if hour_column is not None:
        required.append(hour_column)
    _require_columns(frame, required, value_column)
    values = _finite_nonnegative(frame, value_column, "曲线")
    if hour_column is not None:
        hours = pd.to_numeric(frame[hour_column], errors="coerce")
        if hours.isna().any() or not hours.between(0, 23).all():
            raise ValueError(f"{value_column}的hour必须为0..23")
        duplicate = frame.duplicated([*group_columns, hour_column])
        if duplicate.any():
            raise ValueError(f"{value_column}曲线小时键重复")
        counts = frame.groupby(list(group_columns), observed=True)[hour_column].nunique()
        if not counts.eq(24).all():
            raise ValueError(f"{value_column}至少一条曲线不是24点")
    grouped = pd.Series(values, index=frame.index).groupby(
        [frame[column] for column in group_columns], observed=True
    )
    totals = grouped.sum() if closure == "sum" else grouped.max()
    deviations = np.abs(totals.to_numpy(float) - 1.0)
    ok = bool((deviations <= tolerance).all())
    return (
        ok,
        f"{value_column}按{closure}标幺{'通过' if ok else '失败'}",
        {
            "curve_count": int(len(totals)),
            "max_abs_deviation": float(deviations.max(initial=0.0)),
        },
    )


def _validate_metadata(
    metadata: Mapping[str, Any],
    base_years: tuple[int, ...],
    scenarios: tuple[str, ...],
    expected_bootstrap: int,
) -> tuple[bool, str, dict[str, Any]]:
    # This is the release gate shared with reporting/workbook preparation.  It
    # rejects a successful-looking stale directory, including 3.4.0, before numerical checks
    # can make that directory appear reusable.
    validate_run_manifest(metadata, require_success=True, require_parquet=True)
    actual_years = tuple(int(value) for value in metadata.get("years", []))
    actual_scenarios = tuple(str(value) for value in metadata.get("planning_scenarios", []))
    conditions = {
        "status_success": metadata.get("status") == "success",
        "method_M1": metadata.get("method") == "M1",
        "annual_energy_method_M1": metadata.get("annual_energy_method") == "M1",
        "shape_method_3_4_1": metadata.get("shape_extrapolation_method")
        == "four_window_then_local_smoothing_then_gated_peak_valley_mean_scaling",
        "four_window_final_share_gated": metadata.get(
            "four_window_final_share_policy"
        )
        == "exact_if_peak_valley_skipped_diagnostic_if_applied",
        "peak_valley_extrapolation_enabled": metadata.get(
            "peak_valley_extrapolation_enabled"
        )
        is True,
        "peak_valley_gate_policy_current": metadata.get("peak_valley_gate_policy")
        == (
            "trend_eligible_and_nonflat_and_abs_delta_within_closed_interval_0.065_0.25"
        ),
        "peak_valley_delta_lower_bound_0_065": metadata.get(
            "peak_valley_delta_lower_bound"
        )
        == 0.065,
        "peak_valley_delta_upper_bound_0_25": metadata.get(
            "peak_valley_delta_upper_bound"
        )
        == 0.25,
        "flat_load_factor_threshold_0_80": metadata.get(
            "flat_load_factor_threshold"
        )
        == 0.80,
        "flat_cv_threshold_0_15": metadata.get("flat_cv_threshold") == 0.15,
        "late_peak_diagnostic_only": metadata.get("late_peak_control_policy")
        == "diagnostic_only",
        "province_shared_day_type_factor": metadata.get("day_type_factor_scope")
        == PROVINCE_SHARED_FACTOR_SCOPE,
        "formal_four_window_horizon_8": metadata.get("formal_horizon") == 8,
        "formal_peak_valley_phi_08": metadata.get(
            "formal_peak_valley_phi"
        )
        == 0.8,
        "formal_hourly_parquet": str(
            metadata.get("hourly_output", {}).get("format", "")
            if isinstance(metadata.get("hourly_output"), Mapping)
            else ""
        ).lower()
        == "parquet",
        "annual_basis_P50": metadata.get("annual_energy_basis") == "P50_natural_total",
        "years_exact": actual_years == base_years,
        "scenarios_exact": actual_scenarios == scenarios,
        "bootstrap_exact": int(metadata.get("bootstrap_scenarios_per_context", -1))
        == int(expected_bootstrap),
    }
    ok = all(conditions.values())
    return ok, "运行元数据正式口径" + ("通过" if ok else "不一致"), conditions


def _validate_result_schema(store: _ResultStore) -> tuple[bool, str, dict[str, Any]]:
    """Validate the 3.5.0 table contract while loading only its small schemas.

    ``component_daily_energy`` can be very large in a formal run.  Its declared
    columns and version stamp are checked from the manifest, while only
    ``shape_future`` and ``day_type_factors`` are loaded because their values
    are part of the method contract rather than merely a storage contract.
    """

    validate_tables_manifest_contract(store.table_manifest)
    schema_only: dict[str, pd.DataFrame] = {}
    for name in REQUIRED_FORMAL_TABLES:
        entry = store.table_manifest[name]
        if name in {"shape_future", "day_type_factors"}:
            schema_only[name] = store.table(name)
        else:
            schema_only[name] = pd.DataFrame(columns=list(entry["columns"]))
        # Every manifest target must exist even when its contents are not read.
        store.table_path(name)
    validate_result_tables_contract(schema_only)
    return (
        True,
        "正式结果表符合3.5.0核心表、字段与共享日型系数契约",
        {
            "required_table_count": len(REQUIRED_FORMAL_TABLES),
            "shape_future_rows_checked": len(schema_only["shape_future"]),
            "day_type_factor_rows_checked": len(schema_only["day_type_factors"]),
        },
    )


def _validate_annual_energy_and_trajectories(
    annual: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    base_years: tuple[int, ...],
    scenarios: tuple[str, ...],
    trajectories: tuple[str, ...],
    rtol: float,
    atol: float,
) -> tuple[bool, str, dict[str, Any]]:
    _require_columns(
        annual,
        ["method", "scenario", "year", "P50_natural_MWh"],
        "年度电量区间",
    )
    annual = annual.loc[annual.method.astype(str).eq("M1")].copy()
    coverage_ok, _, coverage = _expected_product(
        annual,
        ["year", "scenario"],
        [base_years, scenarios],
        "M1年度电量P50",
    )
    targets = annual[["year", "scenario", "P50_natural_MWh"]].rename(
        columns={
            "scenario": "planning_scenario",
            "P50_natural_MWh": "target_P50_natural_MWh",
        }
    )
    target_values = pd.to_numeric(targets.target_P50_natural_MWh, errors="coerce")
    if target_values.isna().any() or (target_values <= 0).any():
        raise ValueError("M1年度P50自然负荷必须为有限正数")

    required = [
        "method",
        "planning_scenario",
        "year",
        "trajectory",
        "annual_energy_MWh",
    ]
    _require_columns(metrics, required, "各轨迹年度指标")
    metrics = metrics.loc[metrics.method.astype(str).eq("M1")].copy()
    metric_coverage_ok, _, metric_coverage = _expected_product(
        metrics,
        ["year", "planning_scenario", "trajectory"],
        [base_years, scenarios, trajectories],
        "各轨迹年度指标",
    )
    joined = metrics.merge(
        targets,
        on=["year", "planning_scenario"],
        how="left",
        validate="many_to_one",
    )
    actual = pd.to_numeric(joined.annual_energy_MWh, errors="coerce").to_numpy(float)
    expected = pd.to_numeric(
        joined.target_P50_natural_MWh, errors="coerce"
    ).to_numpy(float)
    finite = np.isfinite(actual).all() and np.isfinite(expected).all()
    close = finite and np.isclose(actual, expected, rtol=rtol, atol=atol).all()
    max_error = float(np.max(np.abs(actual - expected))) if finite and len(actual) else float("nan")
    quantile_ok = True
    if "annual_energy_quantile" in joined:
        quantile_ok = set(joined.annual_energy_quantile.astype(str)) == {"P50"}
    ok = bool(coverage_ok and metric_coverage_ok and close and quantile_ok)
    return (
        ok,
        "所有保存轨迹年度电量均闭合到同情景M1 P50_natural_total"
        if ok
        else "年度P50目标或轨迹闭合失败",
        {
            "annual_coverage": coverage,
            "trajectory_coverage": metric_coverage,
            "max_abs_energy_error_MWh": max_error,
            "quantile_field_is_P50": bool(quantile_ok),
        },
    )


def _validate_hourly(
    hourly: pd.DataFrame,
    *,
    base_years: tuple[int, ...],
    scenarios: tuple[str, ...],
    trajectories: tuple[str, ...],
    tolerance: float,
) -> tuple[bool, str, dict[str, Any]]:
    keys = ["year", "planning_scenario", "trajectory"]
    _require_columns(hourly, [*keys, "timestamp", "load_pu"], "年度逐时标幺轨迹")
    frame = hourly.copy()
    frame["timestamp"] = pd.to_datetime(frame.timestamp, errors="coerce")
    if frame.timestamp.isna().any():
        raise ValueError("年度逐时标幺轨迹含无效时间戳")
    frame["year"] = pd.to_numeric(frame.year, errors="raise").astype(int)
    if not frame.timestamp.dt.year.eq(frame.year).all():
        raise ValueError("逐时时间戳与year不一致")
    values = _finite_nonnegative(frame, "load_pu", "年度逐时标幺轨迹")
    if np.any(values > 1.0 + tolerance):
        raise ValueError("年度逐时标幺值超过1")
    if frame.duplicated([*keys, "timestamp"]).any():
        raise ValueError("年度逐时轨迹时间戳键重复")
    grouped = frame.groupby(keys, observed=True)
    summary = grouped.agg(
        rows=("timestamp", "size"),
        unique_timestamps=("timestamp", "nunique"),
        max_pu=("load_pu", "max"),
    ).reset_index()
    coverage_ok, _, coverage = _expected_product(
        summary,
        keys,
        [base_years, scenarios, trajectories],
        "年度逐时轨迹上下文",
    )
    expected_hours = summary.year.map(
        lambda year: 8784 if pd.Timestamp(f"{int(year)}-12-31").is_leap_year else 8760
    )
    hours_ok = summary.rows.eq(expected_hours).all() and summary.unique_timestamps.eq(
        expected_hours
    ).all()
    max_ok = np.isclose(
        summary.max_pu.to_numpy(float), 1.0, rtol=0.0, atol=tolerance
    ).all()
    ok = bool(coverage_ok and hours_ok and max_ok)
    return (
        ok,
        "年度逐时曲线覆盖完整且各轨迹年最大值为1" if ok else "年度逐时曲线维度或标幺失败",
        {
            "coverage": coverage,
            "context_count": int(len(summary)),
            "hour_counts_passed": bool(hours_ok),
            "max_normalization_passed": bool(max_ok),
        },
    )


def _validate_typical_days(
    typical: pd.DataFrame,
    factors: pd.DataFrame,
    *,
    base_years: tuple[int, ...],
    scenarios: tuple[str, ...],
    trajectories: tuple[str, ...],
    tolerance: float,
) -> tuple[bool, str, dict[str, Any]]:
    context = ["year", "planning_scenario", "trajectory"]
    cell = [*context, "season", "final_day_type"]
    required = [
        *cell,
        "hour",
        "typical_day_pu",
        "sample_days",
        "holiday_sample_days",
        "holiday_merged_with_weekend",
    ]
    _require_columns(typical, required, "自适应典型日")
    curve_ok, _, curve_evidence = _validate_grouped_curve(
        typical,
        group_columns=cell,
        value_column="typical_day_pu",
        closure="sum",
        tolerance=tolerance,
    )
    contexts = typical[context].drop_duplicates()
    context_ok, _, context_evidence = _expected_product(
        contexts,
        context,
        [base_years, scenarios, trajectories],
        "典型日上下文",
    )
    cells = typical[
        [
            *cell,
            "sample_days",
            "holiday_sample_days",
            "holiday_merged_with_weekend",
        ]
    ].drop_duplicates()
    if cells.duplicated(cell).any():
        raise ValueError("典型日单元的样本或合并标记不一致")
    expected_types_ok = True
    sample_coverage_ok = True
    for keys, part in cells.groupby([*context, "season"], sort=False, observed=True):
        holiday_days = set(pd.to_numeric(part.holiday_sample_days, errors="coerce"))
        merge_flags = set(part.holiday_merged_with_weekend.astype(bool))
        if len(holiday_days) != 1 or len(merge_flags) != 1:
            expected_types_ok = False
            continue
        holiday_days_value = int(next(iter(holiday_days)))
        merged = bool(next(iter(merge_flags)))
        expected_merged = holiday_days_value < 8
        expected_types = (
            {"工作日", "周末及节假日（合并）"}
            if expected_merged
            else {"工作日", "周末", "节假日"}
        )
        if merged != expected_merged or set(part.final_day_type.astype(str)) != expected_types:
            expected_types_ok = False
        year = int(keys[0])
        season = str(keys[-1])
        expected_days = int(calendar_frame(year).season.eq(season).sum())
        actual_days = int(pd.to_numeric(part.sample_days, errors="coerce").sum())
        if actual_days != expected_days:
            sample_coverage_ok = False

    _require_columns(
        factors,
        [*cell, "typical_day_load_factor"],
        "典型日日负荷率",
    )
    if factors.duplicated(cell).any():
        raise ValueError("典型日日负荷率单元重复")
    typical_cells = pd.MultiIndex.from_frame(cells[cell])
    factor_cells = pd.MultiIndex.from_frame(factors[cell])
    factor_coverage_ok = len(typical_cells.difference(factor_cells)) == 0 and len(
        factor_cells.difference(typical_cells)
    ) == 0
    recomputed = (
        typical.groupby(cell, observed=True).typical_day_pu.agg(["mean", "max"])
    )
    recomputed["expected_load_factor"] = recomputed["mean"] / recomputed["max"]
    factor_values = factors.set_index(cell).typical_day_load_factor.reindex(recomputed.index)
    formula_ok = np.isclose(
        pd.to_numeric(factor_values, errors="coerce").to_numpy(float),
        recomputed.expected_load_factor.to_numpy(float),
        rtol=0.0,
        atol=tolerance,
    ).all()
    ok = bool(
        curve_ok
        and context_ok
        and expected_types_ok
        and sample_coverage_ok
        and factor_coverage_ok
        and formula_ok
    )
    return (
        ok,
        "自适应典型日、样本覆盖和典型日日负荷率通过"
        if ok
        else "典型日维度、合并规则、标幺或负荷率失败",
        {
            "curve": curve_evidence,
            "contexts": context_evidence,
            "cell_count": int(len(cells)),
            "adaptive_day_types_passed": bool(expected_types_ok),
            "season_day_coverage_passed": bool(sample_coverage_ok),
            "load_factor_coverage_passed": bool(factor_coverage_ok),
            "load_factor_formula_passed": bool(formula_ok),
        },
    )


def _validate_monthly_peaks(
    monthly: pd.DataFrame,
    *,
    base_years: tuple[int, ...],
    scenarios: tuple[str, ...],
    trajectories: tuple[str, ...],
    tolerance: float,
) -> tuple[bool, str, dict[str, Any]]:
    context = ["year", "planning_scenario", "trajectory"]
    _require_columns(monthly, [*context, "month", "monthly_max_pu"], "12个月月峰曲线")
    values = _finite_nonnegative(monthly, "monthly_max_pu", "12个月月峰曲线")
    if np.any(values > 1.0 + tolerance):
        raise ValueError("月最大负荷标幺值超过1")
    if monthly.duplicated([*context, "month"]).any():
        raise ValueError("月最大负荷曲线月份键重复")
    summary = monthly.groupby(context, observed=True).agg(
        month_count=("month", "nunique"), max_pu=("monthly_max_pu", "max")
    ).reset_index()
    coverage_ok, _, coverage = _expected_product(
        summary,
        context,
        [base_years, scenarios, trajectories],
        "月峰曲线上下文",
    )
    months_ok = summary.month_count.eq(12).all()
    month_sets_ok = all(
        set(part.month.astype(int)) == set(range(1, 13))
        for _, part in monthly.groupby(context, observed=True)
    )
    max_ok = np.isclose(
        summary.max_pu.to_numpy(float), 1.0, rtol=0.0, atol=tolerance
    ).all()
    ok = bool(coverage_ok and months_ok and month_sets_ok and max_ok)
    return (
        ok,
        "12个月月最大负荷曲线维度完整且年最大值为1" if ok else "月峰曲线维度或标幺失败",
        {
            "coverage": coverage,
            "context_count": int(len(summary)),
            "month_sets_passed": bool(months_ok and month_sets_ok),
            "max_normalization_passed": bool(max_ok),
        },
    )


def _validate_display(
    store: _ResultStore,
    annual_targets: pd.DataFrame,
    *,
    display_years: tuple[int, ...],
    tolerance: float,
) -> tuple[bool, str, dict[str, Any]]:
    curves = store.table("display_workday_curves_2025_2045")
    factors = store.table("display_typical_day_load_factors_2025_2045")
    monthly_metrics = store.table("display_monthly_metrics_2025_2045")
    annual_metrics = store.table("display_annual_metrics_2025_2045")
    monthly_peaks = store.table("display_monthly_peak_curves_2025_2045")
    scale = store.table("display_m1_p50_annual_scale_2025_2045")
    filter_columns = ["year", "planning_scenario", "trajectory"]
    for label, frame in (
        ("展示曲线", curves),
        ("展示典型日日负荷率", factors),
        ("展示月度指标", monthly_metrics),
        ("展示年度指标", annual_metrics),
        ("展示月峰曲线", monthly_peaks),
    ):
        _require_columns(frame, filter_columns, label)
        if set(pd.to_numeric(frame.year, errors="coerce").astype(int)) != set(display_years):
            raise ValueError(f"{label}年份不是要求的展示期")
        if set(frame.planning_scenario.astype(str)) != {"基准"}:
            raise ValueError(f"{label}包含非基准规划情景")
        if set(frame.trajectory.astype(str)) != {"P50_real"}:
            raise ValueError(f"{label}包含非P50_real轨迹")

    curve_key = ["year", "season", "hour"]
    _require_columns(curves, [*curve_key, "final_day_type", "typical_day_pu"], "展示曲线")
    expected_rows = len(display_years) * len(SEASONS) * 24
    curve_rows_ok = len(curves) == expected_rows
    if display_years == DISPLAY_YEARS:
        curve_rows_ok = curve_rows_ok and len(curves) == 2016
    expected_curve = pd.MultiIndex.from_product(
        [display_years, SEASONS, range(24)], names=curve_key
    )
    actual_curve = pd.MultiIndex.from_frame(curves[curve_key])
    curve_grid_ok = len(expected_curve.difference(actual_curve)) == 0 and len(
        actual_curve.difference(expected_curve)
    ) == 0
    workday_only = set(curves.final_day_type.astype(str)) == {"工作日"}
    curve_norm_ok = np.isclose(
        curves.groupby(["year", "season"], observed=True).typical_day_pu.sum(),
        1.0,
        rtol=0.0,
        atol=tolerance,
    ).all()

    factors_ok = len(factors) == len(display_years) * 4 and set(
        factors.final_day_type.astype(str)
    ) == {"工作日"}
    _require_columns(monthly_metrics, ["month", "monthly_imbalance_rate"], "展示月度指标")
    monthly_metrics_ok = len(monthly_metrics) == len(display_years) * 12 and all(
        set(part.month.astype(int)) == set(range(1, 13))
        for _, part in monthly_metrics.groupby("year")
    )
    _finite_nonnegative(monthly_metrics, "monthly_imbalance_rate", "展示月度指标")
    _require_columns(annual_metrics, ["Tmax_h"], "展示年度指标")
    annual_metrics_ok = len(annual_metrics) == len(display_years)
    _finite_nonnegative(annual_metrics, "Tmax_h", "展示年度指标")
    _require_columns(monthly_peaks, ["month", "monthly_max_pu"], "展示月峰曲线")
    monthly_peaks_ok = len(monthly_peaks) == len(display_years) * 12 and np.isclose(
        monthly_peaks.groupby("year").monthly_max_pu.max(),
        1.0,
        rtol=0.0,
        atol=tolerance,
    ).all()

    _require_columns(
        scale,
        [
            "method",
            "planning_scenario",
            "year",
            "annual_scale_MWh",
            "annual_energy_quantile",
            "annual_energy_basis",
        ],
        "M1基准P50年度规模",
    )
    scale_contract_ok = (
        len(scale) == len(display_years)
        and set(scale.method.astype(str)) == {"M1"}
        and set(scale.planning_scenario.astype(str)) == {"基准"}
        and set(scale.annual_energy_quantile.astype(str)) == {"P50"}
        and set(scale.annual_energy_basis.astype(str)) == {"P50_natural_total"}
    )
    target = annual_targets.loc[
        annual_targets.method.astype(str).eq("M1")
        & annual_targets.scenario.astype(str).eq("基准")
        & pd.to_numeric(annual_targets.year, errors="coerce").isin(display_years),
        ["year", "P50_natural_MWh"],
    ]
    scale_check = scale[["year", "annual_scale_MWh"]].merge(
        target, on="year", how="outer", validate="one_to_one", indicator=True
    )
    p50_values = pd.to_numeric(scale_check.P50_natural_MWh, errors="coerce").to_numpy(float)
    scale_values = pd.to_numeric(scale_check.annual_scale_MWh, errors="coerce").to_numpy(float)
    scale_matches_p50 = (
        scale_check["_merge"].eq("both").all()
        and np.isfinite(p50_values).all()
        and np.isfinite(scale_values).all()
        and np.isclose(scale_values, p50_values, rtol=1e-10, atol=1e-3).all()
    )

    display_dir = store.root / f"展示_{min(display_years)}-{max(display_years)}_基准P50"
    expected_files = {
        "四季工作日标幺曲线_长表.csv",
        "四季工作日典型日日负荷率.csv",
        "月度指标与月不均衡系数.csv",
        "年度指标与年利用小时数.csv",
        "12个月月最大负荷标幺曲线.csv",
        "M1基准情景年度电量P50.csv",
    }
    physical_files_ok = display_dir.is_dir() and expected_files.issubset(
        {path.name for path in display_dir.glob("*.csv")}
    )
    ok = bool(
        curve_rows_ok
        and curve_grid_ok
        and workday_only
        and curve_norm_ok
        and factors_ok
        and monthly_metrics_ok
        and annual_metrics_ok
        and monthly_peaks_ok
        and scale_contract_ok
        and scale_matches_p50
        and physical_files_ok
    )
    return (
        ok,
        "展示结果严格为展示期基准P50_real四季工作日，M1规模取P50"
        if ok
        else "展示范围、维度、标幺或M1 P50规模合同失败",
        {
            "expected_workday_curve_rows": expected_rows,
            "actual_workday_curve_rows": len(curves),
            "workday_grid_passed": bool(curve_grid_ok),
            "workday_normalization_passed": bool(curve_norm_ok),
            "supporting_display_dimensions_passed": bool(
                factors_ok and monthly_metrics_ok and annual_metrics_ok and monthly_peaks_ok
            ),
            "M1_baseline_scale_contract_passed": bool(scale_contract_ok),
            "M1_baseline_scale_matches_upstream_P50": bool(scale_matches_p50),
            "display_files_present": bool(physical_files_ok),
        },
    )


def _validate_component_curves(
    store: _ResultStore, tolerance: float
) -> tuple[bool, str, dict[str, Any]]:
    outcomes: dict[str, Any] = {}
    baseline = store.table("baseline_shapes")
    _require_columns(baseline, SHAPE_COLS, "行业基准形状")
    baseline_values = baseline[list(SHAPE_COLS)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    baseline_ok = (
        np.isfinite(baseline_values).all()
        and np.all(baseline_values >= 0)
        and np.isclose(baseline_values.sum(axis=1), 1.0, rtol=0.0, atol=tolerance).all()
    )
    outcomes["baseline_shapes"] = bool(baseline_ok)

    specs = (
        (
            "province_typical_2023",
            ["season", "day_group"],
            ("province_typical_share",),
        ),
        (
            "base_year_industry_components",
            ["industry", "season", "day_group"],
            ("normalized_load",),
        ),
        (
            "correction_component_shape",
            ["season", "day_group"],
            ("province_shape_proxy", "correction_normalized_shape"),
        ),
        (
            "future_industry_components",
            ["method", "planning_scenario", "year", "industry", "season", "day_group"],
            ("normalized_load",),
        ),
        (
            "future_total_component_shapes",
            ["method", "planning_scenario", "year", "season", "day_group"],
            ("total_normalized_load",),
        ),
        (
            "proxy_sector_typical_shapes",
            ["method", "planning_scenario", "year", "proxy_sector", "season", "day_group"],
            ("proxy_normalized_load",),
        ),
        (
            "province_season_daytype_typical",
            ["method", "planning_scenario", "year", "season", "day_group"],
            ("normalized_load",),
        ),
        (
            "typical_workdays",
            ["method", "planning_scenario", "year", "season"],
            ("normalized_hour_share",),
        ),
    )
    for table_name, groups, values in specs:
        frame = store.table(table_name)
        for value_column in values:
            result, _, evidence = _validate_grouped_curve(
                frame,
                group_columns=groups,
                value_column=value_column,
                closure="sum",
                tolerance=tolerance,
            )
            outcomes[f"{table_name}.{value_column}"] = {
                "passed": bool(result),
                **evidence,
            }
    ok = baseline_ok and all(
        value is True or (isinstance(value, Mapping) and bool(value.get("passed")))
        for value in outcomes.values()
    )
    return (
        bool(ok),
        "行业、修正组成及派生典型曲线全部标幺" if ok else "至少一个组成或派生曲线未标幺",
        outcomes,
    )


def _validate_formal_call_index(
    store: _ResultStore,
    *,
    study_years: tuple[int, ...],
    year_sources: Mapping[str, pd.DataFrame],
) -> tuple[bool, str, dict[str, Any]]:
    directory = store.root / f"正式调用_{min(study_years)}-{max(study_years)}"
    index_path = directory / "调用索引.csv"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    index = pd.read_csv(index_path, low_memory=False)
    _require_columns(index, ["result_name", "file", "formal_year_filter"], "正式调用索引")
    study = set(study_years)
    source_coverage: dict[str, bool] = {}
    for name, frame in year_sources.items():
        source_coverage[name] = study.issubset(_years_from(frame))
    files_exist: list[bool] = []
    for recorded in index.file.astype(str):
        path = store.root / Path(recorded)
        if path.is_file():
            files_exist.append(True)
            continue
        fallback = store.root / "数据表" / path.name
        hourly_fallback = store.root / "逐时曲线" / path.name
        files_exist.append(fallback.is_file() or hourly_fallback.is_file())
    only_index = directory.is_dir() and {
        path.name for path in directory.iterdir() if path.is_file()
    } == {"调用索引.csv"}
    filter_text = index.formal_year_filter.astype(str)
    filter_text_ok = filter_text.str.contains(str(min(study_years))).all() and filter_text.str.contains(
        str(max(study_years))
    ).all()
    ok = bool(all(source_coverage.values()) and all(files_exist) and filter_text_ok)
    mode = "index_only_references_saved_full_tables" if only_index else "materialized_or_mixed"
    return (
        ok,
        f"{min(study_years)}—{max(study_years)}可直接从已保存底表按year筛选读取"
        if ok
        else "正式调用索引或指定年份底表覆盖不完整",
        {
            "materialization_mode": mode,
            "large_tables_copied_by_validator": False,
            "study_years": [min(study_years), max(study_years)],
            "source_year_coverage": source_coverage,
            "indexed_file_count": int(len(index)),
            "all_index_targets_exist": bool(all(files_exist)),
            "formal_filter_text_passed": bool(filter_text_ok),
        },
    )


def validate_formal_output(
    output_dir: str | Path,
    *,
    base_years: Iterable[int] = FORMAL_BASE_YEARS,
    study_years: Iterable[int] = FORMAL_STUDY_YEARS,
    display_years: Iterable[int] = DISPLAY_YEARS,
    scenarios: Sequence[str] = PLANNING_SCENARIOS,
    trajectories: Sequence[str] = SAVED_TRAJECTORIES,
    expected_bootstrap: int = 1000,
    curve_tolerance: float = 1e-10,
    energy_rtol: float = 1e-10,
    energy_atol_MWh: float = 1e-3,
    check_component_curves: bool = True,
) -> dict[str, Any]:
    """Validate one saved output directory without running or mutating it."""

    base_year_values = tuple(int(value) for value in base_years)
    study_year_values = tuple(int(value) for value in study_years)
    display_year_values = tuple(int(value) for value in display_years)
    scenario_values = tuple(str(value) for value in scenarios)
    trajectory_values = tuple(str(value) for value in trajectories)
    if not base_year_values or not study_year_values or not display_year_values:
        raise ValueError("年份集合不得为空")
    if not set(study_year_values).issubset(base_year_values):
        raise ValueError("正式调用期必须包含在完整底座年份中")
    if not set(display_year_values).issubset(base_year_values):
        raise ValueError("展示期必须包含在完整底座年份中")

    collector = _Collector()
    try:
        store = _ResultStore(output_dir)
    except Exception as exc:
        return {
            "passed": False,
            "output_dir": str(Path(output_dir).resolve()),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "summary": {"passed": 0, "failed": 1, "warnings": 0},
            "checks": [
                {
                    "name": "open_result_store",
                    "status": "failed",
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            ],
        }

    collector.run(
        "metadata_contract",
        lambda: _validate_metadata(
            store.metadata, base_year_values, scenario_values, expected_bootstrap
        ),
    )
    collector.run("result_schema_3_4_1", lambda: _validate_result_schema(store))

    loaded: dict[str, pd.DataFrame] = {}
    for name in (
        "annual_energy_interval",
        "trajectory_annual_metrics",
        "trajectory_monthly_metrics",
        "adaptive_typical_day_curves",
        "typical_day_load_factors",
        "monthly_maximum_curves_pu",
    ):
        try:
            loaded[name] = store.table(name)
        except Exception as exc:
            collector.failed(f"load_{name}", f"{type(exc).__name__}: {exc}")

    if {"annual_energy_interval", "trajectory_annual_metrics"}.issubset(loaded):
        collector.run(
            "trajectory_annual_energy_closure",
            lambda: _validate_annual_energy_and_trajectories(
                loaded["annual_energy_interval"],
                loaded["trajectory_annual_metrics"],
                base_years=base_year_values,
                scenarios=scenario_values,
                trajectories=trajectory_values,
                rtol=energy_rtol,
                atol=energy_atol_MWh,
            ),
        )

    try:
        hourly = store.hourly()
        loaded["normalized_hourly_trajectories"] = hourly
        collector.run(
            "normalized_hourly_trajectories",
            lambda: _validate_hourly(
                hourly,
                base_years=base_year_values,
                scenarios=scenario_values,
                trajectories=trajectory_values,
                tolerance=curve_tolerance,
            ),
        )
    except Exception as exc:
        collector.failed("load_normalized_hourly_trajectories", f"{type(exc).__name__}: {exc}")

    if {"adaptive_typical_day_curves", "typical_day_load_factors"}.issubset(loaded):
        collector.run(
            "adaptive_typical_days",
            lambda: _validate_typical_days(
                loaded["adaptive_typical_day_curves"],
                loaded["typical_day_load_factors"],
                base_years=base_year_values,
                scenarios=scenario_values,
                trajectories=trajectory_values,
                tolerance=curve_tolerance,
            ),
        )
    if "monthly_maximum_curves_pu" in loaded:
        collector.run(
            "monthly_maximum_curves",
            lambda: _validate_monthly_peaks(
                loaded["monthly_maximum_curves_pu"],
                base_years=base_year_values,
                scenarios=scenario_values,
                trajectories=trajectory_values,
                tolerance=curve_tolerance,
            ),
        )
    if "annual_energy_interval" in loaded:
        collector.run(
            "display_contract_and_M1_P50_scale",
            lambda: _validate_display(
                store,
                loaded["annual_energy_interval"],
                display_years=display_year_values,
                tolerance=curve_tolerance,
            ),
        )
    if check_component_curves:
        collector.run(
            "all_component_and_derived_curves_normalized",
            lambda: _validate_component_curves(store, curve_tolerance),
        )

    formal_sources = {
        name: frame
        for name, frame in loaded.items()
        if "year" in frame.columns
        and name
        in {
            "annual_energy_interval",
            "trajectory_annual_metrics",
            "trajectory_monthly_metrics",
            "adaptive_typical_day_curves",
            "typical_day_load_factors",
            "monthly_maximum_curves_pu",
            "normalized_hourly_trajectories",
        }
    }
    if formal_sources:
        collector.run(
            "formal_2030_2060_direct_read_access",
            lambda: _validate_formal_call_index(
                store, study_years=study_year_values, year_sources=formal_sources
            ),
        )

    passed_count = sum(item["status"] == "passed" for item in collector.checks)
    failed_count = sum(item["status"] == "failed" for item in collector.checks)
    warning_count = sum(item["status"] == "warning" for item in collector.checks)
    formal_access = next(
        (
            item
            for item in collector.checks
            if item["name"] == "formal_2030_2060_direct_read_access"
        ),
        None,
    )
    return {
        "passed": failed_count == 0,
        "output_dir": str(store.root),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_scope": {
            "base_years": [min(base_year_values), max(base_year_values)],
            "formal_study_years": [min(study_year_values), max(study_year_values)],
            "display_years": [min(display_year_values), max(display_year_values)],
            "planning_scenarios": list(scenario_values),
            "saved_trajectories": list(trajectory_values),
            "expected_bootstrap_per_context": int(expected_bootstrap),
        },
        "formal_call_access": formal_access,
        "summary": {
            "passed": int(passed_count),
            "failed": int(failed_count),
            "warnings": int(warning_count),
        },
        "checks": collector.checks,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="正式结果根目录")
    parser.add_argument(
        "--report",
        type=Path,
        help="可选：把JSON校验报告另存到指定路径；默认只打印",
    )
    parser.add_argument(
        "--skip-component-curves",
        action="store_true",
        help="仅用于诊断精简结果；正式验收不应跳过组成曲线",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = validate_formal_output(
        args.output_dir,
        check_component_curves=not args.skip_component_curves,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DISPLAY_YEARS",
    "FORMAL_BASE_YEARS",
    "FORMAL_STUDY_YEARS",
    "PLANNING_SCENARIOS",
    "SAVED_TRAJECTORIES",
    "main",
    "validate_formal_output",
]
