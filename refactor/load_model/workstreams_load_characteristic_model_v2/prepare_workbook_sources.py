"""Prepare the strictly validated data contract for the 2030 display workbook.

This program never writes an Excel workbook. It consumes one successful 3.5.0
run containing 2030, validates the current 24-industry plus correction-component contract,
and writes only the compact CSV sources used by ``build_workbook.mjs``.

Any retired table or column in the result manifest blocks preparation before a
presentation source is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

try:  # Package import and direct-script execution are both supported.
    from .result_tables import (
        FORBIDDEN_FORMAL_TABLES,
        FORBIDDEN_SHAPE_FUTURE_COLUMNS,
        METHOD_REVISION,
        PROVINCE_SHARED_FACTOR_SCOPE,
        REQUIRED_DAY_TYPE_FACTOR_COLUMNS,
        REQUIRED_FORMAL_TABLES,
        REQUIRED_SHAPE_FUTURE_COLUMNS,
        RESULT_SCHEMA_VERSION,
        validate_run_manifest,
        validate_tables_manifest_contract,
    )
    from .reporting_outputs import REPORT_METHOD_STATEMENTS
except ImportError:  # pragma: no cover - exercised by the formal CLI path
    from result_tables import (
        FORBIDDEN_FORMAL_TABLES,
        FORBIDDEN_SHAPE_FUTURE_COLUMNS,
        METHOD_REVISION,
        PROVINCE_SHARED_FACTOR_SCOPE,
        REQUIRED_DAY_TYPE_FACTOR_COLUMNS,
        REQUIRED_FORMAL_TABLES,
        REQUIRED_SHAPE_FUTURE_COLUMNS,
        RESULT_SCHEMA_VERSION,
        validate_run_manifest,
        validate_tables_manifest_contract,
    )
    from reporting_outputs import REPORT_METHOD_STATEMENTS


DISPLAY_YEAR = 2030
ANCHOR_YEARS = (2025, 2030, 2040)
SCENARIOS = ("低碳", "基准", "高电气化")
TRACKS = ("P10", "P50", "P90")
SEASONS = ("春季", "夏季", "秋季", "冬季")
DAY_GROUPS = ("工作日", "非工作日")
PROXY_SECTORS = ("第一产业", "第二产业", "第三产业", "居民生活")
CORRECTION_COMPONENT = "修正组成"
EXPECTED_INDUSTRIES = 24
EXPECTED_COMPONENTS = 25
EXPECTED_BOOTSTRAP_SCENARIOS = 1000
EXPECTED_MODEL_VERSION = RESULT_SCHEMA_VERSION
EXPECTED_METHOD_REVISION = METHOD_REVISION
EXPECTED_ANNUAL_ENERGY_METHOD = "M1"
EXPECTED_OUTPUT_NAME = "负荷特性模型（完善版）_2030展示.xlsx"
SOURCE_ORDER = (
    "权重与假设",
    "四类代理曲线",
    "修正组成",
    "2030年度指标",
    "2030月度指标",
    "12点月最大曲线",
    "2030典型日",
    "质量检查",
    "输入文件清单",
)

FORBIDDEN_TABLE_NAMES = FORBIDDEN_FORMAL_TABLES.union(
    {
        "correction_weights",
        "correction_curve_2023",
        "correction_coverage_2023",
        "run_group_cell_judgement",
    }
)
FORBIDDEN_COLUMNS = {
    "run_group",
    "shift_hours",
    "load_duration_rank",
    "exceedance_probability",
}

# 3.5.0 workbook sources consume only compact presentation inputs.  The full
# core-table presence is checked separately by ``validate_tables_manifest_contract``;
# deliberately avoid loading large component-daily tables just to prepare a
# workbook.
TABLE_CONTRACTS: dict[str, set[str]] = {
    "input_manifest": {"logical_name", "path", "size_bytes", "sha256"},
    "industry_inventory": {
        "industry",
        "broad_sector",
        "is_traditional",
        "official_energy_2025",
        "inventory_role",
    },
    "user_quality": {
        "industry",
        "user",
        "total_days",
        "valid_days",
        "valid_rate",
    },
    "cell_quality": {"industry", "season", "final_day_type", "trend_eligible"},
    "baseline_shapes": {
        "industry",
        "season",
        "final_day_type",
        "year",
        "is_future_baseline",
    },
    "shape_future": set(REQUIRED_SHAPE_FUTURE_COLUMNS).union(
        {
            "industry",
            "season",
            "final_day_type",
            "year",
            "H",
            "phi",
            "formal_H8",
            "formal_phi08",
        }
    ),
    "day_type_factors": set(REQUIRED_DAY_TYPE_FACTOR_COLUMNS),
    "annual_weight_ledger": {
        "method",
        "planning_scenario",
        "year",
        "annual_energy_quantile",
        "component",
        "industry",
        "component_type",
        "broad_sector",
        "base_share_2025",
        "target_multiplier_2040",
        "electricity_share",
        "weight_source",
        "rationale",
        "assembly_relative_error",
        "assembly_passed",
    },
    "component_energy_closure": {
        "method",
        "planning_scenario",
        "year",
        "relative_error",
    },
    "proxy_sector_typical_shapes": {
        "method",
        "planning_scenario",
        "year",
        "proxy_sector",
        "season",
        "day_group",
        "hour",
        "proxy_sector_share",
        "proxy_normalized_load",
    },
    "correction_component_shape": {
        "season",
        "hour",
        "province_shape_proxy",
        "C24_contribution",
        "R_contribution",
        "omega_R",
        "correction_component_shape_proxy",
        "reconstruction_max_abs_error",
        "industry_shape_source",
        "correction_weight_year",
    },
    "correction_component_diagnostics": {
        "season",
        "reconstruction_max_abs_error",
    },
    "daily_energy_checks": {
        "method",
        "planning_scenario",
        "year",
        "component_count",
        "relative_error",
        "passed",
    },
    "deterministic_annual_metrics": {
        "method",
        "planning_scenario",
        "year",
        "annual_energy_MWh",
        "annual_max_MW",
        "annual_min_MW",
        "Tmax_h",
        "annual_load_factor",
        "year_hours",
        "mean_daily_load_factor",
    },
    "representative_annual_metrics": {
        "method",
        "planning_scenario",
        "year",
        "representative_track",
        "bootstrap_scenario_id",
        "annual_energy_MWh",
        "annual_max_MW",
        "annual_min_MW",
        "Tmax_h",
        "annual_load_factor",
        "year_hours",
        "mean_daily_load_factor",
    },
    "deterministic_monthly_metrics": {
        "method",
        "planning_scenario",
        "year",
        "month",
        "monthly_energy_MWh",
        "monthly_average_MW",
        "monthly_max_MW",
        "monthly_peak_timestamp",
        "peak_day_load_factor",
        "monthly_imbalance_rate",
    },
    "representative_monthly_metrics": {
        "method",
        "planning_scenario",
        "year",
        "month",
        "representative_track",
        "bootstrap_scenario_id",
        "monthly_energy_MWh",
        "monthly_average_MW",
        "monthly_max_MW",
        "monthly_peak_timestamp",
        "peak_day_load_factor",
        "monthly_imbalance_rate",
    },
    "monthly_maximum_curves_pu": {
        "method",
        "planning_scenario",
        "year",
        "trajectory",
        "month",
        "monthly_max_pu",
        "monthly_peak_timestamp",
    },
    "adaptive_typical_day_curves": {
        "method",
        "planning_scenario",
        "year",
        "trajectory",
        "season",
        "final_day_type",
        "hour",
        "typical_day_pu",
        "sample_days",
    },
}


class WorkbookSourceError(ValueError):
    """Raised when a formal result cannot drive the display workbook."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fail(message: str) -> None:
    raise WorkbookSourceError(message)


def _as_bool(series: pd.Series, *, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.fillna(False).astype(bool)
    tokens = series.astype("string").str.strip().str.lower()
    valid = tokens.isin({"true", "false", "1", "0", "yes", "no", "是", "否"})
    if not valid.all():
        _fail(f"{label}包含无法识别的布尔值")
    return tokens.isin({"true", "1", "yes", "是"})


def _ensure_year(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    result = frame.copy()
    result["year"] = pd.to_numeric(result.year, errors="raise").astype(int)
    result = result.loc[result.year.eq(DISPLAY_YEAR)].copy()
    if result.empty:
        _fail(f"{label}没有{DISPLAY_YEAR}年记录")
    return result


def _scenario_coverage(frame: pd.DataFrame, *, label: str) -> None:
    observed = set(frame.planning_scenario.astype(str).unique())
    if observed != set(SCENARIOS):
        _fail(f"{label}规划情景不完整：{sorted(observed)}")


def _validate_metadata(metadata: Mapping[str, Any]) -> None:
    try:
        validate_run_manifest(
            metadata, require_success=True, require_parquet=True
        )
    except RuntimeError as exc:
        _fail(str(exc))
    expected = {
        "status": "success",
        "model_version": EXPECTED_MODEL_VERSION,
        "method_revision": EXPECTED_METHOD_REVISION,
        "method": EXPECTED_ANNUAL_ENERGY_METHOD,
        "annual_energy_method": EXPECTED_ANNUAL_ENERGY_METHOD,
        "component_count": EXPECTED_COMPONENTS,
        "new_load_deducted": False,
        "new_load_independently_shaped": False,
        "new_load_in_correction_component": True,
        "external_new_load_overlay": False,
        "formal_horizon": 8,
        "formal_peak_valley_phi": 0.8,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            _fail(f"运行元数据{key}={metadata.get(key)!r}，预期{value!r}")
    years = {int(value) for value in metadata.get("years", [])}
    if DISPLAY_YEAR not in years:
        _fail("2030展示工作簿要求运行结果包含2030年")
    if set(str(value) for value in metadata.get("planning_scenarios", [])) != set(
        SCENARIOS
    ):
        _fail("运行元数据必须完整覆盖低碳、基准、高电气化三情景")
    if int(metadata.get("bootstrap_scenarios_per_context", 0)) != (
        EXPECTED_BOOTSTRAP_SCENARIOS
    ):
        _fail("2030展示工作簿要求每个情景1000条Bootstrap场景")
    hourly = metadata.get("hourly_output")
    if not isinstance(hourly, Mapping) or str(hourly.get("format")).lower() != "parquet":
        _fail("2030正式展示要求逐时结果已成功写为Parquet")


def _validate_method_contract(
    metadata: Mapping[str, Any], tables: Mapping[str, pd.DataFrame]
) -> None:
    """所有带method字段的正式表必须与元数据中的主方法严格一致。"""

    expected = str(metadata.get("method", "")).strip()
    if expected != EXPECTED_ANNUAL_ENERGY_METHOD:
        _fail(
            f"年度电量主方法必须为{EXPECTED_ANNUAL_ENERGY_METHOD}，实际{expected or '<空>'}"
        )
    for name, frame in tables.items():
        if "method" not in frame.columns:
            continue
        observed = set(frame.method.dropna().astype(str).str.strip().unique())
        if observed != {expected}:
            _fail(f"结果表{name}的方法值{sorted(observed)}与主方法{expected}不一致")


def _validate_result_manifest(manifest: Mapping[str, Any]) -> None:
    try:
        validate_tables_manifest_contract(manifest)
    except RuntimeError as exc:
        _fail(str(exc))
    retired = FORBIDDEN_TABLE_NAMES.intersection(manifest)
    if retired:
        _fail(f"结果清单包含已退役表：{sorted(retired)}")
    missing_tables = sorted(set(TABLE_CONTRACTS).difference(manifest))
    if missing_tables:
        _fail(f"2030展示缺少结果表：{missing_tables}")
    for table_name, entry in manifest.items():
        if not isinstance(entry, Mapping):
            _fail(f"结果清单{table_name}条目不是对象")
        columns = [str(value) for value in entry.get("columns", [])]
        forbidden_columns = set(FORBIDDEN_COLUMNS)
        if table_name == "shape_future":
            forbidden_columns.update(FORBIDDEN_SHAPE_FUTURE_COLUMNS)
        retired_columns = sorted(set(columns).intersection(forbidden_columns))
        if retired_columns:
            _fail(f"结果表{table_name}包含已退役字段：{retired_columns}")
    for table_name, required in TABLE_CONTRACTS.items():
        columns = set(str(value) for value in manifest[table_name].get("columns", []))
        missing = sorted(required.difference(columns))
        if missing:
            _fail(f"结果表{table_name}缺少字段：{missing}")


def _load_table(data_dir: Path, manifest: Mapping[str, Any], name: str) -> pd.DataFrame:
    entry = manifest[name]
    path = (data_dir / str(entry["file"])).resolve()
    if path.parent != data_dir.resolve() or not path.is_file():
        _fail(f"结果表{name}文件缺失或越出数据目录：{path}")
    if path.suffix.lower() != ".csv":
        _fail(f"结果表{name}必须是CSV")
    if entry.get("sha256") and _sha256(path) != str(entry["sha256"]):
        _fail(f"结果表{name}的SHA256与清单不一致")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if len(frame) != int(entry.get("rows", -1)):
        _fail(f"结果表{name}行数与清单不一致")
    if list(frame.columns) != [str(value) for value in entry.get("columns", [])]:
        _fail(f"结果表{name}列顺序与清单不一致")
    return frame


def _derive_weight_table(
    ledger: pd.DataFrame, industry_inventory: pd.DataFrame
) -> pd.DataFrame:
    frame = _ensure_year(ledger, label="年度25组成权重账本")
    frame = frame.loc[
        frame["annual_energy_quantile"].astype(str).str.strip().eq("P50")
    ].copy()
    if frame.empty:
        _fail("年度25组成权重账本没有2030年P50记录")
    _scenario_coverage(frame, label="年度25组成权重账本")
    counts = frame.groupby("planning_scenario").component.nunique()
    if not counts.eq(EXPECTED_COMPONENTS).all():
        _fail("每个规划情景必须恰有24行业加1个修正组成")
    if frame.duplicated(["planning_scenario", "component"]).any():
        _fail("年度25组成权重账本的情景/组成必须唯一")
    for column in (
        "base_share_2025",
        "target_multiplier_2040",
        "electricity_share",
        "assembly_relative_error",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[["base_share_2025", "electricity_share"]].isna().any().any():
        _fail("年度25组成权重账本存在缺失份额")
    share_sums = frame.groupby("planning_scenario")["electricity_share"].sum()
    if not np.isclose(
        share_sums.to_numpy(float), 1.0, rtol=0.0, atol=1e-12
    ).all():
        _fail("年度25组成权重账本的情景内组成份额未闭合到1")
    if not _as_bool(frame.assembly_passed, label="年度装配通过标记").all():
        _fail("年度25组成账本存在未通过的装配闭合")

    invariant = [
        "industry",
        "component_type",
        "broad_sector",
        "base_share_2025",
        "target_multiplier_2040",
        "weight_source",
        "rationale",
    ]
    for component, part in frame.groupby("component", sort=False):
        for column in invariant:
            if part[column].nunique(dropna=False) != 1:
                _fail(f"组成{component}的{column}在三情景间不一致")
    unique = frame.loc[frame.planning_scenario.eq("基准")].copy()
    if len(unique) != EXPECTED_COMPONENTS:
        _fail("年度25组成权重账本缺少完整的2030基准情景P50记录")
    unique = unique[["component", *invariant, "electricity_share"]].copy()
    traditional = unique.component_type.eq("传统细分行业")
    correction = unique.component_type.eq(CORRECTION_COMPONENT)
    if traditional.sum() != EXPECTED_INDUSTRIES or correction.sum() != 1:
        _fail("权重账本必须恰有24个传统细分行业和1个修正组成")
    if set(unique.loc[traditional, "broad_sector"].astype(str)) != set(PROXY_SECTORS):
        _fail("24行业必须完整归属一产、二产、三产、居民四类代理")
    if set(unique.loc[correction, "component"].astype(str)) != {CORRECTION_COMPONENT}:
        _fail("第25项必须命名为修正组成")
    if unique.loc[traditional, "target_multiplier_2040"].isna().any():
        _fail("24行业缺少2040/2025目标倍数")

    unique["share_2025"] = unique.base_share_2025.astype(float)
    unique["share_2030"] = unique.electricity_share.astype(float)
    unique["share_2040"] = np.nan
    unique.loc[traditional, "share_2040"] = (
        unique.loc[traditional, "share_2025"]
        * unique.loc[traditional, "target_multiplier_2040"].astype(float)
    )
    unique.loc[correction, "share_2040"] = 1.0 - float(
        unique.loc[traditional, "share_2040"].sum()
    )
    for year in ANCHOR_YEARS:
        total = float(unique[f"share_{year}"].sum())
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
            _fail(f"{year}年24行业加修正组成份额未闭合：{total}")
    if (unique[["share_2025", "share_2030", "share_2040"]] < 0).any().any():
        _fail("权重锚点存在负份额")

    inventory = industry_inventory.copy()
    traditional_inventory = inventory.loc[
        _as_bool(inventory.is_traditional, label="行业清单传统标记")
    ].copy()
    if (
        len(traditional_inventory) != EXPECTED_INDUSTRIES
        or traditional_inventory.industry.duplicated().any()
    ):
        _fail("行业清单必须恰有24个唯一传统细分行业")
    result = unique.merge(
        traditional_inventory[["industry", "official_energy_2025"]],
        on="industry",
        how="left",
        validate="one_to_one",
    )
    result.loc[result.component_type.eq(CORRECTION_COMPONENT), "official_energy_2025"] = np.nan
    sector_order = {value: index for index, value in enumerate(PROXY_SECTORS)}
    result["_component_order"] = np.where(
        result.component_type.eq("传统细分行业"), 0, 1
    )
    result["_sector_order"] = result.broad_sector.map(sector_order).fillna(99)
    result = result.sort_values(
        ["_component_order", "_sector_order", "component"], kind="mergesort"
    ).reset_index(drop=True)
    result["display_order"] = np.arange(1, len(result) + 1)
    result["model_role"] = np.where(
        result.component_type.eq("传统细分行业"),
        "24行业基本建模层",
        "第25个剩余电量组成",
    )
    return result[
        [
            "display_order",
            "component",
            "component_type",
            "broad_sector",
            "model_role",
            "official_energy_2025",
            "share_2025",
            "share_2030",
            "share_2040",
            "target_multiplier_2040",
            "weight_source",
            "rationale",
        ]
    ]


def _prepare_proxy_shapes(frame: pd.DataFrame) -> pd.DataFrame:
    result = _ensure_year(frame, label="四类代理曲线")
    _scenario_coverage(result, label="四类代理曲线")
    if set(result.proxy_sector.astype(str)) != set(PROXY_SECTORS):
        _fail("四类代理曲线必须恰有一产、二产、三产、居民")
    result["hour"] = pd.to_numeric(result.hour, errors="raise").astype(int)
    key = ["planning_scenario", "proxy_sector", "season", "day_group"]
    counts = result.groupby(key).hour.nunique()
    if (
        not counts.eq(24).all()
        or set(result.hour) != set(range(24))
    ):
        _fail("四类代理曲线每个情景×类别×季节×日型单元必须完整覆盖24小时")
    normalized = pd.to_numeric(result.proxy_normalized_load, errors="coerce")
    if normalized.isna().any() or (normalized < 0).any():
        _fail("四类代理标幺曲线存在缺失或负值")
    sums = result.assign(_value=normalized).groupby(key)._value.sum()
    if not np.isclose(sums.to_numpy(float), 1.0, rtol=0.0, atol=1e-10).all():
        _fail("四类代理标幺曲线24点和不为1")
    result["proxy_sector_share"] = pd.to_numeric(
        result.proxy_sector_share, errors="raise"
    )
    result["role"] = "由24行业按自然负荷电量份额加权派生；不作为参数或电量分配层"
    result["_scenario_order"] = result.planning_scenario.map(
        {value: index for index, value in enumerate(SCENARIOS)}
    )
    result["_sector_order"] = result.proxy_sector.map(
        {value: index for index, value in enumerate(PROXY_SECTORS)}
    )
    result["_season_order"] = result.season.map(
        {value: index for index, value in enumerate(SEASONS)}
    )
    result["_day_order"] = result.day_group.map(
        {
            value: index
            for index, value in enumerate(
                ("工作日", "普通周末", "周末", "法定节假日", "节假日", "合并非工作日", "非工作日")
            )
        }
    )
    columns = [
        "method",
        "planning_scenario",
        "year",
        "proxy_sector",
        "season",
        "day_group",
        "hour",
        "proxy_sector_share",
        "proxy_normalized_load",
        "role",
    ]
    return result.sort_values(
        ["_scenario_order", "_sector_order", "_season_order", "_day_order", "hour"],
        kind="mergesort",
    )[columns].reset_index(drop=True)


def _prepare_correction_shape(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["hour"] = pd.to_numeric(result.hour, errors="raise").astype(int)
    day_column = next(
        (
            column
            for column in ("final_day_type", "day_type", "day_group")
            if column in result.columns
        ),
        None,
    )
    if day_column is None:
        _fail("修正组成缺少final_day_type/day_type/day_group")
    result["final_day_type"] = result[day_column].astype(str)
    key = ["season", "final_day_type"]
    counts = result.groupby(key).hour.nunique()
    if (
        not counts.eq(24).all()
        or result.duplicated([*key, "hour"]).any()
        or set(result.hour) != set(range(24))
        or set(result.season.astype(str)) != set(SEASONS)
    ):
        _fail("修正组成必须按4季×自适应日型完整且唯一覆盖24小时")
    shape = pd.to_numeric(
        result.correction_component_shape_proxy, errors="coerce"
    )
    contribution = pd.to_numeric(result.R_contribution, errors="coerce")
    if (
        shape.isna().any()
        or contribution.isna().any()
        or (shape < -1e-12).any()
        or (contribution < -1e-12).any()
    ):
        _fail("修正组成标幺形状或贡献非法")
    sums = result.assign(_shape=shape).groupby(key)._shape.sum()
    if not np.isclose(sums.to_numpy(float), 1.0, rtol=0.0, atol=1e-10).all():
        _fail("修正组成标幺曲线24点和不为1")
    reconstructed = pd.to_numeric(result.C24_contribution, errors="coerce") + (
        pd.to_numeric(result.omega_R, errors="coerce") * shape
    )
    province = pd.to_numeric(result.province_shape_proxy, errors="coerce")
    if not np.allclose(reconstructed, province, rtol=0.0, atol=1e-10):
        _fail("修正组成未能由C24+omega_R*p_R复原全省形状代理")
    if set(result.industry_shape_source.astype(str)) != {
        "2023_industry_aggregate_medoid"
    }:
        _fail("修正组成必须由2023行业聚合medoid重新反演")
    if set(
        pd.to_numeric(result.correction_weight_year, errors="raise").astype(int)
    ) != {2023}:
        _fail("修正组成权重年份必须为2023")
    result["shape_source_year"] = 2023
    result["weight_year"] = 2023
    result["anchor_interpretation"] = (
        "2023行业聚合medoid为唯一基准；2022只作历史比较"
    )
    result["_season_order"] = result.season.map(
        {value: index for index, value in enumerate(SEASONS)}
    )
    result["_day_order"] = result.final_day_type.map(
        {value: index for index, value in enumerate(("工作日", "普通周末", "法定节假日", "合并非工作日"))}
    )
    columns = [
        "shape_source_year",
        "weight_year",
        "season",
        "final_day_type",
        "hour",
        "province_shape_proxy",
        "C24_contribution",
        "R_contribution",
        "omega_R",
        "correction_component_shape_proxy",
        "reconstruction_max_abs_error",
        "industry_shape_source",
        "correction_weight_year",
        "anchor_interpretation",
    ]
    return result.sort_values(
        ["_season_order", "_day_order", "hour"], kind="mergesort"
    )[columns].reset_index(drop=True)


def _prepare_annual_metrics(
    deterministic: pd.DataFrame, representative: pd.DataFrame
) -> pd.DataFrame:
    det = _ensure_year(deterministic, label="2030确定性年度指标")
    rep = _ensure_year(representative, label="2030真实代表年度指标")
    _scenario_coverage(det, label="2030确定性年度指标")
    _scenario_coverage(rep, label="2030真实代表年度指标")
    if len(det) != len(SCENARIOS) or det.planning_scenario.duplicated().any():
        _fail("2030确定性年度指标必须每情景恰有1行")
    if set(rep.representative_track.astype(str)) != set(TRACKS):
        _fail("2030真实代表年度指标必须覆盖P10/P50/P90")
    if rep.duplicated(["planning_scenario", "representative_track"]).any() or len(rep) != 9:
        _fail("2030真实代表年度指标必须每情景每轨迹恰有1行")
    det["trajectory"] = "确定性基线"
    det["bootstrap_scenario_id"] = np.nan
    rep["trajectory"] = rep.representative_track.astype(str) + "真实轨迹"
    result = pd.concat([det, rep], ignore_index=True, sort=False)
    required = [
        "method",
        "planning_scenario",
        "year",
        "trajectory",
        "bootstrap_scenario_id",
        "annual_energy_MWh",
        "annual_max_MW",
        "annual_peak_timestamp",
        "annual_min_MW",
        "peak_valley_difference_MW",
        "Tmax_h",
        "annual_load_factor",
        "year_hours",
        "mean_daily_load_factor",
        "result_type",
    ]
    share_columns = sorted(
        column for column in result.columns if str(column).startswith("share_")
    )
    result["_scenario_order"] = result.planning_scenario.map(
        {value: index for index, value in enumerate(SCENARIOS)}
    )
    result["_trajectory_order"] = result.trajectory.map(
        {"确定性基线": 0, "P10真实轨迹": 1, "P50真实轨迹": 2, "P90真实轨迹": 3}
    )
    return result.sort_values(
        ["_scenario_order", "_trajectory_order"], kind="mergesort"
    )[[*required, *share_columns]].reset_index(drop=True)


def _prepare_monthly_metrics(
    deterministic: pd.DataFrame, representative: pd.DataFrame
) -> pd.DataFrame:
    det = _ensure_year(deterministic, label="2030确定性月度指标")
    rep = _ensure_year(representative, label="2030真实代表月度指标")
    _scenario_coverage(det, label="2030确定性月度指标")
    _scenario_coverage(rep, label="2030真实代表月度指标")
    if len(det) != len(SCENARIOS) * 12 or det.duplicated(
        ["planning_scenario", "month"]
    ).any():
        _fail("2030确定性月度指标必须每情景完整覆盖12个月")
    if len(rep) != len(SCENARIOS) * len(TRACKS) * 12 or rep.duplicated(
        ["planning_scenario", "representative_track", "month"]
    ).any():
        _fail("2030真实代表月度指标必须每情景每轨迹完整覆盖12个月")
    if set(rep.representative_track.astype(str)) != set(TRACKS):
        _fail("2030真实代表月度指标缺少P10/P50/P90")
    det["trajectory"] = "确定性基线"
    det["bootstrap_scenario_id"] = np.nan
    rep["trajectory"] = rep.representative_track.astype(str) + "真实轨迹"
    result = pd.concat([det, rep], ignore_index=True, sort=False)
    required = [
        "method",
        "planning_scenario",
        "year",
        "month",
        "trajectory",
        "bootstrap_scenario_id",
        "monthly_energy_MWh",
        "monthly_average_MW",
        "monthly_max_MW",
        "monthly_peak_timestamp",
        "peak_day",
        "peak_day_average_MW",
        "peak_day_min_MW",
        "peak_day_load_factor",
        "monthly_imbalance_rate",
        "result_type",
    ]
    result["month"] = pd.to_numeric(result.month, errors="raise").astype(int)
    result["_scenario_order"] = result.planning_scenario.map(
        {value: index for index, value in enumerate(SCENARIOS)}
    )
    result["_trajectory_order"] = result.trajectory.map(
        {"确定性基线": 0, "P10真实轨迹": 1, "P50真实轨迹": 2, "P90真实轨迹": 3}
    )
    return result.sort_values(
        ["_scenario_order", "_trajectory_order", "month"], kind="mergesort"
    )[required].reset_index(drop=True)


def _prepare_monthly_peak_curve(frame: pd.DataFrame) -> pd.DataFrame:
    result = _ensure_year(frame, label="12点月最大负荷曲线")
    _scenario_coverage(result, label="12点月最大负荷曲线")
    result["month"] = pd.to_numeric(result.month, errors="raise").astype(int)
    trajectories = (
        "deterministic_baseline",
        "P10_real",
        "P50_real",
        "P90_real",
    )
    if set(result.trajectory.astype(str)) != set(trajectories):
        _fail("12点月最大负荷曲线必须覆盖确定性及P10/P50/P90真实轨迹")
    if len(result) != len(SCENARIOS) * len(trajectories) * 12 or result.duplicated(
        ["planning_scenario", "trajectory", "month"]
    ).any():
        _fail("12点月最大负荷曲线必须每情景每轨迹恰有1—12月")
    for context, part in result.groupby(["planning_scenario", "trajectory"]):
        if set(part.month) != set(range(1, 13)):
            _fail(f"{context}的12点月最大负荷曲线月份不完整")
    values = pd.to_numeric(result.monthly_max_pu, errors="coerce")
    if values.isna().any() or (values < 0).any() or (values > 1 + 1e-12).any():
        _fail("月最大负荷标幺值必须位于0—1")
    maxima = result.assign(_value=values).groupby(
        ["planning_scenario", "trajectory"]
    )._value.max()
    if not np.isclose(maxima.to_numpy(float), 1.0, rtol=0.0, atol=1e-10).all():
        _fail("每条12点月最大负荷曲线的最大值必须为1")
    result["_scenario_order"] = result.planning_scenario.map(
        {value: index for index, value in enumerate(SCENARIOS)}
    )
    result["_trajectory_order"] = result.trajectory.map(
        {value: index for index, value in enumerate(trajectories)}
    )
    return result.sort_values(
        ["_scenario_order", "_trajectory_order", "month"], kind="mergesort"
    ).drop(columns=["_scenario_order", "_trajectory_order"]).reset_index(drop=True)




def _prepare_adaptive_typical_days(frame: pd.DataFrame) -> pd.DataFrame:
    """Prepare the already-computed adaptive typical-day result for 2030."""

    result = _ensure_year(frame, label="2030自适应典型日")
    _scenario_coverage(result, label="2030自适应典型日")
    trajectory_order = (
        "deterministic_baseline",
        "P10_real",
        "P50_real",
        "P90_real",
    )
    if set(result.trajectory.astype(str)) != set(trajectory_order):
        _fail("2030典型日必须覆盖确定性及P10/P50/P90真实轨迹")
    result["hour"] = pd.to_numeric(result.hour, errors="raise").astype(int)
    key = ["planning_scenario", "trajectory", "season", "final_day_type"]
    if result.duplicated([*key, "hour"]).any():
        _fail("2030自适应典型日小时键重复")
    counts = result.groupby(key).hour.nunique()
    if not counts.eq(24).all() or set(result.hour) != set(range(24)):
        _fail("2030自适应典型日每个单元必须完整覆盖0—23时")
    values = pd.to_numeric(result.typical_day_pu, errors="coerce")
    if values.isna().any() or (values < 0).any():
        _fail("2030自适应典型日含缺失或负标幺值")
    closure = result.assign(_value=values).groupby(key)._value.sum()
    if not np.isclose(closure.to_numpy(float), 1.0, rtol=0.0, atol=1e-10).all():
        _fail("2030自适应典型日24点和不为1")
    result["normalized_load"] = values
    result["day_group"] = result.final_day_type
    result["valid_days"] = pd.to_numeric(
        result.sample_days, errors="coerce"
    ).fillna(0).astype(int)
    result["result_type"] = "3.5.0自适应日型标幺曲线"
    trajectory_labels = {
        "deterministic_baseline": "确定性基线",
        "P10_real": "P10真实轨迹",
        "P50_real": "P50真实轨迹",
        "P90_real": "P90真实轨迹",
    }
    result["trajectory_code"] = result.trajectory.astype(str)
    result["trajectory"] = result.trajectory.map(trajectory_labels)
    result["_scenario_order"] = result.planning_scenario.map(
        {value: index for index, value in enumerate(SCENARIOS)}
    )
    result["_trajectory_order"] = result.trajectory_code.map(
        {value: index for index, value in enumerate(trajectory_order)}
    )
    result["_season_order"] = result.season.map(
        {value: index for index, value in enumerate(SEASONS)}
    )
    columns = [
        "method",
        "planning_scenario",
        "year",
        "trajectory",
        "trajectory_code",
        "season",
        "day_group",
        "final_day_type",
        "hour",
        "normalized_load",
        "valid_days",
        "result_type",
    ]
    return result.sort_values(
        ["_scenario_order", "_trajectory_order", "_season_order", "day_group", "hour"],
        kind="mergesort",
    )[columns].reset_index(drop=True)


def _quality_rows(
    *,
    metadata: Mapping[str, Any],
    weights: pd.DataFrame,
    inventory: pd.DataFrame,
    user_quality: pd.DataFrame,
    cell_judgement: pd.DataFrame,
    baseline_shapes: pd.DataFrame,
    shape_future: pd.DataFrame,
    correction_shape: pd.DataFrame,
    correction_diagnostics: pd.DataFrame,
    day_type_factors: pd.DataFrame,
    component_closure: pd.DataFrame,
    daily_checks: pd.DataFrame,
    typical_days: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add(category: str, check: str, value: object, expected: object, evidence: str) -> None:
        rows.append(
            {
                "category": category,
                "check": check,
                "value": value,
                "expected": expected,
                "status": "通过",
                "evidence_table": evidence,
            }
        )

    add("运行", "模型版本", metadata["model_version"], EXPECTED_MODEL_VERSION, "run_metadata.json")
    add("运行", "年度电量主方法", metadata["method"], EXPECTED_ANNUAL_ENERGY_METHOD, "run_metadata.json")
    add("运行", "展示年份", DISPLAY_YEAR, DISPLAY_YEAR, "run_metadata.json")
    add("运行", "规划情景数", len(SCENARIOS), 3, "run_metadata.json")
    add(
        "运行",
        "每情景场景数",
        metadata["bootstrap_scenarios_per_context"],
        EXPECTED_BOOTSTRAP_SCENARIOS,
        "run_metadata.json",
    )
    add("运行", "逐时文件格式", metadata["hourly_output"]["format"], "parquet", "run_metadata.json")
    add("组成", "传统细分行业数", int(weights.component_type.eq("传统细分行业").sum()), 24, "annual_weight_ledger")
    add("组成", "修正组成数", int(weights.component_type.eq(CORRECTION_COMPONENT).sum()), 1, "annual_weight_ledger")
    for year in ANCHOR_YEARS:
        error = abs(float(weights[f"share_{year}"].sum()) - 1.0)
        add("组成", f"{year}年25组成份额闭合误差", error, "≤1e-12", "权重与假设")
    add(
        "组成",
        "2030修正组成份额",
        float(weights.loc[weights.component.eq(CORRECTION_COMPONENT), "share_2030"].iloc[0]),
        "0—1之间",
        "权重与假设",
    )
    add("行业形状", "行业清单唯一数", int(inventory.industry.nunique()), 24, "industry_inventory")
    cell_day_column = (
        "final_day_type" if "final_day_type" in cell_judgement.columns else "day_group"
    )
    cell_key = ["industry", "season", cell_day_column]
    add(
        "行业形状",
        "行业单元判别数",
        int(cell_judgement[cell_key].drop_duplicates().shape[0]),
        "192—288（由自适应日型决定）",
        "cell_quality",
    )
    add(
        "行业形状",
        "行业基准曲线数",
        int(baseline_shapes[cell_key].drop_duplicates().shape[0]),
        "192—288（由自适应日型决定）",
        "baseline_shapes",
    )
    formal = shape_future.loc[
        _as_bool(shape_future.formal_H8, label="H8正式标记")
        & _as_bool(shape_future.formal_phi08, label="phi0.8正式标记")
        & pd.to_numeric(shape_future.year, errors="coerce").eq(DISPLAY_YEAR)
    ]
    add(
        "行业形状",
        "2030正式H8/phi0.8单元数",
        int(formal[cell_key].drop_duplicates().shape[0]),
        "192—288（由自适应日型决定）",
        "shape_future",
    )
    modeled_quality = user_quality.loc[
        pd.to_numeric(user_quality.total_days, errors="coerce").fillna(0).gt(0)
    ]
    add(
        "数据质量",
        "进入24行业质量审计的用户数",
        int(modeled_quality[["industry", "user"]].drop_duplicates().shape[0]),
        ">0",
        "user_quality",
    )
    add(
        "数据质量",
        "建模用户最低有效率",
        float(pd.to_numeric(modeled_quality.valid_rate, errors="coerce").min()),
        "≥0",
        "user_quality",
    )
    correction_share = pd.to_numeric(
        correction_shape.omega_R, errors="raise"
    )
    if not (correction_share.gt(0).all() and correction_share.lt(1).all()) or (
        correction_share.max() - correction_share.min() > 1e-12
    ):
        _fail("修正组成2023识别份额必须各季节日型一致且位于0—1")
    add(
        "修正组成",
        "2023同年识别份额",
        float(correction_share.iloc[0]),
        "各季节日型一致且0—1之间",
        "correction_component_shape",
    )
    max_reconstruction_error = float(
        pd.to_numeric(
            correction_diagnostics.reconstruction_max_abs_error, errors="raise"
        ).max()
    )
    if max_reconstruction_error > 1e-10:
        _fail("修正组成逐时复原误差超限")
    if "passed" in correction_diagnostics:
        diagnostics_passed = _as_bool(
            correction_diagnostics.passed, label="修正组成诊断标记"
        ).all()
    else:
        diagnostics_passed = (
            _as_bool(
                correction_diagnostics.nonnegative_passed,
                label="修正非负标记",
            ).all()
            and _as_bool(
                correction_diagnostics.normalization_passed,
                label="修正归一标记",
            ).all()
            and _as_bool(
                correction_diagnostics.reconstruction_passed,
                label="修正复原标记",
            ).all()
        )
    if not diagnostics_passed:
        _fail("修正组成诊断存在未通过项")
    add(
        "修正组成",
        "逐时复原最大绝对误差",
        max_reconstruction_error,
        "≤1e-10",
        "correction_component_diagnostics",
    )
    if "industry" in day_type_factors.columns:
        _fail("day_type_factors不得含industry维度")
    factor_scope = set(day_type_factors.factor_scope.dropna().astype(str))
    if factor_scope != {PROVINCE_SHARED_FACTOR_SCOPE}:
        _fail(f"全省共享日型系数范围不符：{sorted(factor_scope)}")
    source_years = set(
        pd.to_numeric(day_type_factors.source_year, errors="raise").astype(int)
    )
    if source_years != {2023}:
        _fail(f"全省共享日型系数来源年份不符：{sorted(source_years)}")
    add(
        "日电量分配",
        "全省共享季节—日型系数",
        len(day_type_factors),
        "2023全省识别并统一应用于全部25个组成",
        "day_type_factors",
    )
    closure = _ensure_year(component_closure, label="组成年度电量闭合")
    _scenario_coverage(closure, label="组成年度电量闭合")
    max_component_error = float(pd.to_numeric(closure.relative_error, errors="raise").max())
    if max_component_error > 1e-12:
        _fail("组成年度电量闭合误差超限")
    add("闭合", "年度组成电量最大相对误差", max_component_error, "≤1e-12", "component_energy_closure")
    daily = _ensure_year(daily_checks, label="逐日25组成闭合")
    _scenario_coverage(daily, label="逐日25组成闭合")
    if not _as_bool(daily.passed, label="逐日25组成闭合标记").all():
        _fail("至少一个自然日的25组成电量未闭合")
    if not pd.to_numeric(daily.component_count, errors="raise").eq(25).all():
        _fail("至少一个自然日不含完整25组成")
    max_daily_error = float(pd.to_numeric(daily.relative_error, errors="raise").max())
    add("闭合", "逐日25组成最大相对误差", max_daily_error, "≤各行容差", "daily_energy_checks")
    add("口径", "字段与表名口径扫描", "无退役结构", "仅现行24行业+修正组成", "tables_manifest.json")
    add(
        "典型日",
        "2030典型日轨迹覆盖",
        int(typical_days.trajectory.nunique()),
        "确定性+真实P10/P50/P90共4轨迹",
        "adaptive_typical_day_curves",
    )
    add("口径", "行业日聚合", "用户日归一化后行业同日等权", "固定3.5.0方法", "industry_daily_aggregate")
    add(
        "口径",
        "未来形状外推",
        "始终先完成四时段份额调整；仅门控通过时执行峰谷差率均值中心缩放",
        "门控关闭时中间曲线即最终；门控通过时峰谷差率最终优先",
        "shape_future",
    )
    add(
        "口径",
        "峰谷差率双重门控",
        metadata["peak_valley_gate_policy"],
        "历史充分、2023基准非平坦且2022—2023峰谷差率绝对变化量位于6.5%—25%闭区间",
        "run_metadata.json",
    )
    add(
        "口径",
        "平坦与变化量阈值",
        (
            f"LF>{metadata['flat_load_factor_threshold']}; "
            f"CV<{metadata['flat_cv_threshold']}; "
            f"|ΔRpv|∈[{metadata['peak_valley_delta_lower_bound']},"
            f"{metadata['peak_valley_delta_upper_bound']}]"
        ),
        "LF>0.80; CV<0.15; |ΔRpv|∈[0.065,0.25]",
        "run_metadata.json",
    )
    add(
        "口径",
        "合成后处理",
        "全省合成曲线不追加峰谷差率校正",
        "晚峰与控制时段只诊断",
        "run_metadata.json",
    )
    add("口径", "日型系数范围", PROVINCE_SHARED_FACTOR_SCOPE, PROVINCE_SHARED_FACTOR_SCOPE, "day_type_factors")
    add("口径", "新型负荷处理", "不单扣、不独立塑形；包含在修正组成", "现行口径", "run_metadata.json")
    return pd.DataFrame(rows)


def _write_source(
    output_dir: Path,
    name: str,
    frame: pd.DataFrame,
    written: dict[str, dict[str, object]],
) -> None:
    if name not in SOURCE_ORDER:
        raise AssertionError(f"未知展示源：{name}")
    path = output_dir / f"{name}.csv"
    frame.to_csv(path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d %H:%M:%S")
    written[name] = {
        "file": path.name,
        "rows": int(len(frame)),
        "columns": [str(value) for value in frame.columns],
        "sha256": _sha256(path),
    }


def prepare(data_dir: Path, metadata_path: Path, output_dir: Path) -> dict[str, Any]:
    data_dir = data_dir.resolve()
    metadata_path = metadata_path.resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(data_dir)
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    result_manifest_path = data_dir / "tables_manifest.json"
    if not result_manifest_path.is_file():
        raise FileNotFoundError(result_manifest_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    manifest = json.loads(result_manifest_path.read_text(encoding="utf-8-sig"))
    if not isinstance(metadata, Mapping) or not isinstance(manifest, Mapping):
        _fail("运行元数据和结果清单必须是JSON对象")
    _validate_metadata(metadata)
    _validate_result_manifest(manifest)
    tables = {
        name: _load_table(data_dir, manifest, name) for name in TABLE_CONTRACTS
    }
    _validate_method_contract(metadata, tables)

    weights = _derive_weight_table(
        tables["annual_weight_ledger"], tables["industry_inventory"]
    )
    proxy_shapes = _prepare_proxy_shapes(tables["proxy_sector_typical_shapes"])
    correction_shape = _prepare_correction_shape(
        tables["correction_component_shape"]
    )
    annual_metrics = _prepare_annual_metrics(
        tables["deterministic_annual_metrics"],
        tables["representative_annual_metrics"],
    )
    monthly_metrics = _prepare_monthly_metrics(
        tables["deterministic_monthly_metrics"],
        tables["representative_monthly_metrics"],
    )
    monthly_peak_curve = _prepare_monthly_peak_curve(
        tables["monthly_maximum_curves_pu"]
    )
    typical_days = _prepare_adaptive_typical_days(
        tables["adaptive_typical_day_curves"]
    )
    quality = _quality_rows(
        metadata=metadata,
        weights=weights,
        inventory=tables["industry_inventory"],
        user_quality=tables["user_quality"],
        cell_judgement=tables["cell_quality"],
        baseline_shapes=tables["baseline_shapes"],
        shape_future=tables["shape_future"],
        correction_shape=correction_shape,
        correction_diagnostics=tables["correction_component_diagnostics"],
        day_type_factors=tables["day_type_factors"],
        component_closure=tables["component_energy_closure"],
        daily_checks=tables["daily_energy_checks"],
        typical_days=typical_days,
    )

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, dict[str, object]] = {}
    _write_source(output_dir, "权重与假设", weights, written)
    _write_source(output_dir, "四类代理曲线", proxy_shapes, written)
    _write_source(output_dir, "修正组成", correction_shape, written)
    _write_source(output_dir, "2030年度指标", annual_metrics, written)
    _write_source(output_dir, "2030月度指标", monthly_metrics, written)
    _write_source(output_dir, "12点月最大曲线", monthly_peak_curve, written)
    _write_source(output_dir, "2030典型日", typical_days, written)
    _write_source(output_dir, "质量检查", quality, written)
    _write_source(output_dir, "输入文件清单", tables["input_manifest"].copy(), written)
    if tuple(written) != SOURCE_ORDER:
        raise AssertionError("展示源写入顺序与合同不一致")

    source_manifest: dict[str, Any] = {
        "schema_version": "3.5.0-2030-display",
        "target_year": DISPLAY_YEAR,
        "expected_output_name": EXPECTED_OUTPUT_NAME,
        "source_data_dir": str(data_dir),
        "source_run_metadata": str(metadata_path),
        "source_tables_manifest": str(result_manifest_path.resolve()),
        "source_run_metadata_sha256": _sha256(metadata_path),
        "source_tables_manifest_sha256": _sha256(result_manifest_path),
        "model_version": metadata["model_version"],
        "method_revision": metadata["method_revision"],
        "annual_energy_method": metadata["method"],
        "shape_extrapolation_method": metadata["shape_extrapolation_method"],
        "four_window_final_share_policy": metadata[
            "four_window_final_share_policy"
        ],
        "peak_valley_target_priority": metadata["peak_valley_target_priority"],
        "peak_valley_extrapolation_enabled": metadata[
            "peak_valley_extrapolation_enabled"
        ],
        "peak_valley_gate_policy": metadata["peak_valley_gate_policy"],
        "peak_valley_delta_lower_bound": metadata[
            "peak_valley_delta_lower_bound"
        ],
        "peak_valley_delta_upper_bound": metadata[
            "peak_valley_delta_upper_bound"
        ],
        "flat_load_factor_threshold": metadata["flat_load_factor_threshold"],
        "flat_cv_threshold": metadata["flat_cv_threshold"],
        "late_peak_control_policy": metadata["late_peak_control_policy"],
        "day_type_factor_scope": metadata["day_type_factor_scope"],
        "method_statements": list(REPORT_METHOD_STATEMENTS),
        "planning_scenarios": list(SCENARIOS),
        "bootstrap_scenarios_per_context": EXPECTED_BOOTSTRAP_SCENARIOS,
        "component_count": EXPECTED_COMPONENTS,
        "sheet_sources": written,
    }
    (output_dir / "workbook_sources_manifest.json").write_text(
        json.dumps(source_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return source_manifest


def contract_summary() -> dict[str, Any]:
    return {
        "schema_version": "3.5.0-2030-display",
        "target_year": DISPLAY_YEAR,
        "expected_output_name": EXPECTED_OUTPUT_NAME,
        "required_result_tables": {
            name: sorted(columns) for name, columns in TABLE_CONTRACTS.items()
        },
        "forbidden_result_tables": sorted(FORBIDDEN_TABLE_NAMES),
        "forbidden_columns": sorted(
            set(FORBIDDEN_COLUMNS).union(FORBIDDEN_SHAPE_FUTURE_COLUMNS)
        ),
        "method_statements": list(REPORT_METHOD_STATEMENTS),
        "prepared_sources": list(SOURCE_ORDER),
        "workbook_sheets": [
            "模型说明",
            "权重与假设",
            "四类代理曲线",
            "修正组成",
            "2030年度指标",
            "2030月度指标",
            "12点月最大曲线",
            "2030典型日",
            "图表总览",
            "质量与版本",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, help="正式2030结果的 数据表 目录")
    parser.add_argument("--metadata", type=Path, help="正式2030结果的 run_metadata.json")
    parser.add_argument("--output-dir", type=Path, help="工作簿中间CSV与manifest目录")
    parser.add_argument(
        "--print-contract",
        action="store_true",
        help="只打印输入/输出合同，不读取或写入结果文件",
    )
    args = parser.parse_args()
    if args.print_contract:
        print(json.dumps(contract_summary(), ensure_ascii=False, indent=2))
        return
    if args.data_dir is None or args.metadata is None or args.output_dir is None:
        parser.error("非--print-contract模式必须同时提供--data-dir、--metadata和--output-dir")
    result = prepare(args.data_dir, args.metadata, args.output_dir)
    print(
        json.dumps(
            {
                "schema_version": result["schema_version"],
                "sheet_sources": len(result["sheet_sources"]),
                "output_dir": str(args.output_dir.resolve()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
