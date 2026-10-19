"""仅在全省层面实施同历月循环移动块Bootstrap。"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


BASE_SEED = 20260817


def _month_pools(residual_library: pd.DataFrame) -> dict[int, list[np.ndarray]]:
    required = {"history_year", "month", "date", "residual"}
    missing = required - set(residual_library.columns)
    if missing:
        raise ValueError(f"残差库缺少字段：{sorted(missing)}")
    pools: dict[int, list[np.ndarray]] = {month: [] for month in range(1, 13)}
    for (_, month), part in residual_library.groupby(["history_year", "month"], sort=True):
        ordered = part.sort_values("date")
        values = ordered.residual.to_numpy(float)
        if len(values) > 0 and np.isfinite(values).all():
            pools[int(month)].append(values)
    missing_months = [month for month, values in pools.items() if not values]
    if missing_months:
        raise ValueError(f"Block Bootstrap无有效同历月残差：{missing_months}")
    return pools


def circular_month_sample(
    month_pools: list[np.ndarray],
    target_days: int,
    rng: np.random.Generator,
    block_length: int = 7,
) -> np.ndarray:
    if target_days <= 0 or block_length <= 0 or not month_pools:
        raise ValueError("循环块抽样参数非法")
    sampled: list[float] = []
    blocks = int(math.ceil(target_days / block_length))
    for _ in range(blocks):
        pool = month_pools[int(rng.integers(0, len(month_pools)))]
        start = int(rng.integers(0, len(pool)))
        sampled.extend(float(pool[(start + offset) % len(pool)]) for offset in range(block_length))
    return np.asarray(sampled[:target_days], dtype=float)


def bootstrap_daily_multipliers(
    residual_library: pd.DataFrame,
    baseline_daily_energy: pd.Series,
    target_year: int,
    n_scenarios: int = 1000,
    block_length: int = 7,
    base_seed: int = BASE_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = baseline_daily_energy.copy()
    baseline.index = pd.to_datetime(baseline.index).normalize()
    baseline = baseline.sort_index()
    expected = pd.date_range(f"{target_year}-01-01", f"{target_year}-12-31", freq="D")
    if not baseline.index.equals(expected):
        raise ValueError(f"{target_year}基线日电量日期不完整")
    if not np.isfinite(baseline.to_numpy(float)).all() or (baseline <= 0).any():
        raise ValueError("基线日电量必须为有限正数")
    pools = _month_pools(residual_library)
    rows: list[dict] = []
    checks: list[dict] = []
    for scenario_id in range(int(n_scenarios)):
        for month in range(1, 13):
            month_index = expected[expected.month == month]
            month_baseline = baseline.loc[month_index].to_numpy(float)
            rng = np.random.default_rng(np.random.SeedSequence([int(base_seed), int(target_year), int(scenario_id), int(month)]))
            residual = circular_month_sample(pools[month], len(month_index), rng, block_length)
            raw = np.exp(residual)
            denominator = float(np.dot(month_baseline, raw) / month_baseline.sum())
            multiplier = raw / denominator
            before = float(month_baseline.sum())
            after = float(np.dot(month_baseline, multiplier))
            error = abs(after / before - 1.0)
            if error > 1e-12 or np.any(multiplier <= 0):
                raise AssertionError(f"{target_year}-{month:02d}/场景{scenario_id}月度闭合失败：{error}")
            rows.extend({"target_year": target_year, "scenario_id": scenario_id, "date": stamp, "month": month, "raw_residual": float(value), "daily_multiplier": float(factor), "block_length": block_length, "base_seed": base_seed} for stamp, value, factor in zip(month_index, residual, multiplier))
            checks.append({"target_year": target_year, "scenario_id": scenario_id, "month": month, "baseline_energy_MWh": before, "scenario_energy_MWh": after, "relative_error": error, "block_length": block_length})
    return pd.DataFrame(rows), pd.DataFrame(checks)


def apply_daily_multipliers(hourly_baseline: pd.DataFrame, multipliers: pd.DataFrame) -> pd.DataFrame:
    required_hourly = {"timestamp", "load_MW"}
    if required_hourly - set(hourly_baseline.columns):
        raise ValueError("逐时基线必须包含 timestamp/load_MW")
    selected = multipliers[["scenario_id", "date", "daily_multiplier"]].copy()
    selected["date"] = pd.to_datetime(selected.date).dt.normalize()
    if selected.duplicated(["scenario_id", "date"]).any():
        raise ValueError("日乘数的scenario_id+date必须唯一")
    if not np.isfinite(selected.daily_multiplier).all() or (selected.daily_multiplier <= 0).any():
        raise ValueError("日乘数必须为有限正数")
    base = hourly_baseline.copy()
    base["date"] = pd.to_datetime(base.timestamp).dt.normalize()
    hours_per_day = base.groupby("date").size()
    if not hours_per_day.eq(24).all():
        raise ValueError("逐时基线每个自然日必须恰有24个小时")
    result = selected.merge(base, on="date", how="left", validate="many_to_many")
    if result.load_MW.isna().any():
        raise ValueError("日乘数与逐时基线日期未完全匹配")
    result["load_MW"] = result.load_MW * result.daily_multiplier
    return result.drop(columns="date")
