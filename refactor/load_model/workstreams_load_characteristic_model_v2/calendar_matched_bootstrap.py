"""快速验证用的日型匹配Bootstrap，不改动正式主链路。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .calendar_utils import calendar_frame
from .day_types import HOLIDAY, WEEKEND, WORKDAY
from .province_bootstrap import BASE_SEED


_BASE_SAMPLE_CLASSES = (WORKDAY, WEEKEND)
_CLASS_SEED = {WORKDAY: 1, WEEKEND: 2, HOLIDAY: 3}


def _validate_baseline(
    baseline_daily_energy: pd.Series, target_year: int
) -> tuple[pd.Series, pd.DatetimeIndex]:
    baseline = baseline_daily_energy.copy()
    baseline.index = pd.to_datetime(baseline.index).normalize()
    baseline = baseline.sort_index()
    expected = pd.date_range(
        f"{int(target_year)}-01-01",
        f"{int(target_year)}-12-31",
        freq="D",
    )
    if not baseline.index.equals(expected):
        raise ValueError(f"{target_year}基线日电量日期不完整")
    values = pd.to_numeric(baseline, errors="coerce").to_numpy(float)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("基线日电量必须为有限正数")
    return baseline.astype(float), expected


def _validated_pools(
    residual_library: pd.DataFrame,
) -> tuple[dict[tuple[int, int, str], np.ndarray], tuple[int, ...]]:
    required = {
        "history_year",
        "month",
        "date",
        "residual_day_class",
        "residual",
    }
    missing = required - set(residual_library.columns)
    if missing:
        raise ValueError(f"日型匹配残差库缺少字段：{sorted(missing)}")

    data = residual_library[list(required)].copy()
    data["history_year"] = pd.to_numeric(data.history_year, errors="coerce")
    data["month"] = pd.to_numeric(data.month, errors="coerce")
    data["date"] = pd.to_datetime(data.date, errors="coerce").dt.normalize()
    data["residual"] = pd.to_numeric(data.residual, errors="coerce")
    data["residual_day_class"] = data.residual_day_class.astype(str).str.strip()
    if data[["history_year", "month", "date", "residual"]].isna().any().any():
        raise ValueError("日型匹配残差库含缺失值")
    if not np.isfinite(data.residual.to_numpy(float)).all():
        raise ValueError("日型匹配残差库含非有限残差")
    if (
        not np.equal(data.history_year, np.floor(data.history_year)).all()
        or not np.equal(data.month, np.floor(data.month)).all()
    ):
        raise ValueError("history_year和month必须为整数")
    data["history_year"] = data.history_year.astype(int)
    data["month"] = data.month.astype(int)
    if not data.month.between(1, 12).all():
        raise ValueError("month必须位于1..12")
    allowed = {WORKDAY, WEEKEND, HOLIDAY}
    unknown = sorted(set(data.residual_day_class).difference(allowed))
    if unknown:
        raise ValueError(f"残差库含未知日型：{unknown}")
    if data.duplicated(["history_year", "date"]).any():
        raise ValueError("history_year/date必须唯一")

    pools: dict[tuple[int, int, str], np.ndarray] = {}
    for key, part in data.groupby(
        ["history_year", "month", "residual_day_class"], sort=True
    ):
        values = part.sort_values("date").residual.to_numpy(float)
        if len(values):
            pools[(int(key[0]), int(key[1]), str(key[2]))] = values
    years = tuple(sorted(data.history_year.unique()))
    if not years:
        raise ValueError("残差库没有历史年份")
    if not any(key[2] == HOLIDAY for key in pools):
        raise ValueError("残差库缺少法定节假日残差")
    return pools, years


def _balanced_source_years(
    history_years: tuple[int, ...],
    n_scenarios: int,
    target_year: int,
    base_seed: int,
) -> np.ndarray:
    if int(n_scenarios) <= 0:
        raise ValueError("场景数必须为正整数")
    assigned = np.resize(np.asarray(history_years, dtype=int), int(n_scenarios)).copy()
    rng = np.random.default_rng(
        np.random.SeedSequence([int(base_seed), int(target_year), 97001])
    )
    rng.shuffle(assigned)
    return assigned


def _single_circular_shift(
    values: np.ndarray, target_days: int, rng: np.random.Generator
) -> np.ndarray:
    source = np.asarray(values, dtype=float)
    if source.ndim != 1 or source.size == 0 or int(target_days) <= 0:
        raise ValueError("循环平移抽样参数非法")
    start = int(rng.integers(0, len(source)))
    positions = (start + np.arange(int(target_days))) % len(source)
    return source[positions]


def _select_pool(
    pools: dict[tuple[int, int, str], np.ndarray],
    source_year: int,
    month: int,
    day_class: str,
) -> tuple[np.ndarray, str]:
    key = (int(source_year), int(month), str(day_class))
    if key in pools:
        return pools[key], "same_history_year_month_day_class"
    same_year = sorted(
        (
            (source_month, values)
            for (year, source_month, source_class), values in pools.items()
            if year == int(source_year) and source_class == str(day_class)
        ),
        key=lambda item: item[0],
    )
    if same_year:
        return (
            np.concatenate([values for _, values in same_year]),
            "same_history_year_day_class_all_months_fallback",
        )
    raise ValueError(
        f"source_history_year={int(source_year)}没有可用于"
        f"{int(month)}月/{day_class}的同年残差"
    )


def bootstrap_daily_multipliers_calendar_matched(
    residual_library: pd.DataFrame,
    baseline_daily_energy: pd.Series,
    target_year: int,
    n_scenarios: int = 1000,
    block_length: int = 7,
    base_seed: int = BASE_SEED,
    *,
    freeze_holidays: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按全年历史骨架和目标日型生成快速验证日乘数。

    每个场景固定一个历史来源年份，所有残差只能映射到相同目标日型。
    默认冻结法定节假日残差；``freeze_holidays=False``时，法定节假日
    仅从法定节假日池中抽取，用于比较冻结是否过度收窄峰值月份分布。
    ``block_length``仅作为与原接口兼容的审计字段。
    """

    if int(n_scenarios) <= 0 or int(block_length) <= 0:
        raise ValueError("Bootstrap场景数和块长必须为正整数")
    baseline, expected = _validate_baseline(baseline_daily_energy, target_year)
    pools, history_years = _validated_pools(residual_library)
    assigned_years = _balanced_source_years(
        history_years, n_scenarios, target_year, base_seed
    )

    calendar = calendar_frame(int(target_year)).sort_values("date")
    calendar["date"] = pd.to_datetime(calendar.date).dt.normalize()
    if not calendar.date.reset_index(drop=True).equals(pd.Series(expected, name="date")):
        raise ValueError(f"{target_year}目标日历日期不完整")
    calendar = calendar.set_index("date")
    sample_classes = (
        _BASE_SAMPLE_CLASSES
        if bool(freeze_holidays)
        else (*_BASE_SAMPLE_CLASSES, HOLIDAY)
    )
    holiday_policy = "frozen_zero" if freeze_holidays else "same_day_class_sampled"

    rows: list[dict[str, object]] = []
    checks: list[dict[str, object]] = []
    for scenario_id, source_year in enumerate(assigned_years):
        for month in range(1, 13):
            month_index = expected[expected.month == month]
            month_baseline = baseline.loc[month_index].to_numpy(float)
            target_classes = (
                calendar.loc[month_index, "canonical_day_type"].astype(str).to_numpy()
            )
            residual = np.zeros(len(month_index), dtype=float)
            source_classes = np.full(len(month_index), HOLIDAY, dtype=object)
            pool_sources = np.full(
                len(month_index), "holiday_frozen_zero", dtype=object
            )
            frozen = (
                target_classes == HOLIDAY
                if freeze_holidays
                else np.zeros(len(month_index), dtype=bool)
            )

            for day_class in sample_classes:
                positions = np.flatnonzero(target_classes == day_class)
                if not len(positions):
                    continue
                pool, pool_source = _select_pool(
                    pools, int(source_year), month, day_class
                )
                rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [
                            int(base_seed),
                            int(target_year),
                            int(scenario_id),
                            int(month),
                            _CLASS_SEED[day_class],
                        ]
                    )
                )
                residual[positions] = _single_circular_shift(
                    pool, len(positions), rng
                )
                source_classes[positions] = day_class
                pool_sources[positions] = pool_source

            raw = np.exp(residual)
            multiplier = np.ones(len(month_index), dtype=float)
            adjustable = ~frozen
            if adjustable.any():
                base_adjustable = month_baseline[adjustable]
                denominator = float(np.dot(base_adjustable, raw[adjustable]))
                if denominator <= 0 or not np.isfinite(denominator):
                    raise AssertionError("月度归一化分母非法")
                multiplier[adjustable] = (
                    raw[adjustable] * float(base_adjustable.sum()) / denominator
                )
            before = float(month_baseline.sum())
            after = float(np.dot(month_baseline, multiplier))
            error = abs(after / before - 1.0)
            if (
                error > 1e-12
                or not np.isfinite(multiplier).all()
                or np.any(multiplier <= 0)
                or not np.allclose(multiplier[frozen], 1.0, rtol=0.0, atol=1e-15)
            ):
                raise AssertionError(
                    f"{target_year}-{month:02d}/场景{scenario_id}月度闭合失败：{error}"
                )

            for values in zip(
                month_index,
                residual,
                multiplier,
                target_classes,
                source_classes,
                frozen,
                pool_sources,
                strict=True,
            ):
                stamp, res, factor, target_class, source_class, is_frozen, source = values
                rows.append(
                    {
                        "target_year": int(target_year),
                        "scenario_id": int(scenario_id),
                        "date": stamp,
                        "month": int(month),
                        "raw_residual": float(res),
                        "daily_multiplier": float(factor),
                        "block_length": int(block_length),
                        "base_seed": int(base_seed),
                        "source_history_year": int(source_year),
                        "target_day_class": str(target_class),
                        "residual_source_day_class": str(source_class),
                        "holiday_residual_frozen": bool(is_frozen),
                        "holiday_policy": holiday_policy,
                        "sampling_pool_source": str(source),
                        "sampling_method": "calendar_matched_single_circular_shift",
                    }
                )
            checks.append(
                {
                    "target_year": int(target_year),
                    "scenario_id": int(scenario_id),
                    "source_history_year": int(source_year),
                    "month": int(month),
                    "baseline_energy_MWh": before,
                    "scenario_energy_MWh": after,
                    "relative_error": error,
                    "holiday_days_frozen": int(frozen.sum()),
                    "adjustable_days": int(adjustable.sum()),
                    "holiday_policy": holiday_policy,
                    "block_length": int(block_length),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(checks)
