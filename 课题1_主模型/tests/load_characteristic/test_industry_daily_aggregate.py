from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from workstreams_load_characteristic_model_v2.quality import (
    SHAPE_COLS,
    aggregate_industry_daily,
)


def _row(user: str, date: str, curve: list[float]) -> dict[str, object]:
    return {
        "department": "第二产业",
        "industry": "测试行业",
        "user": user,
        "date": pd.Timestamp(date),
        "year": 2023,
        "season": "夏季",
        "day_type_detail": "普通工作日",
        **dict(zip(SHAPE_COLS, curve, strict=True)),
    }


def test_equal_weight_is_independent_of_user_absolute_scale() -> None:
    a = np.zeros(24)
    b = np.zeros(24)
    a[8] = 1.0
    b[20] = 1.0
    frame = pd.DataFrame(
        [
            _row("A", "2023-07-03", a.tolist()),
            _row("B", "2023-07-03", b.tolist()),
        ]
    )

    daily, _ = aggregate_industry_daily(frame)

    curve = daily.iloc[0][SHAPE_COLS].to_numpy(float)
    assert np.allclose(curve, (a + b) / 2)
    assert daily.iloc[0].aggregation_method == "user_day_normalize_then_equal_weight"


def test_missing_user_is_not_filled_with_zero() -> None:
    complete = np.arange(1, 25, dtype=float)
    complete /= complete.sum()
    frame = pd.DataFrame(
        [
            _row("A", "2023-07-03", complete.tolist()),
            _row("B", "2023-07-03", complete.tolist()),
            _row("A", "2023-07-04", complete.tolist()),
        ]
    )

    daily, diagnostics = aggregate_industry_daily(frame)

    second = daily.loc[daily.date.eq(pd.Timestamp("2023-07-04"))].iloc[0]
    second_diag = diagnostics.loc[
        diagnostics.date.eq(pd.Timestamp("2023-07-04"))
    ].iloc[0]
    assert np.allclose(second[SHAPE_COLS].to_numpy(float), complete)
    assert int(second_diag.active_user_count) == 1
    assert int(second_diag.source_user_count) == 2
    assert int(second_diag.missing_user_count) == 1
    assert bool(second_diag.partial_composition_day)


def test_no_active_user_produces_no_industry_day() -> None:
    empty = pd.DataFrame(
        columns=[
            "department",
            "industry",
            "user",
            "date",
            "year",
            "season",
            "day_type_detail",
            *SHAPE_COLS,
        ]
    )
    daily, diagnostics = aggregate_industry_daily(empty)
    assert daily.empty
    assert diagnostics.empty


def test_aggregate_rejects_an_unnormalized_user_day() -> None:
    curve = np.ones(24)
    with pytest.raises(ValueError, match="未完成归一化"):
        aggregate_industry_daily(
            pd.DataFrame([_row("A", "2023-07-03", curve.tolist())])
        )

