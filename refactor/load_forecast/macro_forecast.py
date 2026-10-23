#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""02 段:宏观特征 ARIMA 外推(原包 02_宏观特征预测/run_stage.py 移植)。

计算行逐行保持一致;差异仅两处:
- 输入从包内固定路径改为显式传参;
- 审计 source 写逻辑名 "111.xlsx"(比对口径 D18 把原包相对路径归一化为同名)。
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from arima_compat import ARIMA
from data_io import FEATURES, TARGET, load_macro_table

warnings.filterwarnings("ignore")


def determine_d(values, max_d: int = 2) -> int:
    """用ADF检验确定ARIMA差分阶数，最多差分两次。"""
    current = np.asarray(values, dtype=float)
    d = 0
    while d < max_d:
        try:
            if adfuller(current)[1] < 0.05:
                return d
        except Exception:
            return d
        current = np.diff(current)
        d += 1
    return d


def auto_arima(values):
    """在p、q均为0到3的候选中按AIC选择ARIMA阶数。

    ``(0, d, 0)`` 被排除，因为纯差分模型不能体现序列的自回归或移动
    平均结构；单个候选拟合失败时跳过，不影响其它阶数组合搜索。
    """
    d = determine_d(values)
    best = None
    for p in range(4):
        for q in range(4):
            if p == 0 and q == 0:
                continue
            try:
                fit = ARIMA(values, order=(p, d, q)).fit(method_kwargs={"warn_convergence": False})
            except Exception:
                continue
            if best is None or float(fit.aic) < best[0]:
                best = (float(fit.aic), (p, d, q), fit)
    if best is None:
        fit = ARIMA(values, order=(0, d, 1)).fit()
        best = (float(fit.aic), (0, d, 1), fit)
    return best


def rolling_forecast(values, steps: int):
    """递归预测：每预测一年，就把该预测值追加回历史序列。"""
    history = list(np.asarray(values, dtype=float))
    forecasts, orders = [], []
    for _ in range(steps):
        _, order, fit = auto_arima(np.asarray(history, dtype=float))
        raw_forecast = fit.forecast(1)
        # statsmodels 0.10 returns (forecast, stderr, confidence_interval),
        # while newer releases return an array-like forecast directly.
        if isinstance(raw_forecast, tuple):
            raw_forecast = raw_forecast[0]
        prediction = float(np.asarray(raw_forecast).reshape(-1)[0])
        history.append(prediction)
        forecasts.append(prediction)
        orders.append(order)
    return forecasts, orders


def mape(actual, predicted) -> float:
    """计算平均绝对百分比误差，结果单位为百分数。"""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs((actual - predicted) / actual)) * 100.0)


def run_macro_feature_forecast(macro_workbook: Path, run_dir: Path) -> Path:
    """执行特征相关性、ARIMA外推和扩展窗口回测,写 4 个判据文件。"""
    output = run_dir / "02_宏观特征预测"
    output.mkdir(parents=True, exist_ok=True)
    frame = load_macro_table(macro_workbook)

    # Pearson衡量线性相关，Spearman先转秩后衡量单调相关；两者只用于
    # 量化关联程度，不代表因果关系。
    correlation_rows = []
    for feature in FEATURES:
        pearson = float(frame[feature].corr(frame[TARGET], method="pearson"))
        spearman = float(frame[feature].rank().corr(frame[TARGET].rank(), method="pearson"))
        correlation_rows.append(
            {
                "特征": feature,
                "Pearson系数": pearson,
                "Spearman系数": spearman,
                "系数和": pearson + spearman,
                "绝对值系数和": abs(pearson) + abs(spearman),
            }
        )
    pd.DataFrame(correlation_rows).to_csv(output / "特征与总电量相关性.csv", index=False, encoding="utf-8-sig")

    # 第03步只读取这个预测文件，因此文件列名是跨阶段接口的一部分。
    forecast_years = list(range(2026, 2031))
    forecast_table = pd.DataFrame({"年份": forecast_years})
    order_audit = {}
    for feature in FEATURES:
        values, orders = rolling_forecast(frame[feature].values, len(forecast_years))
        forecast_table[feature] = values
        order_audit[feature] = [list(order) for order in orders]
    forecast_table["GDP总量"] = forecast_table[FEATURES[:3]].sum(axis=1)
    forecast_table.to_csv(output / "ARIMA特征预测_2026-2030.csv", index=False, encoding="utf-8-sig")

    # 扩展窗口单步回测：模拟“只知道当年以前数据”时外推下一年，
    # 用于评价ARIMA特征外推本身，而不是评价后续MLP总电量模型。
    backtest_rows = []
    for feature in FEATURES:
        actual, predicted = [], []
        for year in range(2020, 2026):
            train = frame.loc[frame.index < year, feature].values
            value, _ = rolling_forecast(train, 1)
            actual.append(float(frame.loc[year, feature]))
            predicted.append(value[0])
        errors = np.asarray(predicted) - np.asarray(actual)
        backtest_rows.append(
            {
                "特征": feature,
                "回测区间": "2020-2025",
                "MAPE_pct": mape(actual, predicted),
                "RMSE": float(np.sqrt(np.mean(errors ** 2))),
                "MAE": float(np.mean(np.abs(errors))),
            }
        )
    pd.DataFrame(backtest_rows).to_csv(output / "ARIMA特征外推回测指标.csv", index=False, encoding="utf-8-sig")

    audit = {
        "source": "111.xlsx",
        "history": [int(frame.index.min()), int(frame.index.max())],
        "forecast_years": forecast_years,
        "features": FEATURES,
        "orders_by_recursive_step": order_audit,
        "method": "ADF确定d，p/q=0..3按AIC择优，逐年滚动并将预测值写回历史",
    }
    (output / "阶段审计.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[02] ARIMA feature forecasting completed")
    return output
