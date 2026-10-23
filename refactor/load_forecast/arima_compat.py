# -*- coding: utf-8 -*-
"""Compatibility wrapper for old and new statsmodels ARIMA implementations.

The original project was developed across several Python environments.  The
wrapper keeps one ``ARIMA(...).fit(...).forecast(...)`` interface.  If an
individual ARIMA fit fails because an old statsmodels version cannot estimate
the selected order, a short local linear extrapolation is used and the very
large AIC prevents that fallback from winning when any real ARIMA fit works.
"""

import numpy as np

try:
    from statsmodels.tsa.arima.model import ARIMA as _ModernARIMA
except ImportError:
    _ModernARIMA = None

try:
    from statsmodels.tsa.arima_model import ARIMA as _LegacyARIMA
except ImportError:
    _LegacyARIMA = None


class _LinearFit:
    """ARIMA全部候选失败时的保底拟合对象。

    它只提供 ``aic`` 和 ``forecast`` 两个接口，使旧版statsmodels或极端
    序列仍能完成流程；AIC设为极大值，正常ARIMA成功时不会选中该保底模型。
    """
    def __init__(self, values, order):
        self.values = np.asarray(values, dtype=float)
        self.order = order
        self.aic = 1e100

    def forecast(self, steps=1):
        n = len(self.values)
        if n < 2:
            return np.repeat(self.values[-1] if n else 0.0, steps)
        k = min(8, n)
        slope, intercept = np.polyfit(np.arange(k, dtype=float), self.values[-k:], 1)
        result = np.asarray([intercept + slope * (k + i) for i in range(steps)])
        return np.maximum(result, 1e-9)


class ARIMA:
    """兼容新版状态空间ARIMA和旧版statsmodels ARIMA API。"""
    def __init__(self, endog, order):
        self.endog = np.asarray(endog, dtype=float)
        self.order = order
        if _ModernARIMA is not None:
            self.model = _ModernARIMA(self.endog, order=order)
        elif _LegacyARIMA is not None:
            self.model = _LegacyARIMA(self.endog, order=order)
        else:
            self.model = None

    def fit(self, method_kwargs=None):
        # 新版fit接受method_kwargs，旧版fit使用disp=0；两种接口统一成同一返回对象。
        if self.model is None:
            return _LinearFit(self.endog, self.order)
        try:
            if _ModernARIMA is not None:
                return self.model.fit(method_kwargs=method_kwargs or {})
            return self.model.fit(disp=0)
        except Exception:
            return _LinearFit(self.endog, self.order)
