"""中长期负荷特性模型 3.5.0。

正式形状链以 2023 行业聚合 medoid 为基准，先在时段内同比例缩放以
命中四时段份额；仅在历史充分、基准非平坦且峰谷差率绝对变化量位于
正式闭区间时，才围绕全日均值缩放 24 点偏差。晚峰和控制时段只作诊断。
"""

from .config import METHOD_REVISION, MODEL_VERSION

__all__ = ["METHOD_REVISION", "MODEL_VERSION"]
