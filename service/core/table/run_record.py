#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/10/21 15:40
@Author  : weiyutao
@File    : run_record.py

runs 表模型与运行状态枚举(SQLAlchemy 2.0 声明式)。
"""

from datetime import datetime
from enum import IntEnum, unique

from sqlalchemy import JSON, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """全项目统一声明基类。"""


@unique
class RunStatus(IntEnum):
    """运行状态机,编码十位递增。"""

    QUEUED = 10     # 排队:已登记,等待执行
    RUNNING = 20    # 运行中:算法计算进行中
    SUCCEEDED = 30  # 成功:结果 CSV 已生成
    FAILED = 40     # 失败:error_message 记录原因


RUN_STATUS_LABELS: dict[int, str] = {
    RunStatus.QUEUED: "排队",
    RunStatus.RUNNING: "运行中",
    RunStatus.SUCCEEDED: "成功",
    RunStatus.FAILED: "失败",
}

RUN_STATUS_NAMES: dict[int, str] = {
    RunStatus.QUEUED: "queued",
    RunStatus.RUNNING: "running",
    RunStatus.SUCCEEDED: "succeeded",
    RunStatus.FAILED: "failed",
}  # 接口回传的状态英文名(对齐 regional_power_system_optimization 的 jobs 契约)


class RunRecord(Base):
    """运行记录表:一次上传触发的算法运行。"""

    __tablename__ = "runs"

    run_id: Mapped[str] = mapped_column(String(32), primary_key=True)       # 运行标识(uuid4 hex)
    topic: Mapped[str] = mapped_column(String(32), default="load_model", server_default="load_model")  # 课题标识(load_model 课题1/load_forecast 课题2)
    params: Mapped[dict] = mapped_column(JSON)                              # 运行参数(起始年/结束年/抽样数/原始文件名)
    status: Mapped[int] = mapped_column(default=RunStatus.QUEUED)           # 状态,RunStatus 编码
    input_dir: Mapped[str] = mapped_column(String)                          # 上传输入目录(绝对路径)
    output_dir: Mapped[str] = mapped_column(String)                         # 算法输出目录(绝对路径)
    result_path: Mapped[str | None] = mapped_column(String, default=None)   # 基准 P50 结果 CSV 路径
    error_message: Mapped[str | None] = mapped_column(Text, default=None)   # 失败原因(中文)
    create_time: Mapped[datetime] = mapped_column(default=datetime.now)     # 创建时刻
    update_time: Mapped[datetime] = mapped_column(default=datetime.now, onupdate=datetime.now)  # 更新时刻
