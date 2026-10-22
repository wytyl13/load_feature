#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/10/21 15:40
@Author  : weiyutao
@File    : run_dto.py

runs 表数据层 DTO:创建、更新、状态响应三类;无筛选接口,Filter 不建。
"""

from pydantic import BaseModel, Field


class RunCreateDTO(BaseModel):
    """创建运行记录的入参。"""

    run_id: str = Field(..., description="运行标识(uuid4 hex)")
    params: dict = Field(..., description="运行参数(起始年/结束年/抽样数/原始文件名)")
    input_dir: str = Field(..., description="上传输入目录绝对路径")
    output_dir: str = Field(..., description="算法输出目录绝对路径")


class RunUpdateDTO(BaseModel):
    """更新运行记录的入参,全可选。"""

    status: int | None = Field(None, description="状态,RunStatus 编码")
    result_path: str | None = Field(None, description="结果 CSV 路径")
    error_message: str | None = Field(None, description="失败原因")


class RunStatusResp(BaseModel):
    """状态轮询响应结构(键名对齐 regional_power_system_optimization 的 jobs 契约)。"""

    job_id: str = Field(..., description="任务标识")
    status: str = Field(..., description="状态(queued 排队/running 运行中/succeeded 成功/failed 失败)")
    error: str | None = Field(None, description="失败原因,非失败态为 null")
