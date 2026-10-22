#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/10/21 15:40
@Author  : weiyutao
@File    : run_schema.py

HTTP 请求体模型;创建接口为 multipart 表单,其字段在路由以 Form 定义,此处只放 JSON 请求体。
"""

from pydantic import BaseModel, Field


class RunIdReq(BaseModel):
    """按 job_id 操作的请求体。"""

    job_id: str = Field(..., description="任务标识")
