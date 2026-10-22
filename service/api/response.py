#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/10/21 15:40
@Author  : weiyutao
@File    : response.py

统一响应壳:所有 JSON 接口经 make_response 包装返回。
"""

from datetime import datetime
from typing import Any

from fastapi.encoders import jsonable_encoder


def make_response(
    success: bool = True, message: str = "", data: Any = None, code: int | None = None
) -> dict:
    """包装统一响应结构。

    Args:
        success (bool): 是否成功。
        message (str): 面向用户的说明,失败时为原因。
        data (Any): 业务数据,经 jsonable_encoder 序列化。
        code (int | None): 业务子码,失败时使用。

    Returns:
        dict: {success, message, data[, code], timestamp}。
    """
    body = {
        "success": success,
        "message": message,
        "data": jsonable_encoder(data),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    if code is not None:
        body["code"] = code
    return body
