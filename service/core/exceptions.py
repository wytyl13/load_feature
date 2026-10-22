#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/10/21 15:40
@Author  : weiyutao
@File    : exceptions.py

业务异常类型:均为 ValueError 子类,服务端异常处理器按类型映射 404/409。
"""


class NotFoundError(ValueError):
    """查询对象不存在,映射 404。"""


class ConflictError(ValueError):
    """对象当前状态不允许该操作,映射 409。"""
