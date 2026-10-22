#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/10/21 15:40
@Author  : weiyutao
@File    : dependencies.py

依赖工厂:service/manager 进程内单例,路由经 Depends 注入。
"""

from functools import lru_cache

from core.manager.run_manager import RunManager
from core.service.run_service import RunService


@lru_cache(maxsize=1)
def get_run_service() -> RunService:
    """runs 表 service 单例。

    Returns:
        RunService: service 实例。
    """
    return RunService()


@lru_cache(maxsize=1)
def get_run_manager() -> RunManager:
    """运行组织 manager 单例。

    Returns:
        RunManager: manager 实例。
    """
    return RunManager(run_service=get_run_service())
