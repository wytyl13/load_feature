#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/10/21 15:40
@Author  : weiyutao
@File    : sql_provider.py

通用异步 CRUD 提供者:面向单个表模型的创建、按主键查、按主键更新。
"""

import logging
from typing import Any, TypeVar

from core.database import session_factory

ModelT = TypeVar("ModelT")


class SqlProvider:
    """按注入的表模型执行异步 CRUD,每方法独立会话。"""

    def __init__(self, model: type[ModelT]) -> None:
        self.logger = logging.getLogger("SqlProvider")
        self.model = model

    async def create(self, data: dict[str, Any]) -> ModelT:
        """插入一条记录并返回持久化后的实例。

        Args:
            data (dict[str, Any]): 字段名到值的映射。

        Returns:
            ModelT: 持久化后的表模型实例。
        """
        async with session_factory() as session:
            obj = self.model(**data)
            session.add(obj)
            await session.commit()
            await session.refresh(obj)
            return obj

    async def get_by_id(self, obj_id: str) -> ModelT | None:
        """按主键查询。

        Args:
            obj_id (str): 主键值。

        Returns:
            ModelT | None: 记录实例,不存在返回 None。
        """
        async with session_factory() as session:
            return await session.get(self.model, obj_id)

    async def update(self, obj_id: str, data: dict[str, Any]) -> ModelT | None:
        """按主键更新给定字段。

        Args:
            obj_id (str): 主键值。
            data (dict[str, Any]): 待更新字段到值的映射。

        Returns:
            ModelT | None: 更新后的实例,记录不存在返回 None。
        """
        async with session_factory() as session:
            obj = await session.get(self.model, obj_id)
            if obj is None:
                return None
            for key, value in data.items():
                setattr(obj, key, value)
            await session.commit()
            await session.refresh(obj)
            return obj
