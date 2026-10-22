#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/10/21 15:40
@Author  : weiyutao
@File    : run_service.py

runs 单表 CRUD,不含跨表逻辑。
"""

import logging

from core.provider.sql_provider import SqlProvider
from core.schema.run_dto import RunCreateDTO, RunUpdateDTO
from core.table.run_record import RunRecord


class RunService:
    """runs 表的创建、查询与更新。"""

    def __init__(self) -> None:
        self.logger = logging.getLogger("RunService")
        self.model = RunRecord
        self.sql_provider = SqlProvider(model=RunRecord)

    async def create(self, dto: RunCreateDTO) -> RunRecord:
        """插入一条运行记录。

        Args:
            dto (RunCreateDTO): 创建入参。

        Returns:
            RunRecord: 持久化后的运行记录。
        """
        return await self.sql_provider.create(dto.model_dump(exclude_none=True))

    async def get(self, run_id: str) -> RunRecord | None:
        """按 run_id 查询运行记录。

        Args:
            run_id (str): 运行标识。

        Returns:
            RunRecord | None: 运行记录,不存在返回 None。
        """
        return await self.sql_provider.get_by_id(run_id)

    async def update(self, run_id: str, dto: RunUpdateDTO) -> RunRecord | None:
        """按 run_id 更新给定字段。

        Args:
            run_id (str): 运行标识。
            dto (RunUpdateDTO): 更新入参,只更新显式传入的字段。

        Returns:
            RunRecord | None: 更新后的记录,不存在返回 None。
        """
        return await self.sql_provider.update(run_id, dto.model_dump(exclude_unset=True))
