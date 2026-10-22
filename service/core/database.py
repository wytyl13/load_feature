#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/10/21 15:40
@Author  : weiyutao
@File    : database.py

异步数据库引擎与会话工厂(SQLite + aiosqlite),进程内单例;建表在服务启动时执行一次。
"""

from core.paths import DB_PATH
from core.table.run_record import Base
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

engine = create_async_engine(f"sqlite+aiosqlite:///{DB_PATH}")
session_factory = async_sessionmaker(engine, expire_on_commit=False)


async def init_db() -> None:
    """建运行目录与全部表;幂等,重复调用无副作用。

    Returns: None
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def dispose_engine() -> None:
    """关闭连接池,服务停机时调用。

    Returns: None
    """
    await engine.dispose()
