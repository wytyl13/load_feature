#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/10/21 15:40
@Author  : weiyutao
@File    : load_model_server.py

FastAPI app 装配:lifespan 建表、跨域、请求日志中间件、服务状态端点、全局异常、路由注册。
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.response import make_response
from api.routers.forecast_router import router as forecast_router
from api.routers.run_router import router as run_router
from core.database import dispose_engine, init_db
from core.exceptions import ConflictError, NotFoundError

logger = logging.getLogger("load_model_service")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时初始化数据库,停机时关闭连接池。"""
    await init_db()
    logger.info("数据库初始化完成")
    yield
    await dispose_engine()


app = FastAPI(title="负荷特性主模型服务", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_credentials=True,
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """请求日志:方法、路径、状态码与耗时。"""
    start = time.perf_counter()
    response = await call_next(request)
    logger.info(
        "%s %s -> %s (%.0f ms)",
        request.method,
        request.url.path,
        response.status_code,
        (time.perf_counter() - start) * 1000,
    )
    return response


@app.exception_handler(NotFoundError)
async def not_found_handler(request: Request, exc: NotFoundError) -> JSONResponse:
    """查询对象不存在 → 404。"""
    return JSONResponse(status_code=404, content=make_response(success=False, message=str(exc)))


@app.exception_handler(ConflictError)
async def conflict_handler(request: Request, exc: ConflictError) -> JSONResponse:
    """状态不允许该操作 → 409。"""
    return JSONResponse(status_code=409, content=make_response(success=False, message=str(exc)))


@app.exception_handler(ValueError)
async def bad_request_handler(request: Request, exc: ValueError) -> JSONResponse:
    """参数/数据类错误 → 400。"""
    return JSONResponse(status_code=400, content=make_response(success=False, message=str(exc)))


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """请求体校验失败 → 400,报指明字段。"""
    return JSONResponse(
        status_code=400,
        content=make_response(success=False, message=f"请求参数校验失败:{exc.errors()[:3]}"),
    )


@app.exception_handler(Exception)
async def global_handler(request: Request, exc: Exception) -> JSONResponse:
    """未捕获异常 → 500;堆栈只进日志。"""
    logger.error("未捕获异常 %s %s: %s", request.method, request.url.path, exc, exc_info=exc)
    return JSONResponse(
        status_code=500,
        content=make_response(success=False, message=f"Internal Server Error: {exc}"),
    )


@app.get("/api/health")
async def health():
    """【服务状态】服务存活探测。"""
    return make_response(data={"status": "ok", "message": "服务正常"})


app.include_router(run_router)
app.include_router(forecast_router)
