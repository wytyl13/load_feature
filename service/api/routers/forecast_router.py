#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/10/23
@Author  : weiyutao
@File    : forecast_router.py

课题2 运行接口:创建(上传三件)/状态轮询/结果查看/结果下载;薄路由,业务在 manager。
契约与课题1 run_router 一致(job_id/英文状态名/results 只返 views)。
"""

from fastapi import APIRouter, BackgroundTasks, Depends, File, Query, UploadFile
from fastapi.responses import FileResponse

from api.dependencies import get_forecast_manager
from api.response import make_response
from core.manager.forecast_run_manager import ForecastRunManager
from core.manager.run_manager import UploadedFile

router = APIRouter(prefix="/api/load-forecast/runs", tags=["LoadForecast"])


@router.post("/create")
async def create_forecast_run(
    background_tasks: BackgroundTasks,
    macro_workbook: UploadFile = File(..., description="111.xlsx 宏观与电量(8 指标×1996—2025)"),
    industry_energy: UploadFile = File(..., description="行业电量简表(sheet'行业电量',10 行×2020—2025)"),
    industry_load: UploadFile = File(..., description="行业历史负荷(10 sheet,每 sheet [时间,值])"),
    manager: ForecastRunManager = Depends(get_forecast_manager),
):
    """【创建运行】保存三件输入并登记运行,转后台计算(目标年 2030 固定),返回 run_id。"""
    run_id = await manager.create_run(
        macro_workbook=UploadedFile(macro_workbook.filename or "", await macro_workbook.read()),
        industry_energy=UploadedFile(
            industry_energy.filename or "", await industry_energy.read()
        ),
        industry_load=UploadedFile(industry_load.filename or "", await industry_load.read()),
    )
    background_tasks.add_task(manager.execute_run, run_id)
    return make_response(data={"job_id": run_id})


@router.get("/status")
async def get_forecast_status(
    job_id: str = Query(..., description="任务标识"),
    manager: ForecastRunManager = Depends(get_forecast_manager),
):
    """【状态轮询】返回任务状态与失败原因,不返回结果数据。"""
    resp = await manager.get_status(job_id)
    return make_response(data=resp)


@router.get("/results")
async def get_forecast_results(
    job_id: str = Query(..., description="任务标识"),
    manager: ForecastRunManager = Depends(get_forecast_manager),
):
    """【结果查看】返回 2030 总负荷/行业MW/行业标幺三组 JSON 视图(供前端绘图)。"""
    views = await manager.get_result_views(job_id)
    return make_response(data=views)


@router.get("/download")
async def download_forecast_result(
    job_id: str = Query(..., description="任务标识"),
    file: str = Query("total", description="结果文件:total 总负荷曲线/industry 行业汇总宽表"),
    manager: ForecastRunManager = Depends(get_forecast_manager),
):
    """【结果下载】任务成功后下载结果 CSV(total=总负荷 MW 及标幺 8760/industry=行业标幺及 MW 汇总)。"""
    path = await manager.get_result_path(job_id, file)
    return FileResponse(path, filename=path.name)
