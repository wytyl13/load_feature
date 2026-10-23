#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/10/21 15:40
@Author  : weiyutao
@File    : run_router.py

运行接口:创建(上传)/状态轮询/结果查看/结果下载;薄路由,业务在 manager。
"""

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, Query, UploadFile
from fastapi.responses import FileResponse

from api.dependencies import get_run_manager
from api.response import make_response
from core.manager.run_manager import RunManager, UploadedFile

router = APIRouter(prefix="/api/load-model/runs", tags=["LoadModel"])


@router.post("/create")
async def create_run(
    background_tasks: BackgroundTasks,
    load_workbook: UploadFile = File(..., description="用户日曲线工作簿,必含全省历史曲线 sheet"),
    official_workbook: UploadFile = File(..., description="全社会用电分类表(行业×年)"),
    annual_energy: UploadFile | None = File(None, description="年度电量模板,不传用内置默认"),
    start_year: int = Form(2025, description="规划起始年,2025—2060"),
    end_year: int = Form(2030, description="规划结束年,2025—2060"),
    sample_size: int = Form(20, description="Bootstrap 抽样数,20 快速/1000 正式"),
    scenario: str = Form("基准", description="规划情景:低碳/基准/高电气化,单算该情景"),
    manager: RunManager = Depends(get_run_manager),
):
    """【创建运行】保存上传文件并登记运行,转后台计算,返回 run_id。"""
    annual_file = (
        UploadedFile(annual_energy.filename or "", await annual_energy.read())
        if annual_energy
        else None
    )
    run_id = await manager.create_run(
        load_workbook=UploadedFile(load_workbook.filename or "", await load_workbook.read()),
        official_workbook=UploadedFile(
            official_workbook.filename or "", await official_workbook.read()
        ),
        annual_energy=annual_file,
        params={
            "start_year": start_year,
            "end_year": end_year,
            "sample_size": sample_size,
            "scenario": scenario,
        },
    )
    background_tasks.add_task(manager.execute_run, run_id)
    return make_response(data={"job_id": run_id})


@router.get("/status")
async def get_run_status(
    job_id: str | None = Query(None, description="任务标识;不传则返回最近 5 条历史任务状态"),
    manager: RunManager = Depends(get_run_manager),
):
    """【状态轮询】传 job_id 返回单任务状态与失败原因;不传返回最近 5 条历史任务状态列表。"""
    if job_id is None:
        return make_response(data=await manager.get_recent_status())
    resp = await manager.get_status(job_id)
    return make_response(data=resp)


@router.get("/results")
async def get_run_results(
    job_id: str = Query(..., description="任务标识"),
    manager: RunManager = Depends(get_run_manager),
):
    """【结果查看】返回指定情景 P50 曲线 JSON 视图(按年分组,供前端绘图)。"""
    views = await manager.get_result_views(job_id)
    return make_response(data=views)


@router.get("/download")
async def download_result(
    job_id: str = Query(..., description="任务标识"),
    manager: RunManager = Depends(get_run_manager),
):
    """【结果下载】任务成功后下载指定情景 P50 逐时标幺曲线 CSV(timestamp, load_pu)。"""
    path = await manager.get_result_path(job_id)
    return FileResponse(path, filename=path.name)
