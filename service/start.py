#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/10/21 15:40
@Author  : weiyutao
@File    : start.py

服务启动脚本:argparse 解析监听参数后 uvicorn 启动。
在 service/ 目录下执行;算法包 load-model-algo 须已安装到 conda load_feature 环境。
"""

import argparse
import logging

import uvicorn

from api.server.load_model_server import app


def main() -> None:
    """解析监听参数并启动 uvicorn。"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="负荷特性主模型服务")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址,默认 127.0.0.1")
    parser.add_argument("--port", type=int, default=8000, help="监听端口,默认 8000")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
