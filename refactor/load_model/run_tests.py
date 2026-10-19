"""从任意工作目录运行交付测试，自动启用随包真实数据测试。"""
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
environment = os.environ.copy()
environment['V2_TEST_LOAD_WORKBOOK'] = str(ROOT / 'inputs/副本山东负荷特性8760数据表.xlsx')
environment['V2_TEST_OFFICIAL_WORKBOOK'] = str(ROOT / 'inputs/山东2020-2025_全社会用电分类表（全口径）.xlsx')
raise SystemExit(subprocess.call(
    [sys.executable, '-m', 'pytest', '-c', str(ROOT / 'pytest.ini'), str(ROOT / 'tests'), '-q', *sys.argv[1:]],
    cwd=ROOT, env=environment))
