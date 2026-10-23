#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/10/23
@Author  : weiyutao
@File    : compare_forecast_run.py

课题2 软件集成版(20261022)重构输出比对工具(DESIGN D18 归一化口径)。

用法:
    python compare_forecast_run.py make <run_dir> <清单文件> [--only <段目录>]
    python compare_forecast_run.py compare <run_dir> <清单文件> [--only <段目录>]

口径(D18,类比 D14):
- 根 = 一次运行的输出目录 outputs/<run>;
- csv/json/log 逐位一致,先做路径归一化再 md5:
  run_dir 绝对路径(正/反斜杠/JSON 转义形态)→ @RUN@;
  审计内 111.xlsx 源路径(01_输入数据与配置/宏观与电量/111.xlsx 各形态)→ 111.xlsx;
  审计内生成器路径(05_行业标幺8760生成/weekly_generator.py 各形态)→ weekly_generator.py;
- run_manifest.json 不入 md5(含 started_at/finished_at/elapsed_seconds/python 路径),
  改结构断言(status=passed、5 段名序、逐段 return_code=0;仅全量模式执行);
- svg 不入 md5(衍生图),仅核对总数;
- --only <段目录> 分段比对:只取该前缀的判据文件,pipeline.log 与 manifest 断言跳过。
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

STAGE_DIRS = [
    "02_宏观特征预测",
    "03_总电量与行业电量预测",
    "04_历史负荷预处理",
    "05_行业标幺8760生成",
    "06_有名值换算与总负荷合成",
]
TEXT_SUFFIXES = {".csv", ".json", ".log"}
FIG_SUFFIXES = {".svg"}
# 审计 json 内嵌的相对路径(正/反斜杠/JSON 转义形态)→ 逻辑名,消除输入布局差异
PATH_ALIASES = [
    ("01_输入数据与配置/宏观与电量/111.xlsx", "111.xlsx"),
    ("01_输入数据与配置\\宏观与电量\\111.xlsx", "111.xlsx"),
    ("01_输入数据与配置\\\\宏观与电量\\\\111.xlsx", "111.xlsx"),
    ("05_行业标幺8760生成/weekly_generator.py", "weekly_generator.py"),
    ("05_行业标幺8760生成\\weekly_generator.py", "weekly_generator.py"),
    ("05_行业标幺8760生成\\\\weekly_generator.py", "weekly_generator.py"),
]


def normalize(text: str, run_dir: Path) -> str:
    """run_dir 各路径形态→@RUN@;固定相对路径→逻辑名。"""
    for rep in (str(run_dir), str(run_dir).replace("\\", "/"), str(run_dir).replace("\\", "\\\\")):
        if rep:
            text = text.replace(rep, "@RUN@")
    for old, new in PATH_ALIASES:
        text = text.replace(old, new)
    return text


def text_digest(path: Path, run_dir: Path) -> str:
    raw = path.read_bytes().decode("utf-8-sig")
    return hashlib.md5(normalize(raw, run_dir).encode("utf-8")).hexdigest()


def collect(run_dir: Path, only: str | None) -> tuple[dict[str, str], int]:
    """扫描受判输出,返回 {相对路径: 摘要} 与 svg 计数。"""
    digests: dict[str, str] = {}
    fig_count = 0
    stages = [only] if only else STAGE_DIRS
    for rel_dir in stages:
        base = run_dir / rel_dir
        if not base.is_dir():
            raise SystemExit(f"缺少输出目录: {base}")
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(run_dir).as_posix()
            suffix = path.suffix.lower()
            if suffix in FIG_SUFFIXES:
                fig_count += 1
            elif suffix in TEXT_SUFFIXES:
                digests[rel] = text_digest(path, run_dir)
            else:
                raise SystemExit(f"未预期输出类型,需明确口径: {rel}")
    if not only:
        digests["pipeline.log"] = text_digest(run_dir / "pipeline.log", run_dir)
    return digests, fig_count


def assert_manifest(run_dir: Path) -> None:
    """run_manifest.json 结构断言(内容不入 md5 判据)。"""
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "passed":
        raise SystemExit(f"run_manifest status={manifest.get('status')} != passed")
    stages = manifest.get("stages", [])
    names = [item["name"] for item in stages]
    if names != STAGE_DIRS:
        raise SystemExit(f"run_manifest 段名序 {names} 与预期不符")
    bad = [item["name"] for item in stages if item.get("return_code") != 0]
    if bad:
        raise SystemExit(f"run_manifest 存在非零退出段: {bad}")


def make_manifest(run_dir: Path, out: Path, only: str | None) -> None:
    digests, fig_count = collect(run_dir, only)
    scope = f";段={only}" if only else ""
    lines = [f"# 课题2 软件集成版全链输出基准(D18 归一化口径{scope})",
             f"# FIGCOUNT {fig_count}"]
    lines += [f"{md5} *{rel}" for rel, md5 in sorted(digests.items())]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"清单: {out}  判据文件 {len(digests)}  svg {fig_count}")


def compare_manifest(run_dir: Path, baseline: Path, only: str | None) -> int:
    digests, fig_count = collect(run_dir, only)
    if not only:
        assert_manifest(run_dir)
    base_lines = [ln for ln in baseline.read_text(encoding="utf-8").splitlines() if ln.strip()]
    base_fig = int(next(ln.split()[2] for ln in base_lines if ln.startswith("# FIGCOUNT")))
    base = dict(ln.split(" *", 1)[::-1] for ln in base_lines if not ln.startswith("#"))
    if only:
        base = {rel: md5 for rel, md5 in base.items() if rel.startswith(only + "/")}
        base_fig_expected = None
    else:
        base_fig_expected = base_fig

    missing = sorted(set(base) - set(digests))
    extra = sorted(set(digests) - set(base))
    diff = sorted(rel for rel in set(base) & set(digests) if base[rel] != digests[rel])
    for rel in missing:
        print(f"MISSING {rel}")
    for rel in extra:
        print(f"EXTRA   {rel}")
    for rel in diff:
        print(f"DIFF    {rel}\n  基准 {base[rel]}\n  实际 {digests[rel]}")
    ok = not (missing or extra or diff)
    if base_fig_expected is not None and fig_count != base_fig_expected:
        print(f"FIGCOUNT 基准 {base_fig_expected} != 实际 {fig_count}")
        ok = False
    print(f"比对: {'通过,零差异' if ok else '不通过'} "
          f"(判据 {len(base)} 文件,差异 {len(diff)},缺 {len(missing)},多 {len(extra)},svg {fig_count})")
    return 0 if ok else 1


def main() -> None:
    args = sys.argv[1:]
    only = None
    if "--only" in args:
        index = args.index("--only")
        only = args[index + 1]
        del args[index:index + 2]
    if len(args) != 3 or args[0] not in ("make", "compare"):
        raise SystemExit(__doc__)
    mode, run_dir, target = args[0], Path(args[1]).resolve(), Path(args[2])
    if mode == "make":
        make_manifest(run_dir, target, only)
    else:
        raise SystemExit(compare_manifest(run_dir, target, only))


if __name__ == "__main__":
    main()
