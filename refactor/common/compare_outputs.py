#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/10/20 13:30
@Author  : weiyutao
@File    : compare_outputs.py

课题2 重构输出比对工具(DESIGN D14 归一化口径)。

用法:
    python compare_outputs.py make <包根> <清单文件>     生成归一化 md5 清单
    python compare_outputs.py compare <包根> <清单文件>  同口径重算并与清单比对

口径(D14):
- csv/json/txt 逐位一致,先做路径归一化再 md5:包根绝对路径(正/反斜杠)、
  refactor/load_forecast(正/反斜杠)、课题2_预测 → @PKG@;
- xlsx 不比字节(工作簿内嵌时间戳),按单元格值哈希(openpyxl 逐 sheet 逐 cell,
  字符串单元格同样做路径归一化);
- png/svg 不入 md5 判据(衍生图,svg 内嵌生成日期),仅核对总数;
- 主链外产物不入清单:预测结果下两份对照 xlsx(增速/传统回归,主链外脚本产物)、
  验证结果_原始数据 下作者日志 txt。
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

# 16 段主链会再生成的输出目录;03_气温敏感性 目录内混有输入,只取指定前缀的输出文件
GATED_DIRS = [
    "预测结果_2020-2030",
    "03_气温敏感性",
    "行业标幺8760曲线",
    "2030行业8760标幺曲线_三假前当后周",
    "2030行业有名值曲线",
    "扣除重点行业剩余标幺曲线",
    "10_扩展行业/验证结果_原始数据",
]
TEMP_OUTPUT_PREFIXES = ("气温基温判定", "气温基温扫描", "气温弹性")
# 交付包内残留的旧版审计产物目录(标签无 _hybrid/_pm1 后缀,现行代码不再生成),不入判据
SKIP_DIR_SEGMENTS = {"充换电服务业_2022+2023", "充换电服务业_仅2023",
                     "数据中心_2022+2023", "数据中心_仅2023"}
SKIP_FILES = {
    "预测结果_2020-2030/ARIMA增速预测_2026-2035.xlsx",      # 主链外脚本产物
    "预测结果_2020-2030/ARIMA预测2026-2030_传统回归.xlsx",  # 主链外脚本产物
}
TEXT_SUFFIXES = {".csv", ".json", ".txt"}
FIG_SUFFIXES = {".png", ".svg"}


def normalize(text: str, root: Path) -> str:
    """把包根的各种路径形态替换为占位符,消除重构目录名差异。

    含 JSON 转义形态(双反斜杠),主预测运行状态.json 等输出内嵌绝对路径。
    """
    for rep in (str(root), str(root).replace("\\", "/"), str(root).replace("\\", "\\\\"),
                "refactor/load_forecast", "refactor\\load_forecast", "refactor\\\\load_forecast",
                "课题2_预测"):
        if rep:
            text = text.replace(rep, "@PKG@")
    return text


def text_digest(path: Path, root: Path) -> str:
    raw = path.read_bytes().decode("utf-8-sig")
    return hashlib.md5(normalize(raw, root).encode("utf-8")).hexdigest()


def xlsx_digest(path: Path, root: Path) -> str:
    """按单元格值哈希 xlsx;时间戳/样式不参与。"""
    from openpyxl import load_workbook

    wb = load_workbook(path, read_only=True, data_only=True)
    parts = []
    for ws in wb.worksheets:
        cells = []
        for row in ws.iter_rows():
            for cell in row:
                if cell.value is not None:
                    value = normalize(str(cell.value), root) if isinstance(cell.value, str) else cell.value
                    cells.append(f"{cell.coordinate}={value!r}")
        parts.append(ws.title + "|" + ";".join(cells))
    wb.close()
    return hashlib.md5("\n".join(parts).encode("utf-8")).hexdigest()


def collect(root: Path) -> tuple[dict[str, str], int]:
    """扫描受判输出,返回 {相对路径: 摘要或 FIG} 与图片计数。"""
    digests: dict[str, str] = {}
    fig_count = 0
    for rel_dir in GATED_DIRS:
        base = root / rel_dir
        if not base.is_dir():
            raise SystemExit(f"缺少输出目录: {base}")
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(root).as_posix()
            if rel in SKIP_FILES:
                continue
            if any(part in SKIP_DIR_SEGMENTS for part in path.parts):
                continue
            suffix = path.suffix.lower()
            if rel_dir == "03_气温敏感性":
                if not path.name.startswith(TEMP_OUTPUT_PREFIXES):
                    continue  # 该目录其余文件是输入
            elif rel_dir == "10_扩展行业/验证结果_原始数据" and suffix == ".txt":
                continue  # 作者日志,非链路产物
            if suffix in FIG_SUFFIXES:
                fig_count += 1
            elif suffix in TEXT_SUFFIXES:
                digests[rel] = text_digest(path, root)
            elif suffix == ".xlsx":
                digests[rel] = "VALUES:" + xlsx_digest(path, root)
            else:
                raise SystemExit(f"未预期输出类型,需明确口径: {rel}")
    return digests, fig_count


def make_manifest(root: Path, out: Path) -> None:
    digests, fig_count = collect(root)
    lines = [f"# 课题2 全链输出基准(D14 归一化口径;包根={root.name})",
             f"# FIGCOUNT {fig_count}"]
    lines += [f"{md5} *{rel}" for rel, md5 in sorted(digests.items())]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"清单: {out}  判据文件 {len(digests)}  图片 {fig_count}")


def compare_manifest(root: Path, baseline: Path) -> int:
    digests, fig_count = collect(root)
    base_lines = [ln for ln in baseline.read_text(encoding="utf-8").splitlines() if ln.strip()]
    base_fig = int(next(ln.split()[2] for ln in base_lines if ln.startswith("# FIGCOUNT")))
    base = dict(ln.split(" *", 1)[::-1] for ln in base_lines if not ln.startswith("#"))

    missing = sorted(set(base) - set(digests))
    extra = sorted(set(digests) - set(base))
    diff = sorted(rel for rel in set(base) & set(digests) if base[rel] != digests[rel])
    for rel in missing:
        print(f"MISSING {rel}")
    for rel in extra:
        print(f"EXTRA   {rel}")
    for rel in diff:
        print(f"DIFF    {rel}\n  基准 {base[rel]}\n  实际 {digests[rel]}")
    if fig_count != base_fig:
        print(f"FIGCOUNT 基准 {base_fig} != 实际 {fig_count}")
    ok = not (missing or extra or diff) and fig_count == base_fig
    print(f"比对: {'通过,零差异' if ok else '不通过'} "
          f"(判据 {len(base)} 文件,差异 {len(diff)},缺 {len(missing)},多 {len(extra)},图 {fig_count}/{base_fig})")
    return 0 if ok else 1


def main() -> None:
    if len(sys.argv) != 4 or sys.argv[1] not in ("make", "compare"):
        raise SystemExit(__doc__)
    mode, root, target = sys.argv[1], Path(sys.argv[2]).resolve(), Path(sys.argv[3])
    if mode == "make":
        make_manifest(root, target)
    else:
        raise SystemExit(compare_manifest(root, target))


if __name__ == "__main__":
    main()
