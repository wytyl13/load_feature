"""Deprecated compatibility module for the former workbook payload builder.

Current workbook sources are produced exclusively by
``prepare_workbook_sources.py`` and consumed by ``build_workbook.mjs``.
No result tables are transformed or written here.
"""

from __future__ import annotations

DEPRECATION_MESSAGE = (
    "workbook_payload已停用；请使用prepare_workbook_sources.py生成经校验的CSV与manifest，"
    "再使用build_workbook.mjs构建2030展示工作簿。"
)


def main() -> None:
    raise SystemExit(DEPRECATION_MESSAGE)


__all__: list[str] = []


if __name__ == "__main__":
    main()
