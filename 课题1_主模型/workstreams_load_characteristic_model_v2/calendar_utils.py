"""历史正式日历与规划期代理日历。"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

try:  # Keep compatibility with legacy scripts importing this module directly.
    from .day_types import HOLIDAY, WEEKEND, WORKDAY, canonical_day_type
except ImportError:  # pragma: no cover - exercised only by legacy top-level imports
    from day_types import HOLIDAY, WEEKEND, WORKDAY, canonical_day_type


CALENDAR_VERSION = (
    "official-2019-2026-cross-year__planning-proxy-v2-lunar-day1-eight-days"
)


def _date_range(start: str | date, end: str | date) -> set[date]:
    return set(pd.date_range(start, end, freq="D").date)


OFFICIAL_HOLIDAY_SPANS = {
    2019: [("2019-01-01", "2019-01-01"), ("2019-02-04", "2019-02-10"), ("2019-04-05", "2019-04-07"), ("2019-05-01", "2019-05-04"), ("2019-06-07", "2019-06-09"), ("2019-09-13", "2019-09-15"), ("2019-10-01", "2019-10-07")],
    2020: [("2020-01-01", "2020-01-01"), ("2020-01-24", "2020-02-02"), ("2020-04-04", "2020-04-06"), ("2020-05-01", "2020-05-05"), ("2020-06-25", "2020-06-27"), ("2020-10-01", "2020-10-08")],
    2021: [("2021-01-01", "2021-01-03"), ("2021-02-11", "2021-02-17"), ("2021-04-03", "2021-04-05"), ("2021-05-01", "2021-05-05"), ("2021-06-12", "2021-06-14"), ("2021-09-19", "2021-09-21"), ("2021-10-01", "2021-10-07")],
    2022: [("2022-01-01", "2022-01-03"), ("2022-01-31", "2022-02-06"), ("2022-04-03", "2022-04-05"), ("2022-04-30", "2022-05-04"), ("2022-06-03", "2022-06-05"), ("2022-09-10", "2022-09-12"), ("2022-10-01", "2022-10-07")],
    2023: [("2022-12-31", "2023-01-02"), ("2023-01-21", "2023-01-27"), ("2023-04-05", "2023-04-05"), ("2023-04-29", "2023-05-03"), ("2023-06-22", "2023-06-24"), ("2023-09-29", "2023-10-06")],
    2025: [("2025-01-01", "2025-01-01"), ("2025-01-28", "2025-02-04"), ("2025-04-04", "2025-04-06"), ("2025-05-01", "2025-05-05"), ("2025-05-31", "2025-06-02"), ("2025-10-01", "2025-10-08")],
    2026: [("2026-01-01", "2026-01-03"), ("2026-02-15", "2026-02-23"), ("2026-04-04", "2026-04-06"), ("2026-05-01", "2026-05-05"), ("2026-06-19", "2026-06-21"), ("2026-09-25", "2026-09-27"), ("2026-10-01", "2026-10-07")],
}

OFFICIAL_ADJUSTED_WORKDAYS = {
    2019: ["2019-02-02", "2019-02-03", "2019-04-28", "2019-05-05", "2019-09-29", "2019-10-12"],
    2020: ["2020-01-19", "2020-04-26", "2020-05-09", "2020-06-28", "2020-09-27", "2020-10-10"],
    2021: ["2021-02-07", "2021-02-20", "2021-04-25", "2021-05-08", "2021-09-18", "2021-09-26", "2021-10-09"],
    2022: ["2022-01-29", "2022-01-30", "2022-04-02", "2022-04-24", "2022-05-07", "2022-10-08", "2022-10-09"],
    2023: ["2023-01-28", "2023-01-29", "2023-04-23", "2023-05-06", "2023-06-25", "2023-10-07", "2023-10-08"],
    2025: ["2025-01-26", "2025-02-08", "2025-04-27", "2025-09-28", "2025-10-11"],
    2026: ["2026-01-04", "2026-02-14", "2026-02-28", "2026-05-09", "2026-09-20", "2026-10-10"],
}

PLANNING_LUNAR_FESTIVALS = {
    2027:("2027-02-07","2027-06-09","2027-09-15"), 2028:("2028-01-26","2028-05-28","2028-10-03"),
    2029:("2029-02-13","2029-06-16","2029-09-22"), 2030:("2030-02-02","2030-06-05","2030-09-12"),
    2031:("2031-01-23","2031-06-24","2031-10-01"), 2032:("2032-02-11","2032-06-12","2032-09-19"),
    2033:("2033-01-31","2033-06-01","2033-09-08"), 2034:("2034-02-19","2034-06-20","2034-09-27"),
    2035:("2035-02-08","2035-06-10","2035-09-16"), 2036:("2036-01-28","2036-05-30","2036-10-04"),
    2037:("2037-02-15","2037-06-18","2037-09-24"), 2038:("2038-02-04","2038-06-07","2038-09-13"),
    2039:("2039-01-24","2039-05-27","2039-10-02"), 2040:("2040-02-12","2040-06-14","2040-09-20"),
    2041:("2041-02-01","2041-06-03","2041-09-10"), 2042:("2042-01-22","2042-06-22","2042-09-28"),
    2043:("2043-02-10","2043-06-11","2043-09-17"), 2044:("2044-01-30","2044-05-31","2044-10-05"),
    2045:("2045-02-17","2045-06-19","2045-09-25"), 2046:("2046-02-06","2046-06-08","2046-09-15"),
    2047:("2047-01-26","2047-05-29","2047-10-04"), 2048:("2048-02-14","2048-06-15","2048-09-22"),
    2049:("2049-02-02","2049-06-04","2049-09-11"), 2050:("2050-01-23","2050-06-23","2050-09-30"),
    2051:("2051-02-11","2051-06-13","2051-09-19"), 2052:("2052-02-01","2052-06-01","2052-09-07"),
    2053:("2053-02-19","2053-06-20","2053-09-26"), 2054:("2054-02-08","2054-06-10","2054-09-16"),
    2055:("2055-01-28","2055-05-30","2055-10-05"), 2056:("2056-02-15","2056-06-17","2056-09-24"),
    2057:("2057-02-04","2057-06-06","2057-09-13"), 2058:("2058-01-24","2058-06-25","2058-10-02"),
    2059:("2059-02-12","2059-06-14","2059-09-21"), 2060:("2060-02-02","2060-06-03","2060-09-09"),
}

_OFFICIAL_HOLIDAYS = {
    year: set().union(*(_date_range(a, b) for a, b in spans))
    for year, spans in OFFICIAL_HOLIDAY_SPANS.items()
}
_OFFICIAL_ADJUSTED = {
    year: {pd.Timestamp(value).date() for value in values}
    for year, values in OFFICIAL_ADJUSTED_WORKDAYS.items()
}
# Holiday spans may cross a Gregorian year boundary (for example the 2023
# New-Year arrangement starts on 2022-12-31).  Classification must therefore
# be keyed by the actual date, not only by the dictionary bucket that stores
# the published arrangement.
_OFFICIAL_HOLIDAY_DATES = set().union(*_OFFICIAL_HOLIDAYS.values())
_OFFICIAL_ADJUSTED_DATES = set().union(*_OFFICIAL_ADJUSTED.values())


def season_of(month: int) -> str:
    return {12: "冬季", 1: "冬季", 2: "冬季", 3: "春季", 4: "春季", 5: "春季", 6: "夏季", 7: "夏季", 8: "夏季", 9: "秋季", 10: "秋季", 11: "秋季"}[int(month)]


def future_calendar_proxy(year: int) -> tuple[set[date], set[date]]:
    """确定性规划代理，不宣称为国务院未来安排。"""
    if year not in PLANNING_LUNAR_FESTIVALS:
        raise ValueError(f"规划期农历节日映射缺失：{year}")
    lunar_new_year, dragon, mid_autumn = [pd.Timestamp(v).date() for v in PLANNING_LUNAR_FESTIVALS[year]]
    # 规划期春节代理固定从农历正月初一开始，连续8天。
    spring_start = lunar_new_year
    spans = [
        (date(year, 1, 1), date(year, 1, 3)),
        (spring_start, spring_start + timedelta(days=7)),
        (date(year, 4, 4), date(year, 4, 6)),
        (date(year, 5, 1), date(year, 5, 5)),
        (dragon, dragon + timedelta(days=2)),
        (mid_autumn, mid_autumn + timedelta(days=2)),
        (date(year, 10, 1), date(year, 10, 7)),
    ]
    holidays = set().union(*(_date_range(a, b) for a, b in spans))
    adjusted: set[date] = set()
    for start, end in spans:
        for candidate in (start - timedelta(days=1), end + timedelta(days=1)):
            if candidate.year == year and candidate.weekday() >= 5 and candidate not in holidays:
                adjusted.add(candidate)
    return holidays, adjusted


def classify_date(value: date | pd.Timestamp) -> tuple[str, str, str]:
    d = pd.Timestamp(value).date()
    if d in _OFFICIAL_HOLIDAY_DATES:
        holidays, adjusted, source = (
            _OFFICIAL_HOLIDAY_DATES,
            _OFFICIAL_ADJUSTED_DATES,
            "已公布官方日历",
        )
    elif d in _OFFICIAL_ADJUSTED_DATES or d.year in _OFFICIAL_HOLIDAYS:
        holidays, adjusted, source = (
            _OFFICIAL_HOLIDAY_DATES,
            _OFFICIAL_ADJUSTED_DATES,
            "已公布官方日历",
        )
    else:
        holidays, adjusted = future_calendar_proxy(d.year)
        source = "规划日历代理"
    if d in holidays:
        detailed = HOLIDAY
    elif d in adjusted:
        detailed = "调休工作日"
    elif d.weekday() >= 5:
        detailed = WEEKEND
    else:
        detailed = "普通工作日"
    group = WORKDAY if detailed in {"普通工作日", "调休工作日"} else "非工作日"
    return detailed, group, source


def calendar_frame(year: int) -> pd.DataFrame:
    dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    rows = []
    for stamp in dates:
        detailed, group, source = classify_date(stamp)
        rows.append(
            {
                "date": stamp.normalize(),
                "year": year,
                "month": stamp.month,
                "season": season_of(stamp.month),
                "day_type": detailed,
                "canonical_day_type": canonical_day_type(detailed),
                "day_group": group,
                "calendar_source": source,
                "calendar_version": CALENDAR_VERSION,
            }
        )
    return pd.DataFrame(rows)
