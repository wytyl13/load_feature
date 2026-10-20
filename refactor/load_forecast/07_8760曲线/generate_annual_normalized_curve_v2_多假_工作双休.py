#!/usr/bin/env python3
"""Generate an annual hourly normalized load curve from historical hourly load.

原始谱系 A 的周→日→小时生成器：先提取历史周峰和工作日/双休日典型日，
再对春节、五一、十一做前一周/当周/后一周修正，最后拼接目标年 8760 点并按年峰值标幺化。

v2 changes (2026-09-10, based on generate_annual_normalized_curve.py):
1. Holiday mechanism generalized from Spring Festival only to
   春节(spring) + 五一(may) + 十一(national).  Each holiday uses the SAME
   前1周/当周/后1周 three-week treatment as the original script:
   - the three weeks are interpolated out of the multi-year baseline
     (linear ramp between the nearest normal week on each side),
   - a per-holiday impact factor (raw week peak / baseline, averaged over
     history years) is applied to the target year's three weeks.
   --holiday-pattern core limits each holiday to its anchor week only;
   --holiday-pattern span selects the weeks actually overlapping the official
   holiday date span (五一 1-2 weeks, 十一 2 weeks) so the trough hugs
   May 1 / Oct 1, with each target week's factor sourced from the history
   special weeks having the closest holiday-day count.
2. Typical-day hourly templates are now split into 工作日(Mon-Fri) and
   双休日(Sat-Sun): every week of the target year has its own pair of
   24h curves.  For this (weekly peak-day) variant the 工作日 curve is the
   week's highest-peak weekday and the 双休日 curve the week's highest-peak
   weekend day (per typical year, averaged and re-normalized; a partial
   week missing one group falls back to the other group's curve).
3. The intra-week 7-point day-peak layer is unchanged (single curve per
   week, holidays use actual historical holiday-week shapes).

Method:
1. Split each year into calendar weeks, Monday-Sunday, clipped by year boundary.
2. Remove Feb 29 from leap years.
3. Build a no-holiday weekly peak baseline by interpolating all holiday
   special weeks from their neighboring normal weeks.
4. Estimate per-holiday impact factors from actual week peak / baseline.
5. Generate the target year:
   annual_hour_norm = week_peak_norm * day_peak_norm * typical_day_hour_norm,
   where typical_day_hour_norm is chosen per day: 工作日 or 双休日 curve.

Input: 历史负荷数据{year}.csv (columns 时间,值) inside --input-dir.
Default output: <input_dir>/生成结果/{target}精简结果_多假_工作双休
"""

from __future__ import annotations

import argparse
import csv
import math
from datetime import date, datetime, timedelta
from pathlib import Path


DEFAULT_INPUT_DIR = Path(__file__).resolve().parents[1] / "06_生成器输入" / "总负荷去掉九个行业负荷"

SPRING_FESTIVAL_DATES = {
    2020: date(2020, 1, 25),
    2021: date(2021, 2, 12),
    2022: date(2022, 2, 1),
    2023: date(2023, 1, 22),
    2024: date(2024, 2, 10),
    2025: date(2025, 1, 29),
    2026: date(2026, 2, 17),
    2027: date(2027, 2, 6),
    2028: date(2028, 1, 26),
    2029: date(2029, 2, 13),
    2030: date(2030, 2, 3),
    2031: date(2031, 1, 23),
    2032: date(2032, 2, 11),
    2033: date(2033, 1, 31),
    2034: date(2034, 2, 19),
    2035: date(2035, 2, 8),
    2036: date(2036, 1, 28),
    2037: date(2037, 2, 15),
    2038: date(2038, 2, 4),
    2039: date(2039, 1, 24),
    2040: date(2040, 2, 12),
    2041: date(2041, 2, 1),
    2042: date(2042, 1, 22),
    2043: date(2043, 2, 10),
    2044: date(2044, 1, 30),
    2045: date(2045, 2, 17),
    2046: date(2046, 2, 6),
    2047: date(2047, 1, 26),
    2048: date(2048, 2, 14),
    2049: date(2049, 2, 2),
    2050: date(2050, 1, 23),
    2051: date(2051, 2, 11),
    2052: date(2052, 2, 1),
    2053: date(2053, 2, 19),
}

HOLIDAY_LABELS = {"spring": "春节", "may": "五一", "national": "十一"}


def holiday_anchor_date(name: str, year: int) -> date:
    if name == "spring":
        return SPRING_FESTIVAL_DATES[year]
    if name == "may":
        return date(year, 5, 1)
    if name == "national":
        return date(year, 10, 1)
    raise KeyError(f"Unknown holiday: {name}")


def holiday_week_index(name: str, year: int, weeks: list[list[date]]) -> int:
    anchor = holiday_anchor_date(name, year)
    for index, week in enumerate(weeks, 1):
        if anchor in week:
            return index
    raise ValueError(f"{name} anchor date is not in calendar: {year}")


PATTERN_RELS = {"pm1": (-1, 0, 1), "pm0": (-1, 0), "p01": (0, 1), "core": (0,)}
HYBRID_RELS = {"spring": (-1, 0, 1), "may": (0, 1), "national": (0, 1)}


def holiday_rels(pattern: str, name: str = "spring") -> tuple[int, ...]:
    if pattern == "hybrid":
        return HYBRID_RELS[name]
    return PATTERN_RELS[pattern]


def special_weeks_for(anchor_week: int, week_count: int, pattern: str, name: str = "spring") -> set[int]:
    rels = holiday_rels(pattern, name)
    return {w for w in (anchor_week + rel for rel in rels) if 1 <= w <= week_count}


OFFICIAL_HOLIDAY_SPANS: dict[int, dict[str, tuple[date, date]]] = {
    2020: {"may": (date(2020, 5, 1), date(2020, 5, 5)), "national": (date(2020, 10, 1), date(2020, 10, 8))},
    2022: {"may": (date(2022, 4, 30), date(2022, 5, 4)), "national": (date(2022, 10, 1), date(2022, 10, 7))},
    2023: {"may": (date(2023, 4, 29), date(2023, 5, 3)), "national": (date(2023, 9, 29), date(2023, 10, 6))},
}


def holiday_span(name: str, year: int) -> tuple[date, date]:
    """span 模式的假期日期区间: 历史年用官方安排, 缺年份用默认(春节=除夕~初六)."""
    if name == "spring":
        d = SPRING_FESTIVAL_DATES[year]
        return d - timedelta(days=1), d + timedelta(days=6)
    official = OFFICIAL_HOLIDAY_SPANS.get(year, {}).get(name)
    if official:
        return official
    if name == "may":
        return date(year, 5, 1), date(year, 5, 5)
    if name == "national":
        return date(year, 10, 1), date(year, 10, 7)
    raise KeyError(name)


def holiday_span_weeks(name: str, year: int, weeks: list[list[date]]) -> set[int]:
    start, end = holiday_span(name, year)
    return {
        index
        for index, week in enumerate(weeks, 1)
        if any(start <= day <= end for day in week)
    }


def span_days_in_week(name: str, year: int, weeks: list[list[date]], week_index: int) -> int:
    start, end = holiday_span(name, year)
    return sum(1 for day in weeks[week_index - 1] if start <= day <= end)


def resolve_windows_path(path: Path) -> Path:
    path_str = str(path)
    if len(path_str) >= 3 and path_str[1:3] == ":\\":
        drive = path_str[0].lower()
        rest = path_str[3:].replace("\\", "/")
        return Path(f"/mnt/{drive}/{rest}")
    return path


def valid_days(year: int) -> list[date]:
    current = date(year, 1, 1)
    end = date(year, 12, 31)
    days = []
    while current <= end:
        if not (current.month == 2 and current.day == 29):
            days.append(current)
        current += timedelta(days=1)
    return days


def split_weeks(year: int) -> list[list[date]]:
    weeks: list[list[date]] = []
    current_week: list[date] = []
    for day in valid_days(year):
        current_week.append(day)
        if day.weekday() == 6:
            weeks.append(current_week)
            current_week = []
    if current_week:
        weeks.append(current_week)
    return weeks


def norm_max(values: list[float]) -> list[float]:
    max_value = max(values) if values else 0.0
    if max_value == 0:
        return [0.0 for _ in values]
    return [value / max_value for value in values]


def mean_curves(curves: list[list[float]]) -> list[float]:
    length = len(curves[0])
    return [sum(curve[i] for curve in curves) / len(curves) for i in range(length)]


def resample(values: list[float], target_length: int) -> list[float]:
    if len(values) == target_length:
        return values[:]
    if len(values) == 1:
        return [values[0]] * target_length
    if target_length == 1:
        return [values[0]]

    old_last = len(values) - 1
    new_last = target_length - 1
    output = []
    for i in range(target_length):
        x = i * old_last / new_last
        lo = int(math.floor(x))
        hi = min(lo + 1, old_last)
        weight = x - lo
        output.append(values[lo] * (1 - weight) + values[hi] * weight)
    return output


def _contiguous_blocks(weeks: set[int]) -> list[list[int]]:
    blocks: list[list[int]] = []
    for w in sorted(weeks):
        if blocks and w == blocks[-1][-1] + 1:
            blocks[-1].append(w)
        else:
            blocks.append([w])
    return blocks


def holiday_adjusted_scalars(values: list[float], special: set[int]) -> list[float]:
    """Replace each contiguous special-week block by the linear ramp between
    its nearest normal weeks (edge blocks copy the inner side).  For a single
    3-week block this reproduces the original spring_adjusted_scalars."""
    adjusted = values[:]
    n = len(values)
    for block in _contiguous_blocks(special):
        li = block[0] - 2          # 0-based index of week (block[0]-1)
        ri = block[-1]             # 0-based index of week (block[-1]+1)
        li = li if li >= 0 else None
        ri = ri if ri <= n - 1 else None
        for w in block:
            i = w - 1
            if li is None and ri is None:
                v = values[i]
            elif li is None:
                v = values[ri]
            elif ri is None:
                v = values[li]
            else:
                weight = (i - li) / (ri - li)
                v = values[li] * (1 - weight) + values[ri] * weight
            adjusted[i] = v
    return adjusted


def holiday_adjusted_vectors(vectors: list[list[float]], special: set[int]) -> list[list[float]]:
    adjusted = [vector[:] for vector in vectors]
    n = len(vectors)
    for block in _contiguous_blocks(special):
        li = block[0] - 2
        ri = block[-1]
        li = li if li >= 0 else None
        ri = ri if ri <= n - 1 else None
        for w in block:
            i = w - 1
            if li is None and ri is None:
                adjusted[i] = vectors[i][:]
            elif li is None:
                adjusted[i] = vectors[ri][:]
            elif ri is None:
                adjusted[i] = vectors[li][:]
            else:
                weight = (i - li) / (ri - li)
                adjusted[i] = [
                    vectors[li][j] * (1 - weight) + vectors[ri][j] * weight
                    for j in range(len(vectors[li]))
                ]
    return adjusted


def load_history(input_dir: Path, years: list[int]) -> dict[int, dict[date, list[float]]]:
    result: dict[int, dict[date, list[float]]] = {}
    for year in years:
        path = input_dir / f"历史负荷数据{year}.csv"
        if not path.exists():
            raise FileNotFoundError(path)

        day_map: dict[date, list[float | None]] = {}
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                dt = datetime.strptime(row["时间"], "%Y-%m-%d %H:%M:%S")
                if dt.month == 2 and dt.day == 29:
                    continue
                day_map.setdefault(dt.date(), [None] * 24)[dt.hour] = float(row["值"])

        clean_days: dict[date, list[float]] = {}
        for day, values in day_map.items():
            if all(value is not None for value in values):
                clean_days[day] = [float(value) for value in values]
        result[year] = clean_days
    return result


def max_peak_day(days: list[date], daily_peak: dict[date, float]) -> date:
    return max(days, key=lambda day: daily_peak[day])


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted = {}
            for key, value in row.items():
                formatted[key] = f"{value:.8f}" if isinstance(value, float) else value
            writer.writerow(formatted)


def save_svg(
    path: Path,
    hourly_rows: list[dict[str, object]],
    weeks_target: list[list[date]],
    target_year: int,
    target_special: dict[int, str],
) -> None:
    width, height = 1600, 520
    left, right, top, bottom = 70, 24, 36, 60
    plot_width = width - left - right
    plot_height = height - top - bottom
    ymin, ymax = 0.0, 1.05
    values = [float(row["全年小时标幺"]) for row in hourly_rows]

    points = []
    for i, value in enumerate(values):
        x = left + i * plot_width / (len(values) - 1)
        y = top + (ymax - value) * plot_height / (ymax - ymin)
        points.append(f"{x:.2f},{y:.2f}")

    start = date(target_year, 1, 1)
    spring_weeks = [w for w, name in target_special.items() if name == "spring"]
    if spring_weeks:
        spring_anchor = holiday_week_index("spring", target_year, weeks_target)
        special = {w: w - spring_anchor for w in spring_weeks}
    else:
        special = {}
    spring_spans = []
    for week_index, relative in sorted(special.items()):
        days = weeks_target[week_index - 1]
        i1 = (days[0] - start).days * 24
        i2 = ((days[-1] - start).days + 1) * 24 - 1
        x1 = left + i1 * plot_width / (len(values) - 1)
        x2 = left + i2 * plot_width / (len(values) - 1)
        spring_spans.append((relative, x1, x2))

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="24" text-anchor="middle" font-family="Arial" font-size="20">'
        f"{target_year} Annual Normalized Hourly Load Curve</text>",
    ]
    labels = {-1: "Spring Festival -1 week", 0: "Spring Festival week", 1: "Spring Festival +1 week"}
    for relative, x1, x2 in spring_spans:
        svg.append(
            f'<rect x="{x1:.2f}" y="{top}" width="{x2 - x1:.2f}" height="{plot_height}" '
            'fill="#fde2e2" opacity="0.75"/>'
        )
        svg.append(
            f'<text x="{(x1 + x2) / 2:.2f}" y="{top + 18}" text-anchor="middle" '
            f'font-family="Arial" font-size="12" fill="#9b1c1c">{labels[relative]}</text>'
        )

    for tick in (0, 0.2, 0.4, 0.6, 0.8, 1.0):
        y = top + (ymax - tick) * plot_height / (ymax - ymin)
        svg.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        svg.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-family="Arial" font-size="12" fill="#333">{tick:.1f}</text>'
        )

    for month in range(1, 13):
        idx = (date(target_year, month, 1) - start).days * 24
        x = left + idx * plot_width / (len(values) - 1)
        svg.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height - bottom}" stroke="#f0f0f0"/>')
        svg.append(f'<text x="{x + 4:.2f}" y="{height - bottom + 22}" font-family="Arial" font-size="12">{month}</text>')

    svg.extend(
        [
            f'<polyline fill="none" stroke="#d62728" stroke-width="1.4" points="{" ".join(points)}"/>',
            f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#222"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#222"/>',
            f'<text x="{width / 2}" y="{height - 16}" text-anchor="middle" font-family="Arial" font-size="14">Month of {target_year}</text>',
            f'<text x="20" y="{height / 2}" transform="rotate(-90 20 {height / 2})" text-anchor="middle" font-family="Arial" font-size="14">Normalized load</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(svg), encoding="utf-8")


def generate(
    input_dir: Path,
    output_dir: Path,
    target_year: int,
    hist_years: list[int],
    typical_years: list[int],
    holidays: list[str],
    pattern: str,
    single_typical: bool = False,
) -> dict[str, object]:
    # hist_years 决定周峰层级；typical_years 决定 24 小时日形状，二者可以独立指定。
    all_source_years = sorted(set(hist_years + typical_years))
    load_by_year_day = load_history(input_dir, all_source_years)
    weeks_all = {year: split_weeks(year) for year in all_source_years}
    weeks_hist = {year: weeks_all[year] for year in hist_years}
    weeks_target = split_weeks(target_year)
    target_week_count = len(weeks_target)

    # 每年每个假期的特殊周集合: pm1=锚点周前/当/后3周, core=仅锚点周, span=与假期日期区间重叠的周
    special_by_year: dict[int, dict[str, set[int]]] = {}
    for year in all_source_years:
        count = len(weeks_all[year])
        per: dict[str, set[int]] = {}
        for name in holidays:
            if pattern == "span":
                per[name] = holiday_span_weeks(name, year, weeks_all[year])
            else:
                per[name] = special_weeks_for(holiday_week_index(name, year, weeks_all[year]), count, pattern, name)
        special_by_year[year] = per
    hist_special = {year: set().union(*special_by_year[year].values()) for year in hist_years}
    typical_special = {year: set().union(*special_by_year[year].values()) for year in typical_years}

    # 目标年特殊周 -> 假期名, 断言无重叠
    target_special: dict[int, str] = {}
    target_special_by_holiday: dict[str, set[int]] = {}
    for name in holidays:
        if pattern == "span":
            weeks_n = holiday_span_weeks(name, target_year, weeks_target)
        else:
            weeks_n = special_weeks_for(
                holiday_week_index(name, target_year, weeks_target), target_week_count, pattern, name
            )
        target_special_by_holiday[name] = weeks_n
        for w in weeks_n:
            if w in target_special:
                raise ValueError(f"Holiday special weeks overlap at week {w}")
            target_special[w] = name

    def resolve_sources(name: str, target_week: int, years: list[int]) -> list[tuple[int, int]]:
        """目标特殊周的源(年,周): pm1/core=各样本年同相对位置周; span=假期天数最接近的样本特殊周(并列全取)."""
        if pattern != "span":
            rel = target_week - holiday_week_index(name, target_year, weeks_target)
            return [
                (year, holiday_week_index(name, year, weeks_all[year]) + rel)
                for year in years
                if 1 <= holiday_week_index(name, year, weeks_all[year]) + rel <= len(weeks_all[year])
            ]
        n_target = span_days_in_week(name, target_year, weeks_target, target_week)
        scored = [
            (abs(span_days_in_week(name, year, weeks_all[year], w) - n_target), year, w)
            for year in years
            for w in sorted(special_by_year[year][name])
        ]
        best = min(s[0] for s in scored)
        return [(year, w) for diff, year, w in scored if diff == best]

    daily_peak_all = {
        year: {day: max(values) for day, values in load_by_year_day[year].items()}
        for year in all_source_years
    }
    daily_peak = {
        year: {day: max(values) for day, values in load_by_year_day[year].items()}
        for year in hist_years
    }

    # ---- 周峰层: 无假期基准 + 各假期影响系数 ----
    # 周峰先除以各历史年的年峰值，融合的是形状而不是历史 MW 量级。
    raw_week_norm: dict[int, list[float]] = {}
    for year in hist_years:
        weekly_peak = [
            max(daily_peak[year][day] for day in week if day in daily_peak[year])
            for week in weeks_hist[year]
        ]
        annual_peak = max(weekly_peak)
        raw_week_norm[year] = [value / annual_peak for value in weekly_peak]

    base_week_by_year = {
        year: holiday_adjusted_scalars(raw_week_norm[year], hist_special[year])
        for year in hist_years
    }
    max_week_count = max(len(weeks_hist[year]) for year in hist_years)
    base_week_raw = []
    for week_index in range(1, max_week_count + 1):
        values = [base_week_by_year[year][week_index - 1] for year in hist_years]
        base_week_raw.append(sum(values) / len(values))
    base_week_curve = norm_max(base_week_raw)

    holiday_factor: dict[tuple[str, int], float] = {}
    for name in holidays:
        for w in sorted(target_special_by_holiday[name]):
            srcs = resolve_sources(name, w, hist_years)
            holiday_factor[(name, w)] = sum(
                raw_week_norm[year][w_ - 1] / base_week_by_year[year][w_ - 1]
                for year, w_ in srcs
            ) / len(srcs)

    # ---- 日峰层(周内7点分布, 不分工作日/双休日): 逻辑同原脚本 ----
    # 日峰系数描述周内每天相对该周峰的强弱，后面与典型日小时曲线相乘。
    daily_raw_by_year: dict[int, list[list[float]]] = {}
    for year in hist_years:
        curves = []
        for week in weeks_hist[year]:
            values = [daily_peak[year][day] for day in week if day in daily_peak[year]]
            curves.append(resample(norm_max(values), 7))
        daily_raw_by_year[year] = curves
    daily_base_by_year = {
        year: holiday_adjusted_vectors(daily_raw_by_year[year], hist_special[year])
        for year in hist_years
    }
    normal_daily: dict[int, list[float]] = {}
    for week_index in range(1, max_week_count + 1):
        normal_daily[week_index] = norm_max(
            mean_curves([daily_base_by_year[year][week_index - 1] for year in hist_years])
        )
    holiday_daily: dict[tuple[str, int], list[float]] = {}
    for name in holidays:
        for w in sorted(target_special_by_holiday[name]):
            srcs = resolve_sources(name, w, hist_years)
            holiday_daily[(name, w)] = norm_max(
                mean_curves([daily_raw_by_year[year][w_ - 1] for year, w_ in srcs])
            )

    # ---- 典型日层: 工作日/双休日两条曲线, 每周各一条 ----
    # 工作日和双休日分别取代表性日，避免用一条曲线抹平周末行为差异。
    def week_group_curves(year: int, week: list[date]) -> tuple[list[float], list[float]]:
        """(工作日曲线, 双休日曲线): 各取组内日峰最高一天的24h曲线; 组空则互为兜底."""
        days = [day for day in week if day in load_by_year_day[year]]
        work = [day for day in days if day.weekday() <= 4]
        wknd = [day for day in days if day.weekday() >= 5]
        work_curve = norm_max(load_by_year_day[year][max_peak_day(work, daily_peak_all[year])]) if work else None
        wknd_curve = norm_max(load_by_year_day[year][max_peak_day(wknd, daily_peak_all[year])]) if wknd else None
        if work_curve is None:
            work_curve = wknd_curve
        if wknd_curve is None:
            wknd_curve = work_curve
        return work_curve, wknd_curve

    typical_raw_work: dict[int, list[list[float]]] = {}
    typical_raw_wknd: dict[int, list[list[float]]] = {}
    for year in typical_years:
        curves_work = []
        curves_wknd = []
        for week in weeks_all[year]:
            c_work, c_wknd = week_group_curves(year, week)
            curves_work.append(c_work)
            curves_wknd.append(c_wknd)
        typical_raw_work[year] = curves_work
        typical_raw_wknd[year] = curves_wknd

    if single_typical:
        # 领导原版口径: 每周单条典型日曲线(周内日峰最高日), 不分工作日/双休日
        for year in typical_years:
            single = []
            for week in weeks_all[year]:
                days = [day for day in week if day in load_by_year_day[year]]
                single.append(norm_max(load_by_year_day[year][max_peak_day(days, daily_peak_all[year])]))
            typical_raw_work[year] = single
            typical_raw_wknd[year] = single

    typical_base_work = {
        year: holiday_adjusted_vectors(typical_raw_work[year], typical_special[year])
        for year in typical_years
    }
    typical_base_wknd = {
        year: holiday_adjusted_vectors(typical_raw_wknd[year], typical_special[year])
        for year in typical_years
    }
    normal_typical_work: dict[int, list[float]] = {}
    normal_typical_wknd: dict[int, list[float]] = {}
    for week_index in range(1, max_week_count + 1):
        normal_typical_work[week_index] = norm_max(
            mean_curves([typical_base_work[year][week_index - 1] for year in typical_years])
        )
        normal_typical_wknd[week_index] = norm_max(
            mean_curves([typical_base_wknd[year][week_index - 1] for year in typical_years])
        )
    holiday_typical_work: dict[tuple[str, int], list[float]] = {}
    holiday_typical_wknd: dict[tuple[str, int], list[float]] = {}
    for name in holidays:
        for w in sorted(target_special_by_holiday[name]):
            srcs = resolve_sources(name, w, typical_years)
            holiday_typical_work[(name, w)] = norm_max(
                mean_curves([typical_raw_work[year][w_ - 1] for year, w_ in srcs])
            )
            holiday_typical_wknd[(name, w)] = norm_max(
                mean_curves([typical_raw_wknd[year][w_ - 1] for year, w_ in srcs])
            )

    # ---- 目标年合成: 周峰 = 基准 × 假期系数; 小时 = 周峰 × 日峰 × 典型日(工作/双休) ----
    # 三层相乘得到 8760 小时形状，最终再次除以全年最大值，保证最大标幺值为 1。
    raw_target_week = []
    for week_index in range(1, target_week_count + 1):
        base_value = base_week_curve[week_index - 1]
        if week_index in target_special:
            name = target_special[week_index]
            factor = holiday_factor[(name, week_index)]
        else:
            factor = 1.0
        raw_target_week.append(base_value * factor)
    final_week = norm_max(raw_target_week)

    hourly_rows = []
    week_rows = []
    for week_index, days in enumerate(weeks_target, 1):
        if week_index in target_special:
            name = target_special[week_index]
            day_curve = holiday_daily[(name, week_index)]
            if pattern == "span":
                week_type = name
            else:
                rel = week_index - holiday_week_index(name, target_year, weeks_target)
                week_type = f"{name}{rel:+d}"
            factor_used = holiday_factor[(name, week_index)]
            work_curve = holiday_typical_work[(name, week_index)]
            wknd_curve = holiday_typical_wknd[(name, week_index)]
        else:
            day_curve = normal_daily[week_index]
            week_type = "normal"
            factor_used = 1.0
            work_curve = normal_typical_work[week_index]
            wknd_curve = normal_typical_wknd[week_index]

        day_coeff = resample(day_curve, len(days))
        week_rows.append(
            {
                "年份": target_year,
                "周序": week_index,
                "开始日期": str(days[0]),
                "结束日期": str(days[-1]),
                "天数": len(days),
                "类型": week_type,
                "无假期基准周峰标幺": base_week_curve[week_index - 1],
                "假期影响系数": factor_used,
                "假期修正后周峰": raw_target_week[week_index - 1],
                "最终周峰标幺": final_week[week_index - 1],
            }
        )

        for day_index, day in enumerate(days, 1):
            is_weekend = day.weekday() >= 5
            hour_curve = wknd_curve if is_weekend else work_curve
            day_type = "双休日" if is_weekend else "工作日"
            for hour in range(24):
                hourly_rows.append(
                    {
                        "时间": f"{day} {hour:02d}:00:00",
                        "年份": target_year,
                        "周序": week_index,
                        "周开始": str(days[0]),
                        "周结束": str(days[-1]),
                        "周内日序": day_index,
                        "小时": hour,
                        "类型": week_type,
                        "日类型": day_type,
                        "周峰标幺": final_week[week_index - 1],
                        "日峰标幺": day_coeff[day_index - 1],
                        "典型日小时标幺": hour_curve[hour],
                        "全年小时标幺": final_week[week_index - 1] * day_coeff[day_index - 1] * hour_curve[hour],
                    }
                )

    annual_max = max(float(row["全年小时标幺"]) for row in hourly_rows)
    for row in hourly_rows:
        row["全年小时标幺"] = float(row["全年小时标幺"]) / annual_max

    values = [float(row["全年小时标幺"]) for row in hourly_rows]
    summary_rows = [
        {"项目": "目标年", "值": str(target_year)},
        {"项目": "历史样本年", "值": "、".join(str(year) for year in hist_years)},
        {"项目": "典型日小时曲线来源", "值": "、".join(str(year) for year in typical_years)},
        {"项目": "假期处理", "值": "、".join(HOLIDAY_LABELS[name] for name in holidays) + f"（{pattern}）"},
        {"项目": "典型日小时曲线方法", "值": "每周两条：工作日=周内日峰最高的周一至周五，双休日=周内日峰最高的周六/周日；按典型年平均后再标幺"},
    ]
    for name in holidays:
        if pattern == "span":
            s0, s1 = holiday_span(name, target_year)
            date_txt = f"{s0}~{s1}"
        else:
            date_txt = str(holiday_anchor_date(name, target_year))
        summary_rows.append({"项目": f"{HOLIDAY_LABELS[name]}日期", "值": date_txt})
        summary_rows.append(
            {
                "项目": f"{HOLIDAY_LABELS[name]}特殊周",
                "值": "、".join(f"第{w}周" for w in sorted(target_special_by_holiday[name])),
            }
        )
        for w in sorted(target_special_by_holiday[name]):
            srcs = resolve_sources(name, w, hist_years)
            summary_rows.append(
                {"项目": f"{HOLIDAY_LABELS[name]}第{w}周影响系数", "值": f"{holiday_factor[(name, w)]:.8f}"}
            )
            summary_rows.append(
                {
                    "项目": f"{HOLIDAY_LABELS[name]}第{w}周源周",
                    "值": "、".join(f"{year}年第{w_}周" for year, w_ in srcs),
                }
            )
    summary_rows.extend(
        [
            {"项目": "小时点数", "值": str(len(values))},
            {"项目": "最大标幺值", "值": f"{max(values):.8f}"},
            {"项目": "最小标幺值", "值": f"{min(values):.8f}"},
            {"项目": "最大负荷利用小时数", "值": f"{sum(values) / max(values):.8f}"},
            {"项目": "平均负荷率", "值": f"{sum(values) / len(values):.8f}"},
        ]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    hourly_path = output_dir / f"{target_year}年全年小时标幺曲线.csv"
    week_path = output_dir / f"{target_year}年周峰值与假期修正.csv"
    summary_path = output_dir / f"{target_year}年汇总说明.csv"
    svg_path = output_dir / f"{target_year}年全年小时标幺曲线.svg"

    write_csv(
        hourly_path,
        ["时间", "年份", "周序", "周开始", "周结束", "周内日序", "小时", "类型", "日类型", "周峰标幺", "日峰标幺", "典型日小时标幺", "全年小时标幺"],
        hourly_rows,
    )
    write_csv(
        week_path,
        ["年份", "周序", "开始日期", "结束日期", "天数", "类型", "无假期基准周峰标幺", "假期影响系数", "假期修正后周峰", "最终周峰标幺"],
        week_rows,
    )
    write_csv(summary_path, ["项目", "值"], summary_rows)
    save_svg(svg_path, hourly_rows, weeks_target, target_year, target_special)

    return {
        "output_dir": str(output_dir),
        "hourly_csv": str(hourly_path),
        "week_csv": str(week_path),
        "summary_csv": str(summary_path),
        "svg": str(svg_path),
        "hours": len(values),
        "utilization_hours": sum(values) / max(values),
    }


def parse_years(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_holidays(value: str) -> list[str]:
    names = [part.strip() for part in value.split(",") if part.strip()]
    for name in names:
        if name not in HOLIDAY_LABELS:
            raise ValueError(f"Unknown holiday {name!r}; choose from {sorted(HOLIDAY_LABELS)}")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate annual normalized hourly load curve with multi-holiday and weekday/weekend templates.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory containing 历史负荷数据YYYY.csv")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory")
    parser.add_argument("--target-year", type=int, default=2030, help="Target calendar year")
    parser.add_argument("--history-years", default="2020,2022,2023", help="Years for week/day peak templates")
    parser.add_argument("--typical-years", default="2022,2023", help="Years for typical-day hourly templates")
    parser.add_argument("--holidays", default="spring,may,national", help="Comma list from: spring,may,national")
    parser.add_argument("--holiday-pattern", default="pm1", choices=["pm1", "pm0", "p01", "core", "span", "hybrid"],
                        help="pm1=前1/当/后1三周(同春节), pm0=前1+当, p01=当+后1, core=仅当周, span=与假期日期区间重叠的周, hybrid=春节前当后+五一前当+十一当后")
    parser.add_argument("--single-typical", action="store_true", help="每周单条典型日曲线(领导原版口径), 不分工作日/双休日")
    args = parser.parse_args()

    input_dir = resolve_windows_path(args.input_dir)
    output_dir = resolve_windows_path(args.output_dir) if args.output_dir else input_dir / "生成结果" / f"{args.target_year}精简结果_多假_工作双休"
    result = generate(
        input_dir=input_dir,
        output_dir=output_dir,
        target_year=args.target_year,
        hist_years=parse_years(args.history_years),
        typical_years=parse_years(args.typical_years),
        holidays=parse_holidays(args.holidays),
        pattern=args.holiday_pattern,
        single_typical=args.single_typical,
    )

    print("Generation complete.")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
