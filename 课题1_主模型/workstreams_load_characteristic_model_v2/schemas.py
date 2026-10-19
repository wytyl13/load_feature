"""Typed records shared by the input and industry-mapping layer.

The records deliberately keep source row numbers and labels.  Official
electricity values are therefore traceable to the exact workbook cell instead
of relying on fuzzy label matching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Mapping


class OfficialClassification(str, Enum):
    SUMMARY = "汇总项"
    NEW_LOAD = "新型负荷"
    EXCLUDED = "排除项"


class AccountingRole(str, Enum):
    """Whether and how an official workbook row participates in closure."""

    SUMMARY = "summary"
    LEAF = "leaf"
    DETAIL = "detail"
    SPLIT_PARENT = "split_parent"
    SPLIT_COMPONENT = "split_component"


@dataclass(frozen=True)
class InputFileManifest:
    logical_name: str
    path: Path
    size_bytes: int
    modified_time_utc: str
    sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "logical_name": self.logical_name,
            "path": str(self.path),
            "size_bytes": self.size_bytes,
            "modified_time_utc": self.modified_time_utc,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class SampleSheetRecord:
    raw_sheet_name: str
    sheet_name: str
    broad_sector: str
    industry: str
    user_number: int


@dataclass(frozen=True)
class SampleIndustryMapping:
    sample_industry: str
    broad_sector: str
    is_traditional: bool
    official_source_row: int
    official_label: str
    notes: str = ""

@dataclass(frozen=True)
class Official2025Row:
    source_row: int
    raw_label: str
    label: str
    value_2025: float
    unit: str
    display_decimal_places: int


@dataclass(frozen=True)
class OfficialMappingEntry:
    source_row: int
    official_label: str
    classification: str
    accounting_role: AccountingRole
    department: str = ""
    split_parent_label: str = ""
    representative_lookup_allowed: bool = False
    notes: str = ""


@dataclass(frozen=True)
class DepartmentBalance:
    department: str
    target_label: str
    target_value_2025: float
    mapped_value_2025: float
    absolute_error: float
    relative_error: float
    passed: bool


@dataclass(frozen=True)
class OfficialBalanceResult:
    unit: str
    total_label: str
    total_value_2025: float
    mapped_value_2025: float
    absolute_tolerance: float
    relative_tolerance: float
    absolute_error: float
    relative_error: float
    passed: bool
    amount_by_classification: Mapping[str, float] = field(default_factory=dict)
    amount_by_department: Mapping[str, float] = field(default_factory=dict)
    department_balances: tuple[DepartmentBalance, ...] = ()


@dataclass(frozen=True)
class SampleIndustryEnergy:
    sample_industry: str
    broad_sector: str
    is_traditional: bool
    official_source_row: int
    official_label: str
    official_energy_2025: float
    official_energy_basis: str
    unit: str


__all__ = [
    "AccountingRole",
    "DepartmentBalance",
    "InputFileManifest",
    "Official2025Row",
    "OfficialBalanceResult",
    "OfficialClassification",
    "OfficialMappingEntry",
    "SampleIndustryEnergy",
    "SampleIndustryMapping",
    "SampleSheetRecord",
]
