from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

XLSX_NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _xlsx_column_index(cell_ref: str) -> int:
    """Convert an A1-style cell reference into a zero-based column index."""
    letters = "".join(ch for ch in cell_ref if ch.isalpha()).upper()
    index = 0
    for letter in letters:
        index = (index * 26) + (ord(letter) - 64)
    return max(index - 1, 0)


def _load_shared_strings(workbook: ZipFile) -> list[str]:
    """Load shared strings from an XLSX workbook when present."""
    if "xl/sharedStrings.xml" not in workbook.namelist():
        return []
    root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
    strings: list[str] = []
    for item in root.findall("x:si", XLSX_NS):
        strings.append("".join(text.text or "" for text in item.iterfind(".//x:t", XLSX_NS)))
    return strings


def _xlsx_cell_text(cell: ET.Element, shared_strings: list[str]) -> str:
    """Resolve one XLSX cell into display text."""
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        inline = cell.find("x:is", XLSX_NS)
        return "".join(inline.itertext()) if inline is not None else ""

    value = cell.find("x:v", XLSX_NS)
    raw = "" if value is None or value.text is None else value.text
    if cell_type == "s":
        try:
            return shared_strings[int(raw)]
        except (IndexError, ValueError):
            return ""
    if cell_type == "b":
        return "1" if raw == "1" else "0"
    return raw


def parse_inline_xlsx_rows(file_path: Path) -> list[dict[str, str]]:
    """Read a minimal XLSX sheet into row dictionaries."""
    with ZipFile(file_path) as workbook:
        shared_strings = _load_shared_strings(workbook)
        sheet = ET.fromstring(workbook.read("xl/worksheets/sheet1.xml"))

    row_elements = sheet.find("x:sheetData", XLSX_NS)
    if row_elements is None:
        return []

    records: list[list[str]] = []
    for row in row_elements.findall("x:row", XLSX_NS):
        values: list[str] = []
        for cell in row.findall("x:c", XLSX_NS):
            cell_ref = cell.attrib.get("r", "")
            col_index = _xlsx_column_index(cell_ref) if cell_ref else len(values)
            while len(values) < col_index:
                values.append("")
            values.append(_xlsx_cell_text(cell, shared_strings))
        records.append(values)

    if not records:
        return []

    headers = records[0]
    return [
        {headers[index]: values[index] if index < len(values) else "" for index in range(len(headers))}
        for values in records[1:]
    ]


def parse_coefficients_csv(file_path: Path) -> tuple[dict[str, str], list[tuple[str, float]]]:
    """Read exported coefficient metadata and coefficient values."""
    metadata: dict[str, str] = {}
    coeffs: list[tuple[str, float]] = []
    if not file_path.exists():
        return metadata, coeffs

    for line in file_path.read_text().splitlines():
        if not line.strip():
            continue
        name, _, value = line.partition(",")
        key = name.strip()
        raw_value = value.strip()
        if key.startswith("Z"):
            try:
                coeffs.append((key, float(raw_value)))
            except ValueError:
                continue
        else:
            metadata[key] = raw_value
    return metadata, coeffs


def parse_name_value_csv_rows(file_path: Path) -> list[tuple[str, str]]:
    """Read a two-column name/value CSV while preserving row order and formatting."""
    if not file_path.exists():
        return []

    rows: list[tuple[str, str]] = []
    with file_path.open(newline="") as handle:
        reader = csv.reader(handle, dialect="excel")
        for row in reader:
            if not row:
                continue
            name = str(row[0]).strip()
            value = str(row[1]).strip() if len(row) > 1 else ""
            if not name:
                continue
            rows.append((name, value))
    return rows


def parse_run_manifest(summary_file: Path) -> dict[str, object]:
    """Load the sibling run manifest when present."""
    manifest_path = summary_file.with_name("run_manifest.json")
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return {}


def detect_subset_workbook_kind(
    rows: list[dict[str, str]],
    workbook_path: Path,
    manifest: dict[str, object],
) -> str:
    """Infer the subset-workbook kind from row content and manifest hints."""
    spec_kind = None
    for spec in manifest.get("gui_workbook_specs", []) or []:
        if spec.get("file") == workbook_path.name:
            spec_kind = str(spec.get("kind") or "")
            break

    result_kinds = {row.get("result_kind", "") for row in rows if row.get("result_kind")}
    result_kind = next(iter(result_kinds), "")
    if result_kind == "drop_importance":
        return "drop_importance"
    if result_kind == "subset_path":
        path_kinds = {row.get("path_kind", "") for row in rows if row.get("path_kind")}
        if "greedy_refit" in path_kinds:
            return "subset_path_greedy"
        if "single_drop_ranked_refit" in path_kinds:
            return "subset_path_ranked"
        return spec_kind or "subset_path_greedy"
    if result_kind == "global_consistent_subset":
        return "global_consistent_subset"
    if result_kind == "global_consistent_subset_aggregate":
        return "global_consistent_subset_aggregate"
    if result_kind == "mode_consistency":
        if any(row.get("global_order", "").strip() for row in rows):
            return "global_mode_order"
        group_types = {row.get("group_type", "") for row in rows if row.get("group_type")}
        if "surf_id" in group_types:
            return "mode_consistency_by_surf_id"
        return "mode_consistency_overall"
    return spec_kind or result_kind or workbook_path.stem
