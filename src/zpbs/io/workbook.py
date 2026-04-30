from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import BadZipFile, ZipFile

XLSX_NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
COMPACT_SUMMARY_COLUMNS = (
    "source_file",
    "output_coefficients_csv",
    "force_id",
    "surf_id",
    "surface_token",
    "applied_reference_radius_um",
    "applied_normalization_radius_um",
    "observed_aperture_radius_um",
    "sphere_rms_um",
    "sphere_mae_um",
    "zpbs_residual_rms_um",
    "zpbs_residual_cond",
    "vertex_mismatch_z_um",
    "vertex_um",
    "vertex_residual_um",
)


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
    if not file_path.is_file():
        raise ValueError(f"Summary workbook is not a file: {file_path}")

    try:
        with ZipFile(file_path) as workbook:
            shared_strings = _load_shared_strings(workbook)
            sheet = ET.fromstring(workbook.read("xl/worksheets/sheet1.xml"))
    except BadZipFile as exc:
        raise ValueError(f"Summary workbook is not a valid .xlsx file: {file_path}") from exc
    except KeyError as exc:
        raise ValueError(f"Summary workbook is missing xl/worksheets/sheet1.xml: {file_path}") from exc
    except ET.ParseError as exc:
        raise ValueError(f"Summary workbook contains invalid XML: {file_path}") from exc

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


def is_compact_summary_rows(rows: list[dict[str, str]]) -> bool:
    """Return True when workbook rows match the maintained compact summary schema."""
    if not rows:
        return False
    headers = tuple(rows[0].keys())
    return headers == COMPACT_SUMMARY_COLUMNS
