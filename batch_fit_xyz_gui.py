"""PyQt frontend for launching and inspecting batch XYZ fits.

Public-release note:
The subset-inspection workflow is intentionally disabled for the standalone
public GUI build. The implementation remains in this file for possible future
re-enablement, but the tab is hidden and the loader is hard-gated by an explicit
release toggle below.
"""

from __future__ import annotations

import json
import math
import os
import shlex
import sys
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str((Path(__file__).resolve().parent / ".matplotlib").resolve()))

# isort: off
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import QProcess, Qt
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from batch_fit_xyz import (
    load_xyz_point_cloud,
    parse_boolish,
    parse_optional_int,
    parse_surface_metadata,
    radial_bin_profile,
    run_fit_pipeline,
    sphere_profile_z,
    uses_posterior_sign_convention,
)
# isort: on

SCRIPT_DIR = Path(__file__).resolve().parent
BATCH_SCRIPT = SCRIPT_DIR / "batch_fit_xyz.py"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "batch_outputs"
FOCUS_SURF_IDS = ("AA", "AP", "PA", "PP")
XLSX_NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

# Public release toggle: keep subset-analysis code present but inaccessible in
# the shipped GUI unless someone intentionally re-enables it in code.
ENABLE_SUBSET_INSPECTION_UI = False


@dataclass(frozen=True)
class PathResolution:
    """Resolved viewer path plus the remap strategy that produced it."""

    original: Path
    resolved: Path
    strategy: str
    exists: bool


def _xlsx_column_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha()).upper()
    index = 0
    for letter in letters:
        index = (index * 26) + (ord(letter) - 64)
    return max(index - 1, 0)


def _load_shared_strings(workbook: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in workbook.namelist():
        return []
    root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
    strings: list[str] = []
    for item in root.findall("x:si", XLSX_NS):
        strings.append("".join(text.text or "" for text in item.iterfind(".//x:t", XLSX_NS)))
    return strings


def _xlsx_cell_text(cell: ET.Element, shared_strings: list[str]) -> str:
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
    """Read the batch workbook, including Excel-rewritten variants with shared strings."""
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


def parse_run_manifest(summary_file: Path) -> dict[str, object]:
    """Load run_manifest.json if it sits next to the selected summary workbook."""
    manifest_path = summary_file.with_name("run_manifest.json")
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return {}


def _coerce_path(value: object) -> Path | None:
    """Return a Path for non-empty path-like values."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text)


def _append_unique_candidate(
    candidates: list[tuple[Path, str]],
    seen: set[str],
    path: Path | None,
    strategy: str,
) -> None:
    """Preserve insertion order while avoiding duplicate remap candidates."""
    if path is None:
        return
    key = str(path)
    if key in seen:
        return
    seen.add(key)
    candidates.append((path, strategy))


def _select_path_resolution(candidates: list[tuple[Path, str]]) -> PathResolution:
    """Choose the first existing candidate and otherwise keep the best fallback."""
    for path, strategy in candidates:
        if path.exists():
            return PathResolution(original=candidates[0][0], resolved=path, strategy=strategy, exists=True)
    fallback_path, fallback_strategy = candidates[-1]
    return PathResolution(
        original=candidates[0][0],
        resolved=fallback_path,
        strategy=fallback_strategy,
        exists=False,
    )


def remap_path_prefix(
    source_path: Path,
    *,
    original_root: Path | None,
    replacement_root: Path | None,
) -> Path | None:
    """Rebase one path onto a new root when it lived under the original root."""
    if original_root is None or replacement_root is None:
        return None
    try:
        relative = source_path.relative_to(original_root)
    except ValueError:
        return None
    return replacement_root / relative


def infer_original_run_dir(summary_file: Path, manifest: dict[str, object]) -> Path | None:
    """Infer the run directory recorded by the original machine."""
    manifest_summary_path = _coerce_path(manifest.get("summary_report_path"))
    if manifest_summary_path is not None:
        return manifest_summary_path.parent
    return None


def infer_original_source_root(
    rows: list[dict[str, str]],
    manifest: dict[str, object],
) -> Path | None:
    """Prefer the manifest input directory and fall back to a common source-file prefix."""
    manifest_input_dir = _coerce_path(manifest.get("input_dir"))
    if manifest_input_dir is not None:
        return manifest_input_dir

    source_paths = [
        Path(row["source_file"])
        for row in rows
        if row.get("source_file", "").strip()
    ]
    absolute_paths = [path for path in source_paths if path.is_absolute()]
    if not absolute_paths:
        return None

    try:
        common_path = Path(os.path.commonpath([str(path) for path in absolute_paths]))
    except ValueError:
        return None
    return common_path


def resolve_summary_coefficients_path(
    coeff_path_text: str,
    *,
    summary_file: Path,
    manifest: dict[str, object],
) -> PathResolution:
    """Resolve a coefficients CSV after a run folder has been copied elsewhere."""
    original = Path(coeff_path_text)
    candidates: list[tuple[Path, str]] = []
    seen: set[str] = set()
    current_run_dir = summary_file.parent

    _append_unique_candidate(candidates, seen, original, "workbook")
    if not original.is_absolute():
        _append_unique_candidate(candidates, seen, current_run_dir / original, "summary-relative")

    remapped_run_path = remap_path_prefix(
        original,
        original_root=infer_original_run_dir(summary_file, manifest),
        replacement_root=current_run_dir,
    )
    _append_unique_candidate(candidates, seen, remapped_run_path, "run-folder-remap")
    _append_unique_candidate(
        candidates,
        seen,
        current_run_dir / "coefficients" / original.name,
        "coefficients-filename-fallback",
    )
    return _select_path_resolution(candidates)


def resolve_summary_source_path(
    source_path_text: str,
    *,
    summary_file: Path,
    rows: list[dict[str, str]],
    manifest: dict[str, object],
    original_source_root: Path | None,
    local_source_root: Path | None,
) -> PathResolution:
    """Resolve a raw XYZ path when the batch workbook moves to another machine."""
    original = Path(source_path_text)
    candidates: list[tuple[Path, str]] = []
    seen: set[str] = set()

    _append_unique_candidate(candidates, seen, original, "workbook")
    if not original.is_absolute():
        _append_unique_candidate(candidates, seen, summary_file.parent / original, "summary-relative")

    effective_original_root = original_source_root or infer_original_source_root(rows, manifest)
    remapped_source_path = remap_path_prefix(
        original,
        original_root=effective_original_root,
        replacement_root=local_source_root,
    )
    _append_unique_candidate(candidates, seen, remapped_source_path, "source-root-remap")
    _append_unique_candidate(
        candidates,
        seen,
        local_source_root / original.name if local_source_root is not None else None,
        "source-filename-fallback",
    )
    return _select_path_resolution(candidates)


def prepare_summary_row_for_preview(
    row: dict[str, str],
    *,
    summary_file: Path,
    rows: list[dict[str, str]],
    manifest: dict[str, object],
    original_source_root_text: str = "",
    local_source_root_text: str = "",
) -> tuple[dict[str, str], dict[str, str]]:
    """Attach resolved source/coeff paths to one summary row for live replay."""
    original_source_root = _coerce_path(original_source_root_text)
    local_source_root = _coerce_path(local_source_root_text)

    source_resolution = resolve_summary_source_path(
        row.get("source_file", ""),
        summary_file=summary_file,
        rows=rows,
        manifest=manifest,
        original_source_root=original_source_root,
        local_source_root=local_source_root,
    )
    coeff_resolution = resolve_summary_coefficients_path(
        row.get("output_coefficients_csv", ""),
        summary_file=summary_file,
        manifest=manifest,
    )

    preview_row = dict(row)
    preview_row["_resolved_source_file"] = str(source_resolution.resolved)
    preview_row["_resolved_output_coefficients_csv"] = str(coeff_resolution.resolved)

    details = {
        "original_source_file": str(source_resolution.original),
        "resolved_source_file": str(source_resolution.resolved),
        "source_resolution_strategy": source_resolution.strategy,
        "source_exists": "1" if source_resolution.exists else "0",
        "original_coeff_file": str(coeff_resolution.original),
        "resolved_coeff_file": str(coeff_resolution.resolved),
        "coeff_resolution_strategy": coeff_resolution.strategy,
        "coeff_exists": "1" if coeff_resolution.exists else "0",
        "effective_original_source_root": str(original_source_root or infer_original_source_root(rows, manifest) or ""),
        "local_source_root": str(local_source_root or ""),
    }
    return preview_row, details


def parse_optional_float(text: str) -> float | None:
    """Parse optional numeric strings from summary workbooks."""
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def _nice_axis_step(span: float, *, target_ticks: int = 6) -> float:
    """Choose a stable 1/2/5 step size for tick spacing."""
    safe_span = max(float(span), 1e-12)
    raw_step = safe_span / max(target_ticks - 1, 1)
    exponent = math.floor(math.log10(raw_step))
    fraction = raw_step / (10**exponent)
    if fraction <= 1.0:
        nice_fraction = 1.0
    elif fraction <= 2.0:
        nice_fraction = 2.0
    elif fraction <= 5.0:
        nice_fraction = 5.0
    else:
        nice_fraction = 10.0
    return nice_fraction * (10**exponent)


def _snap_axis_bound(value: float, *, direction: str) -> float:
    """Snap one axis bound outward using the bound's own magnitude, not the full span."""
    magnitude = max(abs(float(value)), 1e-12)
    exponent = math.floor(math.log10(magnitude))
    quantum = 10 ** (exponent - 1)
    if direction == "down":
        return float(math.floor(value / quantum) * quantum)
    if direction == "up":
        return float(math.ceil(value / quantum) * quantum)
    raise ValueError(f"Unsupported snap direction: {direction}")


def snapped_axis_limits(values: np.ndarray, *, include_zero: bool = False) -> tuple[float, float, float]:
    """Return outward-rounded y-axis limits based on the actual extrema."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return -1.0, 1.0, 0.5

    raw_min = float(np.min(arr))
    raw_max = float(np.max(arr))
    min_value = raw_min
    max_value = raw_max
    if include_zero:
        min_value = min(min_value, 0.0)
        max_value = max(max_value, 0.0)

    if math.isclose(min_value, max_value):
        base = max(abs(min_value), 1e-3)
        half_window = _nice_axis_step(base * 0.4, target_ticks=3)
        y_min = min_value - half_window
        y_max = max_value + half_window
        step = _nice_axis_step(y_max - y_min)
        return float(y_min), float(y_max), float(step)

    y_min = _snap_axis_bound(min_value, direction="down")
    y_max = _snap_axis_bound(max_value, direction="up")
    if math.isclose(y_min, y_max):
        step = _nice_axis_step(max(abs(y_min), 1e-3))
        y_min -= step
        y_max += step
    step = _nice_axis_step(y_max - y_min)
    return float(y_min), float(y_max), float(step)


def parse_int_list_text(text: str) -> list[int]:
    """Parse JSON-like integer lists stored in workbook cells."""
    stripped = text.strip()
    if not stripped:
        return []
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        return []
    return [int(value) for value in data]


def format_metric(value: object, *, precision: int = 2) -> str:
    """Compact display formatting for tables and details."""
    if value is None:
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        magnitude = abs(float(value))
        if magnitude >= 1000 or (0 < magnitude < 0.01):
            return f"{float(value):.{precision}e}"
        return f"{float(value):.{precision}f}"
    text = str(value)
    numeric = parse_optional_float(text) if text.strip() else None
    if numeric is None:
        return text
    return format_metric(numeric, precision=precision)


def detect_subset_workbook_kind(
    rows: list[dict[str, str]],
    workbook_path: Path,
    manifest: dict[str, object],
) -> str:
    """Prefer row-level result data and use manifest specs only to break ties."""
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


class NumericTableWidgetItem(QTableWidgetItem):
    """QTableWidget item with stable numeric sorting."""

    def __init__(self, text: str, sort_value: float | int | str) -> None:
        super().__init__(text)
        self.sort_value = sort_value

    def __lt__(self, other: object) -> bool:
        if isinstance(other, NumericTableWidgetItem):
            return self.sort_value < other.sort_value
        return super().__lt__(other)


class SubsetPlotCanvas(FigureCanvas):
    """Dedicated canvas for subset omission workbooks."""

    def __init__(self, parent: QWidget | None = None) -> None:
        self.figure = Figure(figsize=(9.0, 7.0), constrained_layout=True)
        super().__init__(self.figure)
        self.setParent(parent)

    def clear_message(self, message: str) -> None:
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        axis.axis("off")
        axis.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
        self.draw_idle()

    def plot_drop_importance(self, rows: list[dict[str, str]], selected_row: dict[str, str] | None) -> str:
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        ordered_rows = sorted(rows, key=lambda row: int(float(row.get("impact_rank", "0"))))
        modes = [int(float(row.get("removed_mode_noll", "0"))) for row in ordered_rows]
        deltas = [float(row.get("delta_rms_vs_full", "0")) for row in ordered_rows]
        colors = ["#94a3b8"] * len(ordered_rows)
        selected_text = "No selected mode."
        if selected_row is not None:
            selected_mode = int(float(selected_row.get("removed_mode_noll", "0")))
            for index, mode in enumerate(modes):
                if mode == selected_mode:
                    colors[index] = "#dc2626"
                    break
            selected_text = (
                f"Removed mode Z{selected_mode} | "
                f"delta_rms_vs_full={format_metric(float(selected_row.get('delta_rms_vs_full', '0')))} | "
                f"subset_rms={format_metric(float(selected_row.get('subset_rms', '0')))}"
            )
        axis.bar(modes, deltas, color=colors, width=0.85)
        axis.set_title("Drop Importance: delta RMS vs full fit", fontsize=11)
        axis.set_xlabel("Removed Mode (Noll)")
        axis.set_ylabel("delta_rms_vs_full")
        axis.grid(True, axis="y", alpha=0.2)
        axis.text(0.01, 0.99, selected_text, transform=axis.transAxes, ha="left", va="top", fontsize=9)
        self.draw_idle()
        return selected_text

    def plot_subset_path(
        self,
        rows: list[dict[str, str]],
        selected_row: dict[str, str] | None,
        *,
        title: str,
    ) -> str:
        self.figure.clear()
        grid = self.figure.add_gridspec(2, 1, height_ratios=[1.2, 0.9])
        ax_line = self.figure.add_subplot(grid[0, 0])
        ax_text = self.figure.add_subplot(grid[1, 0])

        ordered_rows = sorted(rows, key=lambda row: int(float(row.get("step_index", "0"))))
        counts = [int(float(row.get("active_mode_count", "0"))) for row in ordered_rows]
        deltas = [float(row.get("delta_rms_vs_full", "0")) for row in ordered_rows]
        ax_line.plot(counts, deltas, color="#2563eb", linewidth=1.2, marker="o", markersize=3)
        ax_line.set_title(title, fontsize=11)
        ax_line.set_xlabel("Active Mode Count")
        ax_line.set_ylabel("delta_rms_vs_full")
        ax_line.grid(True, alpha=0.2)

        selected_text = "Select a path step."
        if selected_row is not None:
            selected_count = int(float(selected_row.get("active_mode_count", "0")))
            selected_delta = float(selected_row.get("delta_rms_vs_full", "0"))
            ax_line.scatter([selected_count], [selected_delta], color="#dc2626", s=42, zorder=5)
            active_modes = parse_int_list_text(selected_row.get("active_modes_noll", ""))
            removed_mode = selected_row.get("removed_mode_noll", "").strip() or "None"
            selected_text = "\n".join(
                [
                    f"Selected step {selected_row.get('step_index', '')} | active_mode_count={selected_count}",
                    f"subset_rms={format_metric(float(selected_row.get('rms', '0')))} | full_fit_rms={format_metric(float(ordered_rows[0].get('rms', '0')))}",
                    f"removed_mode={removed_mode} | delta_rms_vs_full={format_metric(selected_delta)}",
                    f"active_modes={active_modes}",
                ]
            )

        ax_text.axis("off")
        ax_text.text(0.0, 1.0, selected_text, va="top", ha="left", fontsize=9, family="monospace", wrap=True)
        self.draw_idle()
        return selected_text

    def plot_mode_consistency(
        self,
        rows: list[dict[str, str]],
        selected_row: dict[str, str] | None,
        *,
        title: str,
    ) -> str:
        self.figure.clear()
        grid = self.figure.add_gridspec(2, 1, height_ratios=[1.0, 0.8])
        ax_heatmap = self.figure.add_subplot(grid[0, 0])
        ax_line = self.figure.add_subplot(grid[1, 0])

        ordered_rows = sorted(rows, key=lambda row: int(float(row.get("mode_noll", "0"))))
        metrics = [
            ("median_rank", "Median Rank"),
            ("top1_count", "Top1"),
            ("top3_count", "Top3"),
            ("top5_count", "Top5"),
            ("p95_delta_rms", "p95 dRMS"),
        ]
        matrix: list[list[float]] = []
        for field, _label in metrics:
            values = np.array([float(row.get(field, "0")) for row in ordered_rows], dtype=float)
            if values.size == 0 or np.allclose(values.max(), values.min()):
                matrix.append(np.zeros(len(ordered_rows), dtype=float).tolist())
            else:
                matrix.append(((values - values.min()) / (values.max() - values.min())).tolist())
        image = ax_heatmap.imshow(np.array(matrix), aspect="auto", cmap="magma", interpolation="nearest")
        modes = [int(float(row.get("mode_noll", "0"))) for row in ordered_rows]
        ax_heatmap.set_title(title, fontsize=11)
        ax_heatmap.set_xticks(range(len(modes)))
        ax_heatmap.set_xticklabels([str(mode) for mode in modes], rotation=90, fontsize=7)
        ax_heatmap.set_yticks(range(len(metrics)))
        ax_heatmap.set_yticklabels([label for _field, label in metrics], fontsize=8)
        self.figure.colorbar(image, ax=ax_heatmap, fraction=0.025, pad=0.02, label="Normalized metric")

        p95_values = [float(row.get("p95_delta_rms", "0")) for row in ordered_rows]
        ax_line.plot(modes, p95_values, color="#2563eb", linewidth=1.1)
        ax_line.set_xlabel("Mode (Noll)")
        ax_line.set_ylabel("p95_delta_rms")
        ax_line.grid(True, alpha=0.2)

        selected_text = "Select a mode row."
        if selected_row is not None:
            selected_mode = int(float(selected_row.get("mode_noll", "0")))
            try:
                selected_index = modes.index(selected_mode)
            except ValueError:
                selected_index = -1
            if selected_index >= 0:
                ax_heatmap.axvline(selected_index, color="#22c55e", linewidth=1.2)
            ax_line.scatter(
                [selected_mode],
                [float(selected_row.get("p95_delta_rms", "0"))],
                color="#dc2626",
                s=42,
                zorder=5,
            )
            selected_text = (
                f"Mode Z{selected_mode} | median_rank={format_metric(float(selected_row.get('median_rank', '0')))} | "
                f"top1/top3/top5={selected_row.get('top1_count', '')}/{selected_row.get('top3_count', '')}/{selected_row.get('top5_count', '')} | "
                f"p95_delta_rms={format_metric(float(selected_row.get('p95_delta_rms', '0')))}"
            )

        ax_line.text(0.01, 0.99, selected_text, transform=ax_line.transAxes, ha="left", va="top", fontsize=9)
        self.draw_idle()
        return selected_text

    def plot_global_mode_order(
        self,
        rows: list[dict[str, str]],
        *,
        prefix_count: int,
    ) -> str:
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        ordered_rows = sorted(rows, key=lambda row: int(float(row.get("global_order", "0"))))
        orders = [int(float(row.get("global_order", "0"))) for row in ordered_rows]
        modes = [int(float(row.get("mode_noll", "0"))) for row in ordered_rows]
        scores = [float(row.get("p95_delta_rms", "0")) for row in ordered_rows]
        colors = ["#2563eb" if order <= prefix_count else "#cbd5e1" for order in orders]
        axis.bar(orders, scores, color=colors)
        axis.set_title("Global Mode Order", fontsize=11)
        axis.set_xlabel("Global Removal Order")
        axis.set_ylabel("p95_delta_rms")
        axis.grid(True, axis="y", alpha=0.2)
        for order, mode, score in zip(orders, modes, scores, strict=False):
            if order <= prefix_count:
                axis.text(order, score, f"Z{mode}", rotation=90, ha="center", va="bottom", fontsize=7)
        text = f"Highlighted removal prefix: {prefix_count} modes"
        axis.text(0.01, 0.99, text, transform=axis.transAxes, ha="left", va="top", fontsize=9)
        self.draw_idle()
        return text

    def plot_global_subset_aggregate(
        self,
        rows: list[dict[str, str]],
        *,
        selected_count: int | None,
    ) -> str:
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        ordered_rows = sorted(rows, key=lambda row: int(float(row.get("removed_mode_count", "0"))))
        counts = [int(float(row.get("removed_mode_count", "0"))) for row in ordered_rows]
        median_values = [float(row.get("median_delta_rms", "0")) for row in ordered_rows]
        p95_values = [float(row.get("p95_delta_rms", "0")) for row in ordered_rows]
        max_values = [float(row.get("max_delta_rms", "0")) for row in ordered_rows]
        axis.plot(counts, median_values, color="#2563eb", linewidth=1.1, label="median_delta_rms")
        axis.plot(counts, p95_values, color="#dc2626", linewidth=1.1, label="p95_delta_rms")
        axis.plot(counts, max_values, color="#059669", linewidth=1.1, label="max_delta_rms")
        detail = "Select a removed-mode count."
        if selected_count is not None:
            match = next((row for row in ordered_rows if int(float(row.get("removed_mode_count", "0"))) == selected_count), None)
            if match is not None:
                axis.axvline(selected_count, color="#7c3aed", linewidth=1.0, alpha=0.8)
                detail = (
                    f"removed_mode_count={selected_count} | median={format_metric(float(match.get('median_delta_rms', '0')))} | "
                    f"p95={format_metric(float(match.get('p95_delta_rms', '0')))} | max={format_metric(float(match.get('max_delta_rms', '0')))}"
                )
        axis.set_title("Global Consistent Subset Aggregate", fontsize=11)
        axis.set_xlabel("Removed Mode Count")
        axis.set_ylabel("delta_rms_vs_full")
        axis.grid(True, alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
        axis.text(0.01, 0.99, detail, transform=axis.transAxes, ha="left", va="top", fontsize=9)
        self.draw_idle()
        return detail

    def plot_global_subset_source(self, rows: list[dict[str, str]], selected_row: dict[str, str] | None) -> str:
        self.figure.clear()
        grid = self.figure.add_gridspec(2, 1, height_ratios=[1.2, 0.9])
        ax_line = self.figure.add_subplot(grid[0, 0])
        ax_text = self.figure.add_subplot(grid[1, 0])

        ordered_rows = sorted(rows, key=lambda row: int(float(row.get("removed_mode_count", "0"))))
        removed_counts = [int(float(row.get("removed_mode_count", "0"))) for row in ordered_rows]
        deltas = [float(row.get("delta_rms_vs_full", "0")) for row in ordered_rows]
        ax_line.plot(removed_counts, deltas, color="#2563eb", linewidth=1.2, marker="o", markersize=3)
        ax_line.set_title("Global Consistent Subset: per-source path", fontsize=11)
        ax_line.set_xlabel("Removed Mode Count")
        ax_line.set_ylabel("delta_rms_vs_full")
        ax_line.grid(True, alpha=0.2)

        selected_text = "Select a removal step."
        if selected_row is not None:
            removed_count = int(float(selected_row.get("removed_mode_count", "0")))
            ax_line.scatter([removed_count], [float(selected_row.get("delta_rms_vs_full", "0"))], color="#dc2626", s=42)
            selected_text = "\n".join(
                [
                    f"removed_mode_count={removed_count} | active_mode_count={selected_row.get('active_mode_count', '')}",
                    f"subset_rms={format_metric(float(selected_row.get('rms', '0')))} | delta_rms_vs_full={format_metric(float(selected_row.get('delta_rms_vs_full', '0')))}",
                    f"removed_modes={parse_int_list_text(selected_row.get('removed_modes_noll', ''))}",
                    f"active_modes={parse_int_list_text(selected_row.get('active_modes_noll', ''))}",
                ]
            )

        ax_text.axis("off")
        ax_text.text(0.0, 1.0, selected_text, va="top", ha="left", fontsize=9, family="monospace", wrap=True)
        self.draw_idle()
        return selected_text


class FitPreviewCanvas(FigureCanvas):
    """Matplotlib canvas for interactive inspection of one summary row."""

    def __init__(self, parent: QWidget | None = None) -> None:
        self.figure = Figure(figsize=(9.0, 7.0), constrained_layout=True)
        super().__init__(self.figure)
        self.setParent(parent)

    def clear_message(self, message: str) -> None:
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        axis.axis("off")
        axis.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
        self.draw_idle()

    def plot_selection(
        self,
        row: dict[str, str],
        manifest: dict[str, object],
        rho_axis_limit_um: float | None = None,
    ) -> tuple[str, dict[str, str]]:
        source_file = Path(row.get("_resolved_source_file", row["source_file"]))
        coeff_file = Path(row.get("_resolved_output_coefficients_csv", row["output_coefficients_csv"]))
        metadata = parse_surface_metadata(source_file)

        reference_radius = None
        if row.get("roc_mode") != "fit-per-file":
            reference_radius = parse_optional_float(row.get("applied_reference_radius_um", ""))

        normalization_mode = row.get("normalization_mode", "per-file")
        normalization_radius = None
        if normalization_mode == "common-per-surf-id":
            normalization_radius = parse_optional_float(row.get("applied_normalization_radius_um", ""))

        maxfev = int(float(str(manifest.get("maxfev", 10000))))
        rcond_raw = manifest.get("rcond")
        rcond = None if rcond_raw in (None, "", "null") else float(str(rcond_raw))
        round_radii_um = parse_boolish(row.get("round_radii_um", manifest.get("round_radii_um")), default=False)
        zernike_coeff_sigfigs = parse_optional_int(
            row.get("zernike_coeff_sigfigs", manifest.get("zernike_coeff_sigfigs"))
        )

        x, y, z = load_xyz_point_cloud(source_file)
        fit_data = run_fit_pipeline(
            x,
            y,
            z,
            surf_id=metadata.surf_id,
            method=row.get("method", "lstsq"),
            n_modes=int(float(row.get("n_modes", "45"))),
            maxfev=maxfev,
            rcond=rcond,
            reference_radius_um=reference_radius,
            normalization_radius_um=normalization_radius,
            round_radii_um=round_radii_um,
            zernike_coeff_sigfigs=zernike_coeff_sigfigs,
        )

        coeff_meta, coeffs = parse_coefficients_csv(coeff_file)
        top_coeffs = sorted(coeffs[1:], key=lambda item: abs(item[1]), reverse=True)[:8]

        sphere_z = sphere_profile_z(
            fit_data["rho"],
            z0_fit=fit_data["z0_fit"],
            radius_um=fit_data["applied_reference_radius_um"],
            posterior_surface=uses_posterior_sign_convention(metadata.surf_id),
            np=np,
        )
        rho_meas, z_meas = radial_bin_profile(fit_data["rho"], fit_data["z"], bins=64, np=np)
        rho_sphere, z_sphere = radial_bin_profile(fit_data["rho"], sphere_z, bins=64, np=np)
        rho_fit, z_fit = radial_bin_profile(fit_data["rho"], fit_data["zernike_surface"], bins=64, np=np)
        rho_resid, z_resid = radial_bin_profile(
            fit_data["rho"], fit_data["zernike_surface_residuals"], bins=64, np=np
        )
        rho_sphere_resid, z_sphere_resid = radial_bin_profile(
            fit_data["rho"], fit_data["sphere_residuals"], bins=64, np=np
        )

        sample_stride = max(1, len(fit_data["x"]) // 3500)
        self.figure.clear()
        grid = self.figure.add_gridspec(2, 2, width_ratios=[1.2, 1.0], height_ratios=[1.1, 1.0])
        ax_profile = self.figure.add_subplot(grid[0, 0])
        ax_map = self.figure.add_subplot(grid[0, 1])
        ax_residual = self.figure.add_subplot(grid[1, 0])
        ax_sphere_residual = self.figure.add_subplot(grid[1, 1])

        ax_profile.plot(rho_meas, z_meas, color="#111827", linewidth=1.2, label="Measured")
        ax_profile.plot(rho_sphere, z_sphere, color="#2563eb", linewidth=1.0, label="Sphere")
        ax_profile.plot(rho_fit, z_fit, color="#dc2626", linewidth=1.0, label="Zernike")
        ax_profile.set_title("Radial Profile vs Fit", fontsize=10)
        ax_profile.set_xlabel("rho (um)")
        ax_profile.set_ylabel("z (um)")
        if rho_axis_limit_um is not None:
            ax_profile.set_xlim(0.0, rho_axis_limit_um)
        ax_profile.grid(True, alpha=0.2)
        ax_profile.legend(fontsize=8, frameon=False)

        scatter = ax_map.scatter(
            fit_data["x"][::sample_stride],
            fit_data["y"][::sample_stride],
            c=fit_data["z"][::sample_stride],
            s=7,
            cmap="viridis",
            linewidths=0,
        )
        ax_map.set_title("Measured Surface Map", fontsize=10)
        ax_map.set_xlabel("x (um)")
        ax_map.set_ylabel("y (um)")
        ax_map.grid(True, alpha=0.2)
        self.figure.colorbar(scatter, ax=ax_map, fraction=0.046, pad=0.04, label="z (um)")

        ax_residual.plot(rho_resid, z_resid, color="#059669", linewidth=1.0)
        ax_residual.axhline(0.0, color="#6b7280", linewidth=0.8, alpha=0.8)
        ax_residual.set_title("Zernike Residual vs Radius", fontsize=10)
        ax_residual.set_xlabel("rho (um)")
        ax_residual.set_ylabel("Residual (um)")
        if rho_axis_limit_um is not None:
            ax_residual.set_xlim(0.0, rho_axis_limit_um)
        zernike_y_min, zernike_y_max, zernike_step = snapped_axis_limits(
            fit_data["zernike_surface_residuals"]
        )
        ax_residual.set_ylim(zernike_y_min, zernike_y_max)
        ax_residual.set_yticks(np.arange(zernike_y_min, zernike_y_max + (0.5 * zernike_step), zernike_step))
        ax_residual.grid(True, alpha=0.2)

        ax_sphere_residual.plot(rho_sphere_resid, z_sphere_resid, color="#7c3aed", linewidth=1.0)
        ax_sphere_residual.axhline(0.0, color="#6b7280", linewidth=0.8, alpha=0.8)
        ax_sphere_residual.set_title("Sphere Fit Residual vs Radius", fontsize=10)
        ax_sphere_residual.set_xlabel("rho (um)")
        ax_sphere_residual.set_ylabel("Residual (um)")
        if rho_axis_limit_um is not None:
            ax_sphere_residual.set_xlim(0.0, rho_axis_limit_um)
        sphere_y_min, sphere_y_max, sphere_step = snapped_axis_limits(fit_data["sphere_residuals"])
        ax_sphere_residual.set_ylim(sphere_y_min, sphere_y_max)
        ax_sphere_residual.set_yticks(np.arange(sphere_y_min, sphere_y_max + (0.5 * sphere_step), sphere_step))
        ax_sphere_residual.grid(True, alpha=0.2)

        coeff_lines_list = [
            "    ".join(f"{name}={value:.2e}" for name, value in top_coeffs[index : index + 2])
            for index in range(0, len(top_coeffs), 2)
        ]
        coeff_lines = "\n".join(coeff_lines_list) or "No coefficients"
        text = "\n".join(
            [
                f"File: {source_file.name}",
                f"Surface: {row['surf_id']} ({row['surface_token']})",
                f"Force: {row['force_id']}",
                f"Method: {row.get('method', 'lstsq')}",
                f"ROC mode: {row.get('roc_mode', '')}",
                f"Norm mode: {normalization_mode}",
                f"Round radii: {'on' if round_radii_um else 'off'}",
                f"Coeff precision: {zernike_coeff_sigfigs} sig figs" if zernike_coeff_sigfigs else "Coeff precision: full",
                f"Fitted sphere radius: {fit_data['fitted_sphere_radius_um']:.2f} um",
                f"Applied norm radius: {fit_data['norm_radius_um']:.2f} um",
                f"Observed aperture radius: {fit_data['observed_aperture_radius_um']:.2f} um",
                f"Sphere SSE: {fit_data['sphere_sse']:.2e}",
                f"Surface RMS: {fit_data['surface_zernike_rms']:.2e}",
                f"Residual RMS: {fit_data['sphere_residual_zernike_rms']:.2e}",
                f"Coeff file: {coeff_file.name}",
                f"Coeff metadata radius: {coeff_meta.get('Norm. Radius (mm)', 'n/a')} mm",
                "",
                "Top coeffs:",
                coeff_lines,
            ]
        )

        self.figure.suptitle(
            f"{row['run_name']} | {row['force_id']} | {row['surface_token']}",
            fontsize=11,
        )
        self.draw_idle()

        details = {
            "source_file": str(source_file),
            "coeff_file": str(coeff_file),
            "sphere_sse": f"{fit_data['sphere_sse']:.2e}",
            "surface_rms": f"{fit_data['surface_zernike_rms']:.2e}",
            "residual_rms": f"{fit_data['sphere_residual_zernike_rms']:.2e}",
            "applied_norm_radius_um": f"{fit_data['norm_radius_um']:.2f}",
            "observed_aperture_radius_um": f"{fit_data['observed_aperture_radius_um']:.2f}",
        }
        return text, details


class BatchFitWindow(QMainWindow):
    """Launcher and summary viewer for the public batch-fitting release."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Batch Fit Launcher and Viewer")
        self.resize(1380, 940)
        self.process: QProcess | None = None
        self.summary_rows: list[dict[str, str]] = []
        self.summary_manifest: dict[str, object] = {}
        self.summary_workbook_path: Path | None = None
        self.common_rho_axis_limit_um: float | None = None
        self.subset_rows: list[dict[str, str]] = []
        self.subset_manifest: dict[str, object] = {}
        self.subset_workbook_path: Path | None = None
        self.subset_workbook_kind = ""
        self.subset_workbook_specs: list[dict[str, str]] = []
        self.subset_spec_paths: dict[str, Path] = {}
        self._subset_updating_spec_combo = False

        tabs = QTabWidget(self)
        self.setCentralWidget(tabs)
        tabs.addTab(self._build_runner_tab(), "Run Batch")
        tabs.addTab(self._build_viewer_tab(), "Inspect Summary")
        if ENABLE_SUBSET_INSPECTION_UI:
            tabs.addTab(self._build_subset_tab(), "Inspect Subsets")

        self.refresh_command_preview()

    def _build_runner_tab(self) -> QWidget:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)
        layout.addWidget(self._build_header())
        layout.addWidget(self._build_paths_group())
        layout.addWidget(self._build_fit_group())
        layout.addWidget(self._build_runtime_group())
        layout.addWidget(self._build_command_group(), stretch=1)
        layout.addLayout(self._build_button_row())
        return tab

    def _build_viewer_tab(self) -> QWidget:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)
        layout.addWidget(self._build_summary_loader_group())
        layout.addWidget(self._build_summary_path_remap_group())

        splitter = QSplitter(self)
        splitter.setChildrenCollapsible(False)

        left = QWidget(self)
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self._build_selection_group())
        left_layout.addWidget(self._build_details_group(), stretch=1)
        splitter.addWidget(left)

        right = QWidget(self)
        right_layout = QVBoxLayout(right)
        self.preview_canvas = FitPreviewCanvas(right)
        self.preview_canvas.clear_message("Load a batch summary workbook to inspect one fit.")
        right_layout.addWidget(self.preview_canvas, stretch=1)
        splitter.addWidget(right)
        splitter.setSizes([360, 980])

        layout.addWidget(splitter, stretch=1)
        return tab

    def _build_subset_tab(self) -> QWidget:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)
        layout.addWidget(self._build_subset_loader_group())

        splitter = QSplitter(self)
        splitter.setChildrenCollapsible(False)

        left = QWidget(self)
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self._build_subset_manifest_group())
        left_layout.addWidget(self._build_subset_selection_group())
        left_layout.addWidget(self._build_subset_table_group(), stretch=1)
        left_layout.addWidget(self._build_subset_details_group(), stretch=1)
        splitter.addWidget(left)

        right = QWidget(self)
        right_layout = QVBoxLayout(right)
        self.subset_canvas = SubsetPlotCanvas(right)
        self.subset_canvas.clear_message("Load a subset workbook to inspect analysis results.")
        right_layout.addWidget(self.subset_canvas, stretch=1)
        splitter.addWidget(right)
        splitter.setSizes([470, 870])

        layout.addWidget(splitter, stretch=1)
        return tab

    def _build_header(self) -> QWidget:
        widget = QWidget(self)
        layout = QVBoxLayout(widget)
        title = QLabel("Batch XYZ Fitter")
        title.setStyleSheet("font-size: 20px; font-weight: 600;")
        subtitle = QLabel(
            "Wrapper for batch_fit_xyz.py. The runner only includes AA/AP/PA/PP files, and the viewer can load a summary workbook to inspect individual fits live."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #475569;")
        layout.addWidget(title)
        layout.addWidget(subtitle)
        return widget

    def _build_paths_group(self) -> QGroupBox:
        group = QGroupBox("Paths", self)
        form = QFormLayout(group)

        self.input_dir_edit = QLineEdit(self)
        self.output_dir_edit = QLineEdit(str(DEFAULT_OUTPUT_DIR), self)
        self.h5_path_edit = QLineEdit("", self)
        self.run_name_edit = QLineEdit(self._default_run_name(), self)
        self.glob_edit = QLineEdit("*_FVS_*.xyz", self)

        form.addRow("Input Directory", self._with_browse_button(self.input_dir_edit, directory=True))
        form.addRow("Output Directory", self._with_browse_button(self.output_dir_edit, directory=True))
        form.addRow("HDF5 Path", self._with_browse_button(self.h5_path_edit, directory=False, save=True))
        form.addRow("Run Name", self.run_name_edit)
        form.addRow("Glob", self.glob_edit)

        for widget in (
            self.input_dir_edit,
            self.output_dir_edit,
            self.h5_path_edit,
            self.run_name_edit,
            self.glob_edit,
        ):
            widget.textChanged.connect(self.refresh_command_preview)

        return group

    def _build_fit_group(self) -> QGroupBox:
        group = QGroupBox("Fit Configuration", self)
        grid = QGridLayout(group)

        self.method_combo = QComboBox(self)
        self.method_combo.addItems(["lstsq", "curve_fit"])

        self.roc_mode_combo = QComboBox(self)
        self.roc_mode_combo.addItems(["fit-per-file", "fixed", "average-best-fit"])

        self.normalization_mode_combo = QComboBox(self)
        self.normalization_mode_combo.addItems(["per-file", "common-per-surf-id"])

        self.fixed_roc_spin = QDoubleSpinBox(self)
        self.fixed_roc_spin.setRange(0.0, 1_000_000_000.0)
        self.fixed_roc_spin.setDecimals(6)
        self.fixed_roc_spin.setValue(0.0)
        self.fixed_roc_spin.setEnabled(False)

        self.n_modes_spin = QSpinBox(self)
        self.n_modes_spin.setRange(1, 45)
        self.n_modes_spin.setValue(45)

        self.maxfev_spin = QSpinBox(self)
        self.maxfev_spin.setRange(1, 5_000_000)
        self.maxfev_spin.setValue(10000)

        self.rcond_edit = QLineEdit(self)
        self.limit_spin = QSpinBox(self)
        self.limit_spin.setRange(0, 1_000_000)
        self.limit_spin.setSpecialValueText("No limit")
        self.limit_spin.setValue(0)
        self.round_radii_check = QCheckBox("Round radii to nearest um before fitting", self)
        self.round_coeffs_check = QCheckBox("Round Zernike coefficients to 6 sig figs", self)
        self.round_radii_check.setChecked(True)
        self.round_coeffs_check.setChecked(True)

        grid.addWidget(QLabel("Method"), 0, 0)
        grid.addWidget(self.method_combo, 0, 1)
        grid.addWidget(QLabel("ROC Mode"), 0, 2)
        grid.addWidget(self.roc_mode_combo, 0, 3)
        grid.addWidget(QLabel("Fixed ROC (um)"), 1, 0)
        grid.addWidget(self.fixed_roc_spin, 1, 1)
        grid.addWidget(QLabel("Normalization Mode"), 1, 2)
        grid.addWidget(self.normalization_mode_combo, 1, 3)
        grid.addWidget(QLabel("N Modes"), 2, 0)
        grid.addWidget(self.n_modes_spin, 2, 1)
        grid.addWidget(QLabel("Max Function Evals"), 2, 2)
        grid.addWidget(self.maxfev_spin, 2, 3)
        grid.addWidget(QLabel("Least-Squares rcond"), 3, 0)
        grid.addWidget(self.rcond_edit, 3, 1)
        grid.addWidget(QLabel("Limit"), 3, 2)
        grid.addWidget(self.limit_spin, 3, 3)
        grid.addWidget(self.round_radii_check, 4, 0, 1, 2)
        grid.addWidget(self.round_coeffs_check, 4, 2, 1, 2)

        self.method_combo.currentTextChanged.connect(self.refresh_command_preview)
        self.roc_mode_combo.currentTextChanged.connect(self._on_roc_mode_changed)
        self.normalization_mode_combo.currentTextChanged.connect(self.refresh_command_preview)
        self.fixed_roc_spin.valueChanged.connect(self.refresh_command_preview)
        self.n_modes_spin.valueChanged.connect(self.refresh_command_preview)
        self.maxfev_spin.valueChanged.connect(self.refresh_command_preview)
        self.rcond_edit.textChanged.connect(self.refresh_command_preview)
        self.limit_spin.valueChanged.connect(self.refresh_command_preview)
        self.round_radii_check.stateChanged.connect(self.refresh_command_preview)
        self.round_coeffs_check.stateChanged.connect(self.refresh_command_preview)
        return group

    def _build_runtime_group(self) -> QGroupBox:
        group = QGroupBox("Runtime Options", self)
        layout = QHBoxLayout(group)

        self.recursive_check = QCheckBox("Recursive search", self)
        self.qa_report_check = QCheckBox("Generate QA report", self)
        self.qa_report_check.setChecked(True)
        self.fail_fast_check = QCheckBox("Fail fast", self)
        self.h5_enabled_check = QCheckBox("Write HDF5", self)
        self.h5_enabled_check.setChecked(True)
        self.recursive_check.setToolTip("Search subdirectories under the input directory, not just the top level.")
        self.qa_report_check.setToolTip("Write an HTML gallery with thumbnail plots for quick visual inspection.")
        self.fail_fast_check.setToolTip("Stop the batch immediately on the first file error instead of continuing.")
        self.h5_enabled_check.setToolTip("Append the batch results into the shared HDF5 file for later analysis.")

        for widget in (
            self.recursive_check,
            self.qa_report_check,
            self.fail_fast_check,
            self.h5_enabled_check,
        ):
            layout.addWidget(widget)
            widget.stateChanged.connect(self.refresh_command_preview)
        layout.addStretch(1)
        return group

    def _build_command_group(self) -> QGroupBox:
        group = QGroupBox("Command and Output", self)
        layout = QVBoxLayout(group)

        self.command_preview = QPlainTextEdit(self)
        self.command_preview.setReadOnly(True)
        self.command_preview.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.command_preview.setMaximumBlockCount(200)

        self.log_output = QPlainTextEdit(self)
        self.log_output.setReadOnly(True)
        self.log_output.setLineWrapMode(QPlainTextEdit.NoWrap)

        layout.addWidget(QLabel("Command Preview"))
        layout.addWidget(self.command_preview)
        layout.addWidget(QLabel("Process Log"))
        layout.addWidget(self.log_output, stretch=1)
        return group

    def _build_button_row(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        self.refresh_run_name_button = QPushButton("New Run Name", self)
        self.run_button = QPushButton("Run", self)
        self.stop_button = QPushButton("Stop", self)
        self.stop_button.setEnabled(False)

        self.refresh_run_name_button.clicked.connect(self._reset_run_name)
        self.run_button.clicked.connect(self.start_process)
        self.stop_button.clicked.connect(self.stop_process)

        layout.addWidget(self.refresh_run_name_button)
        layout.addStretch(1)
        layout.addWidget(self.run_button)
        layout.addWidget(self.stop_button)
        return layout

    def _build_summary_loader_group(self) -> QGroupBox:
        group = QGroupBox("Summary Workbook", self)
        layout = QHBoxLayout(group)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(6)
        self.summary_file_edit = QLineEdit(self)
        self.summary_load_button = QPushButton("Load Summary", self)
        self.summary_load_button.clicked.connect(self.load_summary_workbook)
        layout.addWidget(QLabel("Batch summary(.xlsx)"), stretch=0)
        layout.addWidget(
            self._with_browse_button(
                self.summary_file_edit,
                directory=False,
                save=False,
                xlsx=True,
                on_selected=self.load_summary_workbook,
            ),
            stretch=1,
        )
        layout.addWidget(self.summary_load_button, stretch=0)
        return group

    def _build_selection_group(self) -> QGroupBox:
        group = QGroupBox("Selection", self)
        layout = QVBoxLayout(group)

        surf_widget = QWidget(self)
        surf_layout = QHBoxLayout(surf_widget)
        surf_layout.setContentsMargins(0, 0, 0, 0)
        surf_layout.addWidget(QLabel("Surface Family"))
        self.surf_id_buttons: dict[str, QPushButton] = {}
        for surf_id in FOCUS_SURF_IDS:
            button = QPushButton(surf_id, self)
            button.setCheckable(True)
            button.setAutoExclusive(True)
            button.setEnabled(False)
            button.clicked.connect(self._refresh_force_options)
            self.surf_id_buttons[surf_id] = button
            surf_layout.addWidget(button)
        surf_layout.addStretch(1)

        force_widget = QWidget(self)
        force_layout = QHBoxLayout(force_widget)
        force_layout.setContentsMargins(0, 0, 0, 0)
        force_layout.addWidget(QLabel("Force"))
        self.prev_force_button = QPushButton("Prev", self)
        self.prev_force_button.setMaximumWidth(60)
        self.force_id_combo = QComboBox(self)
        self.force_id_combo.currentTextChanged.connect(self._refresh_surface_token_options)
        self.next_force_button = QPushButton("Next", self)
        self.next_force_button.setMaximumWidth(60)
        self.prev_force_button.clicked.connect(lambda: self._navigate_force(-1))
        self.next_force_button.clicked.connect(lambda: self._navigate_force(1))
        force_layout.addWidget(self.prev_force_button)
        force_layout.addWidget(self.force_id_combo, stretch=1)
        force_layout.addWidget(self.next_force_button)

        token_widget = QWidget(self)
        token_layout = QHBoxLayout(token_widget)
        token_layout.setContentsMargins(0, 0, 0, 0)
        token_layout.addWidget(QLabel("Surface State"))
        self.surface_suffix_buttons: dict[str, QPushButton] = {}
        for suffix, label in (("I", "Initial"), ("D", "Deformed")):
            button = QPushButton(label, self)
            button.setCheckable(True)
            button.setAutoExclusive(True)
            button.setEnabled(False)
            button.clicked.connect(self.plot_current_selection)
            self.surface_suffix_buttons[suffix] = button
            token_layout.addWidget(button)
        token_layout.addStretch(1)

        layout.addWidget(surf_widget)
        layout.addWidget(force_widget)
        layout.addWidget(token_widget)
        return group

    def _build_summary_path_remap_group(self) -> QGroupBox:
        group = QGroupBox("Viewer Path Remapping", self)
        self.summary_path_remap_group = group
        layout = QHBoxLayout(group)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(6)

        self.summary_original_source_root_edit = QLineEdit(self)
        self.summary_local_source_root_edit = QLineEdit(self)

        self.summary_original_source_root_edit.setPlaceholderText("Auto-filled from run_manifest.json or source_file paths")
        self.summary_local_source_root_edit.setPlaceholderText("Choose the local folder containing the raw .xyz files")
        self.summary_original_source_root_edit.editingFinished.connect(self._refresh_current_summary_preview_if_loaded)
        self.summary_local_source_root_edit.editingFinished.connect(self._refresh_current_summary_preview_if_loaded)
        self.summary_original_source_root_edit.setToolTip(
            "Original raw-data root recorded by the saved run. Auto-filled when available."
        )
        self.summary_local_source_root_edit.setToolTip(
            "Local raw-data root used only when the saved workbook references .xyz files from another machine."
        )

        layout.addWidget(QLabel("Original XYZ Root"), stretch=0)
        layout.addWidget(self.summary_original_source_root_edit, stretch=1)
        layout.addWidget(QLabel("Local XYZ Root"), stretch=0)
        layout.addWidget(
            self._with_browse_button(
                self.summary_local_source_root_edit,
                directory=True,
                on_selected=self._refresh_current_summary_preview_if_loaded,
            ),
            stretch=1,
        )
        return group

    def _build_details_group(self) -> QGroupBox:
        group = QGroupBox("Selection Details", self)
        layout = QVBoxLayout(group)
        self.details_output = QPlainTextEdit(self)
        self.details_output.setReadOnly(True)
        self.details_output.setLineWrapMode(QPlainTextEdit.NoWrap)
        layout.addWidget(self.details_output)
        return group

    def _build_subset_loader_group(self) -> QGroupBox:
        group = QGroupBox("Subset Workbook", self)
        layout = QHBoxLayout(group)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(6)
        self.subset_file_edit = QLineEdit(self)
        self.subset_load_button = QPushButton("Load Workbook", self)
        self.subset_load_button.clicked.connect(self.load_subset_workbook)
        layout.addWidget(QLabel("Subset workbook(.xlsx)"), stretch=0)
        layout.addWidget(
            self._with_browse_button(
                self.subset_file_edit,
                directory=False,
                save=False,
                xlsx=True,
                on_selected=self.load_subset_workbook,
            ),
            stretch=1,
        )
        layout.addWidget(self.subset_load_button, stretch=0)
        return group

    def _build_subset_manifest_group(self) -> QGroupBox:
        group = QGroupBox("Subset Run", self)
        layout = QVBoxLayout(group)
        self.subset_manifest_label = QLabel("No subset workbook loaded.")
        self.subset_manifest_label.setWordWrap(True)
        self.subset_manifest_label.setStyleSheet("color: #475569;")
        layout.addWidget(self.subset_manifest_label)
        return group

    def _build_subset_selection_group(self) -> QGroupBox:
        group = QGroupBox("Subset Selection", self)
        layout = QVBoxLayout(group)

        workbook_row = QWidget(self)
        workbook_layout = QHBoxLayout(workbook_row)
        workbook_layout.setContentsMargins(0, 0, 0, 0)
        workbook_layout.addWidget(QLabel("Workbook Kind"))
        self.subset_spec_combo = QComboBox(self)
        self.subset_spec_combo.currentIndexChanged.connect(self._on_subset_spec_changed)
        workbook_layout.addWidget(self.subset_spec_combo, stretch=1)
        layout.addWidget(workbook_row)

        surf_widget = QWidget(self)
        surf_layout = QHBoxLayout(surf_widget)
        surf_layout.setContentsMargins(0, 0, 0, 0)
        surf_layout.addWidget(QLabel("Surface Family"))
        self.subset_surf_buttons: dict[str, QPushButton] = {}
        for surf_id in FOCUS_SURF_IDS:
            button = QPushButton(surf_id, self)
            button.setCheckable(True)
            button.setAutoExclusive(True)
            button.setEnabled(False)
            button.clicked.connect(self._refresh_subset_force_options)
            self.subset_surf_buttons[surf_id] = button
            surf_layout.addWidget(button)
        surf_layout.addStretch(1)
        layout.addWidget(surf_widget)

        force_widget = QWidget(self)
        force_layout = QHBoxLayout(force_widget)
        force_layout.setContentsMargins(0, 0, 0, 0)
        force_layout.addWidget(QLabel("Force"))
        self.subset_prev_force_button = QPushButton("Prev", self)
        self.subset_prev_force_button.setMaximumWidth(60)
        self.subset_force_combo = QComboBox(self)
        self.subset_force_combo.currentTextChanged.connect(self._refresh_subset_state_options)
        self.subset_next_force_button = QPushButton("Next", self)
        self.subset_next_force_button.setMaximumWidth(60)
        self.subset_prev_force_button.clicked.connect(lambda: self._navigate_subset_force(-1))
        self.subset_next_force_button.clicked.connect(lambda: self._navigate_subset_force(1))
        force_layout.addWidget(self.subset_prev_force_button)
        force_layout.addWidget(self.subset_force_combo, stretch=1)
        force_layout.addWidget(self.subset_next_force_button)
        layout.addWidget(force_widget)

        state_widget = QWidget(self)
        state_layout = QHBoxLayout(state_widget)
        state_layout.setContentsMargins(0, 0, 0, 0)
        state_layout.addWidget(QLabel("Surface State"))
        self.subset_state_buttons: dict[str, QPushButton] = {}
        for suffix, label in (("I", "Initial"), ("D", "Deformed")):
            button = QPushButton(label, self)
            button.setCheckable(True)
            button.setAutoExclusive(True)
            button.setEnabled(False)
            button.clicked.connect(self.refresh_subset_view)
            self.subset_state_buttons[suffix] = button
            state_layout.addWidget(button)
        state_layout.addStretch(1)
        layout.addWidget(state_widget)

        filter_widget = QWidget(self)
        filter_layout = QHBoxLayout(filter_widget)
        filter_layout.setContentsMargins(0, 0, 0, 0)
        filter_layout.addWidget(QLabel("Surf Filter"))
        self.subset_group_filter_combo = QComboBox(self)
        self.subset_group_filter_combo.currentTextChanged.connect(self.refresh_subset_view)
        filter_layout.addWidget(self.subset_group_filter_combo, stretch=1)
        filter_layout.addWidget(QLabel("Removed Count"))
        self.subset_removed_count_combo = QComboBox(self)
        self.subset_removed_count_combo.currentTextChanged.connect(self._render_subset_from_current_table_selection)
        filter_layout.addWidget(self.subset_removed_count_combo, stretch=1)
        layout.addWidget(filter_widget)
        return group

    def _build_subset_table_group(self) -> QGroupBox:
        group = QGroupBox("Subset Table", self)
        layout = QVBoxLayout(group)
        self.subset_table = QTableWidget(self)
        self.subset_table.setColumnCount(0)
        self.subset_table.setRowCount(0)
        self.subset_table.setSortingEnabled(True)
        self.subset_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.subset_table.setSelectionMode(QTableWidget.SingleSelection)
        self.subset_table.itemSelectionChanged.connect(self._render_subset_from_current_table_selection)
        layout.addWidget(self.subset_table)
        return group

    def _build_subset_details_group(self) -> QGroupBox:
        group = QGroupBox("Subset Details", self)
        layout = QVBoxLayout(group)
        self.subset_details_output = QPlainTextEdit(self)
        self.subset_details_output.setReadOnly(True)
        self.subset_details_output.setLineWrapMode(QPlainTextEdit.NoWrap)
        layout.addWidget(self.subset_details_output)
        return group

    def _with_browse_button(
        self,
        line_edit: QLineEdit,
        *,
        directory: bool,
        save: bool = False,
        xlsx: bool = False,
        on_selected: Callable[[], None] | None = None,
    ) -> QWidget:
        widget = QWidget(self)
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        button = QPushButton("Browse", widget)

        def choose_path() -> None:
            if directory:
                selected = QFileDialog.getExistingDirectory(self, "Select Directory", line_edit.text() or str(SCRIPT_DIR))
            elif save:
                selected, _ = QFileDialog.getSaveFileName(
                    self,
                    "Select Output File",
                    line_edit.text() or str(SCRIPT_DIR / "results.h5"),
                    "HDF5 Files (*.h5);;All Files (*)",
                )
            else:
                selected, _ = QFileDialog.getOpenFileName(
                    self,
                    "Select File",
                    line_edit.text() or str(SCRIPT_DIR),
                    "Excel Files (*.xlsx)" if xlsx else "All Files (*)",
                )
            if selected:
                line_edit.setText(selected)
                if on_selected is not None:
                    on_selected()

        button.clicked.connect(choose_path)
        layout.addWidget(line_edit, stretch=1)
        layout.addWidget(button)
        return widget

    def _default_run_name(self) -> str:
        return datetime.now().strftime("gui_run_%Y%m%d_%H%M%S")

    def _reset_run_name(self) -> None:
        self.run_name_edit.setText(self._default_run_name())

    def _on_roc_mode_changed(self) -> None:
        self.fixed_roc_spin.setEnabled(self.roc_mode_combo.currentText() == "fixed")
        self.refresh_command_preview()

    def build_command(self) -> list[str]:
        args = [str(BATCH_SCRIPT), self.input_dir_edit.text().strip()]
        if self.glob_edit.text().strip():
            args.extend(["--glob", self.glob_edit.text().strip()])
        if self.output_dir_edit.text().strip():
            args.extend(["--output-dir", self.output_dir_edit.text().strip()])
        if self.run_name_edit.text().strip():
            args.extend(["--run-name", self.run_name_edit.text().strip()])
        if self.recursive_check.isChecked():
            args.append("--recursive")

        args.extend(["--method", self.method_combo.currentText()])
        args.extend(["--roc-mode", self.roc_mode_combo.currentText()])
        if self.roc_mode_combo.currentText() == "fixed":
            args.extend(["--fixed-roc-um", f"{self.fixed_roc_spin.value():.6f}"])
        args.extend(["--normalization-mode", self.normalization_mode_combo.currentText()])
        args.extend(["--n-modes", str(self.n_modes_spin.value())])
        args.extend(["--maxfev", str(self.maxfev_spin.value())])
        if not self.round_radii_check.isChecked():
            args.append("--no-round-radii-um")
        if not self.round_coeffs_check.isChecked():
            args.append("--no-round-zernike-coeffs")

        rcond_text = self.rcond_edit.text().strip()
        if rcond_text:
            args.extend(["--rcond", rcond_text])
        if self.limit_spin.value() > 0:
            args.extend(["--limit", str(self.limit_spin.value())])
        if self.fail_fast_check.isChecked():
            args.append("--fail-fast")
        if self.qa_report_check.isChecked():
            args.append("--qa-report")
        if self.h5_enabled_check.isChecked() and self.h5_path_edit.text().strip():
            args.extend(["--h5-path", self.h5_path_edit.text().strip()])
        return args

    def refresh_command_preview(self) -> None:
        command = [sys.executable, *self.build_command()]
        self.command_preview.setPlainText(shlex.join(command))

    def validate_inputs(self) -> bool:
        input_dir = self.input_dir_edit.text().strip()
        output_dir = self.output_dir_edit.text().strip()
        if not input_dir:
            QMessageBox.warning(self, "Missing Input", "Select an input directory.")
            return False
        if not Path(input_dir).exists():
            QMessageBox.warning(self, "Invalid Input", f"Input directory does not exist:\n{input_dir}")
            return False
        if not output_dir:
            QMessageBox.warning(self, "Missing Output", "Select an output directory.")
            return False
        if self.roc_mode_combo.currentText() == "fixed" and self.fixed_roc_spin.value() <= 0:
            QMessageBox.warning(self, "Invalid Fixed ROC", "Fixed ROC must be greater than zero.")
            return False
        if self.rcond_edit.text().strip():
            try:
                float(self.rcond_edit.text().strip())
            except ValueError:
                QMessageBox.warning(self, "Invalid rcond", "Least-squares rcond must be numeric.")
                return False
        return True

    def start_process(self) -> None:
        if self.process is not None:
            QMessageBox.information(self, "Already Running", "A batch process is already running.")
            return
        if not self.validate_inputs():
            return

        self.log_output.clear()
        self.refresh_command_preview()

        process = QProcess(self)
        process.setProgram(sys.executable)
        process.setArguments(self.build_command())
        process.setWorkingDirectory(str(SCRIPT_DIR))
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(self._consume_output)
        process.finished.connect(self._process_finished)
        process.started.connect(self._process_started)

        self.process = process
        process.start()

    def stop_process(self) -> None:
        if self.process is not None:
            self.log_output.appendPlainText("\nStopping process...\n")
            self.process.kill()

    def _process_started(self) -> None:
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.log_output.appendPlainText(f"$ {self.command_preview.toPlainText()}\n")

    def _consume_output(self) -> None:
        if self.process is None:
            return
        chunk = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if chunk:
            self.log_output.appendPlainText(chunk.rstrip("\n"))

    def _process_finished(self, exit_code: int, _exit_status: QProcess.ExitStatus) -> None:
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.log_output.appendPlainText(
            f"\nCompleted successfully with exit code {exit_code}.\n"
            if exit_code == 0
            else f"\nProcess exited with code {exit_code}.\n"
        )
        self.process = None

    def _refresh_current_summary_preview_if_loaded(self) -> None:
        if self.summary_rows:
            self.plot_current_selection()

    def _update_summary_path_remap_hint(self) -> None:
        if self.summary_workbook_path is None:
            tooltip = (
                "Coefficient paths under the original run folder are remapped automatically to the loaded workbook folder. "
                "Set Local XYZ Root only when the raw .xyz tree moved to a different absolute path."
            )
            self.summary_path_remap_group.setToolTip(tooltip)
            self.summary_original_source_root_edit.setToolTip(
                "Original raw-data root recorded by the saved run. Auto-filled when available."
            )
            self.summary_local_source_root_edit.setToolTip(
                "Local raw-data root used only when the saved workbook references .xyz files from another machine."
            )
            return

        original_run_dir = infer_original_run_dir(self.summary_workbook_path, self.summary_manifest)
        current_run_dir = self.summary_workbook_path.parent
        original_xyz_root = self.summary_original_source_root_edit.text().strip() or "n/a"
        if original_run_dir is None:
            coeff_line = f"Results folder remap: current workbook folder = {current_run_dir}"
        else:
            coeff_line = f"Results folder remap: {original_run_dir} -> {current_run_dir}"
        group_tooltip = (
            f"{coeff_line}\n"
            f"Original XYZ root: {original_xyz_root}\n"
            "Set Local XYZ Root only if the saved workbook references raw data on another machine."
        )
        self.summary_path_remap_group.setToolTip(group_tooltip)
        self.summary_original_source_root_edit.setToolTip(
            f"Original XYZ root inferred from the saved run.\nCurrent value: {original_xyz_root}"
        )
        self.summary_local_source_root_edit.setToolTip(
            f"{coeff_line}\nSet this only when the raw .xyz files live elsewhere on this machine."
        )

    def load_summary_workbook(self) -> None:
        path_text = self.summary_file_edit.text().strip()
        if not path_text:
            QMessageBox.warning(self, "Missing Summary", "Choose a batch summary workbook.")
            return
        path = Path(path_text)
        if not path.exists():
            QMessageBox.warning(self, "Missing Summary", f"Summary workbook not found:\n{path}")
            return

        rows = parse_inline_xlsx_rows(path)
        rows = [row for row in rows if row.get("surf_id") in FOCUS_SURF_IDS]
        if not rows:
            QMessageBox.warning(self, "No Rows", "No AA/AP/PA/PP rows were found in the workbook.")
            return

        self.summary_rows = rows
        self.summary_manifest = parse_run_manifest(path)
        self.summary_workbook_path = path
        inferred_source_root = infer_original_source_root(rows, self.summary_manifest)
        self.summary_original_source_root_edit.setText(str(inferred_source_root or ""))
        self._update_summary_path_remap_hint()
        observed_radii = [
            parse_optional_float(row.get("observed_aperture_radius_um", ""))
            for row in rows
            if row.get("observed_aperture_radius_um", "").strip()
        ]
        max_observed_radius = max((value for value in observed_radii if value is not None), default=None)
        self.common_rho_axis_limit_um = None
        if max_observed_radius is not None:
            self.common_rho_axis_limit_um = float(int(np.ceil(max_observed_radius / 100.0) * 100.0))
        self._refresh_surf_options()
        self.details_output.setPlainText(
            f"Loaded {len(rows)} focus rows from:\n{path}\n\nRun manifest keys:\n"
            + "\n".join(sorted(str(key) for key in self.summary_manifest))
        )
        if self.current_selected_row() is None:
            self.preview_canvas.clear_message("Summary loaded, but no matching AA/AP/PA/PP row is currently selectable.")
        else:
            self.plot_current_selection()

    def _refresh_surf_options(self) -> None:
        available = {row["surf_id"] for row in self.summary_rows}
        current = self._selected_surf_id()
        target = current if current in available else next((surf_id for surf_id in FOCUS_SURF_IDS if surf_id in available), None)
        for surf_id, button in self.surf_id_buttons.items():
            button.blockSignals(True)
            button.setEnabled(surf_id in available)
            button.setChecked(surf_id == target)
            button.blockSignals(False)
        self._refresh_force_options()

    def _refresh_force_options(self) -> None:
        surf_id = self._selected_surf_id()
        matching = [row for row in self.summary_rows if row["surf_id"] == surf_id]
        force_ids = sorted({row["force_id"] for row in matching}, key=self._force_sort_key)
        current = self.force_id_combo.currentText()
        self.force_id_combo.blockSignals(True)
        self.force_id_combo.clear()
        self.force_id_combo.addItems(force_ids)
        if current in force_ids:
            self.force_id_combo.setCurrentText(current)
        elif force_ids:
            self.force_id_combo.setCurrentIndex(0)
        self.force_id_combo.blockSignals(False)
        has_force_selection = bool(force_ids)
        self.prev_force_button.setEnabled(has_force_selection)
        self.next_force_button.setEnabled(has_force_selection)
        self._refresh_surface_token_options()

    def _refresh_surface_token_options(self) -> None:
        surf_id = self._selected_surf_id()
        force_id = self.force_id_combo.currentText()
        matching = [
            row
            for row in self.summary_rows
            if row["surf_id"] == surf_id and row["force_id"] == force_id
        ]
        available_suffixes = {row["surface_token"][-1].upper() for row in matching if row.get("surface_token")}
        current = self._selected_surface_suffix()
        target = current if current in available_suffixes else next((suffix for suffix in ("I", "D") if suffix in available_suffixes), None)
        for suffix, button in self.surface_suffix_buttons.items():
            button.blockSignals(True)
            button.setEnabled(suffix in available_suffixes)
            button.setChecked(suffix == target)
            button.blockSignals(False)
        self._update_force_navigation_buttons()
        self.plot_current_selection()

    def current_selected_row(self) -> dict[str, str] | None:
        surf_id = self._selected_surf_id()
        force_id = self.force_id_combo.currentText()
        surface_suffix = self._selected_surface_suffix()
        if not surf_id or not force_id or not surface_suffix:
            return None
        surface_token = f"{surf_id}{surface_suffix}"
        for row in self.summary_rows:
            if (
                row["surf_id"] == surf_id
                and row["force_id"] == force_id
                and row["surface_token"] == surface_token
            ):
                return row
        return None

    def plot_current_selection(self) -> None:
        row = self.current_selected_row()
        if row is None:
            self.preview_canvas.clear_message("Choose a valid AA/AP/PA/PP selection to preview a fit.")
            self.details_output.setPlainText("No matching row for the current surface family, force, and token selection.")
            return
        try:
            if self.summary_workbook_path is None:
                raise FileNotFoundError("No summary workbook is loaded.")
            preview_row, resolution_details = prepare_summary_row_for_preview(
                row,
                summary_file=self.summary_workbook_path,
                rows=self.summary_rows,
                manifest=self.summary_manifest,
                original_source_root_text=self.summary_original_source_root_edit.text(),
                local_source_root_text=self.summary_local_source_root_edit.text(),
            )
            if resolution_details["source_exists"] != "1":
                raise FileNotFoundError(
                    "Resolved source .xyz file not found.\n"
                    f"Original: {resolution_details['original_source_file']}\n"
                    f"Resolved: {resolution_details['resolved_source_file']}\n\n"
                    "If this run folder was copied from another machine, set Local XYZ Root to the matching raw-data folder."
                )
            text, details = self.preview_canvas.plot_selection(
                preview_row,
                self.summary_manifest,
                rho_axis_limit_um=self.common_rho_axis_limit_um,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Plot Failed", str(exc))
            return

        detail_lines = [
            f"Run: {row['run_name']}",
            f"Source file: {details['source_file']}",
            f"Source remap: {resolution_details['source_resolution_strategy']} | exists={resolution_details['source_exists']}",
            f"Original source file: {resolution_details['original_source_file']}",
            f"Coeff file: {details['coeff_file']}",
            f"Coeff remap: {resolution_details['coeff_resolution_strategy']} | exists={resolution_details['coeff_exists']}",
            f"Original coeff file: {resolution_details['original_coeff_file']}",
            f"Sphere SSE: {details['sphere_sse']}",
            f"Surface RMS: {details['surface_rms']}",
            f"Residual RMS: {details['residual_rms']}",
            f"Applied normalization radius (um): {details['applied_norm_radius_um']}",
            f"Observed aperture radius (um): {details['observed_aperture_radius_um']}",
            "",
            text,
        ]
        self.details_output.setPlainText("\n".join(detail_lines))

    def _selected_surf_id(self) -> str:
        for surf_id, button in self.surf_id_buttons.items():
            if button.isChecked():
                return surf_id
        return ""

    def _selected_surface_suffix(self) -> str:
        for suffix, button in self.surface_suffix_buttons.items():
            if button.isChecked():
                return suffix
        return ""

    def _navigate_force(self, step: int) -> None:
        if self.force_id_combo.count() == 0:
            return
        current_index = self.force_id_combo.currentIndex()
        if current_index < 0:
            current_index = 0
        new_index = min(max(current_index + step, 0), self.force_id_combo.count() - 1)
        if new_index != current_index:
            self.force_id_combo.setCurrentIndex(new_index)
        self._update_force_navigation_buttons()

    def _update_force_navigation_buttons(self) -> None:
        count = self.force_id_combo.count()
        current_index = self.force_id_combo.currentIndex()
        has_items = count > 0 and current_index >= 0
        self.prev_force_button.setEnabled(has_items and current_index > 0)
        self.next_force_button.setEnabled(has_items and current_index < count - 1)

    def load_subset_workbook(self) -> None:
        # Public release guard: the subset-analysis workflow is intentionally
        # disabled even though the implementation remains in the source tree.
        if not ENABLE_SUBSET_INSPECTION_UI:
            QMessageBox.information(
                self,
                "Subset Inspection Disabled",
                "Subset inspection is disabled in this public release.",
            )
            return
        path_text = self.subset_file_edit.text().strip()
        if not path_text:
            QMessageBox.warning(self, "Missing Workbook", "Choose a subset workbook.")
            return
        path = Path(path_text)
        if not path.exists():
            QMessageBox.warning(self, "Missing Workbook", f"Subset workbook not found:\n{path}")
            return

        rows = parse_inline_xlsx_rows(path)
        if not rows:
            QMessageBox.warning(self, "No Rows", "The selected subset workbook contains no rows.")
            return

        filtered_rows: list[dict[str, str]] = []
        for row in rows:
            surf_id = row.get("surf_id", "").strip()
            group_value = row.get("group_value", "").strip()
            group_type = row.get("group_type", "").strip()
            if surf_id and surf_id not in FOCUS_SURF_IDS:
                continue
            if group_type == "surf_id" and group_value and group_value not in FOCUS_SURF_IDS:
                continue
            filtered_rows.append(row)

        manifest = parse_run_manifest(path)
        self.subset_workbook_path = path
        self.subset_rows = filtered_rows
        self.subset_manifest = manifest
        self.subset_workbook_kind = detect_subset_workbook_kind(filtered_rows, path, manifest)
        self.subset_workbook_specs = list(manifest.get("gui_workbook_specs", []) or [])
        self._populate_subset_spec_combo()
        self._update_subset_manifest_label()
        self._configure_subset_controls()
        self.refresh_subset_view()

    def _populate_subset_spec_combo(self) -> None:
        self._subset_updating_spec_combo = True
        self.subset_spec_combo.blockSignals(True)
        self.subset_spec_combo.clear()
        self.subset_spec_paths = {}

        if self.subset_workbook_specs and self.subset_workbook_path is not None:
            run_dir = self.subset_workbook_path.parent
            for spec in self.subset_workbook_specs:
                file_name = str(spec.get("file", ""))
                kind = str(spec.get("kind", ""))
                description = str(spec.get("description", ""))
                label = f"{kind} | {description}" if description else kind
                target_path = run_dir / file_name
                self.subset_spec_combo.addItem(label, str(target_path))
                self.subset_spec_paths[label] = target_path
            current_index = 0
            for index in range(self.subset_spec_combo.count()):
                current_path = Path(str(self.subset_spec_combo.itemData(index)))
                if current_path == self.subset_workbook_path:
                    current_index = index
                    break
            self.subset_spec_combo.setCurrentIndex(current_index)
            self.subset_spec_combo.setEnabled(True)
        elif self.subset_workbook_path is not None:
            self.subset_spec_combo.addItem(self.subset_workbook_kind or self.subset_workbook_path.name, str(self.subset_workbook_path))
            self.subset_spec_combo.setCurrentIndex(0)
            self.subset_spec_combo.setEnabled(False)
        else:
            self.subset_spec_combo.setEnabled(False)

        self.subset_spec_combo.blockSignals(False)
        self._subset_updating_spec_combo = False

    def _on_subset_spec_changed(self) -> None:
        if self._subset_updating_spec_combo:
            return
        selected_path = self.subset_spec_combo.currentData()
        if not selected_path:
            return
        target_path = Path(str(selected_path))
        if self.subset_workbook_path is not None and target_path == self.subset_workbook_path:
            return
        self.subset_file_edit.setText(str(target_path))
        self.load_subset_workbook()

    def _update_subset_manifest_label(self) -> None:
        if not self.subset_rows:
            self.subset_manifest_label.setText("No subset workbook loaded.")
            return
        run_name = str(self.subset_manifest.get("run_name") or self.subset_rows[0].get("run_name", ""))
        normalization_mode = str(
            self.subset_manifest.get("normalization_mode") or self.subset_rows[0].get("normalization_mode", "")
        )
        targets = sorted({row.get("target", "") for row in self.subset_rows if row.get("target")})
        target = str(self.subset_manifest.get("target_pattern") or ", ".join(targets) or "n/a")
        source_count = self.subset_manifest.get("source_count")
        if source_count in (None, "", "null"):
            source_count = len({row.get("source_label", "") for row in self.subset_rows if row.get("source_label")})
        self.subset_manifest_label.setText(
            f"Run: {run_name} | Kind: {self.subset_workbook_kind} | "
            f"Normalization: {normalization_mode or 'n/a'} | Target: {target} | "
            f"Source count: {source_count or 'n/a'}"
        )

    def _configure_subset_controls(self) -> None:
        per_source_kind = self.subset_workbook_kind in {
            "drop_importance",
            "subset_path_greedy",
            "subset_path_ranked",
            "global_consistent_subset",
        }
        surf_filter_kind = self.subset_workbook_kind == "mode_consistency_by_surf_id"
        removed_count_kind = self.subset_workbook_kind in {"global_mode_order", "global_consistent_subset_aggregate"}

        if per_source_kind:
            self._refresh_subset_surf_options()
        else:
            for button in self.subset_surf_buttons.values():
                button.setEnabled(False)
                button.setChecked(False)
            self.subset_force_combo.clear()
            for button in self.subset_state_buttons.values():
                button.setEnabled(False)
                button.setChecked(False)
            self._update_subset_force_navigation_buttons()

        self.subset_group_filter_combo.blockSignals(True)
        self.subset_group_filter_combo.clear()
        if surf_filter_kind:
            self.subset_group_filter_combo.addItems(["All", *FOCUS_SURF_IDS])
            self.subset_group_filter_combo.setEnabled(True)
        else:
            self.subset_group_filter_combo.addItem("All")
            self.subset_group_filter_combo.setEnabled(False)
        self.subset_group_filter_combo.blockSignals(False)

        self.subset_removed_count_combo.blockSignals(True)
        self.subset_removed_count_combo.clear()
        if removed_count_kind:
            if self.subset_workbook_kind == "global_mode_order":
                max_count = len(self.subset_rows)
                self.subset_removed_count_combo.addItems([str(count) for count in range(0, max_count + 1)])
            else:
                counts = sorted(
                    {
                        int(float(row.get("removed_mode_count", "0")))
                        for row in self.subset_rows
                        if row.get("removed_mode_count", "").strip()
                    }
                )
                self.subset_removed_count_combo.addItems([str(count) for count in counts])
            self.subset_removed_count_combo.setEnabled(True)
            if self.subset_removed_count_combo.count() > 0:
                self.subset_removed_count_combo.setCurrentIndex(0)
        else:
            self.subset_removed_count_combo.setEnabled(False)
        self.subset_removed_count_combo.blockSignals(False)

    def _refresh_subset_surf_options(self) -> None:
        available = {row["surf_id"] for row in self.subset_rows if row.get("surf_id") in FOCUS_SURF_IDS}
        current = self._selected_subset_surf_id()
        target = current if current in available else next((surf_id for surf_id in FOCUS_SURF_IDS if surf_id in available), None)
        for surf_id, button in self.subset_surf_buttons.items():
            button.blockSignals(True)
            button.setEnabled(surf_id in available)
            button.setChecked(surf_id == target)
            button.blockSignals(False)
        self._refresh_subset_force_options()

    def _refresh_subset_force_options(self) -> None:
        surf_id = self._selected_subset_surf_id()
        matching = [row for row in self.subset_rows if row.get("surf_id") == surf_id]
        force_ids = sorted({row["force_id"] for row in matching if row.get("force_id")}, key=self._force_sort_key)
        current = self.subset_force_combo.currentText()
        self.subset_force_combo.blockSignals(True)
        self.subset_force_combo.clear()
        self.subset_force_combo.addItems(force_ids)
        if current in force_ids:
            self.subset_force_combo.setCurrentText(current)
        elif force_ids:
            self.subset_force_combo.setCurrentIndex(0)
        self.subset_force_combo.blockSignals(False)
        self._update_subset_force_navigation_buttons()
        self._refresh_subset_state_options()

    def _refresh_subset_state_options(self) -> None:
        surf_id = self._selected_subset_surf_id()
        force_id = self.subset_force_combo.currentText()
        matching = [
            row
            for row in self.subset_rows
            if row.get("surf_id") == surf_id and row.get("force_id") == force_id
        ]
        available_suffixes = {
            row["surface_token"][-1].upper() for row in matching if row.get("surface_token")
        }
        current = self._selected_subset_state_suffix()
        target = current if current in available_suffixes else next((suffix for suffix in ("I", "D") if suffix in available_suffixes), None)
        for suffix, button in self.subset_state_buttons.items():
            button.blockSignals(True)
            button.setEnabled(suffix in available_suffixes)
            button.setChecked(suffix == target)
            button.blockSignals(False)
        self._update_subset_force_navigation_buttons()
        self.refresh_subset_view()

    def _selected_subset_surf_id(self) -> str:
        for surf_id, button in self.subset_surf_buttons.items():
            if button.isChecked():
                return surf_id
        return ""

    def _selected_subset_state_suffix(self) -> str:
        for suffix, button in self.subset_state_buttons.items():
            if button.isChecked():
                return suffix
        return ""

    def _navigate_subset_force(self, step: int) -> None:
        if self.subset_force_combo.count() == 0:
            return
        current_index = self.subset_force_combo.currentIndex()
        if current_index < 0:
            current_index = 0
        new_index = min(max(current_index + step, 0), self.subset_force_combo.count() - 1)
        if new_index != current_index:
            self.subset_force_combo.setCurrentIndex(new_index)
        self._update_subset_force_navigation_buttons()

    def _update_subset_force_navigation_buttons(self) -> None:
        count = self.subset_force_combo.count()
        current_index = self.subset_force_combo.currentIndex()
        has_items = count > 0 and current_index >= 0
        self.subset_prev_force_button.setEnabled(has_items and current_index > 0)
        self.subset_next_force_button.setEnabled(has_items and current_index < count - 1)

    def _current_subset_source_rows(self) -> list[dict[str, str]]:
        surf_id = self._selected_subset_surf_id()
        force_id = self.subset_force_combo.currentText()
        suffix = self._selected_subset_state_suffix()
        if not surf_id or not force_id or not suffix:
            return []
        token = f"{surf_id}{suffix}"
        return [
            row
            for row in self.subset_rows
            if row.get("surf_id") == surf_id
            and row.get("force_id") == force_id
            and row.get("surface_token") == token
        ]

    def _subset_display_rows(self) -> list[dict[str, str]]:
        if self.subset_workbook_kind in {
            "drop_importance",
            "subset_path_greedy",
            "subset_path_ranked",
            "global_consistent_subset",
        }:
            return self._current_subset_source_rows()
        if self.subset_workbook_kind == "mode_consistency_by_surf_id":
            selected_filter = self.subset_group_filter_combo.currentText()
            if selected_filter and selected_filter != "All":
                return [row for row in self.subset_rows if row.get("group_value") == selected_filter]
        return list(self.subset_rows)

    def _table_columns_for_subset_kind(self, kind: str) -> list[tuple[str, str]]:
        if kind == "drop_importance":
            return [
                ("impact_rank", "Rank"),
                ("removed_mode_noll", "Removed"),
                ("full_fit_abs_coefficient_um", "|Coeff|"),
                ("subset_rms", "Subset RMS"),
                ("delta_rms_vs_full", "dRMS"),
                ("delta_max_abs_residual_um_vs_full", "dMax Abs"),
            ]
        if kind in {"subset_path_greedy", "subset_path_ranked"}:
            return [
                ("step_index", "Step"),
                ("active_mode_count", "Active"),
                ("removed_mode_noll", "Removed"),
                ("rms", "Subset RMS"),
                ("delta_rms_vs_full", "dRMS"),
                ("delta_max_abs_residual_um_vs_full", "dMax Abs"),
            ]
        if kind in {"mode_consistency_overall", "mode_consistency_by_surf_id"}:
            return [
                ("mode_noll", "Mode"),
                ("sample_count", "N"),
                ("median_rank", "Median Rank"),
                ("top1_count", "Top1"),
                ("top3_count", "Top3"),
                ("top5_count", "Top5"),
                ("p95_delta_rms", "p95 dRMS"),
                ("max_delta_rms", "max dRMS"),
            ]
        if kind == "global_mode_order":
            return [
                ("global_order", "Order"),
                ("mode_noll", "Mode"),
                ("median_rank", "Median Rank"),
                ("top1_count", "Top1"),
                ("top3_count", "Top3"),
                ("p95_delta_rms", "p95 dRMS"),
                ("max_delta_rms", "max dRMS"),
            ]
        if kind == "global_consistent_subset":
            return [
                ("removed_mode_count", "Removed"),
                ("active_mode_count", "Active"),
                ("last_removed_mode_noll", "Last Removed"),
                ("rms", "Subset RMS"),
                ("delta_rms_vs_full", "dRMS"),
                ("delta_max_abs_residual_um_vs_full", "dMax Abs"),
            ]
        if kind == "global_consistent_subset_aggregate":
            return [
                ("removed_mode_count", "Removed"),
                ("active_mode_count", "Active"),
                ("median_delta_rms", "Median dRMS"),
                ("p95_delta_rms", "p95 dRMS"),
                ("max_delta_rms", "max dRMS"),
                ("worst_case_surface_token", "Worst Token"),
            ]
        return []

    def _populate_subset_table(self, rows: list[dict[str, str]]) -> dict[str, str] | None:
        columns = self._table_columns_for_subset_kind(self.subset_workbook_kind)
        self.subset_table.blockSignals(True)
        self.subset_table.setSortingEnabled(False)
        self.subset_table.clear()
        self.subset_table.setColumnCount(len(columns))
        self.subset_table.setHorizontalHeaderLabels([label for _field, label in columns])
        self.subset_table.setRowCount(len(rows))

        for row_index, row in enumerate(rows):
            for column_index, (field, _label) in enumerate(columns):
                raw_value = row.get(field, "")
                numeric = parse_optional_float(str(raw_value)) if str(raw_value).strip() else None
                display = format_metric(numeric if numeric is not None else raw_value)
                sort_value: float | int | str = numeric if numeric is not None else str(raw_value)
                item = NumericTableWidgetItem(display, sort_value)
                if column_index == 0:
                    item.setData(Qt.UserRole, row)
                self.subset_table.setItem(row_index, column_index, item)

        self.subset_table.resizeColumnsToContents()
        self.subset_table.setSortingEnabled(True)
        selected_row = rows[0] if rows else None
        if rows:
            self.subset_table.selectRow(0)
        self.subset_table.blockSignals(False)
        return selected_row

    def _current_subset_table_row(self) -> dict[str, str] | None:
        selected_items = self.subset_table.selectedItems()
        if not selected_items:
            return None
        return selected_items[0].data(Qt.UserRole)

    def refresh_subset_view(self) -> None:
        if not self.subset_rows:
            self.subset_canvas.clear_message("Load a subset workbook to inspect omission results.")
            self.subset_details_output.setPlainText("No subset workbook loaded.")
            return

        display_rows = self._subset_display_rows()
        if not display_rows:
            self.subset_canvas.clear_message("No rows match the current subset filters.")
            self.subset_details_output.setPlainText("No subset rows match the current selection.")
            self.subset_table.blockSignals(True)
            self.subset_table.clear()
            self.subset_table.setRowCount(0)
            self.subset_table.setColumnCount(0)
            self.subset_table.blockSignals(False)
            return

        self._populate_subset_table(display_rows)
        self._render_subset_from_current_table_selection()

    def _render_subset_from_current_table_selection(self) -> None:
        if not self.subset_rows:
            return
        display_rows = self._subset_display_rows()
        if not display_rows:
            return

        selected_row = self._current_subset_table_row()
        if selected_row is None:
            selected_row = display_rows[0]

        detail = ""
        if self.subset_workbook_kind == "drop_importance":
            detail = self.subset_canvas.plot_drop_importance(display_rows, selected_row)
        elif self.subset_workbook_kind in {"subset_path_greedy", "subset_path_ranked"}:
            title = "Subset Path: Greedy Refit" if self.subset_workbook_kind == "subset_path_greedy" else "Subset Path: Ranked Refit"
            detail = self.subset_canvas.plot_subset_path(display_rows, selected_row, title=title)
        elif self.subset_workbook_kind in {"mode_consistency_overall", "mode_consistency_by_surf_id"}:
            title = (
                "Mode Consistency by Surface Family"
                if self.subset_workbook_kind == "mode_consistency_by_surf_id"
                else "Mode Consistency Overall"
            )
            detail = self.subset_canvas.plot_mode_consistency(display_rows, selected_row, title=title)
        elif self.subset_workbook_kind == "global_mode_order":
            prefix_text = self.subset_removed_count_combo.currentText().strip()
            prefix_count = int(prefix_text) if prefix_text else 0
            detail = self.subset_canvas.plot_global_mode_order(display_rows, prefix_count=prefix_count)
            if selected_row is not None:
                detail = (
                    f"{detail}\nSelected row: order={selected_row.get('global_order', '')}, "
                    f"mode=Z{selected_row.get('mode_noll', '')}, "
                    f"p95_delta_rms={format_metric(parse_optional_float(selected_row.get('p95_delta_rms', '')) or 0.0)}"
                )
        elif self.subset_workbook_kind == "global_consistent_subset_aggregate":
            selected_text = self.subset_removed_count_combo.currentText().strip()
            selected_count = int(selected_text) if selected_text else None
            detail = self.subset_canvas.plot_global_subset_aggregate(display_rows, selected_count=selected_count)
            if selected_row is not None:
                detail = (
                    f"{detail}\nSelected row: removed={selected_row.get('removed_mode_count', '')}, "
                    f"median={format_metric(parse_optional_float(selected_row.get('median_delta_rms', '')) or 0.0)}, "
                    f"p95={format_metric(parse_optional_float(selected_row.get('p95_delta_rms', '')) or 0.0)}"
                )
        elif self.subset_workbook_kind == "global_consistent_subset":
            detail = self.subset_canvas.plot_global_subset_source(display_rows, selected_row)
        else:
            self.subset_canvas.clear_message(f"Unsupported subset workbook kind: {self.subset_workbook_kind}")
            detail = f"Unsupported subset workbook kind: {self.subset_workbook_kind}"

        source_label = selected_row.get("source_label", "") if selected_row is not None else ""
        self.subset_details_output.setPlainText(
            "\n".join(
                [
                    f"Workbook: {self.subset_workbook_path}" if self.subset_workbook_path is not None else "Workbook:",
                    f"Kind: {self.subset_workbook_kind}",
                    f"Selected source: {source_label or 'aggregate view'}",
                    "",
                    detail,
                ]
            )
        )

    @staticmethod
    def _force_sort_key(force_id: str) -> tuple[float, str]:
        if force_id.startswith("F") and force_id.endswith("mN"):
            try:
                return (float(force_id[1:-2]), force_id)
            except ValueError:
                pass
        return (float("inf"), force_id)


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Batch Fit Launcher")
    window = BatchFitWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
