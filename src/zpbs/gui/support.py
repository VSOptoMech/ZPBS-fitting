from __future__ import annotations

import json
import math

import numpy as np

METHOD_LABELS = {
    "lstsq": "Direct Least-Squares",
}
ROC_MODE_LABELS = {
    "fit-per-file": "Per-File Best Radius",
    "fixed": "Fixed Reference Radius",
    "average-best-fit": "Average Best-Fit Radius",
}
SPHERE_FIT_MODE_LABELS = {
    "legacy_lsq": "Legacy Whole-Aperture LSQ",
    "center_weighted": "Center-Weighted Sphere",
    "vertex_locked": "Vertex-Locked Sphere",
}
NORMALIZATION_MODE_LABELS = {
    "per-file": "Per-File Aperture",
    "common-per-surf-id": "Shared Aperture by Surface Family",
}
SUBSET_KIND_LABELS = {
    "drop_importance": "Single-Mode Drop Importance",
    "subset_path_greedy": "Greedy Subset Path",
    "subset_path_ranked": "Ranked Subset Path",
    "mode_consistency_overall": "Mode Consistency Overview",
    "mode_consistency_by_surf_id": "Mode Consistency by Surface Family",
    "global_mode_order": "Global Mode Order",
    "global_consistent_subset": "Global Consistent Subset",
    "global_consistent_subset_aggregate": "Global Consistent Subset Aggregate",
}


def display_label(mapping: dict[str, str], value: str) -> str:
    """Return the human-friendly display label for one persisted token."""
    return mapping.get(value, value)


def parse_optional_float(text: str) -> float | None:
    """Parse optional numeric strings from summary workbooks."""
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


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
    """Render compact numeric text for tables and details panes."""
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


def _nice_axis_step(span: float, *, target_ticks: int = 6) -> float:
    """Choose a stable 1/2/5 step size for axis tick spacing."""
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
    """Snap one axis bound outward using the bound magnitude."""
    magnitude = max(abs(float(value)), 1e-12)
    exponent = math.floor(math.log10(magnitude))
    quantum = 10 ** (exponent - 1)
    if direction == "down":
        return float(math.floor(value / quantum) * quantum)
    if direction == "up":
        return float(math.ceil(value / quantum) * quantum)
    raise ValueError(f"Unsupported snap direction: {direction}")


def snapped_axis_limits(
    values: np.ndarray, *, include_zero: bool = False, target_ticks: int = 5
) -> tuple[float, float, float]:
    """Return stable, human-readable axis bounds and a tick step."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return -1.0, 1.0, 0.5

    min_value = float(np.min(arr))
    max_value = float(np.max(arr))
    if include_zero:
        min_value = min(min_value, 0.0)
        max_value = max(max_value, 0.0)

    if math.isclose(min_value, max_value):
        base = max(abs(min_value), 1e-3)
        half_window = _nice_axis_step(base * 0.8, target_ticks=3)
        y_min = min_value - half_window
        y_max = max_value + half_window
        step = _nice_axis_step(y_max - y_min, target_ticks=target_ticks)
        return float(y_min), float(y_max), float(step)

    step = _nice_axis_step(max_value - min_value, target_ticks=target_ticks)
    y_min = float(math.floor(min_value / step) * step)
    y_max = float(math.ceil(max_value / step) * step)
    if math.isclose(y_min, y_max):
        y_min -= step
        y_max += step
    return float(y_min), float(y_max), float(step)
