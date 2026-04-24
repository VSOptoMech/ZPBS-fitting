from __future__ import annotations

import sys

from PyQt5.QtWidgets import QApplication

from ..io.remap import prepare_summary_row_for_preview
from ..io.workbook import parse_inline_xlsx_rows
from .canvases import FitPreviewCanvas, NumericTableWidgetItem, OverviewPlotCanvas, SubsetPlotCanvas
from .support import (
    NORMALIZATION_MODE_LABELS,
    ROC_MODE_LABELS,
    SPHERE_FIT_MODE_LABELS,
    SUBSET_KIND_LABELS,
    display_label,
    format_metric,
    parse_int_list_text,
    parse_optional_float,
    snapped_axis_limits,
)
from .window import BatchFitWindow


def main() -> int:
    """Launch the maintained Qt GUI."""
    app = QApplication(sys.argv)
    app.setApplicationName("Batch Fit Launcher")
    window = BatchFitWindow()
    window.show()
    return app.exec_()


__all__ = [
    "BatchFitWindow",
    "FitPreviewCanvas",
    "NumericTableWidgetItem",
    "OverviewPlotCanvas",
    "SubsetPlotCanvas",
    "main",
    "parse_inline_xlsx_rows",
    "prepare_summary_row_for_preview",
    "NORMALIZATION_MODE_LABELS",
    "ROC_MODE_LABELS",
    "SPHERE_FIT_MODE_LABELS",
    "SUBSET_KIND_LABELS",
    "display_label",
    "format_metric",
    "parse_int_list_text",
    "parse_optional_float",
    "snapped_axis_limits",
]
