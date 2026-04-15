"""Compatibility wrapper for the maintained GUI package."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("MPLCONFIGDIR", str((ROOT_DIR / ".matplotlib").resolve()))

from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import QProcess

from zpbs.gui import (  # noqa: E402
    BatchFitWindow,
    main,
    parse_inline_xlsx_rows,
    prepare_summary_row_for_preview,
    snapped_axis_limits,
)

__all__ = [
    "QApplication",
    "QMessageBox",
    "QProcess",
    "BatchFitWindow",
    "main",
    "parse_inline_xlsx_rows",
    "prepare_summary_row_for_preview",
    "snapped_axis_limits",
]


if __name__ == "__main__":
    raise SystemExit(main())
