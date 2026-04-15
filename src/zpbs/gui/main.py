from __future__ import annotations

import sys

from PyQt5.QtWidgets import QApplication

from .window import BatchFitWindow


def main() -> int:
    """Launch the maintained Qt GUI."""
    app = QApplication(sys.argv)
    app.setApplicationName("Batch Fit Launcher")
    window = BatchFitWindow()
    window.show()
    return app.exec_()
