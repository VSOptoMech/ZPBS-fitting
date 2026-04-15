"""Compatibility wrapper for the maintained batch package."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from zpbs.batch_api import *  # noqa: F401,F403,E402
from zpbs.batch_api import __all__  # noqa: F401,E402
from zpbs.cli.batch_cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
