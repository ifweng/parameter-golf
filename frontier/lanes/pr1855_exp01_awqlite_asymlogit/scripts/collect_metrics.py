#!/usr/bin/env python3
"""PR #1855 wrapper around the shared frontier metrics parser."""

from __future__ import annotations

import runpy
from pathlib import Path


SHARED_COLLECTOR = (
    Path(__file__).resolve().parents[2] / "pr1787" / "scripts" / "collect_metrics.py"
)


if __name__ == "__main__":
    runpy.run_path(str(SHARED_COLLECTOR), run_name="__main__")

