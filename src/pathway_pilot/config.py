"""Project configuration."""

from __future__ import annotations

import os
from pathlib import Path


DEV_DATA_DIR = Path(
    os.getenv("PATHWAY_PILOT_DEV_DATA_DIR", r"C:\Users\B510067\dev_data")
)

