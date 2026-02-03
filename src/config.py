from __future__ import annotations

"""Project-wide constants and paths.

Keeping all paths and stable configuration in one place avoids scattered magic
values across the codebase and makes the pipeline easier to adjust.
"""

from pathlib import Path

# Repository root (../ from this file).
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
SCRIPTS_DIR = BASE_DIR / "scripts"

# Reproducibility and evaluation defaults.
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Original numeric labels -> normalized strings.
LABEL_VALUE_MAP = {
    1: "positive",
    -1: "negative",
    0: "neutral",
}

# Desired class order for reports/plots.
LABEL_ORDER = ["negative", "neutral", "positive"]
