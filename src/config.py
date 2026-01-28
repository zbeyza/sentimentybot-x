from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
SCRIPTS_DIR = BASE_DIR / "scripts"

RANDOM_STATE = 42
TEST_SIZE = 0.2

LABEL_VALUE_MAP = {
    1: "positive",
    -1: "negative",
    0: "neutral",
}

LABEL_ORDER = ["negative", "neutral", "positive"]
