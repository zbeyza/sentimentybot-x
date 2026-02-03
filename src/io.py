from __future__ import annotations

"""Small IO helpers to keep the rest of the code clean."""

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def ensure_dir(path: Path) -> None:
    """Create a directory tree if it doesn't already exist."""
    path.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV with a clear error if the file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    """Save a dataframe as CSV, ensuring parent folder exists."""
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def save_json(data: Dict[str, Any], path: Path) -> None:
    """Write JSON with UTF-8 encoding and readable indentation."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON with a clear error if the file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"JSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
