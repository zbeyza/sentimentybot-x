from __future__ import annotations

"""Text normalization helpers kept intentionally small and explicit."""

import pandas as pd


def normalize_text(text: str) -> str:
    """Lowercase and trim a single text value safely."""
    return str(text).lower().strip()


def lowercase_series(series: pd.Series) -> pd.Series:
    """Lowercase an entire pandas Series."""
    return series.astype(str).str.lower()
