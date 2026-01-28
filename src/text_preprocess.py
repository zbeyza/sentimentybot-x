from __future__ import annotations

import pandas as pd


def normalize_text(text: str) -> str:
    return str(text).lower().strip()


def lowercase_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower()
