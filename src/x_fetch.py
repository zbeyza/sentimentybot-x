from __future__ import annotations

"""Optional X (Twitter) API integration placeholder."""

import os

import pandas as pd
from dotenv import load_dotenv


def fetch_from_x(query: str, max_results: int = 10) -> pd.DataFrame:
    """Return an empty DataFrame unless a token is configured.

    This keeps the project fully offline-first while leaving a clear hook
    for future API work.
    """
    load_dotenv()
    token = os.getenv("X_BEARER_TOKEN")
    if not token:
        print("No API key found. Running in offline mode using CSVs.")
        return pd.DataFrame()

    # Placeholder for X API integration.
    # This function intentionally returns an empty DataFrame to keep offline-first behavior.
    print("X API token found, but online fetching is not implemented yet.")
    return pd.DataFrame()
