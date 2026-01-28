from __future__ import annotations

from typing import Dict, List

import pandas as pd

TURKISH_MONTH_MAP: Dict[str, str] = {
    "December": "Aralık",
    "January": "Ocak",
    "February": "Şubat",
    "March": "Mart",
    "April": "Nisan",
    "May": "Mayıs",
    "June": "Haziran",
    "July": "Temmuz",
    "August": "Ağustos",
    "September": "Eylül",
    "October": "Ekim",
    "November": "Kasım",
}

SEASON_MAP: Dict[str, str] = {
    "Ocak": "Kış",
    "Şubat": "Kış",
    "Mart": "İlkbahar",
    "Nisan": "İlkbahar",
    "Mayıs": "İlkbahar",
    "Haziran": "Yaz",
    "Temmuz": "Yaz",
    "Ağustos": "Yaz",
    "Eylül": "Sonbahar",
    "Ekim": "Sonbahar",
    "Kasım": "Sonbahar",
    "Aralık": "Kış",
}

TURKISH_DAY_MAP: Dict[str, str] = {
    "Monday": "Pazartesi",
    "Tuesday": "Salı",
    "Wednesday": "Çarşamba",
    "Thursday": "Perşembe",
    "Friday": "Cuma",
    "Saturday": "Cumartesi",
    "Sunday": "Pazar",
}

TIME_INTERVAL_MAP: Dict[str, str] = {
    "0-2": "22-02",
    "22-24": "22-02",
    "2-4": "02-06",
    "4-6": "02-06",
    "6-8": "06-10",
    "8-10": "06-10",
    "10-12": "10-14",
    "12-14": "10-14",
    "14-16": "14-18",
    "16-18": "14-18",
    "18-20": "18-22",
    "20-22": "18-22",
}

TIME_INTERVAL_ORDER: List[str] = [
    "22-02",
    "02-06",
    "06-10",
    "10-14",
    "14-18",
    "18-22",
]

DAY_ORDER: List[str] = [
    "Pazartesi",
    "Salı",
    "Çarşamba",
    "Perşembe",
    "Cuma",
    "Cumartesi",
    "Pazar",
]

SEASON_ORDER: List[str] = ["Kış", "İlkbahar", "Yaz", "Sonbahar"]


def _ensure_istanbul_timezone(series: pd.Series) -> pd.Series:
    series = pd.to_datetime(series)
    if series.dt.tz is None:
        series = series.dt.tz_localize("UTC")
    series = series.dt.tz_convert("Europe/Istanbul")
    return series.dt.tz_localize(None)


def add_time_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    if date_col not in df.columns:
        raise ValueError(f"Missing date column: {date_col}")

    df = df.copy()
    df[date_col] = _ensure_istanbul_timezone(df[date_col])

    df["month"] = df[date_col].dt.month_name().replace(TURKISH_MONTH_MAP)
    df["seasons"] = df["month"].map(SEASON_MAP)

    df["days"] = df[date_col].dt.strftime("%A").replace(TURKISH_DAY_MAP)

    df["hour"] = df[date_col].dt.hour
    df["4hour_interval"] = (df["hour"] // 2) * 2

    interval = {
        0: "0-2",
        2: "2-4",
        4: "4-6",
        6: "6-8",
        8: "8-10",
        10: "10-12",
        12: "12-14",
        14: "14-16",
        16: "16-18",
        18: "18-20",
        20: "20-22",
        22: "22-24",
    }

    df["4hour_interval"] = df["4hour_interval"].map(interval)
    df["time_interval"] = df["4hour_interval"].replace(TIME_INTERVAL_MAP)

    df.drop(["4hour_interval", "hour"], axis=1, inplace=True)
    return df
