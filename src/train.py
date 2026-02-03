from __future__ import annotations

"""Training pipeline for the sentiment model."""

from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

from src import config
from src.io import read_csv, save_json
from src.text_preprocess import lowercase_series


# Common Turkish label spellings mapped to English class names.
LABEL_ALIASES = {
    "pozitif": "positive",
    "negatif": "negative",
    "nötr": "neutral",
    "notr": "neutral",
}


@dataclass
class PreparedData:
    X: pd.Series
    y: pd.Series
    label_to_id: Dict[str, int]
    id_to_label: Dict[int, str]


def _normalize_labels(series: pd.Series) -> pd.Series:
    """Normalize label column to standardized string values."""
    if series.dtype.kind in {"i", "f"}:
        return series.map(config.LABEL_VALUE_MAP)

    normalized = series.astype(str).str.lower().map(LABEL_ALIASES).fillna(series.astype(str).str.lower())
    return normalized


def prepare_training_data(df: pd.DataFrame) -> PreparedData:
    """Clean data and return texts + encoded labels."""
    if "tweet" not in df.columns:
        raise ValueError("Missing 'tweet' column in training data.")

    label_col = "label" if "label" in df.columns else "Durum" if "Durum" in df.columns else None
    if label_col is None:
        raise ValueError("Missing label column (expected 'label' or 'Durum').")

    df = df.copy()
    # Keep preprocessing minimal to preserve original signal.
    df["tweet"] = lowercase_series(df["tweet"])
    df[label_col] = _normalize_labels(df[label_col])

    # Fixed label order keeps metrics and saved artifacts stable across runs.
    label_to_id = {label: idx for idx, label in enumerate(config.LABEL_ORDER)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    # Drop rows with missing text or labels.
    df = df.dropna(subset=["tweet", label_col])
    df[label_col] = df[label_col].map(label_to_id)

    if df[label_col].isna().any():
        raise ValueError("Found labels outside the expected set: negative/neutral/positive.")

    return PreparedData(df["tweet"], df[label_col], label_to_id, id_to_label)


def build_pipeline() -> Pipeline:
    """TF-IDF + Logistic Regression pipeline (same core model as the prototype)."""
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
            ("clf", LogisticRegression(max_iter=10000)),
        ]
    )


def split_data(X: pd.Series, y: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Stratified split to keep class balance in the holdout set."""
    return train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )


def train_model() -> Tuple[Pipeline, PreparedData]:
    """Train, evaluate quickly, and persist the model + label map."""
    df = read_csv(config.DATA_DIR / "tweets_labeled.csv")
    prepared = prepare_training_data(df)

    pipeline = build_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    cv_scores = cross_val_score(pipeline, prepared.X, prepared.y, cv=cv, scoring="accuracy")
    print(f"Cross-validated accuracy: {cv_scores.mean():.4f}")

    X_train, X_test, y_train, y_test = split_data(prepared.X, prepared.y)
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    holdout_acc = accuracy_score(y_test, preds)
    print(f"Holdout accuracy: {holdout_acc:.4f}")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, config.MODELS_DIR / "sentiment_model.joblib")
    save_json(
        {"label_to_id": prepared.label_to_id, "id_to_label": prepared.id_to_label},
        config.MODELS_DIR / "label_map.json",
    )

    return pipeline, prepared


if __name__ == "__main__":
    train_model()
