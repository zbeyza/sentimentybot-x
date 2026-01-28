from __future__ import annotations

from typing import Dict

import joblib
import pandas as pd

from src import config
from src.io import load_json, read_csv, save_csv
from src.text_preprocess import lowercase_series


def predict_2021() -> None:
    model_path = config.MODELS_DIR / "sentiment_model.joblib"
    label_map_path = config.MODELS_DIR / "label_map.json"

    if not model_path.exists() or not label_map_path.exists():
        raise FileNotFoundError("Model or label map not found. Train first with --train.")

    df = read_csv(config.DATA_DIR / "tweets_21.csv")
    if "tweet" not in df.columns:
        raise ValueError("Missing 'tweet' column in prediction data.")

    df = df.copy()
    df["tweet"] = lowercase_series(df["tweet"])

    model = joblib.load(model_path)
    label_map: Dict[str, Dict[str, str]] = load_json(label_map_path)
    id_to_label = {int(k): v for k, v in label_map["id_to_label"].items()}

    preds = model.predict(df["tweet"])
    df["pred_label_id"] = preds
    df["pred_label"] = [id_to_label.get(int(x), "unknown") for x in preds]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df["tweet"])
        class_ids = list(model.classes_)
        for idx, class_id in enumerate(class_ids):
            label_name = id_to_label.get(int(class_id), str(class_id))
            df[f"proba_{label_name}"] = proba[:, idx]

    output_path = config.REPORTS_DIR / "predictions_2021.csv"
    save_csv(df, output_path)
    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    predict_2021()
