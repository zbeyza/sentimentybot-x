from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

from src import config
from src.io import read_csv
from src.train import prepare_training_data, split_data


def evaluate_model() -> None:
    model_path = config.MODELS_DIR / "sentiment_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError("Model not found. Train first with --train.")

    df = read_csv(config.DATA_DIR / "tweets_labeled.csv")
    prepared = prepare_training_data(df)

    X_train, X_test, y_train, y_test = split_data(prepared.X, prepared.y)

    model = joblib.load(model_path)
    preds = model.predict(X_test)

    print(classification_report(y_test, preds, target_names=config.LABEL_ORDER))

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        preds,
        display_labels=config.LABEL_ORDER,
        ax=ax,
    )
    fig.tight_layout()
    fig.savefig(config.REPORTS_DIR / "confusion_matrix.png")
    plt.close(fig)


if __name__ == "__main__":
    evaluate_model()
