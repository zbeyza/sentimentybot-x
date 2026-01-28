from __future__ import annotations

import argparse

from src.analysis import run_analysis
from src.evaluate import evaluate_model
from src.predict import predict_2021
from src.train import train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SentimentyBot-X offline pipeline")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    parser.add_argument("--predict", action="store_true", help="Run predictions on 2021 tweets")
    parser.add_argument("--analysis", action="store_true", help="Run time-based negative tweet analysis")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_all = not any([args.train, args.evaluate, args.predict, args.analysis])

    if run_all or args.train:
        train_model()

    if run_all or args.evaluate:
        evaluate_model()

    if run_all or args.predict:
        predict_2021()

    if run_all or args.analysis:
        run_analysis()


if __name__ == "__main__":
    main()
