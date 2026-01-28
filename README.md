# SentimentyBot-X

## Project Overview
SentimentyBot-X is an offline-first sentiment analysis system for Turkish tweets. The project was originally developed as a final group project for the Miuul Data Scientist Bootcamp, where tweets were collected using the Twitter (X) API and manually labeled by the project team.

This repository contains an updated and refactored version of the original project, redesigned to provide a clean, reproducible machine learning workflow for training, evaluation, prediction, and time-based analysis of negative sentiment. The current version operates in offline mode using the curated and labeled datasets produced during the original project.

The refactor focuses on improving code structure, reproducibility, evaluation, and usability, while preserving the core modeling and analysis logic from the original work.

---

## Offline-First Design
While the original project used the Twitter (X) API for real-time data collection, API access is not required for this version.

By default, the system runs entirely from local CSV files stored in the `data/` directory. Optional API integration is included for completeness, but the pipeline is designed to function fully without external dependencies.

---

## Dataset Access & Placement
The datasets used in this project were originally collected via the Twitter (X) API and manually labeled by the project team as part of the Miuul Data Scientist Bootcamp final project.

Due to data usage and distribution considerations, the raw datasets are not included in this repository.

To run the project locally, place the following files under `data/`:

- `tweets_labeled.csv` — labeled training dataset  
- `tweets_21.csv` — unlabeled dataset for inference  

These files are excluded via `.gitignore` to keep the repository lightweight and to avoid distributing raw data.

**The datasets are available upon request.**

---

## How to Run End-to-End
```bash
python -m src.main --train --evaluate --predict --analysis
```

If no flags are provided, the same full pipeline runs by default:
```bash
python -m src.main
```

---

## Artifacts Produced
Models in `models/`:
- `models/sentiment_model.joblib` (trained sklearn pipeline)
- `models/label_map.json` (label encoding map)

Reports in `reports/`:
- `reports/confusion_matrix.png`
- `reports/predictions_2021.csv`
- `reports/neg_by_time_interval.png`
- `reports/neg_by_day.png`
- `reports/neg_by_season.png`

---

