# F1 Podium Finish Prediction

A machine learning pipeline to predict whether a Formula 1 driver will finish on the podium (top 3) using historical race data from 2011 to 2024.

---

## Project Overview

This project frames F1 podium prediction as a **binary classification problem**. Given pre-race information about a driver, constructor, and circuit, the model predicts whether that driver will finish in the top 3.

The pipeline covers end-to-end data engineering, feature engineering, model training, evaluation, and race-by-race prediction analysis.

---

## Dataset

**Source:** [Formula 1 World Championship (1950–2024)](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020) — Kaggle

**Files used:**
- `results.csv`
- `races.csv`
- `qualifying.csv`
- `pit_stops.csv`
- `lap_times.csv`
- `drivers.csv`
- `constructors.csv`
- `driver_standings.csv`
- `constructor_standings.csv`
- `circuits.csv`
- `sprint_results.csv`

**After cleaning:** 5,980 driver-race instances across 286 races and 14 seasons (2011–2024)

---

## Problem Definition

| Property | Details |
|---|---|
| Task | Binary classification |
| Target | `podium` — 1 if driver finishes top 3, 0 otherwise |
| Class imbalance | ~14% positive (podium), ~86% negative |
| Data split | Time-based (not random) |

---

## Train / Validation / Test Split

| Split | Years | Purpose |
|---|---|---|
| Train | 2011 – 2020 | Model fitting |
| Validation | 2021 – 2022 | Hyperparameter tuning |
| Test | 2023 - 2024 | Final evaluation |

A **time-based split** was used to preserve chronological order and prevent data leakage from future races into model training.

---

## Feature Engineering

Features were constructed exclusively from **pre-race information** to prevent leakage:

| Category | Features |
|---|---|
| Race context | `grid`, `round`, `circuitId`, `laps` |
| Driver attributes | `driver_age`, `nationality`, `age_bin` |
| Constructor | `construct_nationality`, `prev_constructor_points`, `prev_constructor_position` |
| Driver standings | `prev_driver_points`, `prev_driver_position`, `prev_driver_wins` |
| Pit stop stats | `num_pitstops`, `avg_pit_duration`, `min_pit_duration` |
| Lap time stats | `avg_lap_ms`, `best_lap_ms`, `std_lap_ms`, `laps_completed_lt` |
| Temporal features | `rolling_podium_rate` (last 5 races), `circuit_starts` |

---

## Data Challenges Addressed

### Class Imbalance
Only 3 of ~20 drivers podium per race (~14% positive rate). Handled via:
- `class_weight="balanced"` for Logistic Regression and SVM
- `scale_pos_weight` for XGBoost
- Evaluation using F1-score, precision, and recall alongside accuracy

### Data Leakage
Post-race features (`position_num`, `quali_position`, final points) were explicitly removed. All temporal features use lagged/shifted values so only pre-race information is used.

### Duplicate Records
Duplicate `(raceId, driverId)` entries were found across multiple merge stages (91 in `result_df`, growing to 1,665 after downstream merges). Fixed by deduplicating at each stage before feature computation.

### Temporal Consistency
All models were evaluated on the same time-based train/val/test split.

---

## Models Evaluated

| Model | Class Imbalance Handling |
|---|---|
| Logistic Regression | `class_weight="balanced"` |
| Decision Tree (Default) | None |
| Decision Tree (Tuned) | None |
| Random Forest | None |
| KNN | None |
| SVM | `class_weight="balanced"` |
| XGBoost | `scale_pos_weight` |
| AdaBoost | None |

---

## Results

| Model | Test Accuracy | F1 (podium) | Precision | Recall |
|---|---|---|---|---|
| **AdaBoost** | **0.9467** | **0.8165** | **0.8450** | 0.7899 |
| Decision Tree (Tuned) | 0.9402 | 0.8014 | 0.7986 | 0.8043 |
| XGBoost | 0.9325 | 0.7905 | 0.7405 | 0.8478 |
| Random Forest | 0.9184 | 0.6939 | 0.7944 | 0.6159 |
| KNN | 0.8792 | 0.5779 | 0.6080 | 0.5507 |
| Decision Tree | 0.8662 | 0.6168 | 0.5410 | 0.7174 |
| SVM | 0.8477 | 0.5954 | 0.4952 | 0.7464 |
| Logistic Regression | 0.8292 | 0.6045 | 0.4633 | **0.8696** |

**AdaBoost** achieved the best overall balance — highest accuracy, F1, and precision — while showing the smallest gap between training (95.7%) and test accuracy (94.9%), indicating strong generalization with no overfitting.

**XGBoost** showed a training accuracy of 99.1% vs test accuracy of 93.1% — a clear sign of overfitting despite strong recall.

---

## Race-by-Race Prediction Analysis

AdaBoost's predictions were evaluated race-by-race across all 2023+ test races using `predict_proba` to rank drivers within each race and identify the predicted top 3.

**Results across 46 test races:**

| Metric | Value |
|---|---|
| Perfect predictions (3/3) | 32 races |
| Good predictions (2/3) | 13 races |
| Partial predictions (1/3) | 1 race |
| Miss (0/3) | 0 races |
| Overall driver hit rate | 89.1% |

The model reliably identified the dominant drivers of the 2023–2024 seasons. Failures were primarily in races with atypical outcomes driven by incidents or strategy anomalies not captured by historical features.

---

## How to Run

1. Clone the repository
2. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020) and place CSV files in `formula-1-world-championship-1950-2020/`
3. Open `f1.ipynb` in Jupyter or VS Code
4. Run all cells top to bottom

---

## Authors

- Dylan Forrest
- Hari Varshan Dharmendra Mohan Prabu
- Chahat Grover
- Rahul Reddy Kesireddy

**Arizona State University — CSE 572 Data Mining**
