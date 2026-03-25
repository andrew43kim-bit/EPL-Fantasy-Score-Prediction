# FPL Points Prediction — ML Pipeline

A machine learning pipeline that predicts Fantasy Premier League (FPL) total points for Premier League players using a combination of FPL data and underlying EPL performance statistics.

---

## Overview

The goal of this project is to build a regression model that can:
- Accurately predict season-long FPL points for individual players
- Rank players by predicted output and cost efficiency (value score)

---

## Dataset

Two datasets are merged on player identity:
- **FPL dataset** — player cost, selected-by percentage, goals, assists, clean sheets, creativity, threat, minutes, etc.
- **EPL raw stats dataset** — appearances, minutes played, goals, assists broken down by home/away/overall

---

## Pipeline

### Data Cleaning & Alignment
- Normalize player names (handle accents, split/merge name formats)
- Filter low-minute players (>= 500 minutes)
- Merge FPL and EPL datasets on player identity

### Feature Engineering
- **Dropped to prevent leakage:** `bonus`, `bps` (directly contribute to total points), `influence`, `ict_index` (FPL composite metrics that can lead to circular prediction)
- **Numerical features:** performance stats (minutes, creativity, threat, clean sheets, goals, assists, etc.)
- **Categorical features:** club, position, nationality

### Preprocessing 
- Numerical: median imputation + standard scaling
- Categorical: one-hot encoding
- Combined via `ColumnTransformer` wrapped in a `Pipeline`

### Models Trained
- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree
- Random Forest

### Hyperparameter Tuning
- `RandomizedSearchCV` for Random Forest
- `GridSearchCV` for Decision Tree, Ridge, and Lasso

---

## Results

5-fold cross-validated RMSE (lower is better):

| Model                      | Mean RMSE |
|----------------------------|-----------|
| **Lasso (tuned)**          | **7.98**  |
| Ridge (baseline)           | 8.63      |
| Ridge (tuned)              | 8.68      |
| Linear Regression          | 9.37      |
| Lasso (baseline)           | 10.52     |
| Random Forest (tuned)      | 12.30     |
| Random Forest (baseline)   | 12.59     |
| Decision Tree (tuned)      | 19.14     |
| Decision Tree (baseline)   | 19.18     |

**Best model: Lasso (alpha=0.1) — RMSE of 7.98 FPL points**

The regularised linear models (Lasso, Ridge) outperform tree-based models, indicating that FPL points have a largely linear relationship with underlying performance statistics.

---

## Top Feature Importances (Tuned Random Forest)

| Feature              | Importance |
|----------------------|------------|
| selected_by_percent  | 0.2874     |
| minutes              | 0.2192     |
| clean_sheets         | 0.1142     |
| appearances_overall  | 0.0450     |
| now_cost             | 0.0431     |
| goals_scored         | 0.0313     |
| threat               | 0.0269     |

---

## Player Value Rankings

Using out-of-fold predictions, players are ranked by:

- **Predicted points** — raw output ranking
- **Value score** = predicted points / cost (£m) — identifies undervalued players

Rankings are available overall and broken down by position (GK, DEF, MID, FWD).

---

## Tech Stack

- Python 3
- pandas, numpy
- scikit-learn
- matplotlib
