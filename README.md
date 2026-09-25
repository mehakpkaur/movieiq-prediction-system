# MovieIQ — Movie Success Prediction System

A machine learning project that analyzes movie metadata (budget, runtime, popularity, 
genre, vote average) to predict whether a movie's revenue will exceed its budget. 
Built as an interactive Streamlit dashboard combining EDA, statistical testing, and a 
tuned classification model — with an honest evaluation of what the model can and can't do.

## Key Finding

The model's raw accuracy (79%) is misleading on this imbalanced dataset (~80% of movies 
are "successful"). After evaluating with ROC-AUC and PR-AUC against a baseline model, the 
tuned Random Forest performs only marginally better than random guessing (ROC-AUC ≈ 0.49, 
PR-AUC 0.823 vs. a 0.807 baseline). This suggests budget, runtime, popularity, vote 
average, and genre alone don't contain enough signal to reliably predict financial 
success — real predictors like marketing spend, star power, and franchise status aren't 
present in this dataset.

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, SciPy, Matplotlib, Seaborn, Streamlit, joblib

## What's Included

- `movieiq.ipynb` — full analysis notebook: EDA, statistical testing (t-test, chi-square), 
  feature engineering, model comparison (Logistic Regression, Random Forest, Gradient 
  Boosting), cross-validation, hyperparameter tuning, and honest evaluation
- `train_model.py` — trains the final tuned model and saves it for the app to use
- `app.py` — Streamlit dashboard: filterable EDA, model performance, feature importance, 
  and a live prediction form
- `requirements.txt` — dependencies
- `movies.csv` — dataset
- `movieiq_model.pkl`, `movieiq_mlb.pkl`, `movieiq_features.pkl` — saved trained artifacts

## How to Run

```bash
pip install -r requirements.txt
python train_model.py   # retrains and saves the model (optional — .pkl files already included)
streamlit run app.py
```

## Methodology Highlights

- Fixed a class-imbalance bug where the model was always predicting "success" and getting 
  a misleadingly high accuracy — resolved with `stratify` and `class_weight="balanced"`
- Checked for data leakage by comparing a pre-release-only feature set against the full 
  feature set
- Compared 3 models with 5-fold cross-validation and tuned hyperparameters with GridSearchCV
- Evaluated with ROC-AUC and PR-AUC against a `DummyClassifier` baseline, instead of relying 
  on raw accuracy
