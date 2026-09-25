"""
MovieIQ - Model Training Script
Run this once (python train_model.py) to train and save the model.
app.py then loads the saved artifacts instead of retraining on every run.
"""
import ast
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, average_precision_score
)

# ---------- 1. Load & prep data ----------
df = pd.read_csv("movies.csv")
df["genres"] = df["genres"].apply(ast.literal_eval)
df["genre_names"] = df["genres"].apply(lambda x: [g["name"] for g in x])
df["success"] = (df["revenue"] > df["budget"]).astype(int)

mlb = MultiLabelBinarizer()
genre_encoded = pd.DataFrame(
    mlb.fit_transform(df["genre_names"]),
    columns=mlb.classes_,
    index=df.index
)

df["budget_log"] = np.log1p(df["budget"])

feature_columns = ["budget_log", "popularity", "runtime", "vote_average"] + list(mlb.classes_)
X = pd.concat([df[["budget_log", "popularity", "runtime", "vote_average"]], genre_encoded], axis=1)
y = df["success"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- 2. Baseline (what "no model" looks like) ----------
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
dummy_pr_auc = average_precision_score(y_test, dummy.predict_proba(X_test)[:, 1])
dummy_roc_auc = roc_auc_score(y_test, dummy.predict_proba(X_test)[:, 1])
print(f"Baseline (always predict majority class) -> ROC-AUC: {dummy_roc_auc:.3f}, PR-AUC: {dummy_pr_auc:.3f}")

# ---------- 3. Compare candidate models ----------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000),
    "Random Forest": RandomForestClassifier(class_weight="balanced", random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}
for name, m in models.items():
    scores = cross_val_score(m, X, y, cv=cv, scoring="roc_auc")
    print(f"{name}: mean ROC-AUC = {scores.mean():.3f} (+/- {scores.std():.3f})")

# ---------- 4. Tune the best model (Random Forest) ----------
param_grid = {"n_estimators": [100, 200], "max_depth": [None, 10, 20]}
grid = GridSearchCV(
    RandomForestClassifier(class_weight="balanced", random_state=42),
    param_grid, cv=cv, scoring="roc_auc"
)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
print("Best params:", grid.best_params_)

# ---------- 5. Final evaluation ----------
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print("\n--- Final Model Performance ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print("PR-AUC:", average_precision_score(y_test, y_proba))
print(f"(Baseline PR-AUC was {dummy_pr_auc:.3f} -- compare against this, not 0)")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---------- 6. Save everything app.py needs ----------
joblib.dump(best_model, "movieiq_model.pkl")
joblib.dump(mlb, "movieiq_mlb.pkl")
joblib.dump(feature_columns, "movieiq_features.pkl")
print("\nSaved: movieiq_model.pkl, movieiq_mlb.pkl, movieiq_features.pkl")
