import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib
from scipy.stats import ttest_ind
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score, confusion_matrix, classification_report

st.set_page_config(page_title="MovieIQ", page_icon="🎬")
st.title("🎬 MovieIQ: Predict Movie Success")

# ---------- Load data ----------
df = pd.read_csv("movies.csv")
df["genres"] = df["genres"].apply(ast.literal_eval)
df["genre_names"] = df["genres"].apply(lambda x: [g["name"] for g in x])
df["success"] = (df["revenue"] > df["budget"]).astype(int)
df["budget_log"] = np.log1p(df["budget"])

# ---------- Load trained artifacts ----------
if not os.path.exists("movieiq_model.pkl"):
    st.error("No trained model found. Run `python train_model.py` first to train and save the model.")
    st.stop()

model = joblib.load("movieiq_model.pkl")
mlb = joblib.load("movieiq_mlb.pkl")
feature_columns = joblib.load("movieiq_features.pkl")

genre_encoded = pd.DataFrame(
    mlb.transform(df["genre_names"]),
    columns=mlb.classes_,
    index=df.index
)
X_all = pd.concat([df[["budget_log", "popularity", "runtime", "vote_average"]], genre_encoded], axis=1)
X_all = X_all[feature_columns]
y_all = df["success"]

# ---------- Sidebar filters ----------
st.sidebar.header("Filter Movies")
all_genres = sorted(mlb.classes_)
genre_filter = st.sidebar.multiselect("Genre", options=all_genres)
min_votes = st.sidebar.slider("Minimum vote average", 0.0, 10.0, 5.0)

filtered_df = df[df["vote_average"] >= min_votes]
if genre_filter:
    filtered_df = filtered_df[filtered_df["genre_names"].apply(
        lambda genres: any(g in genres for g in genre_filter)
    )]

st.write(f"Showing **{len(filtered_df)}** movies matching your filters")
st.dataframe(filtered_df[["title", "genre_names", "budget", "revenue", "vote_average", "success"]].head(20))

# ---------- EDA charts ----------
st.header("Exploratory Data Analysis")
col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots()
    sns.scatterplot(data=filtered_df, x="budget", y="revenue", alpha=0.6, ax=ax)
    ax.set_title("Budget vs Revenue")
    st.pyplot(fig)
with col2:
    fig, ax = plt.subplots()
    sns.boxplot(data=filtered_df, x="success", y="popularity", ax=ax)
    ax.set_title("Popularity vs Success")
    st.pyplot(fig)

# ---------- Statistical test ----------
st.header("Statistical Test Results")
successful = df[df["success"] == 1]["popularity"]
unsuccessful = df[df["success"] == 0]["popularity"]
t_stat, p_value = ttest_ind(successful, unsuccessful)
st.write(f"**T-test (popularity vs success):** t = {t_stat:.3f}, p = {p_value:.4f}")
st.write(
    "→ Statistically significant difference in popularity between successful and unsuccessful movies."
    if p_value < 0.05 else "→ No statistically significant difference found."
)

# ---------- Model performance, honestly reported ----------
st.header("Model Performance")

y_pred = model.predict(X_all)
y_proba = model.predict_proba(X_all)[:, 1]

dummy = DummyClassifier(strategy="most_frequent").fit(X_all, y_all)
dummy_pr_auc = average_precision_score(y_all, dummy.predict_proba(X_all)[:, 1])
model_pr_auc = average_precision_score(y_all, y_proba)

st.write(f"**Model PR-AUC:** {model_pr_auc:.3f}  |  **Baseline (always predict majority class):** {dummy_pr_auc:.3f}")
st.caption(
    "PR-AUC must be compared against the baseline, not against 0 or 1 — "
    "a model that always predicts 'success' already scores close to the baseline "
    "because the dataset is imbalanced (~80% successful movies)."
)

with st.expander("See confusion matrix & classification report (on full dataset)"):
    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_all, y_pred))
    st.text(classification_report(y_all, y_pred))

# ---------- Feature importance ----------
st.subheader("What drives the prediction?")
importances = pd.Series(model.feature_importances_, index=feature_columns).sort_values(ascending=False).head(10)
fig, ax = plt.subplots()
importances.plot(kind="barh", ax=ax)
ax.invert_yaxis()
ax.set_title("Top 10 Feature Importances")
st.pyplot(fig)

# ---------- Prediction form ----------
st.header("Enter Movie Details")

budget = st.number_input("Budget", min_value=0.0, value=50_000_000.0, step=1_000_000.0)
popularity = st.number_input("Popularity", min_value=0.0, value=20.0)
runtime = st.number_input("Runtime (minutes)", min_value=0.0, value=110.0)
vote_average = st.number_input("Vote Average", min_value=0.0, max_value=10.0, value=6.5)
selected_genres = st.multiselect("Select Genres", options=mlb.classes_)

if st.button("Predict Success"):
    genre_input = pd.DataFrame(mlb.transform([selected_genres]), columns=mlb.classes_)
    input_data = pd.DataFrame({
        "budget_log": [np.log1p(budget)],
        "popularity": [popularity],
        "runtime": [runtime],
        "vote_average": [vote_average]
    })
    final_input = pd.concat([input_data, genre_input], axis=1)[feature_columns]

    prediction = model.predict(final_input)
    probability = model.predict_proba(final_input)[0]

    st.info(f"📊 Probability of Success: {probability[1]:.2%}")
    if prediction[0] == 1:
        st.success("🎉 This movie is predicted to be SUCCESSFUL!")
    else:
        st.error("❌ This movie is predicted to FAIL.")
    st.caption(
        "Note: as discussed above, this model performs only marginally above the "
        "baseline rate — treat this prediction as illustrative, not reliable."
    )
