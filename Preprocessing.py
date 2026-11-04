import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================
# LOAD DATASET
# ==========================

df = pd.read_csv("indian_roads_accidents_dataset.csv"
)

# ==========================
# HANDLE MISSING VALUES
# ==========================

df["festival"] = df["festival"].fillna(
    "No Festival"
)

# ==========================
# FEATURES
# ==========================

num_cols = [
    "hour",
    "lanes",
    "temperature",
    "vehicles_involved",
    "casualties",
    "risk_score"
]

cat_cols = [
    "day_of_week",
    "road_type",
    "weather",
    "visibility",
    "traffic_density",
    "festival"
]

bin_cols = [
    "is_weekend",
    "traffic_signal",
    "is_peak_hour"
]

X = df[
    num_cols +
    cat_cols +
    bin_cols
]

y = df["accident_severity"]

# ==========================
# SPLIT DATA
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# ==========================
# PREPROCESSING
# ==========================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            num_cols
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            cat_cols
        )
    ],
    remainder="passthrough"
)

# ==========================
# MODEL
# ==========================

model = RandomForestClassifier(
    n_estimators=20,
    max_depth=8,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# ==========================
# PIPELINE
# ==========================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ]
)

# ==========================
# TRAIN
# ==========================

print("Training Model...")

pipeline.fit(
    X_train,
    y_train
)

# ==========================
# EVALUATION
# ==========================

y_pred = pipeline.predict(X_test)

print("\nAccuracy:")
print(
    accuracy_score(
        y_test,
        y_pred
    )
)

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred
    )
)

print("\nConfusion Matrix:\n")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

# ==========================
# SAVE MODEL
# ==========================

os.makedirs(
    "model",
    exist_ok=True
)

joblib.dump(
    pipeline,
    "road_accident_model.pkl",
    compress=3
)

print("\n Model Saved Successfully")
print("Location: road_accident_model.pkl")
