import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


df = pd.read_csv("data/indian_roads_accidents_dataset.csv")

df["festival"] = df["festival"].fillna("No Festival")

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

X = df[num_cols + cat_cols + bin_cols]

y = df["accident_severity"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ],
    remainder="passthrough"
)

print("Preprocessing Complete")