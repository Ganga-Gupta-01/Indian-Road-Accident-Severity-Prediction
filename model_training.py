from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

model1 = LogisticRegression(
    class_weight="balanced",
    max_iter=3000,
    random_state=42
)

model2 = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model3 = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

ensemble = VotingClassifier(
    estimators=[
        ("lr", model1),
        ("rf", model2),
        ("xgb", model3)
    ],
    voting="soft",
    n_jobs=-1
)