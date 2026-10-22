import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

print("Loading dataset...")

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

data = pd.read_csv(
    "data/GSE33000_AD_vs_control.csv",
    low_memory=False
)

# Expression matrix
X = data.drop(columns=["ID_REF"]).T
X = X.apply(pd.to_numeric, errors="coerce")

# Labels
# 1 = Alzheimer's
# 0 = Non-demented
y = np.array([1] * 310 + [0] * 157)

print("Dataset:", X.shape)

# --------------------------------------------------
# 2. REMOVE COMPLETELY EMPTY FEATURES
# --------------------------------------------------

X = X.dropna(axis=1, how="all")

print("After removing empty features:", X.shape)

# --------------------------------------------------
# 3. DEFINE MODELS
# --------------------------------------------------

models = {

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    ),

    "LightGBM": LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        num_leaves=15,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
        n_jobs=-1
    )
}

# --------------------------------------------------
# 4. 5-FOLD CROSS VALIDATION
# --------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []

for name, model in models.items():

    print("\n================================")
    print("Testing:", name)
    print("================================")

    pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),

        (
            "feature_selection",
            SelectKBest(
                score_func=f_classif,
                k=100
            )
        ),

        (
            "model",
            model
        )
    ])

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=[
            "accuracy",
            "precision",
            "recall",
            "f1"
        ],
        n_jobs=1
    )

    accuracy = scores["test_accuracy"]
    precision = scores["test_precision"]
    recall = scores["test_recall"]
    f1 = scores["test_f1"]

    print(
        "Accuracy:",
        round(accuracy.mean(), 4),
        "+/-",
        round(accuracy.std(), 4)
    )

    print(
        "Precision:",
        round(precision.mean(), 4)
    )

    print(
        "Recall:",
        round(recall.mean(), 4)
    )

    print(
        "F1:",
        round(f1.mean(), 4)
    )

    results.append({
        "Model": name,
        "Accuracy_Mean": accuracy.mean(),
        "Accuracy_STD": accuracy.std(),
        "Precision": precision.mean(),
        "Recall": recall.mean(),
        "F1": f1.mean()
    })

# --------------------------------------------------
# 5. SAVE RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.round(4)

results_df.to_csv(
    "results/cross_validation_results.csv",
    index=False
)

print("\n================================")
print("FINAL CROSS-VALIDATION RESULTS")
print("================================")

print(results_df.to_string(index=False))

print("\nSaved:")
print("results/cross_validation_results.csv")