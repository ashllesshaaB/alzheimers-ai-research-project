import os
import joblib
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier


# ==========================================
# PATHS
# ==========================================

data_file = "data/GSE33000_AD_vs_control.csv"
model_dir = "models"

os.makedirs(model_dir, exist_ok=True)


# ==========================================
# LOAD DATA
# ==========================================

print("Loading dataset...")

data = pd.read_csv(data_file)

print("Dataset:", data.shape)


# ==========================================
# PREPARE X
# ==========================================

# Each row = one probe
# Each column = one sample

X = data.drop(columns=["ID_REF"]).T

X.columns = data["ID_REF"].astype(str)

print("X shape:", X.shape)


# ==========================================
# CREATE LABELS FROM VERIFIED DATASET ORDER
# ==========================================

# 310 Alzheimer's + 157 non-demented

y = [1] * 310 + [0] * 157

print("Number of labels:", len(y))


# ==========================================
# BUILD FINAL PIPELINE
# ==========================================

print("\nBuilding final XGBoost pipeline...")

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
        XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss"
        )
    )
])


# ==========================================
# TRAIN
# ==========================================

print("Training final model...")

pipeline.fit(X, y)

print("Training complete!")


# ==========================================
# SAVE MODEL
# ==========================================

model_path = os.path.join(
    model_dir,
    "final_xgboost_pipeline.pkl"
)

joblib.dump(
    pipeline,
    model_path
)


# ==========================================
# SAVE FEATURE INFORMATION
# ==========================================

selector = pipeline.named_steps["feature_selection"]

selected_indices = selector.get_support(indices=True)

selected_features = X.columns[selected_indices].tolist()

features_path = os.path.join(
    model_dir,
    "selected_features.pkl"
)

joblib.dump(
    selected_features,
    features_path
)


# ==========================================
# SAVE MODEL INFORMATION
# ==========================================

model_info = {
    "model": "XGBoost",
    "samples": 467,
    "alzheimers_samples": 310,
    "control_samples": 157,
    "original_features": X.shape[1],
    "selected_features": 100,
    "cv_accuracy": 0.9209,
    "cv_accuracy_std": 0.0281,
    "cv_precision": 0.9278,
    "cv_recall": 0.9548,
    "cv_f1": 0.9410
}

info_path = os.path.join(
    model_dir,
    "model_info.pkl"
)

joblib.dump(
    model_info,
    info_path
)


# ==========================================
# DONE
# ==========================================

print("\n================================")
print("FINAL MODEL SAVED")
print("================================")

print("\nModel:")
print(model_path)

print("\nSelected features:")
print(features_path)

print("\nModel information:")
print(info_path)

print("\nSelected feature count:", len(selected_features))

print("\nFinal model is ready for the demo!")