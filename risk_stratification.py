import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from xgboost import XGBClassifier

print("Loading dataset...")

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

data = pd.read_csv(
    "data/GSE33000_AD_vs_control.csv",
    low_memory=False
)

# Keep real probe IDs
probe_ids = data["ID_REF"].astype(str)

# Expression values
X = data.drop(columns=["ID_REF"]).T
X = X.apply(pd.to_numeric, errors="coerce")

# Use real probe IDs
X.columns = probe_ids

# Labels
# 1 = Alzheimer's
# 0 = Non-demented
y = [1] * 310 + [0] * 157

# --------------------------------------------------
# 2. TRAIN / TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# 3. REMOVE EMPTY FEATURES
# --------------------------------------------------

valid_features = X_train.columns[
    X_train.notna().any()
]

X_train = X_train[valid_features]
X_test = X_test[valid_features]

# --------------------------------------------------
# 4. HANDLE MISSING VALUES
# --------------------------------------------------

medians = X_train.median()

X_train = X_train.fillna(medians).fillna(0)
X_test = X_test.fillna(medians).fillna(0)

# --------------------------------------------------
# 5. SELECT TOP 100 FEATURES
# --------------------------------------------------

selector = SelectKBest(
    score_func=f_classif,
    k=100
)

X_train_selected = selector.fit_transform(
    X_train,
    y_train
)

X_test_selected = selector.transform(
    X_test
)

print("Selected features:", X_train_selected.shape[1])

# --------------------------------------------------
# 6. TRAIN XGBOOST
# --------------------------------------------------

model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1
)

print("Training XGBoost...")

model.fit(
    X_train_selected,
    y_train
)

print("XGBoost trained!")

# --------------------------------------------------
# 7. CALCULATE PROBABILITY
# --------------------------------------------------

probabilities = model.predict_proba(
    X_test_selected
)[:, 1]

predictions = model.predict(
    X_test_selected
)

# --------------------------------------------------
# 8. CREATE RISK CATEGORIES
# --------------------------------------------------

risk_categories = []

for probability in probabilities:

    if probability < 0.33:
        risk_categories.append("Low")

    elif probability < 0.66:
        risk_categories.append("Moderate")

    else:
        risk_categories.append("High")

# --------------------------------------------------
# 9. CREATE RESULTS TABLE
# --------------------------------------------------

results = pd.DataFrame({
    "Sample": X_test.index,
    "Actual_Class": y_test,
    "Predicted_Class": predictions,
    "Alzheimers_Probability": probabilities,
    "Risk_Category": risk_categories
})

# Convert probability to percentage
results["Alzheimers_Probability"] = (
    results["Alzheimers_Probability"] * 100
).round(2)

# --------------------------------------------------
# 10. SAVE RESULTS
# --------------------------------------------------

results.to_csv(
    "results/risk_stratification_results.csv",
    index=False
)

print("\n===== RISK STRATIFICATION =====")

print(
    results.head(10).to_string(index=False)
)

print("\nRisk category counts:")

print(
    results["Risk_Category"].value_counts()
)

print("\nResults saved:")
print("results/risk_stratification_results.csv")

# --------------------------------------------------
# 11. CREATE GRAPH
# --------------------------------------------------

risk_counts = results["Risk_Category"].value_counts()

# Make sure categories appear in logical order
risk_counts = risk_counts.reindex(
    ["Low", "Moderate", "High"],
    fill_value=0
)

plt.figure(figsize=(7, 5))

plt.bar(
    risk_counts.index,
    risk_counts.values
)

plt.xlabel("Model Risk Category")
plt.ylabel("Number of Test Samples")
plt.title("Alzheimer's Model Risk Stratification")

plt.tight_layout()

plt.savefig(
    "results/risk_stratification.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Graph saved:")
print("results/risk_stratification.png")
