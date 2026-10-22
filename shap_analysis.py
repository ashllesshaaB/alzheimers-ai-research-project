import pandas as pd
import shap
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

# Expression data
X = data.drop(columns=["ID_REF"]).T
X = X.apply(pd.to_numeric, errors="coerce")

# Use real probe IDs as feature names
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

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# --------------------------------------------------
# 3. REMOVE EMPTY FEATURES
# --------------------------------------------------

valid_features = X_train.columns[X_train.notna().any()]

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

# Get real probe IDs
selected_probe_ids = X_train.columns[
    selector.get_support()
]

# Convert selected arrays back to DataFrames
X_train_selected = pd.DataFrame(
    X_train_selected,
    columns=selected_probe_ids,
    index=X_train.index
)

X_test_selected = pd.DataFrame(
    X_test_selected,
    columns=selected_probe_ids,
    index=X_test.index
)

print("Selected features:", len(selected_probe_ids))

# --------------------------------------------------
# 6. LOAD GENE ANNOTATION
# --------------------------------------------------

print("Loading gene annotation...")

mapping = pd.read_csv(
    "results/selected_probe_gene_mapping.csv",
    low_memory=False
)

# Create probe → gene dictionary
probe_to_gene = dict(
    zip(
        mapping["ID"].astype(str),
        mapping["Gene symbol"]
    )
)

# Create readable feature names
gene_names = []

for probe in selected_probe_ids:

    gene = probe_to_gene.get(str(probe))

    if pd.isna(gene) or not gene:
        gene = "Probe_" + str(probe)

    gene_names.append(
        str(gene)
    )

# Make sure names are unique
used_names = {}

final_names = []

for name in gene_names:

    if name not in used_names:
        used_names[name] = 1
        final_names.append(name)

    else:
        used_names[name] += 1
        final_names.append(
            name + "_" + str(used_names[name])
        )

X_train_selected.columns = final_names
X_test_selected.columns = final_names

print("Gene names assigned.")

# --------------------------------------------------
# 7. TRAIN XGBOOST
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
# 8. SHAP
# --------------------------------------------------

print("Calculating SHAP values...")

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(
    X_test_selected
)

print("SHAP calculation complete!")

# --------------------------------------------------
# 9. SHAP SUMMARY PLOT
# --------------------------------------------------

plt.figure()

shap.summary_plot(
    shap_values,
    X_test_selected,
    max_display=20,
    show=False
)

plt.tight_layout()

plt.savefig(
    "results/shap_gene_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\n===== COMPLETE =====")
print("SHAP gene summary saved!")
print("File: results/shap_gene_summary.png")