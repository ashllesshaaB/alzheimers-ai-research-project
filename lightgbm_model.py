import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Loading dataset...")

# Load clean AD vs control dataset
data = pd.read_csv(
    "data/GSE33000_AD_vs_control.csv",
    low_memory=False
)

# Expression data
X = data.drop(columns=["ID_REF"]).T
X = X.apply(pd.to_numeric, errors="coerce")

# Labels
# 1 = Alzheimer's
# 0 = Non-demented
y = [1] * 310 + [0] * 157

print("Dataset:", X.shape)

# Same train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# Remove completely empty features
valid_features = X_train.columns[X_train.notna().any()]

X_train = X_train[valid_features]
X_test = X_test[valid_features]

# Fill missing values using training medians
medians = X_train.median()

X_train = X_train.fillna(medians)
X_test = X_test.fillna(medians)

# Final safety check
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

print("Missing values handled.")

# Select top 100 features
selector = SelectKBest(
    score_func=f_classif,
    k=100
)

X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

print("Selected features:", X_train_selected.shape[1])

# LightGBM model
model = LGBMClassifier(
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

print("Training LightGBM...")

model.fit(X_train_selected, y_train)

print("LightGBM trained!")

# Predictions
y_pred = model.predict(X_test_selected)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\n===== LIGHTGBM RESULTS =====")
print("Accuracy:", round(accuracy, 4))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Non-demented", "Alzheimer's"]
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))