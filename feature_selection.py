import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif

print("Loading clean dataset...")

data = pd.read_csv(
    "data/GSE33000_AD_vs_control.csv",
    low_memory=False
)

# Expression data
X = data.drop(columns=["ID_REF"]).T
X = X.apply(pd.to_numeric, errors="coerce")

# Labels
y = [1] * 310 + [0] * 157

print("Original X shape:", X.shape)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# Remove features that are completely missing
valid_features = X_train.columns[X_train.notna().any()]

X_train = X_train[valid_features]
X_test = X_test[valid_features]

print("After removing empty features:", X_train.shape)

# Fill remaining missing values using training medians
medians = X_train.median()

X_train = X_train.fillna(medians)
X_test = X_test.fillna(medians)

# Safety check
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
print("Feature selection complete!")