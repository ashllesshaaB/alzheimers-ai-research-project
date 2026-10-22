import pandas as pd

data = pd.read_csv(
    "data/GSE33000_AD_vs_control.csv",
    low_memory=False
)

X = data.drop(columns=["ID_REF"]).T
X = X.apply(pd.to_numeric, errors="coerce")

y = [1] * 310 + [0] * 157

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

valid_features = X_train.columns[X_train.notna().any()]

X_train = X_train[valid_features]
X_test = X_test[valid_features]

medians = X_train.median()

X_train = X_train.fillna(medians).fillna(0)
X_test = X_test.fillna(medians).fillna(0)

selector = SelectKBest(f_classif, k=100)

selector.fit(X_train, y_train)

selected = X_train.columns[selector.get_support()]

print("Selected probe IDs:")
for probe in selected:
    print(probe)