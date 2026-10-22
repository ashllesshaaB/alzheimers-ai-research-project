import gzip
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif

print("Loading expression dataset...")

# --------------------------------------------------
# 1. LOAD CLEAN EXPRESSION DATA
# --------------------------------------------------

data = pd.read_csv(
    "data/GSE33000_AD_vs_control.csv",
    low_memory=False
)

# Keep the real probe IDs
probe_ids = data["ID_REF"].astype(str)

# Expression values
X = data.drop(columns=["ID_REF"]).T
X = X.apply(pd.to_numeric, errors="coerce")

# Make the real probe IDs the feature names
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
# 3. REMOVE EMPTY PROBES
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
# 5. SELECT TOP 100 PROBES
# --------------------------------------------------

selector = SelectKBest(
    score_func=f_classif,
    k=100
)

selector.fit(X_train, y_train)

selected_probe_ids = X_train.columns[selector.get_support()]

print("\nSelected real probe IDs:")

for probe in selected_probe_ids:
    print(probe)

print("\nTotal selected probes:", len(selected_probe_ids))

# --------------------------------------------------
# 6. LOAD GPL4372 ANNOTATION
# --------------------------------------------------

print("\nLoading GPL4372 annotation...")

annotation_file = "data/GPL4372.annot.gz"

with gzip.open(
    annotation_file,
    "rt",
    encoding="utf-8",
    errors="replace"
) as f:
    lines = f.readlines()

# Find annotation table
start = None
end = None

for i, line in enumerate(lines):

    if line.startswith("!platform_table_begin"):
        start = i + 1

    elif line.startswith("!platform_table_end"):
        end = i
        break

if start is None or end is None:
    print("Could not find annotation table.")
    raise SystemExit

# Read annotation table
annotation = pd.read_csv(
    annotation_file,
    sep="\t",
    skiprows=start,
    nrows=end - start,
    compression="gzip",
    low_memory=False
)

print("Annotation loaded:", annotation.shape)

# --------------------------------------------------
# 7. CHECK ANNOTATION COLUMNS
# --------------------------------------------------

print("\nAnnotation columns:")
print(annotation.columns.tolist())

# Convert IDs to strings
annotation["ID"] = annotation["ID"].astype(str)

# --------------------------------------------------
# 8. MATCH OUR 100 PROBES
# --------------------------------------------------

mapping = annotation[
    annotation["ID"].isin(selected_probe_ids)
].copy()

# Keep useful gene information
columns_to_keep = [
    "ID",
    "Gene symbol",
    "Gene ID",
    "Gene title"
]

# Only keep columns that actually exist
available_columns = [
    column
    for column in columns_to_keep
    if column in mapping.columns
]

mapping = mapping[available_columns]

# --------------------------------------------------
# 9. DISPLAY RESULTS
# --------------------------------------------------

print("\n===== MAPPING RESULTS =====")

print("Selected probes:", len(selected_probe_ids))

print("Mapped probes:", len(mapping))

print("\nFirst mapped features:")

print(
    mapping.head(20).to_string(index=False)
)

# --------------------------------------------------
# 10. SAVE RESULT
# --------------------------------------------------

mapping.to_csv(
    "results/selected_probe_gene_mapping.csv",
    index=False
)

print("\nSaved successfully!")

print(
    "File: results/selected_probe_gene_mapping.csv"
)