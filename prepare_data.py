import gzip
import pandas as pd

# ==============================
# FILE PATHS
# ==============================

matrix_file = "data/GSE33000_series_matrix.txt.gz"
output_file = "data/GSE33000_AD_vs_control.csv"


# ==============================
# STEP 1: READ SAMPLE LABELS
# ==============================

print("Reading sample labels...")

sample_ids = []
disease_labels = []

with gzip.open(matrix_file, "rt", encoding="utf-8", errors="ignore") as f:

    for line in f:

        if line.startswith("!Sample_geo_accession"):
            values = line.strip().split("\t")
            sample_ids = [
                v.strip('"')
                for v in values[1:]
            ]

        elif line.startswith("!Sample_characteristics_ch2"):
            values = line.strip().split("\t")

            disease_labels = [
                v.strip('"').replace(
                    "disease status: ", ""
                )
                for v in values[1:]
            ]


print("Samples found:", len(sample_ids))
print("Labels found:", len(disease_labels))


# ==============================
# STEP 2: CREATE LABEL TABLE
# ==============================

label_table = pd.DataFrame({
    "Sample": sample_ids,
    "Disease": disease_labels
})

print("\nOriginal disease counts:")
print(label_table["Disease"].value_counts())


# ==============================
# STEP 3: KEEP ONLY AD + CONTROL
# ==============================

keep = label_table[
    label_table["Disease"].isin([
        "Alzheimer's disease",
        "non-demented"
    ])
].copy()

print("\nSamples after removing Huntington's:")
print(keep["Disease"].value_counts())


# ==============================
# STEP 4: READ EXPRESSION DATA
# ==============================

print("\nReading expression matrix...")

expression = pd.read_csv(
    matrix_file,
    sep="\t",
    comment="!",
    compression="gzip"
)

print("Expression shape:", expression.shape)


# ==============================
# STEP 5: SELECT CORRECT SAMPLES
# ==============================

selected_samples = keep["Sample"].tolist()

selected_columns = ["ID_REF"] + selected_samples

clean_data = expression[selected_columns].copy()


# ==============================
# STEP 6: SAVE CLEAN DATA
# ==============================

clean_data.to_csv(
    output_file,
    index=False
)

print("\n================================")
print("DATA PREPARATION COMPLETE")
print("================================")

print("Final dataset:", clean_data.shape)

print("\nFinal samples:")
print("Alzheimer's:", 
      (keep["Disease"] == "Alzheimer's disease").sum())

print("Non-demented:",
      (keep["Disease"] == "non-demented").sum())

print("\nSaved to:")
print(output_file)