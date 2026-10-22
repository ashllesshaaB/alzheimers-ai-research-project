import gzip
import pandas as pd

file_path = "data/GSE33000_series_matrix.txt.gz"

print("Loading expression matrix...")

with gzip.open(file_path, "rt", encoding="utf-8") as f:
    lines = f.readlines()

start = 0
end = len(lines)

for i, line in enumerate(lines):
    if line.startswith("!series_matrix_table_begin"):
        start = i + 1
    elif line.startswith("!series_matrix_table_end"):
        end = i
        break

print("Reading table...")
data = pd.read_csv(
    file_path,
    sep="\t",
    skiprows=start,
    nrows=end - start,
    compression="gzip"
)

print("Dataset loaded!")
print("Shape:", data.shape)
print(data.head())