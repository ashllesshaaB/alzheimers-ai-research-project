import gzip
from collections import Counter

file_path = "data/GSE33000_series_matrix.txt.gz"

print("Reading GSE33000 metadata...")

sample_ids = []
disease_labels = []

with gzip.open(file_path, "rt", encoding="utf-8", errors="ignore") as f:

    for line in f:

        if line.startswith("!Sample_geo_accession"):
            values = line.strip().split("\t")
            sample_ids = [v.strip('"') for v in values[1:]]

        elif line.startswith("!Sample_characteristics_ch2"):
            values = line.strip().split("\t")

            disease_labels = [
                v.strip('"').replace("disease status: ", "")
                for v in values[1:]
            ]

print("\nNumber of samples:", len(sample_ids))
print("Number of disease labels:", len(disease_labels))

print("\nDisease label counts:")

counts = Counter(disease_labels)

for label, count in counts.items():
    print(label, ":", count)

print("\nFirst 20 samples and labels:")

for i in range(min(20, len(sample_ids))):
    print(f"{sample_ids[i]} -> {disease_labels[i]}")

print("\nChecking alignment...")

if len(sample_ids) == len(disease_labels):
    print("SUCCESS: Every sample has exactly one disease label.")
else:
    print("ERROR: Sample and label counts do not match.")