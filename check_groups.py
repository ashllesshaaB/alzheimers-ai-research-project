import gzip
from collections import Counter

file_path = "data/GSE33000_series_matrix.txt.gz"

with gzip.open(file_path, "rt", encoding="utf-8") as f:
    for line in f:
        if "disease status:" in line:
            print("Counting samples...")

            alzheimers = line.count("disease status: Alzheimer's disease")
            controls = line.count("disease status: non-demented")
            huntington = line.count("disease status: Huntington's disease")

            print("Alzheimer's:", alzheimers)
            print("Non-demented:", controls)
            print("Huntington's:", huntington)

            break