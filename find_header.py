import gzip

file_path = "data/GSE33000_series_matrix.txt.gz"

with gzip.open(file_path, "rt", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if "series_matrix_table_begin" in line:
            print("FOUND at line:", i)
            print(line.strip())
            print(next(f).strip()[:500])
            break