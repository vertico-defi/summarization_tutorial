import pyarrow.parquet as pq
import random

# ============================================================
# Helper to sample N random rows efficiently from a Parquet file
# ============================================================

def sample_from_parquet(file_path, num_samples=5):
    # Read only metadata first (no data yet)
    parquet_file = pq.ParquetFile(file_path)
    total_rows = parquet_file.metadata.num_rows
    print(f"\n📦 {file_path} → total rows: {total_rows:,}")

    # Randomly choose row indices
    sample_indices = sorted(random.sample(range(total_rows), num_samples))
    samples = []

    # Go through row groups one by one and extract only needed rows
    current_row_index = 0
    for rg_idx in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(rg_idx)
        num_rows = table.num_rows

        # Select indices that fall in this row group
        rg_indices = [
            i - current_row_index
            for i in sample_indices
            if current_row_index <= i < current_row_index + num_rows
        ]
        if rg_indices:
            df = table.to_pandas().iloc[rg_indices]
            samples.extend(df.to_dict(orient="records"))

        current_row_index += num_rows
        if len(samples) >= num_samples:
            break

    return samples[:num_samples]

# ============================================================
# Sample reviews from both files
# ============================================================

for file_path, lang_label in [
    ("english_reviews.parquet", "English Reviews"),
    ("spanish_reviews.parquet", "Spanish Reviews"),
]:
    print(f"\n=== {lang_label} ===")
    samples = sample_from_parquet(file_path, num_samples=5)
    for i, s in enumerate(samples, 1):
        print(f"\n🔹 Sample {i}")
        print(f"Title: {s.get('review_title', 'N/A')}")
        print(f"Stars: {s.get('stars', 'N/A')}")
        print(f"Category: {s.get('product_category', 'N/A')}")
        print(f"Language: {s.get('language', 'N/A')}")
        print(f"Review: {s.get('review_body', '')[:300]}")
        print("-" * 60)

print("\n🎉 Done! Random samples printed successfully.")
