import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.compute as pc
import os
import random

# ============================================================
# Configuration
# ============================================================

TARGET_CATEGORIES = {"book", "digital_ebook_purchase"}
INPUT_FILES = {
    "english_reviews.parquet": "english_filtered_books.parquet",
    "spanish_reviews.parquet": "spanish_filtered_books.parquet",
}

# ============================================================
# Function to filter and save specific categories
# ============================================================

def filter_and_save(input_path, output_path):
    print(f"\n📦 Processing: {input_path}")
    dataset = ds.dataset(input_path, format="parquet")

    # Create filter expression for target categories
    filter_expr = pc.field("product_category").isin(list(TARGET_CATEGORIES))

    # Apply filter and read matching rows
    filtered_table = dataset.to_table(filter=filter_expr)

    # Save filtered data
    pa.parquet.write_table(filtered_table, output_path)
    print(f"✅ Saved filtered dataset → {output_path}")
    print(f"   Rows kept: {filtered_table.num_rows:,}")
    print(f"   Columns: {filtered_table.num_columns}")

# ============================================================
# Function to randomly sample 5 rows from a parquet file
# ============================================================

def sample_reviews(file_path, num_samples=5):
    print(f"\n📖 Sampling from: {file_path}")
    parquet_file = pq.ParquetFile(file_path)
    total_rows = parquet_file.metadata.num_rows

    if total_rows == 0:
        print("⚠️ No rows found in this dataset.")
        return

    sample_indices = sorted(random.sample(range(total_rows), min(num_samples, total_rows)))
    samples = []

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

    # Print results
    for i, s in enumerate(samples, 1):
        print(f"\n🔹 Sample {i}")
        print(f"Title: {s.get('review_title', 'N/A')}")
        print(f"Stars: {s.get('stars', 'N/A')}")
        print(f"Category: {s.get('product_category', 'N/A')}")
        print(f"Language: {s.get('language', 'N/A')}")
        print(f"Review: {s.get('review_body', '')[:300]}")
        print("-" * 60)

# ============================================================
# Main process
# ============================================================

os.makedirs(".", exist_ok=True)

for input_path, output_path in INPUT_FILES.items():
    filter_and_save(input_path, output_path)
    sample_reviews(output_path, num_samples=5)

print("\n🎉 All done! Filtered Parquet files created and verified successfully.")
