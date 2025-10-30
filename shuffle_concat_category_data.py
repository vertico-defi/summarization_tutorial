import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute as pc
import random
import os

# ============================================================
# Configuration
# ============================================================

INPUT_FILES = [
    "english_filtered_books.parquet",
    "spanish_filtered_books.parquet",
]
OUTPUT_FILE = "bilingual_books.parquet"
SAMPLE_SIZE = 5  # number of random samples to verify after combining

# ============================================================
# Function to load and combine both parquet files
# ============================================================

def combine_and_shuffle_parquets(file_paths, output_path):
    print(f"\n📦 Combining files: {file_paths}")
    
    # Read both datasets into PyArrow tables
    tables = [pq.read_table(path) for path in file_paths]
    
    # Concatenate them vertically
    combined = pa.concat_tables(tables, promote=True)
    total_rows = combined.num_rows
    print(f"✅ Combined dataset has {total_rows:,} rows and {combined.num_columns} columns.")
    
    # Shuffle the rows to mix languages
    shuffled_indices = list(range(total_rows))
    random.shuffle(shuffled_indices)
    shuffled_table = combined.take(pa.array(shuffled_indices))
    
    # Save the shuffled combined table to Parquet
    pq.write_table(shuffled_table, output_path)
    print(f"✅ Saved shuffled bilingual dataset → {output_path}")

    return shuffled_table


# ============================================================
# Function to sample reviews for verification
# ============================================================

def sample_reviews(table, num_samples=SAMPLE_SIZE):
    total_rows = table.num_rows
    print(f"\n📖 Sampling {num_samples} reviews from bilingual dataset...")
    
    if total_rows == 0:
        print("⚠️ No rows in dataset.")
        return

    sample_indices = sorted(random.sample(range(total_rows), min(num_samples, total_rows)))
    df = table.to_pandas().iloc[sample_indices]
    samples = df.to_dict(orient="records")

    for i, s in enumerate(samples, 1):
        print(f"\n🔹 Sample {i}")
        print(f"Language: {s.get('language', 'N/A')}")
        print(f"Category: {s.get('product_category', 'N/A')}")
        print(f"Stars: {s.get('stars', 'N/A')}")
        print(f"Title: {s.get('review_title', 'N/A')}")
        print(f"Review: {s.get('review_body', '')[:300]}")
        print("-" * 60)


# ============================================================
# Main Process
# ============================================================

os.makedirs(".", exist_ok=True)

bilingual_table = combine_and_shuffle_parquets(INPUT_FILES, OUTPUT_FILE)
sample_reviews(bilingual_table, num_samples=SAMPLE_SIZE)

print("\n🎉 Done! Bilingual Parquet file created and verified successfully.")
