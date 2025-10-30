import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow as pa
import os
import re

# ============================================================
# Configuration
# ============================================================

INPUT_FILE = "bilingual_books.parquet"
OUTPUT_FILE = "bilingual_books_filtered_titles.parquet"
MIN_TITLE_WORDS = 3  # must have more than 2 words

# ============================================================
# Helper to count words
# ============================================================

def count_words(text):
    if not isinstance(text, str):
        return 0
    return len(re.findall(r"\w+", text))

# ============================================================
# Load and filter
# ============================================================

print(f"📦 Loading dataset from {INPUT_FILE}...")
dataset = ds.dataset(INPUT_FILE, format="parquet")

# Read only necessary columns to count words efficiently
table = dataset.to_table(columns=["review_title", "review_body", "stars", "language", "product_category"])

# Compute word counts for titles
title_word_counts = [count_words(title) for title in table["review_title"]]

# Create a boolean mask for rows with > MIN_TITLE_WORDS
mask = pa.array([wc >= MIN_TITLE_WORDS for wc in title_word_counts])

# Filter the table
filtered_table = table.filter(mask)
print(f"✅ Filtered dataset: {filtered_table.num_rows:,} rows remaining (titles with ≥ {MIN_TITLE_WORDS} words).")

# Save the filtered version
pq.write_table(filtered_table, OUTPUT_FILE)
print(f"💾 Saved filtered dataset → {OUTPUT_FILE}")

# ============================================================
# Verify by sampling
# ============================================================

print("\n🔍 Checking a few random samples after filtering:")
df = filtered_table.to_pandas()
sample_df = df.sample(n=min(5, len(df)), random_state=42)

for i, row in sample_df.iterrows():
    print(f"\n🔹 Sample {i}")
    print(f"Language: {row.get('language', 'N/A')}")
    print(f"Title: {row.get('review_title', '')}")
    print(f"Word count: {count_words(row.get('review_title', ''))}")
    print(f"Review: {row.get('review_body', '')[:200]}")
    print("-" * 60)

print("\n🎉 Done! Titles filtered successfully.")
