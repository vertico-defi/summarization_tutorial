import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import re

# ============================================================
# Configuration
# ============================================================

INPUT_FILE = "bilingual_books.parquet"
OUTPUT_FILE = "bilingual_books_filtered_titles.parquet"
MIN_TITLE_WORDS = 3
MIN_BODY_WORDS = 10

# ============================================================
# Helper
# ============================================================

def count_words(text):
    if not isinstance(text, str):
        return 0
    return len(re.findall(r"\w+", text))

# ============================================================
# Load + Filter
# ============================================================

print(f"📦 Loading dataset from {INPUT_FILE}...")
df = pd.read_parquet(INPUT_FILE)

print(f"Before filtering: {len(df):,} rows")

# Compute word counts
df["title_words"] = df["review_title"].apply(count_words)
df["body_words"] = df["review_body"].apply(count_words)

# Filter rows
filtered_df = df[(df["title_words"] >= MIN_TITLE_WORDS) & (df["body_words"] >= MIN_BODY_WORDS)]

print(f"✅ After filtering: {len(filtered_df):,} rows remain "
      f"(titles ≥ {MIN_TITLE_WORDS} words & bodies ≥ {MIN_BODY_WORDS} words)")

# Save as parquet
table = pa.Table.from_pandas(filtered_df)
pq.write_table(table, OUTPUT_FILE)
print(f"💾 Saved filtered dataset → {OUTPUT_FILE}")

# ============================================================
# Show random samples
# ============================================================

sample_df = filtered_df.sample(n=min(5, len(filtered_df)), random_state=42)
for i, row in sample_df.iterrows():
    print(f"\n🔹 Sample {i}")
    print(f"Language: {row['language']}")
    print(f"Title: {row['review_title']}")
    print(f"Title words: {row['title_words']}")
    print(f"Body words: {row['body_words']}")
    print(f"Review: {row['review_body'][:200]}")
    print("-" * 60)

print("\n🎉 Done! Filter applied successfully.")
