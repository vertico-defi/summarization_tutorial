import pyarrow.parquet as pq
import pandas as pd
import matplotlib.pyplot as plt
import re
import os

# ============================================================
# Configuration
# ============================================================

INPUT_FILE = "bilingual_books.parquet"
OUTPUT_DIR = "plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Helper function to count words safely
# ============================================================

def count_words(text):
    if not isinstance(text, str):
        return 0
    # Count word-like tokens
    return len(re.findall(r"\w+", text))

# ============================================================
# Load the Parquet file (only necessary columns)
# ============================================================

print(f"📦 Loading {INPUT_FILE} ...")
table = pq.read_table(INPUT_FILE, columns=["review_title", "review_body"])
df = table.to_pandas()
print(f"✅ Loaded {len(df):,} rows.")

# ============================================================
# Compute word counts
# ============================================================

print("🧮 Computing word counts...")
df["title_word_count"] = df["review_title"].apply(count_words)
df["body_word_count"] = df["review_body"].apply(count_words)

# ============================================================
# Plot and save histograms
# ============================================================

plt.figure(figsize=(12, 5))

# Plot title word counts
plt.subplot(1, 2, 1)
plt.hist(df["title_word_count"], bins=30, color="steelblue", edgecolor="black")
plt.title("Review title")
plt.xlabel("Number of words")
plt.ylabel("Counts")

# Plot body word counts
plt.subplot(1, 2, 2)
plt.hist(df["body_word_count"], bins=30, color="steelblue", edgecolor="black")
plt.title("Review body")
plt.xlabel("Number of words")
plt.ylabel("Counts")

plt.tight_layout()

# Save combined figure
combined_path = os.path.join(OUTPUT_DIR, "word_distributions_combined.png")
plt.savefig(combined_path, dpi=300)
print(f"💾 Saved combined histogram → {combined_path}")

plt.show()

# ============================================================
# Save individual plots as well
# ============================================================

# Title histogram
plt.figure(figsize=(6, 5))
plt.hist(df["title_word_count"], bins=30, color="steelblue", edgecolor="black")
plt.title("Review title")
plt.xlabel("Number of words")
plt.ylabel("Counts")
title_path = os.path.join(OUTPUT_DIR, "title_word_distribution.png")
plt.savefig(title_path, dpi=300)
print(f"💾 Saved title histogram → {title_path}")
plt.close()

# Body histogram
plt.figure(figsize=(6, 5))
plt.hist(df["body_word_count"], bins=30, color="steelblue", edgecolor="black")
plt.title("Review body")
plt.xlabel("Number of words")
plt.ylabel("Counts")
body_path = os.path.join(OUTPUT_DIR, "body_word_distribution.png")
plt.savefig(body_path, dpi=300)
print(f"💾 Saved body histogram → {body_path}")
plt.close()

print("\n🎉 Done! Histograms displayed and saved successfully.")
