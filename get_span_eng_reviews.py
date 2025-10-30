from datasets import load_dataset

# ============================================================
# Load the dataset (single config)
# ============================================================

print("⏳ Loading multilingual Amazon reviews dataset...")
dataset = load_dataset("dnagpt/kaggle_amazon_reviews_multi", "default")
print("✅ Dataset loaded successfully!")

# ============================================================
# Inspect structure
# ============================================================

print("\n📊 Dataset splits:")
for split in dataset.keys():
    print(f"  {split:<10} → {len(dataset[split]):,} samples")

print("\n📋 Columns available:")
print(dataset["train"].column_names)

print("\n🧩 Feature types:")
print(dataset["train"].features)

print("\n🔍 Example record:")
print(dataset["train"][0])

# ============================================================
# Filter by language
# ============================================================

def filter_by_language(dataset, lang_code):
    """Filter dataset by language (e.g., 'en' or 'es')."""
    return dataset.filter(lambda x: x["language"] == lang_code)

print("\n⏳ Filtering English samples...")
english_dataset = dataset["train"].filter(lambda x: x["language"] == "en")
print(f"✅ English subset: {len(english_dataset):,} samples")

print("\n⏳ Filtering Spanish samples...")
spanish_dataset = dataset["train"].filter(lambda x: x["language"] == "es")
print(f"✅ Spanish subset: {len(spanish_dataset):,} samples")

# ============================================================
# Show a few random samples
# ============================================================

def show_samples(dataset, lang_label="Dataset", num_samples=3):
    print(f"\n=== {lang_label} Samples ===")
    sample = dataset.shuffle(seed=42).select(range(num_samples))
    for i, s in enumerate(sample):
        print(f"\n🔹 Sample {i+1}")
        print(f"Title: {s.get('review_title', 'N/A')}")
        print(f"Stars: {s.get('stars', 'N/A')}")
        print(f"Language: {s.get('language', 'N/A')}")
        print(f"Category: {s.get('product_category', 'N/A')}")
        print(f"Review: {s.get('review_body', '')[:250]}")

show_samples(english_dataset, "English Dataset")
show_samples(spanish_dataset, "Spanish Dataset")

# ============================================================
# Save datasets as Parquet files
# ============================================================

print("\n💾 Saving filtered datasets to Parquet format...")

english_path = "english_reviews.parquet"
spanish_path = "spanish_reviews.parquet"

english_dataset.to_parquet(english_path)
spanish_dataset.to_parquet(spanish_path)

print(f"✅ English dataset saved to: {english_path}")
print(f"✅ Spanish dataset saved to: {spanish_path}")

print("\n🎉 All done! Datasets are ready for use.")
