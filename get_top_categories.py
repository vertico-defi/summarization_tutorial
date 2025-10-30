import pyarrow.parquet as pq
from collections import Counter

# ============================================================
# Helper function: count product categories efficiently
# ============================================================

def count_categories(file_path):
    print(f"\n📦 Reading categories from: {file_path}")
    parquet_file = pq.ParquetFile(file_path)

    category_counter = Counter()

    for rg_idx in range(parquet_file.num_row_groups):
        # Read only the product_category column for this row group
        table = parquet_file.read_row_group(rg_idx, columns=["product_category"])
        categories = table["product_category"].to_pylist()
        category_counter.update(categories)

    print(f"✅ Finished counting {sum(category_counter.values()):,} total records.")
    return category_counter


# ============================================================
# Get top 20 categories for English and Spanish files
# ============================================================

for file_path, lang_label in [
    ("english_reviews.parquet", "English Reviews"),
    ("spanish_reviews.parquet", "Spanish Reviews"),
]:
    print(f"\n=== {lang_label} ===")
    category_counter = count_categories(file_path)
    top_20 = category_counter.most_common(20)

    print("\n🏷️ Top 20 product categories:")
    for i, (cat, count) in enumerate(top_20, 1):
        print(f"{i:2d}. {cat:<25} {count:,}")

print("\n🎉 Done! Top categories extracted successfully.")
