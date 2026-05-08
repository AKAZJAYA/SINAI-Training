"""
Dataset Cleaning Script for Headline Generation
- Removes duplicate input-output pairs
- Removes very short headlines (≤2 words)
- Reports category distribution
- Saves cleaned train + val sets
"""

import json
import random
from collections import Counter

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TRAIN_INPUT = "headline_dataset_stage3.jsonl"
VAL_INPUT = "headline_dataset_val.jsonl"

TRAIN_OUTPUT = "headline_dataset_stage4.jsonl"      # cleaned train
VAL_OUTPUT = "headline_dataset_val_clean.jsonl"      # cleaned val

MIN_HEADLINE_WORDS = 3   # remove headlines with fewer words
SEED = 42

random.seed(SEED)

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def get_category(item):
    inp = item["input"]
    if "Category:" in inp:
        return inp.split("\n")[0].replace("Category:", "").strip()
    return "General"

def deduplicate(data):
    """Remove duplicate input-output pairs, keeping first occurrence."""
    seen = set()
    unique = []
    for item in data:
        key = (item["input"].strip(), item["output"].strip())
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique

def remove_short_headlines(data, min_words):
    """Remove headlines with fewer than min_words words."""
    return [item for item in data if len(item["output"].strip().split()) >= min_words]

def remove_cross_leakage(train_data, val_data):
    """Remove any training samples whose output also appears in validation."""
    val_outputs = set(item["output"].strip() for item in val_data)
    val_inputs = set(item["input"].strip() for item in val_data)
    
    clean_train = []
    leaked = 0
    for item in train_data:
        # Check if this exact input-output pair exists in validation
        if item["input"].strip() in val_inputs and item["output"].strip() in val_outputs:
            leaked += 1
        else:
            clean_train.append(item)
    
    return clean_train, leaked

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
print("=" * 60)
print("  HEADLINE DATASET CLEANING")
print("=" * 60)

# Load
train_data = load_jsonl(TRAIN_INPUT)
val_data = load_jsonl(VAL_INPUT)
print(f"\n📂 Loaded:")
print(f"   Train: {len(train_data)} samples")
print(f"   Val:   {len(val_data)} samples")

# Step 1: Deduplicate
train_before = len(train_data)
train_data = deduplicate(train_data)
val_data = deduplicate(val_data)
print(f"\n🔹 After deduplication:")
print(f"   Train: {len(train_data)} (removed {train_before - len(train_data)})")
print(f"   Val:   {len(val_data)}")

# Step 2: Remove short headlines
train_before = len(train_data)
val_before = len(val_data)
train_data = remove_short_headlines(train_data, MIN_HEADLINE_WORDS)
val_data = remove_short_headlines(val_data, MIN_HEADLINE_WORDS)
print(f"\n🔹 After removing headlines < {MIN_HEADLINE_WORDS} words:")
print(f"   Train: {len(train_data)} (removed {train_before - len(train_data)})")
print(f"   Val:   {len(val_data)} (removed {val_before - len(val_data)})")

# Step 3: Remove train/val leakage
train_data, leaked = remove_cross_leakage(train_data, val_data)
print(f"\n🔹 After removing train/val leakage:")
print(f"   Removed {leaked} leaked samples from train")
print(f"   Train: {len(train_data)}")

# Step 4: Shuffle train data
random.shuffle(train_data)

# ─────────────────────────────────────────────
# STATS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  CLEANED DATASET STATISTICS")
print("=" * 60)

# Category distribution
train_cats = Counter(get_category(item) for item in train_data)
val_cats = Counter(get_category(item) for item in val_data)

print(f"\n📊 Train categories:")
for cat, cnt in sorted(train_cats.items(), key=lambda x: -x[1]):
    print(f"   {cat:12s}: {cnt:4d} ({cnt/len(train_data)*100:.1f}%)")

print(f"\n📊 Val categories:")
for cat, cnt in sorted(val_cats.items(), key=lambda x: -x[1]):
    print(f"   {cat:12s}: {cnt:4d} ({cnt/len(val_data)*100:.1f}%)")

# Headline word count stats
train_wc = [len(item["output"].split()) for item in train_data]
print(f"\n📏 Headline word counts (train):")
print(f"   Min: {min(train_wc)}, Max: {max(train_wc)}, Avg: {sum(train_wc)/len(train_wc):.1f}")

wc_dist = Counter(train_wc)
print(f"   Distribution:")
for wc in sorted(wc_dist.keys()):
    bar = "█" * (wc_dist[wc] // 10)
    print(f"   {wc:2d} words: {wc_dist[wc]:4d} {bar}")

# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────
save_jsonl(train_data, TRAIN_OUTPUT)
save_jsonl(val_data, VAL_OUTPUT)

print(f"\n💾 Saved:")
print(f"   Train → {TRAIN_OUTPUT} ({len(train_data)} samples)")
print(f"   Val   → {VAL_OUTPUT} ({len(val_data)} samples)")
print(f"\n✅ Done!")
