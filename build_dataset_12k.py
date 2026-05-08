"""
Build 12,000-sample Headline Dataset (Balanced Categories)
============================================================
Extracts articles from lankadeepa.json and creates:
  - headline_dataset_12k_train.jsonl  (~10,800 samples)
  - headline_dataset_12k_val.jsonl    (~1,200 samples)

Category distribution: 50% General / 25% Politics / 25% Business
"""

import json
import random
import re
from collections import Counter

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SOURCE_FILE = "../lankadeepa.json"
TRAIN_OUTPUT = "headline_dataset_12k_train.jsonl"
VAL_OUTPUT = "headline_dataset_12k_val.jsonl"

TOTAL_SAMPLES = 12000
VAL_RATIO = 0.10  # 10% for validation

CATEGORY_TARGETS = {
    "General":  int(TOTAL_SAMPLES * 0.50),   # 6000
    "Business": int(TOTAL_SAMPLES * 0.25),   # 3000
    "Politics": int(TOTAL_SAMPLES * 0.25),   # 3000
}

SEED = 42
random.seed(SEED)

MIN_CONTENT_CHARS = 300
MAX_CONTENT_CHARS = 800
MIN_HEADLINE_WORDS = 3
MAX_HEADLINE_WORDS = 10

# ─────────────────────────────────────────────
# CATEGORY MAPPING
# ─────────────────────────────────────────────
CATEGORY_MAP = {
    "Business": "Business",
    "ව්‍යාපාරික විත්ති": "Business",
    "Politics": "Politics",
    "දේශපාලන": "Politics",
    "Local News": "General",
    "World News": "General",
    "එසැණ පුවත්": "General",
    "Middle East": "General",
    "Sports": "General",
    "Esana": "General",
    "Education": "General",
    "Technology": "General",
    "Medical": "General",
    "Features": "General",
    "විශේෂාංග": "General",
}

# ─────────────────────────────────────────────
# QUALITY FILTERS
# ─────────────────────────────────────────────
def is_valid_article(item):
    headline = item.get("Headline", "").strip()
    content = item.get("News Content", "").strip()
    hw = len(headline.split())
    if hw < MIN_HEADLINE_WORDS or hw > MAX_HEADLINE_WORDS:
        return False
    if len(content) < MIN_CONTENT_CHARS:
        return False
    if any(x in headline for x in ["http", "www", "@", ".com", ".lk"]):
        return False
    clean = re.sub(r'[\s\d\-–—.,;:!?\'\"()\[\]{}]', '', headline)
    if len(clean) < 5:
        return False
    if "<" in content[:100] and ">" in content[:100]:
        return False
    return True


def clean_content(content):
    content = re.sub(r'<[^>]+>', '', content)
    content = re.sub(r'\s+', ' ', content).strip()
    content = re.sub(r'https?://\S+', '', content)
    content = content[:MAX_CONTENT_CHARS]
    if len(content) == MAX_CONTENT_CHARS:
        last_space = content.rfind(' ')
        if last_space > MAX_CONTENT_CHARS - 50:
            content = content[:last_space]
    return content.strip()


def format_sample(item, category):
    headline = item["Headline"].strip()
    content = clean_content(item["News Content"])
    return {
        "input": f"Category: {category}\nArticle: {content}",
        "output": headline,
    }


# ─────────────────────────────────────────────
# LOAD & FILTER
# ─────────────────────────────────────────────
print("=" * 60)
print("  BUILDING 12K HEADLINE DATASET")
print("=" * 60)

print("\n📂 Loading lankadeepa.json...")
with open(SOURCE_FILE, "r", encoding="utf-8") as f:
    raw_data = json.load(f)
print(f"   Total articles: {len(raw_data)}")

categorized = {"General": [], "Business": [], "Politics": []}

for item in raw_data:
    source_cat = item.get("Category", "")
    target_cat = CATEGORY_MAP.get(source_cat)
    if target_cat is None:
        continue
    if not is_valid_article(item):
        continue
    categorized[target_cat].append(item)

print("\n📊 Quality articles available:")
for cat, items in categorized.items():
    target = CATEGORY_TARGETS[cat]
    status = "✅" if len(items) >= target else "⚠️"
    print(f"   {cat:12s}: {len(items):6d} (need {target}) {status}")

# ─────────────────────────────────────────────
# SAMPLE & DEDUPLICATE
# ─────────────────────────────────────────────
print("\n🔹 Sampling and deduplicating...")

all_samples = []
seen_headlines = set()
seen_content_starts = set()

for cat in ["Business", "Politics", "General"]:
    items = categorized[cat]
    target = CATEGORY_TARGETS[cat]
    random.shuffle(items)
    
    cat_samples = []
    for item in items:
        headline = item["Headline"].strip()
        content = item["News Content"].strip()[:100]
        
        if headline in seen_headlines:
            continue
        if content in seen_content_starts:
            continue
        
        seen_headlines.add(headline)
        seen_content_starts.add(content)
        
        sample = format_sample(item, cat)
        cat_samples.append(sample)
        
        if len(cat_samples) >= target:
            break
    
    print(f"   {cat:12s}: sampled {len(cat_samples)}/{target}")
    all_samples.extend(cat_samples)

random.shuffle(all_samples)

# ─────────────────────────────────────────────
# STRATIFIED SPLIT
# ─────────────────────────────────────────────
cat_groups = {"General": [], "Business": [], "Politics": []}
for s in all_samples:
    cat = s["input"].split("\n")[0].replace("Category: ", "").strip()
    cat_groups[cat].append(s)

train_data = []
val_data = []

for cat, items in cat_groups.items():
    cat_val_size = max(1, int(len(items) * VAL_RATIO))
    val_data.extend(items[:cat_val_size])
    train_data.extend(items[cat_val_size:])

random.shuffle(train_data)
random.shuffle(val_data)

# ─────────────────────────────────────────────
# STATS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  DATASET STATISTICS")
print("=" * 60)

train_cats = Counter()
train_wc = []
for s in train_data:
    cat = s["input"].split("\n")[0].replace("Category: ", "").strip()
    train_cats[cat] += 1
    train_wc.append(len(s["output"].split()))

print(f"\n📊 TRAIN SET ({len(train_data)} samples):")
for cat, cnt in sorted(train_cats.items(), key=lambda x: -x[1]):
    print(f"   {cat:12s}: {cnt:5d} ({cnt/len(train_data)*100:.1f}%)")
print(f"   Headline words: min={min(train_wc)}, max={max(train_wc)}, avg={sum(train_wc)/len(train_wc):.1f}")

val_cats = Counter()
for s in val_data:
    cat = s["input"].split("\n")[0].replace("Category: ", "").strip()
    val_cats[cat] += 1

print(f"\n📊 VAL SET ({len(val_data)} samples):")
for cat, cnt in sorted(val_cats.items(), key=lambda x: -x[1]):
    print(f"   {cat:12s}: {cnt:5d} ({cnt/len(val_data)*100:.1f}%)")

# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────
def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

save_jsonl(train_data, TRAIN_OUTPUT)
save_jsonl(val_data, VAL_OUTPUT)

print(f"\n💾 Saved:")
print(f"   Train → {TRAIN_OUTPUT} ({len(train_data)} samples)")
print(f"   Val   → {VAL_OUTPUT} ({len(val_data)} samples)")
print(f"\n✅ Done!")
