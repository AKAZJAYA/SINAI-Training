"""
Build 6,000-sample Headline Dataset (Balanced Categories)
==========================================================
Extracts articles from lankadeepa.json and creates:
  - headline_dataset_6k_train.jsonl  (~5,400 samples)
  - headline_dataset_6k_val.jsonl    (~600 samples)

Category distribution target: 50% General / 25% Politics / 25% Business
"""

import json
import random
import re
from collections import Counter

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SOURCE_FILE = "../lankadeepa.json"
TRAIN_OUTPUT = "headline_dataset_6k_train.jsonl"
VAL_OUTPUT = "headline_dataset_6k_val.jsonl"

TOTAL_SAMPLES = 6000
VAL_RATIO = 0.10  # 10% for validation

# Category distribution (balanced for better learning)
CATEGORY_TARGETS = {
    "General":  int(TOTAL_SAMPLES * 0.50),   # 3000
    "Business": int(TOTAL_SAMPLES * 0.25),   # 1500
    "Politics": int(TOTAL_SAMPLES * 0.25),   # 1500
}

SEED = 42
random.seed(SEED)

# Minimum quality thresholds
MIN_CONTENT_CHARS = 300       # article must have at least 300 chars
MAX_CONTENT_CHARS = 800       # truncate to 800 chars for training
MIN_HEADLINE_WORDS = 3        # headline must have at least 3 words
MAX_HEADLINE_WORDS = 10       # headline must have at most 10 words

# ─────────────────────────────────────────────
# CATEGORY MAPPING
# ─────────────────────────────────────────────
CATEGORY_MAP = {
    # → Business
    "Business": "Business",
    "ව්‍යාපාරික විත්ති": "Business",
    
    # → Politics  
    "Politics": "Politics",
    "දේශපාලන": "Politics",
    
    # → General (news categories only — skip lifestyle/entertainment)
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
    """Check if article meets quality standards."""
    headline = item.get("Headline", "").strip()
    content = item.get("News Content", "").strip()
    
    # Check headline
    hw = len(headline.split())
    if hw < MIN_HEADLINE_WORDS or hw > MAX_HEADLINE_WORDS:
        return False
    
    # Check content length
    if len(content) < MIN_CONTENT_CHARS:
        return False
    
    # Skip if headline contains URLs, email, or excessive English
    if any(x in headline for x in ["http", "www", "@", ".com", ".lk"]):
        return False
    
    # Skip headlines that are just numbers or punctuation
    clean = re.sub(r'[\s\d\-–—.,;:!?\'\"()\[\]{}]', '', headline)
    if len(clean) < 5:
        return False
    
    # Skip content that's mostly HTML or code
    if "<" in content[:100] and ">" in content[:100]:
        return False
    
    return True


def clean_content(content):
    """Clean article content for training."""
    # Remove HTML tags
    content = re.sub(r'<[^>]+>', '', content)
    
    # Remove multiple spaces/newlines
    content = re.sub(r'\s+', ' ', content).strip()
    
    # Remove URL-like patterns
    content = re.sub(r'https?://\S+', '', content)
    
    # Truncate to max chars
    content = content[:MAX_CONTENT_CHARS]
    
    # Don't cut mid-word
    if len(content) == MAX_CONTENT_CHARS:
        last_space = content.rfind(' ')
        if last_space > MAX_CONTENT_CHARS - 50:
            content = content[:last_space]
    
    return content.strip()


def format_sample(item, category):
    """Format article into training sample."""
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
print("  BUILDING 6K HEADLINE DATASET")
print("=" * 60)

print("\n📂 Loading lankadeepa.json...")
with open(SOURCE_FILE, "r", encoding="utf-8") as f:
    raw_data = json.load(f)
print(f"   Total articles: {len(raw_data)}")

# Group by target category
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
# LOAD EXISTING DATA (to avoid duplicates)
# ─────────────────────────────────────────────
print("\n🔹 Loading existing datasets to avoid duplicates...")
existing_headlines = set()
existing_content = set()

for f_path in ["headline_dataset_stage3.jsonl", "headline_dataset_stage4.jsonl", 
               "headline_dataset_val.jsonl"]:
    try:
        with open(f_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                existing_headlines.add(item["output"].strip())
                # Also track content to avoid similar articles
                inp = item["input"]
                if "Article:" in inp:
                    art = inp.split("Article:", 1)[1].strip()[:100]
                    existing_content.add(art)
    except FileNotFoundError:
        pass

print(f"   Existing headlines to skip: {len(existing_headlines)}")

# ─────────────────────────────────────────────
# SAMPLE & DEDUPLICATE
# ─────────────────────────────────────────────
print("\n🔹 Sampling and deduplicating...")

all_samples = []
seen_headlines = set(existing_headlines)
seen_content_starts = set(existing_content)

for cat in ["Business", "Politics", "General"]:
    items = categorized[cat]
    target = CATEGORY_TARGETS[cat]
    
    # Shuffle to randomize selection
    random.shuffle(items)
    
    cat_samples = []
    for item in items:
        headline = item["Headline"].strip()
        content = item["News Content"].strip()[:100]
        
        # Skip duplicates
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

# Shuffle all samples
random.shuffle(all_samples)

# ─────────────────────────────────────────────
# SPLIT TRAIN/VAL
# ─────────────────────────────────────────────
val_size = int(len(all_samples) * VAL_RATIO)
train_size = len(all_samples) - val_size

# Stratified split — ensure proportional categories in val
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

# Train stats
train_cats = Counter()
train_wc = []
train_art_lens = []
for s in train_data:
    cat = s["input"].split("\n")[0].replace("Category: ", "").strip()
    train_cats[cat] += 1
    train_wc.append(len(s["output"].split()))
    if "Article:" in s["input"]:
        art = s["input"].split("Article:", 1)[1].strip()
        train_art_lens.append(len(art))

print(f"\n📊 TRAIN SET ({len(train_data)} samples):")
for cat, cnt in sorted(train_cats.items(), key=lambda x: -x[1]):
    print(f"   {cat:12s}: {cnt:5d} ({cnt/len(train_data)*100:.1f}%)")
print(f"   Headline words: min={min(train_wc)}, max={max(train_wc)}, avg={sum(train_wc)/len(train_wc):.1f}")
print(f"   Article chars:  min={min(train_art_lens)}, max={max(train_art_lens)}, avg={sum(train_art_lens)/len(train_art_lens):.0f}")

# Val stats
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
