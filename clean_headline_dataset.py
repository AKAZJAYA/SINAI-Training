"""
Dataset Cleaning Script for Headline Generation
Fixes: empty articles, short articles, long headlines, informal headlines, category imbalance
"""

import json
import re
import random

INPUT_PATH = "/mnt/user-data/uploads/headline_dataset_cleaned.jsonl"
OUTPUT_PATH = "/home/claude/headline_dataset_stage3.jsonl"

# ─────────────────────────────────────────────
# FILTERING CRITERIA
# ─────────────────────────────────────────────
MIN_ARTICLE_CHARS = 100       # Skip articles shorter than this
MAX_HEADLINE_WORDS = 10       # Allow up to 10 (some valid Sinhala headlines need 9-10)
MIN_HEADLINE_WORDS = 2        # Skip single-word headlines

# Informal/conversational patterns to REMOVE
INFORMAL_PATTERNS = [
    "කියයි", "තියෙනවා", "තියෙනවාලු", "කරනවා", "යනවා",
    "ලෑස්ති නැහැ", "එහෙම", "මොකක්ද", "ඕනේ නෑ",
]

# Junk headline patterns
JUNK_PATTERNS = [
    r"පොලි ටිකල්",       # "poli tikal" - not a headline
    r"\d{8}",              # date-like numbers
    r"^[A-Za-z0-9\s]+$",  # purely English/numeric
]

def is_valid_sample(sample):
    """Check if a sample meets quality criteria."""
    inp = sample.get("input", "")
    headline = sample.get("output", "").strip()
    
    # Extract article text
    if "Article:" in inp:
        article = inp.split("Article:", 1)[1].strip()
    else:
        return False
    
    # 1. Skip empty or very short articles
    if len(article) < MIN_ARTICLE_CHARS:
        return False
    
    # 2. Skip very short or very long headlines
    words = headline.split()
    if len(words) < MIN_HEADLINE_WORDS or len(words) > MAX_HEADLINE_WORDS:
        return False
    
    # 3. Skip informal headlines
    for pattern in INFORMAL_PATTERNS:
        if pattern in headline:
            return False
    
    # 4. Skip junk headlines
    for pattern in JUNK_PATTERNS:
        if re.search(pattern, headline):
            return False
    
    # 5. Skip if headline is just a substring of article (lazy/extractive)
    if headline in article:
        return False
    
    return True

def extract_category(inp):
    """Extract category from input field."""
    if "Category:" in inp:
        cat_line = inp.split("\n")[0]
        return cat_line.replace("Category:", "").strip()
    return "General"

# ─────────────────────────────────────────────
# LOAD AND FILTER
# ─────────────────────────────────────────────
print("Loading dataset...")
all_samples = []
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        sample = json.loads(line.strip())
        all_samples.append(sample)

print(f"Total samples loaded: {len(all_samples)}")

# Filter
valid_samples = [s for s in all_samples if is_valid_sample(s)]
print(f"After quality filtering: {len(valid_samples)}")

# ─────────────────────────────────────────────
# CATEGORY DISTRIBUTION CHECK
# ─────────────────────────────────────────────
cat_counts = {}
cat_samples = {}
for s in valid_samples:
    cat = extract_category(s["input"])
    cat_counts[cat] = cat_counts.get(cat, 0) + 1
    if cat not in cat_samples:
        cat_samples[cat] = []
    cat_samples[cat].append(s)

print(f"\nCategory distribution after filtering:")
for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
    print(f"  {cat}: {count}")

# ─────────────────────────────────────────────
# REFORMAT FOR TRAINING
# ─────────────────────────────────────────────
# Convert to simple input/output format matching training prompt
formatted = []
for s in valid_samples:
    inp = s["input"]
    category = extract_category(inp)
    
    if "Article:" in inp:
        article = inp.split("Article:", 1)[1].strip()
    else:
        continue
    
    # Truncate article to 800 chars (match training)
    article = article[:800]
    
    formatted.append({
        "input": f"Category: {category}\nArticle: {article}",
        "output": s["output"].strip()
    })

# ─────────────────────────────────────────────
# TRAIN/VAL SPLIT (90/10)
# ─────────────────────────────────────────────
random.seed(42)
random.shuffle(formatted)

split_idx = int(len(formatted) * 0.9)
train_data = formatted[:split_idx]
val_data = formatted[split_idx:]

print(f"\nFinal dataset: {len(formatted)} samples")
print(f"  Train: {len(train_data)}")
print(f"  Val:   {len(val_data)}")

# Save training set
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for item in train_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

# Save validation set
val_path = OUTPUT_PATH.replace("stage3", "val")
with open(val_path, "w", encoding="utf-8") as f:
    for item in val_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"\nSaved: {OUTPUT_PATH}")
print(f"Saved: {val_path}")

# ─────────────────────────────────────────────
# SHOW SAMPLE
# ─────────────────────────────────────────────
print("\n--- Sample entries ---")
for item in formatted[:3]:
    print(f"Input: {item['input'][:100]}...")
    print(f"Output: {item['output']}")
    print()
