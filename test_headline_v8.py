"""
Headline Evaluation v7 — Quality-Focused Metrics
=================================================
Key improvements over v6:
  - Evaluates v7 adapter (r=128, 8 epochs)
  - Unicode NFC normalization applied before ALL comparisons
  - Beam search (num_beams=4) for better generation quality
  - max_new_tokens=60 (was 35) — fixes truncation causing 4.6 avg words
  - Removed hard 10-word cut — model decides length naturally
  - Added BERTScore-style character overlap (proxy for semantic sim)
  - Added length penalty analysis
  - Per-sample quality logging improved
  - Normalized exact match (after NFC normalization)
"""

import torch
import unicodedata
import os
import json
from collections import Counter

from unsloth import FastLanguageModel
from transformers import AutoTokenizer
from peft import PeftModel

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

BASE_MODEL       = os.path.join(BASE_DIR, "./models/llama-3-8b")
SINLLAMA_ADAPTER = os.path.join(BASE_DIR, "./models/SinLlama_v01")
HEADLINE_ADAPTER = os.path.join(BASE_DIR, "./models/adapters/headline_sinllama_v8")  # 🔥 v8
TOKENIZER_PATH   = os.path.join(BASE_DIR, "./models/Extended-Sinhala-LLaMA")
VAL_DATA_PATH    = os.path.join(BASE_DIR, "headline_dataset_6k_val.jsonl")

# ─────────────────────────────────────────────
# NORMALIZATION HELPER
# ─────────────────────────────────────────────
def normalize_sinhala(text: str) -> str:
    """
    Unicode NFC normalization — critical for Sinhala.
    Sinhala has multiple valid Unicode sequences for the same glyph.
    Without this, identical-looking text scores 0 in exact match.
    """
    return unicodedata.normalize("NFC", text.strip()) if text else ""

# ─────────────────────────────────────────────
# LOAD TOKENIZER
# ─────────────────────────────────────────────
print("🔹 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
print("🔹 Loading base model...")
model, _ = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=640,
    dtype=None,
    load_in_4bit=True,
)

print("🔹 Resizing embeddings...")
model = model.to("cpu")
model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
model = model.to("cuda")

print("🔹 Loading SinLlama and merging...")
model = PeftModel.from_pretrained(model, SINLLAMA_ADAPTER)
model = model.merge_and_unload()

print("🔹 Loading headline adapter v8 (r=64)...")
model.load_adapter(HEADLINE_ADAPTER)

FastLanguageModel.for_inference(model)
model.eval()

# ─────────────────────────────────────────────
# GENERATION
# ─────────────────────────────────────────────
def generate_headline(article_text: str, category: str) -> str:
    """
    Generate headline using beam search.

    Key fixes vs v6:
      - max_new_tokens=60  (was 35 — caused avg 4.6 words)
      - num_beams=4        (was greedy — beam search is better quality)
      - No hard word truncation at the end
      - NFC normalization on output
    """
    MAX_ARTICLE_CHARS = 1500
    article_text = article_text.strip()[:MAX_ARTICLE_CHARS]

    prompt = f"""### Instruction:
Generate a concise Sinhala news headline for the following article.

Guidelines:
- Formal Sinhala news headline style
- Aim for 6 to 10 words
- Capture the single most important fact, entity, or event
- Use key numbers, names, or locations where present
- Output ONLY the headline — no explanations, no punctuation marks at the end

### Input:
Category: {category}
Article: {article_text}

### Response:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,              # 🔥 was 35 — this was causing truncation
            num_beams=4,                    # 🔥 beam search instead of greedy
            early_stopping=True,
            do_sample=False,
            repetition_penalty=1.05,
            no_repeat_ngram_size=0,
            length_penalty=1.0,             # neutral length preference
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # ── Clean artifacts ──────────────────────────────────────────────────────
    # Take only the first line
    result = result.split("\n")[0].strip()

    # Remove any leaked prompt markers
    for marker in ["###", "Instruction:", "Input:", "Response:", "Category:", "Article:"]:
        if marker in result:
            result = result.split(marker)[0].strip()

    # 🔥 NO hard word truncation here — let the model decide length naturally
    # Apply Unicode normalization
    result = normalize_sinhala(result)

    return result


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

# ── Word-Level ROUGE ────────────────────────────────────────────────────────

def compute_rouge_1(reference: str, hypothesis: str) -> float:
    """ROUGE-1 F1 — unigram word overlap."""
    ref_words = set(reference.split())
    hyp_words = set(hypothesis.split())
    if not ref_words or not hyp_words:
        return 0.0
    overlap = ref_words & hyp_words
    p = len(overlap) / len(hyp_words)
    r = len(overlap) / len(ref_words)
    return (2 * p * r) / (p + r) if (p + r) > 0 else 0.0


def compute_rouge_2(reference: str, hypothesis: str) -> float:
    """ROUGE-2 F1 — bigram word overlap."""
    def bigrams(words):
        return Counter(zip(words[:-1], words[1:])) if len(words) > 1 else Counter()
    ref_bi = bigrams(reference.split())
    hyp_bi = bigrams(hypothesis.split())
    if not ref_bi or not hyp_bi:
        return 0.0
    overlap = sum(min(ref_bi[ng], hyp_bi.get(ng, 0)) for ng in ref_bi)
    p = overlap / sum(hyp_bi.values())
    r = overlap / sum(ref_bi.values())
    return (2 * p * r) / (p + r) if (p + r) > 0 else 0.0


def compute_rouge_l(reference: str, hypothesis: str) -> float:
    """ROUGE-L F1 — word-level LCS."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words or not hyp_words:
        return 0.0
    m, n = len(ref_words), len(hyp_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    p = lcs / n
    r = lcs / m
    return (2 * p * r) / (p + r) if (p + r) > 0 else 0.0


# ── Character N-gram ROUGE (Sinhala-Aware) ──────────────────────────────────

def get_char_ngrams(text: str, n: int) -> Counter:
    """Character n-grams with spaces removed."""
    text = text.replace(" ", "")
    if len(text) < n:
        return Counter()
    return Counter(text[i:i+n] for i in range(len(text) - n + 1))


def compute_rouge_char_ngram(reference: str, hypothesis: str, n: int = 3) -> float:
    """Character n-gram ROUGE F1."""
    ref_ng = get_char_ngrams(reference, n)
    hyp_ng = get_char_ngrams(hypothesis, n)
    if not ref_ng or not hyp_ng:
        return 0.0
    overlap = sum(min(ref_ng[ng], hyp_ng.get(ng, 0)) for ng in ref_ng)
    p = overlap / sum(hyp_ng.values())
    r = overlap / sum(ref_ng.values())
    return (2 * p * r) / (p + r) if (p + r) > 0 else 0.0


def compute_rouge_char_lcs(reference: str, hypothesis: str) -> float:
    """Character-level LCS ROUGE."""
    ref = reference.replace(" ", "")
    hyp = hypothesis.replace(" ", "")
    if not ref or not hyp:
        return 0.0
    m, n = len(ref), len(hyp)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    lcs = prev[n]
    p = lcs / n
    r = lcs / m
    return (2 * p * r) / (p + r) if (p + r) > 0 else 0.0


# ── Exact Match ──────────────────────────────────────────────────────────────

def compute_exact_match(reference: str, hypothesis: str) -> float:
    """
    Exact match after NFC normalization.
    🔥 Both sides normalized — prevents false negatives from encoding differences.
    """
    return 1.0 if normalize_sinhala(reference) == normalize_sinhala(hypothesis) else 0.0


# ── Length Analysis ──────────────────────────────────────────────────────────

def length_ratio(reference: str, hypothesis: str) -> float:
    """Ratio of generated length to reference length (1.0 = perfect)."""
    ref_len = len(reference.split())
    hyp_len = len(hypothesis.split())
    if ref_len == 0:
        return 0.0
    return hyp_len / ref_len


# ─────────────────────────────────────────────
# LOAD VALIDATION DATA
# ─────────────────────────────────────────────
print("\n🔹 Loading validation data...")

val_samples = []
with open(VAL_DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line.strip())
        inp = item["input"]
        category = "General"
        article = ""
        if "Category:" in inp:
            parts = inp.split("\n", 1)
            category = parts[0].replace("Category:", "").strip()
            if len(parts) > 1 and "Article:" in parts[1]:
                article = parts[1].split("Article:", 1)[1].strip()
        val_samples.append({
            "article":  normalize_sinhala(article),       # 🔥 normalize on load
            "category": category,
            "expected": normalize_sinhala(item["output"]), # 🔥 normalize reference
        })

MAX_EVAL = len(val_samples)
eval_samples = val_samples[:MAX_EVAL]
print(f"   Evaluating on {len(eval_samples)} samples")

# ─────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────
all_scores = []

print("\n" + "=" * 80)
print("  HEADLINE GENERATION EVALUATION (v8 — r=64, beam search, NFC norm)")
print("=" * 80)

for i, item in enumerate(eval_samples):
    generated = generate_headline(item["article"], item["category"])
    expected  = item["expected"]

    scores = {
        "index":    i + 1,
        "category": item["category"],
        "expected": expected,
        "generated": generated,
        # Word-level
        "rouge1_word": compute_rouge_1(expected, generated),
        "rouge2_word": compute_rouge_2(expected, generated),
        "rougeL_word": compute_rouge_l(expected, generated),
        # Character-level
        "rouge_char3": compute_rouge_char_ngram(expected, generated, n=3),
        "rouge_char4": compute_rouge_char_ngram(expected, generated, n=4),
        "rouge_charL": compute_rouge_char_lcs(expected, generated),
        # Quality indicators
        "exact_match":   compute_exact_match(expected, generated),
        "length_ratio":  length_ratio(expected, generated),
        "word_count_gen": len(generated.split()),
        "word_count_ref": len(expected.split()),
    }
    all_scores.append(scores)

    # Print first 20 examples with detailed output
    if i < 20:
        print(f"\n--- Test {i+1} [{item['category']}] ---")
        print(f"  Expected  ({scores['word_count_ref']}w): {expected}")
        print(f"  Generated ({scores['word_count_gen']}w): {generated}")
        print(f"  Word  ROUGE → R1: {scores['rouge1_word']:.3f}  R2: {scores['rouge2_word']:.3f}  RL: {scores['rougeL_word']:.3f}")
        print(f"  Char  ROUGE → C3: {scores['rouge_char3']:.3f}  C4: {scores['rouge_char4']:.3f}  CL: {scores['rouge_charL']:.3f}")
        print(f"  Length ratio: {scores['length_ratio']:.2f}  |  EM: {'✓' if scores['exact_match'] else '✗'}")

    # Progress every 50 samples
    if (i + 1) % 50 == 0:
        avg_r1 = sum(s["rouge1_word"] for s in all_scores) / len(all_scores)
        avg_cl = sum(s["rouge_charL"] for s in all_scores) / len(all_scores)
        avg_wc = sum(s["word_count_gen"] for s in all_scores) / len(all_scores)
        print(f"\n  ... {i+1}/{len(eval_samples)} | R1: {avg_r1:.4f} | CL: {avg_cl:.4f} | AvgWords: {avg_wc:.1f} ...")

# ─────────────────────────────────────────────
# RESULTS SUMMARY
# ─────────────────────────────────────────────
n = len(all_scores)

def avg(key):
    return sum(s[key] for s in all_scores) / n

# Per-category grouping
cat_scores = {}
for s in all_scores:
    cat = s["category"]
    if cat not in cat_scores:
        cat_scores[cat] = []
    cat_scores[cat].append(s)

print("\n" + "=" * 80)
print("  RESULTS SUMMARY (v8 — r=64, Beam Search, NFC Normalization)")
print("=" * 80)

print(f"\n  Samples evaluated: {n}")

print(f"\n  ┌─────────────────────────────────────────────────┐")
print(f"  │           WORD-LEVEL ROUGE (Standard)            │")
print(f"  ├─────────────────────────────────────────────────┤")
print(f"  │  ROUGE-1 (Word F1)  : {avg('rouge1_word'):.4f}                    │")
print(f"  │  ROUGE-2 (Word F1)  : {avg('rouge2_word'):.4f}                    │")
print(f"  │  ROUGE-L (Word F1)  : {avg('rougeL_word'):.4f}                    │")
print(f"  └─────────────────────────────────────────────────┘")

print(f"\n  ┌─────────────────────────────────────────────────┐")
print(f"  │      CHARACTER N-GRAM ROUGE (Sinhala-Aware)      │")
print(f"  ├─────────────────────────────────────────────────┤")
print(f"  │  ROUGE Char-3gram   : {avg('rouge_char3'):.4f}                    │")
print(f"  │  ROUGE Char-4gram   : {avg('rouge_char4'):.4f}                    │")
print(f"  │  ROUGE Char-LCS     : {avg('rouge_charL'):.4f}                    │")
print(f"  └─────────────────────────────────────────────────┘")

total_em  = sum(s["exact_match"] for s in all_scores)
avg_wc_g  = avg("word_count_gen")
avg_wc_r  = avg("word_count_ref")
avg_lr    = avg("length_ratio")

print(f"\n  Exact Matches (normalized) : {int(total_em)}/{n} ({total_em/n*100:.1f}%)")
print(f"  Avg generated words        : {avg_wc_g:.1f}  (reference avg: {avg_wc_r:.1f})")
print(f"  Avg length ratio           : {avg_lr:.3f}  (1.0 = perfect length)")

# Per-category table
print(f"\n  Per-category breakdown:")
print(f"  {'Category':12s}  {'R1-W':>6s}  {'R2-W':>6s}  {'RL-W':>6s}  │  {'C3':>6s}  {'C4':>6s}  {'CL':>6s}  │  {'EM':>3s}  {'AvgW':>5s}  {'N':>4s}")
print(f"  {'─'*12}  {'─'*6}  {'─'*6}  {'─'*6}  │  {'─'*6}  {'─'*6}  {'─'*6}  │  {'─'*3}  {'─'*5}  {'─'*4}")
for cat in sorted(cat_scores.keys()):
    items = cat_scores[cat]
    cn = len(items)
    def cat_avg(key):
        return sum(s[key] for s in items) / cn
    cem  = int(sum(s["exact_match"] for s in items))
    awcg = cat_avg("word_count_gen")
    print(f"  {cat:12s}  {cat_avg('rouge1_word'):6.4f}  {cat_avg('rouge2_word'):6.4f}  {cat_avg('rougeL_word'):6.4f}  │  {cat_avg('rouge_char3'):6.4f}  {cat_avg('rouge_char4'):6.4f}  {cat_avg('rouge_charL'):6.4f}  │  {cem:3d}  {awcg:5.1f}  {cn:4d}")

# Quality distribution
def dist(key, threshold_high, threshold_mid):
    high = sum(1 for s in all_scores if s[key] >= threshold_high)
    mid  = sum(1 for s in all_scores if threshold_mid <= s[key] < threshold_high)
    low  = sum(1 for s in all_scores if s[key] < threshold_mid)
    return high, mid, low

high_r1, med_r1, low_r1 = dist("rouge1_word", 0.5, 0.3)
high_c3, med_c3, low_c3 = dist("rouge_char3", 0.5, 0.3)
high_cl, med_cl, low_cl = dist("rouge_charL", 0.5, 0.3)

print(f"\n  Quality distribution:")
print(f"  {'Range':18s}  {'Word R1':>10s}  {'Char C3':>10s}  {'Char CL':>10s}")
print(f"  {'─'*18}  {'─'*10}  {'─'*10}  {'─'*10}")
print(f"  {'≥ 0.5 (Good) 🟢':18s}  {high_r1:3d}/{n} {high_r1/n*100:4.1f}%  {high_c3:3d}/{n} {high_c3/n*100:4.1f}%  {high_cl:3d}/{n} {high_cl/n*100:4.1f}%")
print(f"  {'0.3-0.5 (OK) 🟡':18s}  {med_r1:3d}/{n} {med_r1/n*100:4.1f}%  {med_c3:3d}/{n} {med_c3/n*100:4.1f}%  {med_cl:3d}/{n} {med_cl/n*100:4.1f}%")
print(f"  {'< 0.3 (Low) 🔴':18s}  {low_r1:3d}/{n} {low_r1/n*100:4.1f}%  {low_c3:3d}/{n} {low_c3/n*100:4.1f}%  {low_cl:3d}/{n} {low_cl/n*100:4.1f}%")

# Length distribution
too_short = sum(1 for s in all_scores if s["word_count_gen"] < 4)
ideal     = sum(1 for s in all_scores if 6 <= s["word_count_gen"] <= 10)
too_long  = sum(1 for s in all_scores if s["word_count_gen"] > 10)
short_mid = n - too_short - ideal - too_long

print(f"\n  Length distribution (generated headlines):")
print(f"  < 4 words  (too short) : {too_short:3d}/{n} ({too_short/n*100:.1f}%)")
print(f"  4-5 words  (borderline): {short_mid:3d}/{n} ({short_mid/n*100:.1f}%)")
print(f"  6-10 words (ideal) ✅  : {ideal:3d}/{n} ({ideal/n*100:.1f}%)")
print(f"  > 10 words (too long)  : {too_long:3d}/{n} ({too_long/n*100:.1f}%)")

empty = sum(1 for s in all_scores if len(s["generated"].strip()) == 0)
print(f"\n  Empty generations: {empty}/{n}")
print("=" * 80)

# ─────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────
output_file = os.path.join(BASE_DIR, "headline_eval_results_v8.json")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump({
        "config": {
            "adapter": "headline_sinllama_v8",
            "lora_r": 64,
            "beam_search": True,
            "num_beams": 4,
            "max_new_tokens": 60,
            "nfc_normalization": True,
        },
        "summary": {
            "word_rouge": {
                "rouge1": avg("rouge1_word"),
                "rouge2": avg("rouge2_word"),
                "rougeL": avg("rougeL_word"),
            },
            "char_rouge": {
                "char3gram": avg("rouge_char3"),
                "char4gram": avg("rouge_char4"),
                "charLCS":   avg("rouge_charL"),
            },
            "exact_matches":     int(total_em),
            "avg_word_count_gen": avg_wc_g,
            "avg_word_count_ref": avg_wc_r,
            "avg_length_ratio":  avg_lr,
            "total_samples":     n,
        },
        "per_category": {
            cat: {
                "word_rouge1": sum(s["rouge1_word"] for s in items) / len(items),
                "word_rouge2": sum(s["rouge2_word"] for s in items) / len(items),
                "word_rougeL": sum(s["rougeL_word"] for s in items) / len(items),
                "char3gram":   sum(s["rouge_char3"]  for s in items) / len(items),
                "char4gram":   sum(s["rouge_char4"]  for s in items) / len(items),
                "charLCS":     sum(s["rouge_charL"]  for s in items) / len(items),
                "avg_words_gen": sum(s["word_count_gen"] for s in items) / len(items),
                "count": len(items),
            }
            for cat, items in cat_scores.items()
        },
        "quality_distribution": {
            "word_r1": {"good": high_r1, "ok": med_r1, "low": low_r1},
            "char_c3": {"good": high_c3, "ok": med_c3, "low": low_c3},
            "char_cl": {"good": high_cl, "ok": med_cl, "low": low_cl},
        },
        "length_distribution": {
            "too_short_under4": too_short,
            "borderline_4_5":   short_mid,
            "ideal_6_10":       ideal,
            "too_long_over10":  too_long,
        },
        "results": all_scores,
    }, f, ensure_ascii=False, indent=2)

print(f"\n📄 Full results saved to: {output_file}")
