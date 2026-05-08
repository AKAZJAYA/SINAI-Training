"""
Headline Evaluation v6 — Script-Optimized Model
=================================================
Key improvements over v5:
  - Evaluates v5 adapter (r=64, label smoothing, 5 epochs)
  - Greedy decoding for deterministic reproducible results
  - Same 6K validation set (600 samples)
  - Full Word + Character ROUGE metrics
"""

import torch
from unsloth import FastLanguageModel
from transformers import AutoTokenizer
from peft import PeftModel
import os
import json
from collections import Counter

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

BASE_MODEL = os.path.join(BASE_DIR, "./models/llama-3-8b")
SINLLAMA_ADAPTER = os.path.join(BASE_DIR, "./models/SinLlama_v01")
HEADLINE_ADAPTER = os.path.join(BASE_DIR, "./models/adapters/headline_sinllama_v5")  # 🔥 v5 (r=64)
TOKENIZER_PATH = os.path.join(BASE_DIR, "./models/Extended-Sinhala-LLaMA")
VAL_DATA_PATH = os.path.join(BASE_DIR, "data/headline_dataset_6k_val.jsonl")          # 🔥 600 samples

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
    max_seq_length=512,
    dtype=None,
    load_in_4bit=True,
)

print("🔹 Resizing embeddings...")
model = model.to("cpu")
model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
model = model.to("cuda")

print("🔹 Loading SinLlama...")
model = PeftModel.from_pretrained(model, SINLLAMA_ADAPTER)
model = model.merge_and_unload()

print("🔹 Loading headline adapter v5 (r=64)...")
model.load_adapter(HEADLINE_ADAPTER)

FastLanguageModel.for_inference(model)
model.eval()

# ─────────────────────────────────────────────
# GENERATION — Beam search for better quality
# ─────────────────────────────────────────────
def generate_headline(article_text, category):
    """Generate headline using beam search for better quality."""
    MAX_ARTICLE_CHARS = 800
    article_text = article_text.strip()[:MAX_ARTICLE_CHARS]

    prompt = f"""### Instruction:
Generate a concise Sinhala news headline for the following article.

Rules:
- Formal news headline style
- Maximum 10 words
- Focus on the most important fact
- Include key entity, event, or number from the article
- Output ONLY the headline, nothing else

### Input:
Category: {category}
Article: {article_text}

### Response:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=35,
            do_sample=False,                 # 🔥 greedy — deterministic & reproducible
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Clean
    result = result.split("\n")[0].strip()
    for marker in ["###", "Instruction:", "Input:", "Response:"]:
        if marker in result:
            result = result.split(marker)[0].strip()

    # Limit to 10 words
    words = result.split()
    result = " ".join(words[:10])

    return result.strip()


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

# --- Word-Level ROUGE ---
def compute_rouge_1(reference, hypothesis):
    """ROUGE-1 F1 (unigram overlap)."""
    ref_words = set(reference.split())
    hyp_words = set(hypothesis.split())
    if not ref_words or not hyp_words:
        return 0.0
    overlap = ref_words & hyp_words
    p = len(overlap) / len(hyp_words)
    r = len(overlap) / len(ref_words)
    return (2 * p * r) / (p + r) if (p + r) > 0 else 0.0


def compute_rouge_2(reference, hypothesis):
    """ROUGE-2 F1 (bigram overlap)."""
    def bigrams(words):
        return set(zip(words[:-1], words[1:])) if len(words) > 1 else set()
    ref_bi = bigrams(reference.split())
    hyp_bi = bigrams(hypothesis.split())
    if not ref_bi or not hyp_bi:
        return 0.0
    overlap = ref_bi & hyp_bi
    p = len(overlap) / len(hyp_bi)
    r = len(overlap) / len(ref_bi)
    return (2 * p * r) / (p + r) if (p + r) > 0 else 0.0


def compute_rouge_l(reference, hypothesis):
    """ROUGE-L F1 (word-level LCS)."""
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


# --- Character N-gram ROUGE (Sinhala-Aware) ---
def get_char_ngrams(text, n):
    """Extract character n-grams (spaces removed)."""
    text = text.replace(" ", "")
    if len(text) < n:
        return Counter()
    return Counter(text[i:i+n] for i in range(len(text) - n + 1))


def compute_rouge_char_ngram(reference, hypothesis, n=3):
    """ROUGE F1 based on character n-grams."""
    ref_ngrams = get_char_ngrams(reference, n)
    hyp_ngrams = get_char_ngrams(hypothesis, n)
    if not ref_ngrams or not hyp_ngrams:
        return 0.0
    overlap = sum(min(ref_ngrams[ng], hyp_ngrams.get(ng, 0)) for ng in ref_ngrams)
    ref_total = sum(ref_ngrams.values())
    hyp_total = sum(hyp_ngrams.values())
    p = overlap / hyp_total if hyp_total > 0 else 0
    r = overlap / ref_total if ref_total > 0 else 0
    return (2 * p * r) / (p + r) if (p + r) > 0 else 0.0


def compute_rouge_char_lcs(reference, hypothesis):
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


def compute_exact_match(reference, hypothesis):
    return 1.0 if reference.strip() == hypothesis.strip() else 0.0


# ─────────────────────────────────────────────
# LOAD VALIDATION SET
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
            "article": article,
            "category": category,
            "expected": item["output"].strip(),
        })

MAX_EVAL = len(val_samples)
eval_samples = val_samples[:MAX_EVAL]
print(f"   Evaluating on {len(eval_samples)} samples (full validation set)")

# ─────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────
all_scores = []

print("\n" + "=" * 80)
print("  HEADLINE GENERATION EVALUATION (v6 — r=64 + Label Smoothing)")
print("=" * 80)

for i, item in enumerate(eval_samples):
    generated = generate_headline(item["article"], item["category"])
    expected = item["expected"]

    scores = {
        "index": i + 1,
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
        # Other
        "exact_match": compute_exact_match(expected, generated),
        "word_count": len(generated.split()),
    }
    all_scores.append(scores)

    # Print first 20 examples
    if i < 20:
        print(f"\n--- Test {i+1} [{item['category']}] ---")
        print(f"  Expected : {expected}")
        print(f"  Generated: {generated}")
        print(f"  Word  ROUGE → R1: {scores['rouge1_word']:.3f}  R2: {scores['rouge2_word']:.3f}  RL: {scores['rougeL_word']:.3f}")
        print(f"  Char  ROUGE → C3: {scores['rouge_char3']:.3f}  C4: {scores['rouge_char4']:.3f}  CL: {scores['rouge_charL']:.3f}")
        em_icon = '✓' if scores['exact_match'] else '✗'
        print(f"  EM: {em_icon}")

    # Progress indicator
    if (i + 1) % 50 == 0:
        print(f"\n  ... processed {i+1}/{len(eval_samples)} samples ...")

# ─────────────────────────────────────────────
# RESULTS SUMMARY
# ─────────────────────────────────────────────
n = len(all_scores)

def avg(key):
    return sum(s[key] for s in all_scores) / n

# Category breakdown
cat_scores = {}
for s in all_scores:
    cat = s["category"]
    if cat not in cat_scores:
        cat_scores[cat] = []
    cat_scores[cat].append(s)

print("\n" + "=" * 80)
print("  RESULTS SUMMARY (v6 — Script-Optimized)")
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

total_em = sum(s['exact_match'] for s in all_scores)
avg_wc = avg('word_count')
print(f"\n  Exact Matches      : {int(total_em)}/{n} ({total_em/n*100:.1f}%)")
print(f"  Avg headline words : {avg_wc:.1f}")

# Per-category table
print(f"\n  Per-category breakdown:")
print(f"  {'Category':12s}  {'R1-W':>6s}  {'R2-W':>6s}  {'RL-W':>6s}  │  {'C3':>6s}  {'C4':>6s}  {'CL':>6s}  │  {'EM':>3s}  {'N':>4s}")
print(f"  {'─'*12}  {'─'*6}  {'─'*6}  {'─'*6}  │  {'─'*6}  {'─'*6}  {'─'*6}  │  {'─'*3}  {'─'*4}")
for cat in sorted(cat_scores.keys()):
    items = cat_scores[cat]
    cn = len(items)
    def cat_avg(key):
        return sum(s[key] for s in items) / cn
    cem = int(sum(s['exact_match'] for s in items))
    print(f"  {cat:12s}  {cat_avg('rouge1_word'):6.4f}  {cat_avg('rouge2_word'):6.4f}  {cat_avg('rougeL_word'):6.4f}  │  {cat_avg('rouge_char3'):6.4f}  {cat_avg('rouge_char4'):6.4f}  {cat_avg('rouge_charL'):6.4f}  │  {cem:3d}  {cn:4d}")

# Quality distribution
high_c3 = sum(1 for s in all_scores if s['rouge_char3'] >= 0.5)
med_c3 = sum(1 for s in all_scores if 0.3 <= s['rouge_char3'] < 0.5)
low_c3 = sum(1 for s in all_scores if s['rouge_char3'] < 0.3)

high_r1 = sum(1 for s in all_scores if s['rouge1_word'] >= 0.5)
med_r1 = sum(1 for s in all_scores if 0.3 <= s['rouge1_word'] < 0.5)
low_r1 = sum(1 for s in all_scores if s['rouge1_word'] < 0.3)

high_cl = sum(1 for s in all_scores if s['rouge_charL'] >= 0.5)
med_cl = sum(1 for s in all_scores if 0.3 <= s['rouge_charL'] < 0.5)
low_cl = sum(1 for s in all_scores if s['rouge_charL'] < 0.3)

over_10 = sum(1 for s in all_scores if s['word_count'] > 10)
empty = sum(1 for s in all_scores if len(s['generated'].strip()) == 0)

print(f"\n  Quality distribution:")
print(f"  {'Range':18s}  {'Word R1':>10s}  {'Char C3':>10s}  {'Char CL':>10s}")
print(f"  {'─'*18}  {'─'*10}  {'─'*10}  {'─'*10}")
print(f"  {'≥ 0.5 (Good) 🟢':18s}  {high_r1:3d}/{n} {high_r1/n*100:4.1f}%  {high_c3:3d}/{n} {high_c3/n*100:4.1f}%  {high_cl:3d}/{n} {high_cl/n*100:4.1f}%")
print(f"  {'0.3-0.5 (OK) 🟡':18s}  {med_r1:3d}/{n} {med_r1/n*100:4.1f}%  {med_c3:3d}/{n} {med_c3/n*100:4.1f}%  {med_cl:3d}/{n} {med_cl/n*100:4.1f}%")
print(f"  {'< 0.3 (Low) 🔴':18s}  {low_r1:3d}/{n} {low_r1/n*100:4.1f}%  {low_c3:3d}/{n} {low_c3/n*100:4.1f}%  {low_cl:3d}/{n} {low_cl/n*100:4.1f}%")

print(f"\n  Headlines > 10 words : {over_10}/{n}")
print(f"  Empty generations    : {empty}/{n}")
print("=" * 80)

# ─────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────
output_file = os.path.join(BASE_DIR, "headline_eval_results_v6.json")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump({
        "summary": {
            "word_rouge": {
                "rouge1": avg('rouge1_word'),
                "rouge2": avg('rouge2_word'),
                "rougeL": avg('rougeL_word'),
            },
            "char_rouge": {
                "char3gram": avg('rouge_char3'),
                "char4gram": avg('rouge_char4'),
                "charLCS": avg('rouge_charL'),
            },
            "exact_matches": int(total_em),
            "avg_word_count": avg_wc,
            "total_samples": n,
        },
        "per_category": {
            cat: {
                "word_rouge1": sum(s['rouge1_word'] for s in items) / len(items),
                "word_rouge2": sum(s['rouge2_word'] for s in items) / len(items),
                "word_rougeL": sum(s['rougeL_word'] for s in items) / len(items),
                "char3gram": sum(s['rouge_char3'] for s in items) / len(items),
                "char4gram": sum(s['rouge_char4'] for s in items) / len(items),
                "charLCS": sum(s['rouge_charL'] for s in items) / len(items),
                "count": len(items),
            }
            for cat, items in cat_scores.items()
        },
        "quality_distribution": {
            "word_r1": {"good": high_r1, "ok": med_r1, "low": low_r1},
            "char_c3": {"good": high_c3, "ok": med_c3, "low": low_c3},
            "char_cl": {"good": high_cl, "ok": med_cl, "low": low_cl},
        },
        "results": all_scores,
    }, f, ensure_ascii=False, indent=2)
print(f"\n📄 Full results saved to: {output_file}")
