import torch
from unsloth import FastLanguageModel
from transformers import AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import os
import json

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

BASE_MODEL = os.path.join(BASE_DIR, "./models/llama-3-8b")
SINLLAMA_ADAPTER = os.path.join(BASE_DIR, "./models/SinLlama_v01")
HEADLINE_ADAPTER = os.path.join(BASE_DIR, "./models/adapters/headline_sinllama_v1")  # 🔥 v3
TOKENIZER_PATH = os.path.join(BASE_DIR, "./models/Extended-Sinhala-LLaMA")
VAL_DATA_PATH = os.path.join(BASE_DIR, "data/headline_dataset_val_clean.jsonl")       # 🔥 cleaned

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

print("🔹 Loading headline adapter...")
model.load_adapter(HEADLINE_ADAPTER)

FastLanguageModel.for_inference(model)
model.eval()

# ─────────────────────────────────────────────
# GENERATION FUNCTION
# ─────────────────────────────────────────────
def generate_headline(article_text, category):
    """Generate headline from article text and category."""
    MAX_ARTICLE_CHARS = 800
    article_text = article_text.strip()[:MAX_ARTICLE_CHARS]

    # 🔥 Updated prompt to match training v3 (10 words, added "Focus" rule)
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
            max_new_tokens=30,          # 🔥 reduced from 40 — headlines are short
            do_sample=False,            # greedy for consistency
            repetition_penalty=1.2,     # 🔥 increased from 1.15
            no_repeat_ngram_size=3,     # 🔥 NEW: prevent 3-gram repetition
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only generated tokens (skip prompt)
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Clean: take first line only
    result = result.split("\n")[0].strip()

    # Remove any trailing instruction artifacts
    for marker in ["###", "Instruction:", "Input:", "Response:"]:
        if marker in result:
            result = result.split(marker)[0].strip()

    # 🔥 Limit to 10 words (matches training prompt)
    words = result.split()
    result = " ".join(words[:10])

    return result.strip()


# ─────────────────────────────────────────────
# ROUGE SCORER (Sinhala-compatible)
# ─────────────────────────────────────────────
def compute_rouge_l(reference, hypothesis):
    """Compute ROUGE-L F1 score based on word-level LCS."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    if len(ref_words) == 0 or len(hyp_words) == 0:
        return 0.0

    # LCS length
    m, n = len(ref_words), len(hyp_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs_len = dp[m][n]

    precision = lcs_len / n if n > 0 else 0
    recall = lcs_len / m if m > 0 else 0

    if precision + recall == 0:
        return 0.0
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_rouge_1(reference, hypothesis):
    """Compute ROUGE-1 F1 score (unigram overlap)."""
    ref_words = set(reference.split())
    hyp_words = set(hypothesis.split())

    if len(ref_words) == 0 or len(hyp_words) == 0:
        return 0.0

    overlap = ref_words & hyp_words
    precision = len(overlap) / len(hyp_words) if hyp_words else 0
    recall = len(overlap) / len(ref_words) if ref_words else 0

    if precision + recall == 0:
        return 0.0
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


# 🔥 NEW: ROUGE-2 for bigram overlap
def compute_rouge_2(reference, hypothesis):
    """Compute ROUGE-2 F1 score (bigram overlap)."""
    def get_bigrams(words):
        return set(zip(words[:-1], words[1:])) if len(words) > 1 else set()

    ref_words = reference.split()
    hyp_words = hypothesis.split()

    ref_bigrams = get_bigrams(ref_words)
    hyp_bigrams = get_bigrams(hyp_words)

    if len(ref_bigrams) == 0 or len(hyp_bigrams) == 0:
        return 0.0

    overlap = ref_bigrams & hyp_bigrams
    precision = len(overlap) / len(hyp_bigrams) if hyp_bigrams else 0
    recall = len(overlap) / len(ref_bigrams) if ref_bigrams else 0

    if precision + recall == 0:
        return 0.0
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


# 🔥 NEW: Exact match score
def compute_exact_match(reference, hypothesis):
    """Check if generated headline exactly matches reference."""
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

# 🔥 Use ALL validation samples for accurate evaluation
MAX_EVAL = len(val_samples)
eval_samples = val_samples[:MAX_EVAL]
print(f"   Evaluating on {len(eval_samples)} samples (full validation set)")

# ─────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────
rouge1_scores = []
rouge2_scores = []
rougeL_scores = []
exact_matches = []
word_counts = []
results = []

print("\n" + "=" * 70)
print("  HEADLINE GENERATION EVALUATION (v3)")
print("=" * 70)

for i, item in enumerate(eval_samples):
    generated = generate_headline(item["article"], item["category"])
    expected = item["expected"]

    r1 = compute_rouge_1(expected, generated)
    r2 = compute_rouge_2(expected, generated)
    rl = compute_rouge_l(expected, generated)
    em = compute_exact_match(expected, generated)
    wc = len(generated.split())

    rouge1_scores.append(r1)
    rouge2_scores.append(r2)
    rougeL_scores.append(rl)
    exact_matches.append(em)
    word_counts.append(wc)

    results.append({
        "index": i + 1,
        "category": item["category"],
        "expected": expected,
        "generated": generated,
        "rouge1": r1,
        "rouge2": r2,
        "rougeL": rl,
        "exact_match": em,
    })

    # Print first 15 examples in detail
    if i < 15:
        print(f"\n--- Test {i+1} [{item['category']}] ---")
        print(f"  Expected : {expected}")
        print(f"  Generated: {generated}")
        print(f"  ROUGE-1: {r1:.3f}  |  ROUGE-2: {r2:.3f}  |  ROUGE-L: {rl:.3f}  |  EM: {'✓' if em else '✗'}")

# ─────────────────────────────────────────────
# RESULTS SUMMARY
# ─────────────────────────────────────────────
avg_r1 = sum(rouge1_scores) / len(rouge1_scores)
avg_r2 = sum(rouge2_scores) / len(rouge2_scores)
avg_rl = sum(rougeL_scores) / len(rougeL_scores)
avg_wc = sum(word_counts) / len(word_counts)
total_em = sum(exact_matches)

# Category breakdown
cat_scores = {}
for r in results:
    cat = r["category"]
    if cat not in cat_scores:
        cat_scores[cat] = {"r1": [], "r2": [], "rl": [], "em": []}
    cat_scores[cat]["r1"].append(r["rouge1"])
    cat_scores[cat]["r2"].append(r["rouge2"])
    cat_scores[cat]["rl"].append(r["rougeL"])
    cat_scores[cat]["em"].append(r["exact_match"])

print("\n" + "=" * 70)
print("  RESULTS SUMMARY")
print("=" * 70)
print(f"  Samples evaluated     : {len(eval_samples)}")
print(f"  Average ROUGE-1 (F1)  : {avg_r1:.4f}")
print(f"  Average ROUGE-2 (F1)  : {avg_r2:.4f}")    # 🔥 NEW
print(f"  Average ROUGE-L (F1)  : {avg_rl:.4f}")
print(f"  Exact Matches         : {int(total_em)}/{len(eval_samples)} ({total_em/len(eval_samples)*100:.1f}%)")  # 🔥 NEW
print(f"  Average headline words: {avg_wc:.1f}")

print(f"\n  Per-category scores:")
for cat, scores in sorted(cat_scores.items()):
    cr1 = sum(scores["r1"]) / len(scores["r1"])
    cr2 = sum(scores["r2"]) / len(scores["r2"])
    crl = sum(scores["rl"]) / len(scores["rl"])
    cem = sum(scores["em"])
    n = len(scores["r1"])
    print(f"    {cat:12s}  R1: {cr1:.4f}  R2: {cr2:.4f}  RL: {crl:.4f}  EM: {int(cem)}/{n}  ({n} samples)")

# Quality checks
over_10_words = sum(1 for wc in word_counts if wc > 10)
empty_gen = sum(1 for r in results if len(r["generated"].strip()) == 0)
high_rouge = sum(1 for r1 in rouge1_scores if r1 >= 0.5)
medium_rouge = sum(1 for r1 in rouge1_scores if 0.3 <= r1 < 0.5)
low_rouge = sum(1 for r1 in rouge1_scores if r1 < 0.3)

print(f"\n  Quality checks:")
print(f"    Headlines > 10 words : {over_10_words}/{len(eval_samples)}")
print(f"    Empty generations    : {empty_gen}/{len(eval_samples)}")
print(f"    ROUGE-1 >= 0.5       : {high_rouge}/{len(eval_samples)} ({high_rouge/len(eval_samples)*100:.1f}%) 🟢")
print(f"    ROUGE-1 0.3-0.5      : {medium_rouge}/{len(eval_samples)} ({medium_rouge/len(eval_samples)*100:.1f}%) 🟡")
print(f"    ROUGE-1 < 0.3        : {low_rouge}/{len(eval_samples)} ({low_rouge/len(eval_samples)*100:.1f}%) 🔴")
print("=" * 70)

# Save full results
output_file = os.path.join(BASE_DIR, "headline_eval_results_v3.json")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump({
        "summary": {
            "avg_rouge1": avg_r1,
            "avg_rouge2": avg_r2,
            "avg_rougeL": avg_rl,
            "exact_matches": int(total_em),
            "avg_word_count": avg_wc,
            "total_samples": len(eval_samples),
        },
        "per_category": {
            cat: {
                "avg_rouge1": sum(scores["r1"]) / len(scores["r1"]),
                "avg_rouge2": sum(scores["r2"]) / len(scores["r2"]),
                "avg_rougeL": sum(scores["rl"]) / len(scores["rl"]),
                "exact_matches": int(sum(scores["em"])),
                "count": len(scores["r1"]),
            }
            for cat, scores in cat_scores.items()
        },
        "results": results,
    }, f, ensure_ascii=False, indent=2)
print(f"\n📄 Full results saved to: {output_file}")
