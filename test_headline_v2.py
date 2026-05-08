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
HEADLINE_ADAPTER = os.path.join(BASE_DIR, "./models/adapters/headline_sinllama_v2")
TOKENIZER_PATH = os.path.join(BASE_DIR, "./models/Extended-Sinhala-LLaMA")
VAL_DATA_PATH = os.path.join(BASE_DIR, "data/headline_dataset_val.jsonl")

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

    prompt = f"""### Instruction:
Generate a concise Sinhala news headline for the following article.

Rules:
- Formal news headline style
- Maximum 8 words
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
            max_new_tokens=40,          # headlines are short
            do_sample=False,            # greedy for consistency
            repetition_penalty=1.15,
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

    # Limit to 8 words
    words = result.split()
    result = " ".join(words[:8])

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

# Use subset for quick eval (full set takes long)
MAX_EVAL = 50
eval_samples = val_samples[:MAX_EVAL]
print(f"   Evaluating on {len(eval_samples)} samples")

# ─────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────
rouge1_scores = []
rougeL_scores = []
word_counts = []
results = []

print("\n" + "=" * 70)
print("  HEADLINE GENERATION EVALUATION")
print("=" * 70)

for i, item in enumerate(eval_samples):
    generated = generate_headline(item["article"], item["category"])
    expected = item["expected"]

    r1 = compute_rouge_1(expected, generated)
    rl = compute_rouge_l(expected, generated)
    wc = len(generated.split())

    rouge1_scores.append(r1)
    rougeL_scores.append(rl)
    word_counts.append(wc)

    results.append({
        "index": i + 1,
        "category": item["category"],
        "expected": expected,
        "generated": generated,
        "rouge1": r1,
        "rougeL": rl,
    })

    # Print first 10 examples in detail
    if i < 10:
        print(f"\n--- Test {i+1} [{item['category']}] ---")
        print(f"  Expected : {expected}")
        print(f"  Generated: {generated}")
        print(f"  ROUGE-1: {r1:.3f}  |  ROUGE-L: {rl:.3f}")

# ─────────────────────────────────────────────
# RESULTS SUMMARY
# ─────────────────────────────────────────────
avg_r1 = sum(rouge1_scores) / len(rouge1_scores)
avg_rl = sum(rougeL_scores) / len(rougeL_scores)
avg_wc = sum(word_counts) / len(word_counts)

# Category breakdown
cat_scores = {}
for r in results:
    cat = r["category"]
    if cat not in cat_scores:
        cat_scores[cat] = {"r1": [], "rl": []}
    cat_scores[cat]["r1"].append(r["rouge1"])
    cat_scores[cat]["rl"].append(r["rougeL"])

print("\n" + "=" * 70)
print("  RESULTS SUMMARY")
print("=" * 70)
print(f"  Samples evaluated     : {len(eval_samples)}")
print(f"  Average ROUGE-1 (F1)  : {avg_r1:.4f}")
print(f"  Average ROUGE-L (F1)  : {avg_rl:.4f}")
print(f"  Average headline words: {avg_wc:.1f}")

print(f"\n  Per-category scores:")
for cat, scores in sorted(cat_scores.items()):
    cr1 = sum(scores["r1"]) / len(scores["r1"])
    crl = sum(scores["rl"]) / len(scores["rl"])
    print(f"    {cat:12s}  ROUGE-1: {cr1:.4f}  |  ROUGE-L: {crl:.4f}  ({len(scores['r1'])} samples)")

# Quality checks
over_8_words = sum(1 for wc in word_counts if wc > 8)
empty_gen = sum(1 for r in results if len(r["generated"].strip()) == 0)
high_rouge = sum(1 for r1 in rouge1_scores if r1 >= 0.5)

print(f"\n  Quality checks:")
print(f"    Headlines > 8 words : {over_8_words}/{len(eval_samples)}")
print(f"    Empty generations   : {empty_gen}/{len(eval_samples)}")
print(f"    ROUGE-1 >= 0.5      : {high_rouge}/{len(eval_samples)} ({high_rouge/len(eval_samples)*100:.1f}%)")
print("=" * 70)

# Save full results
output_file = os.path.join(BASE_DIR, "headline_eval_results.json")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump({
        "summary": {
            "avg_rouge1": avg_r1,
            "avg_rougeL": avg_rl,
            "avg_word_count": avg_wc,
            "total_samples": len(eval_samples),
        },
        "results": results,
    }, f, ensure_ascii=False, indent=2)
print(f"\n📄 Full results saved to: {output_file}")
