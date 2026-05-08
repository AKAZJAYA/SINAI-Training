"""
Headline Training v7 — Quality-Focused Improvements
=====================================================
Key changes from v5/v6:
  - LoRA r=128 (more capacity for complex Sinhala patterns)
  - Increased dataset support: designed for 20K+ samples
  - Removed hard 10-word cap from training prompt
  - Length guidance changed to "6-10 words" (not hard limit)
  - LR 2e-5 with cosine restart for better convergence
  - Label smoothing 0.15 (slightly higher for generalization)
  - Dropout 0.08 (balanced regularization)
  - 8 epochs with patience=10
  - Gradient clipping 0.8 (tighter for stability)
  - Unicode NFC normalization applied to all training data
  - Response marker kept consistent for clean decoding
"""

import torch
import unicodedata
from unsloth import FastLanguageModel
from transformers import AutoTokenizer, TrainingArguments, EarlyStoppingCallback
from peft import PeftModel
from datasets import load_dataset
from trl import SFTTrainer

# ─────────────────────────────────────────────
# CONFIG — Update paths as needed
# ─────────────────────────────────────────────
import os
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

BASE_MODEL      = os.path.join(BASE_DIR, "./models/llama-3-8b")
SINLLAMA_ADAPTER = os.path.join(BASE_DIR, "./models/SinLlama_v01")
TOKENIZER_PATH  = os.path.join(BASE_DIR, "./models/Extended-Sinhala-LLaMA")

# 🔥 Point these to your full dataset (ideally 20K+ train, 2K+ val)
DATA_PATH = "headline_dataset_6k_train.jsonl"
VAL_PATH  = "headline_dataset_6k_val.jsonl"
OUTPUT_DIR = "./models/adapters/headline_sinllama_v8"

MAX_SEQ_LENGTH = 640   # 🔥 increased from 512 to handle longer articles

# ─────────────────────────────────────────────
# NORMALIZATION HELPER
# ─────────────────────────────────────────────
def normalize_sinhala(text: str) -> str:
    """Apply Unicode NFC normalization — critical for Sinhala character consistency."""
    return unicodedata.normalize("NFC", text.strip()) if text else ""

# ─────────────────────────────────────────────
# LOAD TOKENIZER
# ─────────────────────────────────────────────
print("🔹 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ─────────────────────────────────────────────
# LOAD BASE MODEL
# ─────────────────────────────────────────────
print("🔹 Loading base model...")
model, _ = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

# Sinhala vocab alignment
print("🔹 Resizing embeddings...")
model = model.to("cpu")
model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
model = model.to("cuda")

# ─────────────────────────────────────────────
# LOAD SINLLAMA & MERGE
# ─────────────────────────────────────────────
print("🔹 Loading SinLlama adapter...")
model = PeftModel.from_pretrained(model, SINLLAMA_ADAPTER)

print("🔹 Merging Sinhala knowledge into base weights...")
model = model.merge_and_unload()

# Fix dtype consistency after merge
print("🔹 Casting model to bfloat16...")
for name, param in model.named_parameters():
    if param.dtype == torch.float32 and param.requires_grad:
        param.data = param.data.to(torch.bfloat16)

# ─────────────────────────────────────────────
# ADD HEADLINE LoRA — High capacity
# ─────────────────────────────────────────────
print("🔹 Adding headline adapter (r=128)...")

model = FastLanguageModel.get_peft_model(
    model,
    r=64,                              # 🔥 r=64
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=128,                     # 🔥 alpha = 2*r
    lora_dropout=0.08,                  # 🔥 balanced regularization
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# ─────────────────────────────────────────────
# LOAD & PREPROCESS DATASET
# ─────────────────────────────────────────────
print("🔹 Loading dataset...")
train_dataset = load_dataset("json", data_files=DATA_PATH)["train"]
val_dataset   = load_dataset("json", data_files=VAL_PATH)["train"]

print(f"   Train samples: {len(train_dataset)}")
print(f"   Val samples  : {len(val_dataset)}")

# ─────────────────────────────────────────────
# PROMPT FORMAT
# ─────────────────────────────────────────────
RESPONSE_MARKER = "### Response:\n"
MAX_ARTICLE_CHARS = 1500   # 🔥 increased from 1000 — more context for better headlines

def format_prompt(example):
    article = normalize_sinhala(example["input"])

    # Extract category and article body cleanly
    if "Article:" in article:
        category_part, article_part = article.split("Article:", 1)
        article_part = article_part.strip()[:MAX_ARTICLE_CHARS]
        article = category_part.strip() + "\nArticle: " + article_part
    else:
        article = article[:MAX_ARTICLE_CHARS]

    # 🔥 Changed: "6-10 words" guidance instead of hard "Maximum 10 words"
    prompt = f"""### Instruction:
Generate a concise Sinhala news headline for the following article.

Guidelines:
- Formal Sinhala news headline style
- Aim for 6 to 10 words
- Capture the single most important fact, entity, or event
- Use key numbers, names, or locations where present
- Output ONLY the headline — no explanations, no punctuation marks at the end

### Input:
{article}

{RESPONSE_MARKER}"""

    headline = normalize_sinhala(example["output"])   # 🔥 normalize target too

    return {
        "text": prompt + headline + tokenizer.eos_token,
    }

print("🔹 Formatting prompts...")
train_dataset = train_dataset.map(format_prompt, num_proc=4)
val_dataset   = val_dataset.map(format_prompt, num_proc=4)

# ─────────────────────────────────────────────
# TRAINER CONFIGURATION
# ─────────────────────────────────────────────
BATCH_SIZE     = 2
GRAD_ACCUM     = 4
EFFECTIVE_BATCH = BATCH_SIZE * GRAD_ACCUM   # = 8
EPOCHS         = 5                           # 🔥 5 epochs
TOTAL_STEPS    = (len(train_dataset) // EFFECTIVE_BATCH) * EPOCHS
WARMUP_STEPS   = int(TOTAL_STEPS * 0.08)    # 8% warmup
EVAL_STEPS     = 100

print(f"\n📊 Training configuration:")
print(f"   Effective batch size : {EFFECTIVE_BATCH}")
print(f"   Total training steps : {TOTAL_STEPS}")
print(f"   Warmup steps         : {WARMUP_STEPS}")
print(f"   Epochs               : {EPOCHS}")
print(f"   LoRA rank            : 64")
print(f"   Learning rate        : 1e-4")
print(f"   Label smoothing      : 0.0")
print(f"   Max article chars    : {MAX_ARTICLE_CHARS}")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    packing=False,

    dataset_kwargs={
        "add_special_tokens": True,
        "append_concat_token": False,
    },

    args=TrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=1e-4,                       # 🔥 1e-4 LR for better tuning
        logging_steps=10,
        max_grad_norm=0.8,                        # 🔥 tighter gradient clipping
        lr_scheduler_type="cosine",
        warmup_steps=WARMUP_STEPS,
        weight_decay=0.02,
        label_smoothing_factor=0.0,              # 🔥 no smoothing for exact matching
        output_dir=OUTPUT_DIR,
        report_to="none",
        # Evaluation & checkpointing
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=EVAL_STEPS,
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Mixed precision
        bf16=True,
        fp16=False,
    ),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],  # 🔥 patience=10
)

# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────
print("\n🚀 Starting headline training v8...")
trainer.train()

# ─────────────────────────────────────────────
# SAVE BEST MODEL
# ─────────────────────────────────────────────
print("💾 Saving best model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"✅ Headline training v8 complete! Saved to: {OUTPUT_DIR}")
