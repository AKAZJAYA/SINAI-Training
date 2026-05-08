"""
Headline Training v4 — Optimized for 6K Balanced Dataset
=========================================================
Key changes from v3:
  - LoRA r=32 (increased from 16 — more data supports more capacity)
  - 3 epochs with patience=5 (more training time with larger dataset)
  - Gradient checkpointing for memory efficiency
  - Cosine schedule with proper warmup
"""

import torch
from unsloth import FastLanguageModel
from transformers import AutoTokenizer, TrainingArguments, EarlyStoppingCallback
from peft import PeftModel
from datasets import load_dataset
from trl import SFTTrainer

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_MODEL = "./models/llama-3-8b"
SINLLAMA_ADAPTER = "./models/SinLlama_v01"
TOKENIZER_PATH = "./models/Extended-Sinhala-LLaMA"

DATA_PATH = "data/headline_dataset_6k_train.jsonl"      # 🔥 6K balanced dataset
VAL_PATH = "data/headline_dataset_6k_val.jsonl"          # 🔥 600 balanced validation
OUTPUT_DIR = "./models/adapters/headline_sinllama_v4"     # 🔥 v4 adapter

MAX_SEQ_LENGTH = 512

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
print("🔹 Loading SinLlama...")
model = PeftModel.from_pretrained(model, SINLLAMA_ADAPTER)

print("🔹 Merging Sinhala knowledge...")
model = model.merge_and_unload()

# Fix dtype consistency after merge
print("🔹 Casting model to bfloat16...")
for name, param in model.named_parameters():
    if param.dtype == torch.float32 and param.requires_grad:
        param.data = param.data.to(torch.bfloat16)

# ─────────────────────────────────────────────
# ADD HEADLINE LoRA — More capacity for 6K data
# ─────────────────────────────────────────────
print("🔹 Adding headline adapter (r=32)...")

model = FastLanguageModel.get_peft_model(
    model,
    r=32,                               # 🔥 back to 32 — 6K data supports higher capacity
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=64,                      # alpha = 2*r
    lora_dropout=0.1,                   # keep strong dropout
    bias="none",
    use_gradient_checkpointing="unsloth",  # 🔥 memory efficient
)

# ─────────────────────────────────────────────
# LOAD DATASET
# ─────────────────────────────────────────────
print("🔹 Loading 6K dataset...")
train_dataset = load_dataset("json", data_files=DATA_PATH)["train"]
val_dataset = load_dataset("json", data_files=VAL_PATH)["train"]

print(f"   Train samples: {len(train_dataset)}")
print(f"   Val samples:   {len(val_dataset)}")

# ─────────────────────────────────────────────
# PROMPT FORMAT
# ─────────────────────────────────────────────
RESPONSE_MARKER = "### Response:\n"

def format_prompt(example):
    MAX_ARTICLE_CHARS = 800
    article = example["input"]

    # Extract and truncate article
    if "Article:" in article:
        category_part, article_part = article.split("Article:", 1)
        article_part = article_part.strip()[:MAX_ARTICLE_CHARS]
        article = category_part + "Article: " + article_part

    prompt = f"""### Instruction:
Generate a concise Sinhala news headline for the following article.

Rules:
- Formal news headline style
- Maximum 10 words
- Focus on the most important fact
- Include key entity, event, or number from the article
- Output ONLY the headline, nothing else

### Input:
{article}

{RESPONSE_MARKER}"""

    headline = example["output"].strip()

    return {
        "text": prompt + headline + tokenizer.eos_token,
    }

train_dataset = train_dataset.map(format_prompt)
val_dataset = val_dataset.map(format_prompt)

# ─────────────────────────────────────────────
# TRAINER — Optimized for 6K dataset
# ─────────────────────────────────────────────
BATCH_SIZE = 2
GRAD_ACCUM = 4
EFFECTIVE_BATCH = BATCH_SIZE * GRAD_ACCUM   # = 8
EPOCHS = 3                                   # 🔥 3 epochs — more data means less overfitting
TOTAL_STEPS = (len(train_dataset) // EFFECTIVE_BATCH) * EPOCHS
WARMUP_STEPS = int(TOTAL_STEPS * 0.05)      # 5% warmup
EVAL_STEPS = 100                              # 🔥 eval every 100 steps (more steps total)

print(f"\n📊 Training config:")
print(f"   Effective batch size: {EFFECTIVE_BATCH}")
print(f"   Total training steps: {TOTAL_STEPS}")
print(f"   Warmup steps: {WARMUP_STEPS}")
print(f"   Eval every: {EVAL_STEPS} steps")
print(f"   Epochs: {EPOCHS}")

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
        learning_rate=1e-4,                      # stable LR
        logging_steps=10,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        warmup_steps=WARMUP_STEPS,
        weight_decay=0.05,                       # strong regularization
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
    # Early stopping with more patience for larger dataset
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────
print("\n🚀 Training headline model v4 (6K dataset)...")
trainer.train()

# ─────────────────────────────────────────────
# SAVE BEST MODEL
# ─────────────────────────────────────────────
print("💾 Saving best model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ Headline training v4 complete!")
