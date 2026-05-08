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

DATA_PATH = "data/headline_dataset_stage4.jsonl"       # 🔥 use cleaned dataset
VAL_PATH = "data/headline_dataset_val.jsonl"      # 🔥 use cleaned validation
OUTPUT_DIR = "./models/adapters/headline_sinllama_v1"

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

# ─────────────────────────────────────────────
# 🔥 FIX: Cast merged model to consistent dtype
# After merge_and_unload() on 4-bit, some layers
# can end up in float32 while LoRA expects bfloat16.
# This fixes the "BFloat16 != float" runtime error.
# ─────────────────────────────────────────────
print("🔹 Casting model to bfloat16 for dtype consistency...")
for name, param in model.named_parameters():
    if param.dtype == torch.float32 and param.requires_grad:
        param.data = param.data.to(torch.bfloat16)

# ─────────────────────────────────────────────
# ADD HEADLINE LoRA
# ─────────────────────────────────────────────
print("🔹 Adding headline adapter...")

model = FastLanguageModel.get_peft_model(
    model,
    r=16,                           # 🔥 reduced from 32 — prevents overfitting with small dataset
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=32,                  # 🔥 alpha = 2*r (standard scaling)
    lora_dropout=0.1,               # 🔥 increased from 0.05 — stronger regularization
    bias="none",
)

# ─────────────────────────────────────────────
# LOAD DATASET
# ─────────────────────────────────────────────
print("🔹 Loading dataset...")
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

    # 🔥 Updated rules: "Maximum 10 words" to match actual headline lengths in dataset
    # 🔥 Added "Focus on the most important fact" for better quality
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
        "text": prompt + headline + tokenizer.eos_token,  # add EOS so model learns to stop
    }

train_dataset = train_dataset.map(format_prompt)
val_dataset = val_dataset.map(format_prompt)

# ─────────────────────────────────────────────
# TRAINER
# ─────────────────────────────────────────────

# Calculate proper steps
BATCH_SIZE = 2
GRAD_ACCUM = 4
EFFECTIVE_BATCH = BATCH_SIZE * GRAD_ACCUM   # = 8
EPOCHS = 2                                   # 🔥 reduced from 3 — eval loss was rising at epoch 2.5
TOTAL_STEPS = (len(train_dataset) // EFFECTIVE_BATCH) * EPOCHS
WARMUP_STEPS = int(TOTAL_STEPS * 0.06)      # 6% warmup

print(f"   Effective batch size: {EFFECTIVE_BATCH}")
print(f"   Total training steps: {TOTAL_STEPS}")
print(f"   Warmup steps: {WARMUP_STEPS}")

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
        learning_rate=1e-4,                  # 🔥 reduced from 2e-4 — more stable convergence
        logging_steps=10,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        warmup_steps=WARMUP_STEPS,
        weight_decay=0.05,                   # 🔥 increased from 0.01 — stronger regularization
        output_dir=OUTPUT_DIR,
        report_to="none",
        # 🔥 More frequent evaluation
        eval_strategy="steps",
        eval_steps=50,                       # 🔥 reduced from 100 — catch overfitting earlier
        save_strategy="steps",
        save_steps=50,
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Mixed precision consistency
        bf16=True,
        fp16=False,
    ),
    # 🔥 NEW: Early stopping to prevent overfitting
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────
print("🚀 Training headline model v3...")
trainer.train()

# ─────────────────────────────────────────────
# SAVE BEST MODEL
# ─────────────────────────────────────────────
print("💾 Saving best model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ Headline training v3 complete!")
