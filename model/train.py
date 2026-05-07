#!/usr/bin/env python3
"""
MedEase BD — Local Training Script
GPU: RTX 3060 12GB | RAM: 16GB DDR5
Trains Gemma 3 4B (or Qwen2.5 / LLaMA 3.2) on bilingual medicine QA data.
"""

import os, sys, json, argparse, torch
from pathlib import Path

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train MedEase BD model locally")
parser.add_argument("--model", default="gemma", choices=["gemma","qwen","llama"],
                    help="Base model to fine-tune")
parser.add_argument("--data",  default="qa_bilingual.json",
                    help="Training data JSON file")
parser.add_argument("--steps", type=int, default=2000,
                    help="Max training steps (2000 ≈ 3-4 hrs on RTX 3060)")
parser.add_argument("--export-gguf", action="store_true",
                    help="Export Q6_K GGUF after training")
args = parser.parse_args()

# ── Model selection ───────────────────────────────────────────────────────────
MODEL_MAP = {
    "gemma": "unsloth/gemma-3-4b-it-bnb-4bit",   # 4B — best multilingual
    "qwen":  "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",  # 3B — fast, good Bengali
    "llama": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit", # 3B — strong reasoning
}
MODEL_ID   = MODEL_MAP[args.model]
OUTPUT_DIR = f"medease-{args.model}-finetuned"
GGUF_DIR   = f"medease-{args.model}-gguf"

print(f"\n{'='*60}")
print(f"  MedEase BD — Fine-tuning {args.model.upper()}")
print(f"  Model  : {MODEL_ID}")
print(f"  Data   : {args.data}")
print(f"  Steps  : {args.steps}")
print(f"  Device : {'CUDA ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (slow!)'}")
print(f"{'='*60}\n")

# ── 1. Install / import unsloth ───────────────────────────────────────────────
try:
    from unsloth import FastLanguageModel
except ImportError:
    print("Installing unsloth… (first run only)")
    os.system("pip install unsloth transformers datasets accelerate peft bitsandbytes trl -q")
    from unsloth import FastLanguageModel

from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer

# ── 2. Load dataset ───────────────────────────────────────────────────────────
print(f"📂 Loading dataset: {args.data}")
with open(args.data, 'r', encoding='utf-8') as f:
    data = json.load(f)
print(f"   ✅ {len(data):,} Q/A pairs loaded")

dataset = Dataset.from_list(data)
dataset = dataset.train_test_split(test_size=0.03, seed=42)
print(f"   Train: {len(dataset['train']):,} | Test: {len(dataset['test']):,}")

# ── 3. Load model (4-bit quantised for 12 GB VRAM) ───────────────────────────
print(f"\n🤖 Loading model: {MODEL_ID}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name   = MODEL_ID,
    max_seq_length = 512,
    load_in_4bit   = True,        # 4-bit keeps VRAM under 8 GB
    dtype          = None,        # auto (bfloat16 on Ampere)
)
print("   ✅ Model loaded")

# ── 4. Apply LoRA (PEFT) ──────────────────────────────────────────────────────
model = FastLanguageModel.get_peft_model(
    model,
    r              = 16,          # rank — 16 is safe for 12 GB
    lora_alpha     = 32,
    lora_dropout   = 0.05,
    target_modules = ["q_proj","k_proj","v_proj","o_proj",
                      "gate_proj","up_proj","down_proj"],
    bias           = "none",
    use_gradient_checkpointing = "unsloth",
    random_state   = 42,
)
print("   ✅ LoRA adapter applied  (rank=16, alpha=32)")

# ── 5. Training arguments ─────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir                  = OUTPUT_DIR,
    per_device_train_batch_size = 2,   # safe for 12 GB
    gradient_accumulation_steps = 8,   # effective batch = 16
    warmup_steps                = 200,
    max_steps                   = args.steps,
    learning_rate               = 2e-4,
    fp16                        = not torch.cuda.is_bf16_supported(),
    bf16                        = torch.cuda.is_bf16_supported(),
    logging_steps               = 50,
    save_steps                  = 500,
    save_total_limit            = 2,
    optim                       = "adamw_8bit",
    seed                        = 42,
    report_to                   = "none",
    dataloader_num_workers      = 0,
)

trainer = SFTTrainer(
    model             = model,
    tokenizer         = tokenizer,
    train_dataset     = dataset["train"],
    dataset_text_field= "text",
    max_seq_length    = 512,
    args              = training_args,
)

# ── 6. Train ──────────────────────────────────────────────────────────────────
print(f"\n🚀 Training started — {args.steps} steps")
print(f"   Estimated time on RTX 3060: {args.steps * 7 // 60} minutes\n")
trainer.train()
print("\n   ✅ Training complete!")

# ── 7. Quick bilingual test ───────────────────────────────────────────────────
print("\n🔍 Running bilingual inference test…")
FastLanguageModel.for_inference(model)

test_cases = [
    ("What is Napa?", "en"),
    ("নাপা কি?", "bn"),
    ("Napa er kaaj ki?", "banglish"),
    ("What is the cheapest brand of Paracetamol?", "en"),
    ("প্যারাসিটামলের সবচেয়ে সস্তা ব্র্যান্ড কোনটি?", "bn"),
    ("Amoxicillin er side effects ki?", "banglish"),
]

for query, lang in test_cases:
    messages = [{"role": "user", "content": query}]
    inp = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(inp, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=120,
                                 temperature=0.1, do_sample=True,
                                 repetition_penalty=1.1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("model\n")[-1].strip()
    print(f"\n[{lang.upper()}] Q: {query}")
    print(f"  A: {response[:300]}")

# ── 8. Save LoRA adapter ──────────────────────────────────────────────────────
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\n💾 LoRA adapter saved → {OUTPUT_DIR}/")

# ── 9. Optional GGUF export ───────────────────────────────────────────────────
if args.export_gguf:
    print(f"\n📦 Exporting Q6_K GGUF → {GGUF_DIR}/")
    model.save_pretrained_gguf(GGUF_DIR, tokenizer, quantization_method="q6_k")
    for f in Path(GGUF_DIR).glob("*.gguf"):
        final = Path(GGUF_DIR) / f"medease-{args.model}-q6k.gguf"
        f.rename(final)
        size_mb = final.stat().st_size / 1024**2
        print(f"   ✅ GGUF saved: {final}  ({size_mb:.0f} MB)")
    print(f"\n🎯 Next: load in Ollama or LM Studio")
    print(f"   ollama create medease-bd -f Modelfile")

print("\n✅ All done!\n")
