#!/usr/bin/env python3
"""
MedEase BD — GGUF Export v4
Properly dequantizes 4-bit BNB weights to float16 before saving.
Fixes: "Can not map tensor 'model.layers.0.mlp.down_proj.weight.absmax'"

Run: python3 export_gguf_v4.py
"""

import os, sys, subprocess, shutil, torch, json, gc
from pathlib import Path

ADAPTER_DIR = "medease-gemma-finetuned"
MERGED_DIR  = "medease-gemma-merged"
GGUF_DIR    = "medease-gemma-gguf"
GGUF_FILE   = f"{GGUF_DIR}/medease-gemma-q6k.gguf"
BASE_MODEL  = "unsloth/gemma-3-4b-it-bnb-4bit"

print("\n" + "="*55)
print("  MedEase BD — GGUF Export v4")
print("="*55 + "\n")

# ── Step 1: Load in 4bit, merge LoRA ─────────────────────────
print("📦 Step 1: Loading model and merging LoRA...")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from safetensors.torch import save_file
import bitsandbytes as bnb

bnb_config = BitsAndBytesConfig(
    load_in_4bit              = True,
    bnb_4bit_compute_dtype    = torch.bfloat16,
    bnb_4bit_use_double_quant = True,
    bnb_4bit_quant_type       = "nf4",
)

base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config = bnb_config,
    device_map          = "auto",
    trust_remote_code   = True,
)
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
model     = PeftModel.from_pretrained(base, ADAPTER_DIR)
model     = model.merge_and_unload()
print("   ✅ LoRA merged (still 4-bit in memory)")

# ── Step 2: Dequantize all Linear4bit layers to float16 ──────
print("\n🔓 Step 2: Dequantizing 4-bit layers to float16...")

def dequantize_model(model):
    """Replace all BNB Linear4bit layers with plain nn.Linear in float16."""
    import torch.nn as nn

    for name, module in model.named_children():
        if isinstance(module, bnb.nn.Linear4bit):
            # Dequantize the weight
            weight_fp16 = bnb.functional.dequantize_4bit(
                module.weight.data,
                module.weight.quant_state,
            ).to(torch.float16)

            new_linear = nn.Linear(
                module.in_features,
                module.out_features,
                bias = module.bias is not None,
            )
            new_linear.weight = nn.Parameter(weight_fp16.cpu())
            if module.bias is not None:
                new_linear.bias = nn.Parameter(module.bias.data.to(torch.float16).cpu())

            setattr(model, name, new_linear)
        else:
            dequantize_model(module)   # recurse into children
    return model

model = model.cpu()   # move to CPU first to avoid VRAM overflow
torch.cuda.empty_cache(); gc.collect()

print("   Dequantizing layers (this takes ~5 min on CPU)...")
model = dequantize_model(model)
print("   ✅ All layers dequantized to float16")

# ── Step 3: Save clean float16 state dict ────────────────────
print(f"\n💾 Step 3: Saving float16 weights → {MERGED_DIR}/")
os.makedirs(MERGED_DIR, exist_ok=True)

# Collect only clean weight tensors (no .absmax / .quant_state artifacts)
print("   Building clean state dict...")
state_dict = {}
for k, v in model.state_dict().items():
    # Skip any bitsandbytes quantization metadata keys
    if any(x in k for x in [".absmax", ".quant_state", ".quant_map",
                              ".nested_absmax", ".nested_quant_map"]):
        print(f"   Skipping BNB key: {k}")
        continue
    state_dict[k] = v.to(torch.float16).contiguous()

print(f"   Clean keys: {len(state_dict)}")

# Shard into ≤4 GB files
MAX_SHARD  = 4 * 1024**3
shards     = []
cur_shard  = {}
cur_size   = 0
index_map  = {}

for k, v in state_dict.items():
    sz = v.numel() * v.element_size()
    if cur_size + sz > MAX_SHARD and cur_shard:
        shards.append(cur_shard)
        cur_shard = {}
        cur_size  = 0
    cur_shard[k] = v
    cur_size    += sz

if cur_shard:
    shards.append(cur_shard)

total = len(shards)
print(f"   Saving {total} shard(s)...")

for i, shard in enumerate(shards, 1):
    fname = f"model-{i:05d}-of-{total:05d}.safetensors"
    save_file(shard, str(Path(MERGED_DIR) / fname))
    size_gb = (Path(MERGED_DIR) / fname).stat().st_size / 1024**3
    print(f"   ✅ {fname}  ({size_gb:.2f} GB)")
    for k in shard:
        index_map[k] = fname

# Save index if multi-shard
if total > 1:
    total_bytes = sum(v.numel()*v.element_size() for v in state_dict.values())
    idx = {"metadata": {"total_size": total_bytes}, "weight_map": index_map}
    with open(f"{MERGED_DIR}/model.safetensors.index.json", "w") as f:
        json.dump(idx, f, indent=2)
else:
    # Single shard — rename to plain model.safetensors
    old = Path(MERGED_DIR) / f"model-00001-of-00001.safetensors"
    new = Path(MERGED_DIR) / "model.safetensors"
    if old.exists():
        old.rename(new)
        print(f"   Renamed → model.safetensors")

# Copy config files
for cfg in ["config.json","generation_config.json","tokenizer_config.json",
            "tokenizer.json","tokenizer.model","special_tokens_map.json",
            "added_tokens.json","chat_template.json"]:
    src = Path(ADAPTER_DIR) / cfg
    dst = Path(MERGED_DIR)  / cfg
    if src.exists() and not dst.exists():
        shutil.copy2(src, dst)

# Fix config — remove quantization_config so llama.cpp accepts it
cfg_path = Path(MERGED_DIR) / "config.json"
with open(cfg_path) as f:
    cfg = json.load(f)
cfg.pop("quantization_config", None)
cfg["torch_dtype"] = "float16"
with open(cfg_path, "w") as f:
    json.dump(cfg, f, indent=2)

print("   ✅ Config files saved")

del model, state_dict
gc.collect()

# ── Step 4: Convert to GGUF F16 ──────────────────────────────
print("\n🔄 Step 4: Converting HF → GGUF (F16)...")
os.makedirs(GGUF_DIR, exist_ok=True)
F16_GGUF   = f"{GGUF_DIR}/medease-f16.gguf"
LLAMACPP   = Path.home() / "llama.cpp"

subprocess.run([sys.executable, "-m", "pip", "install",
                "gguf", "sentencepiece", "numpy", "-q"], check=True)

for script_name in ["convert_hf_to_gguf.py", "convert-hf-to-gguf.py"]:
    script = LLAMACPP / script_name
    if not script.exists():
        continue
    result = subprocess.run([
        sys.executable, str(script),
        MERGED_DIR,
        "--outfile", F16_GGUF,
        "--outtype", "f16",
    ])
    if result.returncode == 0:
        print(f"   ✅ F16 GGUF created")
        break
else:
    print("❌ Conversion failed.")
    sys.exit(1)

# ── Step 5: Quantize F16 → Q6_K ──────────────────────────────
print(f"\n⚡ Step 5: Quantizing → Q6_K...")

BUILD = LLAMACPP / "build"
QBIN  = BUILD / "bin" / "llama-quantize"

if not QBIN.exists():
    BUILD.mkdir(exist_ok=True)
    r = subprocess.run(
        ["cmake", "..", "-DGGML_CUDA=ON", "-DCMAKE_BUILD_TYPE=Release"],
        cwd=BUILD)
    if r.returncode != 0:
        subprocess.run(["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"],
                       cwd=BUILD, check=True)
    subprocess.run(
        ["cmake","--build",".","--config","Release",
         "--target","llama-quantize","-j4"],
        cwd=BUILD, check=True)

result = subprocess.run([str(QBIN), F16_GGUF, GGUF_FILE, "Q6_K"])
if result.returncode != 0:
    GGUF_FILE = f"{GGUF_DIR}/medease-gemma-q4km.gguf"
    print("   Q6_K failed — trying Q4_K_M...")
    subprocess.run([str(QBIN), F16_GGUF, GGUF_FILE, "Q4_K_M"], check=True)

Path(F16_GGUF).unlink(missing_ok=True)

size_mb = Path(GGUF_FILE).stat().st_size / 1024**2
print(f"\n{'='*55}")
print(f"  ✅ DONE!")
print(f"  📁 {GGUF_FILE}")
print(f"  📦 {size_mb:.0f} MB")
print(f"{'='*55}")
print(f"\n👉 Start the web app:")
print(f"   ./start.sh")
print(f"   or: python3 app.py --gguf {GGUF_FILE}\n")
