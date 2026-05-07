#!/usr/bin/env python3
"""
MedEase BD — Interactive Chat (runs the fine-tuned model locally)
Supports: English, বাংলা, and Banglish queries
Usage:
    python chat.py --adapter medease-gemma-finetuned
    python chat.py --gguf    medease-gemma-gguf/medease-gemma-q6k.gguf
"""

import argparse, sys, os, re

parser = argparse.ArgumentParser(description="MedEase BD Chat Interface")
parser.add_argument("--adapter",  default=None, help="Path to LoRA adapter dir")
parser.add_argument("--gguf",     default=None, help="Path to GGUF file (uses llama-cpp)")
parser.add_argument("--model",    default="gemma", choices=["gemma","qwen","llama"],
                    help="Base model (needed with --adapter)")
parser.add_argument("--temp",     type=float, default=0.15, help="Temperature")
parser.add_argument("--max-tokens", type=int, default=300, help="Max response tokens")
args = parser.parse_args()

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║          🏥 MedEase BD — Medicine Assistant                  ║
║   English | বাংলা | Banglish supported                       ║
║   Type your question. 'quit' or Ctrl+C to exit.             ║
╚══════════════════════════════════════════════════════════════╝
"""

EXAMPLES = """
💡 Example queries:
   EN:       What is Napa?  |  Cheapest brand of Paracetamol?
   BN:       নাপা কি?       |  প্যারাসিটামলের সস্তা ব্র্যান্ড কোনটি?
   Banglish: Napa ki?       |  Paracetamol er side effects ki?
"""

# ─────────────────────────────────────────────────────────────────────────────
#  GGUF mode (llama-cpp-python, no GPU required but GPU supported)
# ─────────────────────────────────────────────────────────────────────────────
def run_gguf(gguf_path):
    try:
        from llama_cpp import Llama
    except ImportError:
        print("Installing llama-cpp-python…")
        # Install with CUDA support for RTX 3060
        os.system('CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --upgrade -q')
        from llama_cpp import Llama

    print(f"Loading GGUF: {gguf_path}")
    llm = Llama(
        model_path    = gguf_path,
        n_ctx         = 512,
        n_gpu_layers  = -1,      # push all layers to GPU
        verbose       = False,
    )
    print(BANNER)
    print(EXAMPLES)

    while True:
        try:
            query = input("\n🔍 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! 👋"); break
        if not query or query.lower() in ('quit','exit','bye'):
            print("Goodbye! 👋"); break

        prompt = f"<start_of_turn>user\n{query}<end_of_turn>\n<start_of_turn>model\n"
        output = llm(prompt,
                     max_tokens       = args.max_tokens,
                     temperature      = args.temp,
                     repeat_penalty   = 1.1,
                     stop             = ["<end_of_turn>", "<start_of_turn>"])
        answer = output['choices'][0]['text'].strip()
        print(f"\n💊 MedEase BD:\n{answer}")

# ─────────────────────────────────────────────────────────────────────────────
#  Adapter mode (unsloth + transformers)
# ─────────────────────────────────────────────────────────────────────────────
def run_adapter(adapter_path, model_key):
    import torch
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        os.system("pip install unsloth -q")
        from unsloth import FastLanguageModel

    MODEL_MAP = {
        "gemma": "unsloth/gemma-3-4b-it-bnb-4bit",
        "qwen":  "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
        "llama": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    }

    print(f"Loading base model + adapter from: {adapter_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = adapter_path,
        max_seq_length = 512,
        load_in_4bit   = True,
        dtype          = None,
    )
    FastLanguageModel.for_inference(model)
    print(BANNER)
    print(EXAMPLES)

    history = []   # keep last 2 turns for context

    while True:
        try:
            query = input("\n🔍 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! 👋"); break
        if not query or query.lower() in ('quit','exit','bye'):
            print("Goodbye! 👋"); break
        if query.lower() == 'clear':
            history.clear(); print("Chat history cleared."); continue

        # Build messages with short history
        messages = history[-4:] + [{"role": "user", "content": query}]
        inp = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(inp, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens    = args.max_tokens,
                temperature       = args.temp,
                do_sample         = True,
                repetition_penalty= 1.1,
                pad_token_id      = tokenizer.eos_token_id,
            )
        full = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract last model turn
        answer = full.split("model\n")[-1].strip()
        print(f"\n💊 MedEase BD:\n{answer}")

        history.append({"role": "user",      "content": query})
        history.append({"role": "assistant", "content": answer})

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if args.gguf:
        run_gguf(args.gguf)
    elif args.adapter:
        run_adapter(args.adapter, args.model)
    else:
        print("❌ Provide --gguf <path> or --adapter <path>")
        sys.exit(1)
