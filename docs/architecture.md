# MedEase BD — System Architecture

## Overview

MedEase BD follows an MVC (Model-View-Controller) pattern
combined with a RAG (Retrieval-Augmented Generation) pipeline.

```
┌─────────────────────────────────────────────────────────┐
│                    USER INTERFACE                        │
│         Browser (Web)  │  Expo Go (Mobile)              │
└────────────────┬────────────────────────────────────────┘
                 │ HTTP / SSE
┌────────────────▼────────────────────────────────────────┐
│              CONTROLLER  (controller/app.py)             │
│  ┌──────────────┐  ┌────────────┐  ┌─────────────────┐  │
│  │ /api/chat    │  │/api/search │  │  /api/status    │  │
│  │ POST + SSE   │  │   POST     │  │     GET         │  │
│  └──────┬───────┘  └─────┬──────┘  └─────────────────┘  │
└─────────┼────────────────┼────────────────────────────── ┘
          │                │
┌─────────▼────────┐  ┌────▼─────────────────────────────┐
│   RAG PIPELINE   │  │        DIRECT CSV SEARCH          │
│                  │  │  (no AI — instant results)         │
│ 1. keyword match │  │  Pandas str.contains()            │
│ 2. price sort    │  │  Returns brands + ৳ prices         │
│ 3. fact format   │  └──────────────────────────────────┘
│ 4. inject prompt │
└────────┬─────────┘
         │
┌────────▼──────────────────────────────────────────────┐
│              MODEL  (model/)                           │
│                                                        │
│  Gemma 3 4B Q6_K GGUF                                 │
│  llama-cpp-python  │  n_ctx=4096  │  n_gpu_layers=-1  │
│                                                        │
│  Fine-tuned with LoRA on 36,448 bilingual QA pairs    │
│  Loss: 5.13 → 0.72  │  RTX 3060 12GB  │  4 hours     │
└───────────────────────────────────────────────────────┘
         │
┌────────▼──────────────────────────────────────────────┐
│              DATA  (data/MedData/)                     │
│                                                        │
│  medicine.csv   21,714 records                         │
│  generic.csv     1,711 generics                        │
│  indication.csv  disease mapping                       │
│  drug class.csv  classification                        │
│  manufacturer.csv 241 companies                        │
└───────────────────────────────────────────────────────┘
```

## RAG Query Flow

```
User: "Napa er shosto alternative ki?"
  ↓
[1] RAG Lookup
    → search "napa" in medicine.csv brand column
    → found: Paracetamol brands
    → sort by price ascending
    → format top 8 brands as text block

[2] Prompt injection
    SYSTEM = base_system_prompt + "=== VERIFIED DB FACTS ===" + rag_facts

[3] LLM Generation
    Gemma 3 4B reads injected facts
    Generates Banglish response with real prices

[4] SSE Streaming
    Token by token → Flask response → Browser
```

## File Responsibilities

| File | Responsibility |
|------|---------------|
| `model/generate_dataset.py` | CSV → 36K bilingual QA JSON |
| `model/train.py` | LoRA fine-tuning on GPU |
| `model/export_gguf_v4.py` | HF model → GGUF conversion |
| `model/chat.py` | CLI inference interface |
| `controller/app.py` | Flask routes, RAG, SSE stream |
| `controller/start.sh` | Auto-detect GGUF + launch |
| `view/templates/index.html` | Web UI (3 tabs) |
| `mobile/App.js` | React Native navigation + API_BASE |
| `mobile/screens/ChatScreen.js` | Streaming chat UI |
| `mobile/screens/SearchScreen.js` | Medicine search UI |
| `mvp/mvp.py` | Single-file no-AI version |
