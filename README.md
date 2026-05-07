# 💊 MedEase BD — Bilingual Medicine Assistant

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python)](/)
[![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=flat-square&logo=flask)](/)
[![Model](https://img.shields.io/badge/Model-Gemma%203%204B-4285F4?style=flat-square)](/)
[![Offline](https://img.shields.io/badge/Runs-100%25%20Offline-10B981?style=flat-square)](/)
[![License](https://img.shields.io/badge/License-MIT-64748B?style=flat-square)](/)

> Offline bilingual medicine assistant for Bangladesh — fine-tuned Gemma 3 4B on 21,714 medicines. Supports English, বাংলা & Banglish. No API. No internet required.

---

## 📸 Demo

| AI Chat | Medicine Search |
|---------|----------------|
| Ask in English, বাংলা, or Banglish | Instant CSV search with ৳ prices |

---

## ✨ Features

- 🌐 **Trilingual** — English, বাংলা, Banglish
- 💰 **Cheapest Brand Finder** — Real ৳ prices from BD market
- 🤖 **RAG Pipeline** — Verified DB facts, no hallucination
- ⚡ **Streaming** — Token-by-token responses
- 📱 **Web + Mobile** — Flask web app + React Native
- 🔒 **100% Offline** — No API, no cloud, no internet

---

## 🗂️ Project Structure (MVC)

```
medease-bd/
│
├── 📁 model/                        # M — Data & AI layer
│   ├── generate_dataset.py          # Generate 36K bilingual QA pairs from CSV
│   ├── train.py                     # Fine-tune Gemma 3 4B with LoRA
│   ├── export_gguf_v4.py           # Export trained model to GGUF format
│   └── chat.py                      # CLI chat interface
│
├── 📁 controller/                   # C — Business logic layer
│   ├── app.py                       # Flask server + RAG pipeline + API routes
│   └── start.sh                     # One-click server launcher
│
├── 📁 view/                         # V — Frontend layer
│   └── templates/
│       └── index.html               # Web app UI (served by Flask)
│
├── 📁 mobile/                       # React Native mobile app
│   ├── App.js                       # Navigation + API config
│   ├── app.json
│   ├── package.json
│   └── screens/
│       ├── ChatScreen.js            # AI chat with streaming
│       ├── SearchScreen.js          # Medicine database search
│       └── AboutScreen.js           # Project info
│
├── 📁 mvp/                          # Minimum Viable Product
│   └── mvp.py                       # Single-file app, no AI needed
│
├── 📁 data/                         # Data files
│   └── MedData/                     # Bangladesh medicine CSVs
│       ├── medicine.csv             # 21,714 medicine records
│       ├── generic.csv              # 1,711 generics
│       ├── indication.csv
│       ├── drug class.csv
│       ├── dosage form.csv
│       └── manufacturer.csv
│
├── 📁 docs/                         # Documentation
│   ├── architecture.md              # System design
│   └── screenshots/                 # App screenshots
│
├── .gitignore
├── requirements.txt
├── Modelfile                        # Ollama config
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/medease-bd.git
cd medease-bd
```

### 2. Install dependencies
```bash
python3 -m venv ~/medease_env
source ~/medease_env/bin/activate
pip install -r requirements.txt

# Install llama-cpp with CUDA (Desktop with GPU)
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --upgrade

# OR CPU only (Laptop)
pip install llama-cpp-python
```

### 3. Download the model
```
👉 https://huggingface.co/YOUR_USERNAME/medease-bd-gemma
```
Download `medease-gemma-q6k.gguf` and place it in:
```
medease-bd/medease-gemma-gguf/medease-gemma-q6k.gguf
```

### 4. Run the web app
```bash
source ~/medease_env/bin/activate
python3 controller/app.py --gguf medease-gemma-gguf/medease-gemma-q6k.gguf
```
Open browser → **http://localhost:5000**

### 5. Run MVP (no AI needed)
```bash
pip install flask pandas
python3 mvp/mvp.py
```

---

## 🧠 Training Your Own Model

```bash
# Step 1 — Generate dataset
python3 model/generate_dataset.py --data-dir data/MedData --output qa_bilingual.json

# Step 2 — Fine-tune (requires GPU, ~4 hrs on RTX 3060)
python3 model/train.py --model gemma --steps 2000

# Step 3 — Export to GGUF
sudo apt install -y cmake build-essential
python3 model/export_gguf_v4.py
```

---

## 📱 Mobile App

```bash
# Install Expo CLI
npm install -g expo-cli
cd mobile && npm install

# Set your server IP in App.js
# export const API_BASE = 'http://YOUR_IP:5000';

npx expo start
# Scan QR with Expo Go on your phone
```

---

## 🏗️ Architecture

```
User Query (EN / বাংলা / Banglish)
          ↓
    Flask Controller (app.py)
          ↓
    RAG Lookup → MedData CSVs
    (keyword search, price sort)
          ↓
    Facts injected into prompt
          ↓
    Gemma 3 4B Q6_K (llama-cpp)
          ↓
    SSE Streaming → Browser / Mobile
```

---

## 📊 Dataset & Training

| Metric | Value |
|--------|-------|
| Base Model | Gemma 3 4B (unsloth 4-bit) |
| Method | LoRA rank=16, alpha=32 |
| Training Steps | 2,000 |
| Training Time | ~4 hrs (RTX 3060 12GB) |
| Initial Loss | 5.13 |
| Final Loss | 0.72 |
| QA Pairs | 36,448 (EN + বাংলা + Banglish) |
| Export | Q6_K GGUF (~3.5 GB) |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Model | Gemma 3 4B, Unsloth, LoRA, llama.cpp |
| Backend | Flask 3, llama-cpp-python |
| Data | Pandas, BeautifulSoup4 |
| Web UI | HTML/CSS/JS (SSE streaming) |
| Mobile | React Native, Expo |
| Training OS | Ubuntu 24.04, RTX 3060 12GB |

---

## ⚠️ Disclaimer

For informational purposes only. Always consult a licensed doctor or pharmacist before taking any medicine.

---

## 📄 License

MIT License — free to use for academic and research purposes.

*CloudCamp Bangladesh AI Hackathon 2025 · CSC4233 NLP Fundamentals*
