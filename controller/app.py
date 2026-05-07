#!/usr/bin/env python3
"""
MedEase BD — Local Web App Backend
Flask server that talks to the GGUF model via llama-cpp-python
Run: python app.py --gguf medease-gemma-gguf/medease-gemma-q6k.gguf
"""

import argparse, json, os, sys, re, threading, time
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string, Response, stream_with_context

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--gguf",   default="medease-gemma-gguf/medease-gemma-q6k.gguf",
                    help="Path to GGUF model file")
parser.add_argument("--host",   default="127.0.0.1")
parser.add_argument("--port",   type=int, default=5000)
parser.add_argument("--gpu-layers", type=int, default=-1,
                    help="-1 = all layers on GPU, 0 = CPU only")
args = parser.parse_args()

app = Flask(__name__)

# ── Load model ────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  MedEase BD — Loading model...")
print(f"  GGUF: {args.gguf}")
print(f"{'='*55}\n")

llm = None
model_status = {"loaded": False, "error": None, "loading": True}

def load_model():
    global llm
    try:
        from llama_cpp import Llama
        llm = Llama(
            model_path   = args.gguf,
            n_ctx        = 4096,        # increased from 1024
            n_gpu_layers = args.gpu_layers,
            verbose      = False,
            n_threads    = 4,
        )
        model_status["loaded"]  = True
        model_status["loading"] = False
        print("✅ Model loaded — server ready!\n")
    except Exception as e:
        model_status["error"]   = str(e)
        model_status["loading"] = False
        print(f"❌ Model load error: {e}\n")

threading.Thread(target=load_model, daemon=True).start()

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM = """You are MedEase BD, a helpful bilingual medicine assistant for Bangladesh.
Answer in the same language the user writes in (English, বাংলা, or Banglish).
For every query provide:
- Generic name of the medicine
- Available brand names in Bangladesh
- Cheapest/affordable alternatives with prices in ৳
- Dosage information if asked
- Side effects if asked
- Safety warnings if relevant
Keep answers clear, concise and helpful. Always remind users to consult a doctor for medical advice."""

def extract_price_rag(pkg):
    if not isinstance(pkg, str): return None
    import re as _re
    m = _re.search(r'৳\s*([\d.]+)', pkg)
    return float(m.group(1)) if m else None


def rag_lookup(query):
    """RAG: Fetch verified medicine facts from CSV and inject into prompt."""
    try:
        import pandas as pd
        from collections import defaultdict
        med_path = Path(__file__).parent / "MedData" / "medicine.csv"
        gen_path = Path(__file__).parent / "MedData" / "generic.csv"
        if not med_path.exists():
            return ""
        med = pd.read_csv(med_path)
        gen = pd.read_csv(gen_path)
        q   = query.lower()
        matches = pd.DataFrame()
        for term in [q] + [w for w in q.split() if len(w) >= 3]:
            bm = med['brand name'].str.lower().str.contains(term, na=False)
            gm = med['generic'].str.lower().str.contains(term, na=False)
            matches = med[bm | gm].head(15)
            if not matches.empty:
                break
        if matches.empty:
            return ""
        generic_brands = defaultdict(list)
        for _, row in matches.iterrows():
            price = extract_price_rag(str(row.get('package container', '')))
            generic_brands[str(row['generic']).strip()].append({
                'brand': str(row['brand name']).strip(),
                'manufacturer': str(row['manufacturer']).strip(),
                'strength': str(row['strength']).strip(),
                'dosage_form': str(row['dosage form']).strip(),
                'price': price,
            })
        parts = []
        for gname, brands in generic_brands.items():
            brands.sort(key=lambda x: (x['price'] is None, x['price'] or 9999))
            gen_row = gen[gen['generic name'].str.lower() == gname.lower()]
            drug_class = indication = ""
            if not gen_row.empty:
                drug_class = str(gen_row.iloc[0].get('drug class', '')).strip()
                indication = str(gen_row.iloc[0].get('indication', '')).strip()[:200]
            cheapest = brands[0]
            cheap_str = f"{cheapest['brand']} \u09F3{cheapest['price']:.2f}" if cheapest['price'] else cheapest['brand']
            brand_lines = []
            for b in brands[:8]:
                p = f"\u09F3{b['price']:.2f}" if b['price'] else "N/A"
                brand_lines.append(f"  - {b['brand']} | {b['dosage_form']} {b['strength']} | {b['manufacturer']} | {p}")
            parts.append(f"[{gname}]\nClass: {drug_class} | Used for: {indication}\nCheapest: {cheap_str}\nBrands:\n" + "\n".join(brand_lines))
        return "=== VERIFIED DATABASE FACTS ===\n\n" + "\n\n".join(parts)
    except Exception:
        return ""


def build_prompt(messages):
    sys_msgs  = [m for m in messages if m["role"] == "system"]
    chat_msgs = [m for m in messages if m["role"] != "system"]
    chat_msgs = chat_msgs[-4:]
    trimmed   = sys_msgs + chat_msgs
    prompt = ""
    for m in trimmed:
        role    = m["role"]
        content = m["content"][:1200]
        if role == "system":
            prompt += f"<start_of_turn>system\n{content}<end_of_turn>\n"
        elif role == "user":
            prompt += f"<start_of_turn>user\n{content}<end_of_turn>\n"
        elif role == "assistant":
            prompt += f"<start_of_turn>model\n{content}<end_of_turn>\n"
    prompt += "<start_of_turn>model\n"
    return prompt

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/status")
def status():
    return jsonify(model_status)

@app.route("/api/chat", methods=["POST"])
def chat():
    if not model_status["loaded"]:
        if model_status["error"]:
            return jsonify({"error": f"Model failed to load: {model_status['error']}"}), 500
        return jsonify({"error": "Model still loading, please wait..."}), 503

    data     = request.json
    messages = data.get("messages", [])
    stream   = data.get("stream", False)

    # RAG: look up real database facts for the latest user query
    rag_context = ""
    user_msgs = [m for m in messages if m["role"] == "user"]
    if user_msgs:
        rag_context = rag_lookup(user_msgs[-1]["content"])

    system_with_rag = SYSTEM
    if rag_context:
        system_with_rag = SYSTEM + "\n\nUse these VERIFIED FACTS to answer accurately:\n\n" + rag_context

    full_messages = [{"role": "system", "content": system_with_rag}] + messages
    prompt        = build_prompt(full_messages)

    if stream:
        def generate():
            try:
                for chunk in llm(
                    prompt,
                    max_tokens       = 400,
                    temperature      = 0.15,
                    repeat_penalty   = 1.1,
                    stop             = ["<end_of_turn>", "<start_of_turn>", "\n\n\n"],
                    stream           = True,
                ):
                    token = chunk["choices"][0]["text"]
                    yield f"data: {json.dumps({'token': token})}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return Response(stream_with_context(generate()),
                        mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache",
                                 "X-Accel-Buffering": "no"})
    else:
        try:
            out    = llm(prompt, max_tokens=400, temperature=0.15,
                         repeat_penalty=1.1,
                         stop=["<end_of_turn>","<start_of_turn>"])
            answer = out["choices"][0]["text"].strip()
            return jsonify({"answer": answer})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route("/api/search", methods=["POST"])
def search():
    """Fast medicine search directly from CSV — no LLM needed"""
    import pandas as pd
    from collections import defaultdict

    q = request.json.get("query", "").strip().lower()
    if not q or len(q) < 2:
        return jsonify({"results": []})

    try:
        med_path = Path(__file__).parent / "MedData" / "medicine.csv"
        gen_path = Path(__file__).parent / "MedData" / "generic.csv"
        if not med_path.exists():
            return jsonify({"results": [], "error": "MedData not found"})

        med = pd.read_csv(med_path)
        gen = pd.read_csv(gen_path)

        def extract_price(pkg):
            if not isinstance(pkg, str): return None
            m = re.search(r'৳\s*([\d.]+)', pkg)
            return float(m.group(1)) if m else None

        # Search brand names and generic names
        brand_mask   = med['brand name'].str.lower().str.contains(q, na=False)
        generic_mask = med['generic'].str.lower().str.contains(q, na=False)
        matches      = med[brand_mask | generic_mask].head(30)

        results = []
        seen_generics = defaultdict(list)

        for _, row in matches.iterrows():
            price = extract_price(str(row.get('package container', '')))
            seen_generics[str(row['generic']).strip()].append({
                'brand':        str(row['brand name']).strip(),
                'generic':      str(row['generic']).strip(),
                'manufacturer': str(row['manufacturer']).strip(),
                'strength':     str(row['strength']).strip(),
                'dosage_form':  str(row['dosage form']).strip(),
                'price':        price,
                'package':      str(row['package container']).strip(),
            })

        for generic_name, brands in seen_generics.items():
            brands.sort(key=lambda x: (x['price'] is None, x['price'] or 9999))
            # Get drug class & indication from generic table
            gen_row = gen[gen['generic name'].str.lower() == generic_name.lower()]
            drug_class = ''
            indication = ''
            if not gen_row.empty:
                drug_class = str(gen_row.iloc[0].get('drug class', '')).strip()
                indication = str(gen_row.iloc[0].get('indication', '')).strip()
            results.append({
                'generic':    generic_name,
                'drug_class': drug_class,
                'indication': indication,
                'brands':     brands[:10],
                'cheapest':   brands[0] if brands else None,
            })

        return jsonify({"results": results[:8]})
    except Exception as e:
        return jsonify({"results": [], "error": str(e)})


# ── HTML Template (entire frontend) ──────────────────────────────────────────
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>MedEase BD</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=Hind+Siliguri:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
:root {
  --bg:        #0a0f1e;
  --surface:   #111827;
  --surface2:  #1a2332;
  --border:    #1e3a5f;
  --accent:    #00d4ff;
  --accent2:   #0ea5e9;
  --green:     #10b981;
  --red:       #f43f5e;
  --amber:     #f59e0b;
  --text:      #e2e8f0;
  --muted:     #64748b;
  --pill:      #0f2744;
}
*{margin:0;padding:0;box-sizing:border-box;}
html,body{height:100%;font-family:'Sora',sans-serif;background:var(--bg);color:var(--text);overflow:hidden;}

/* ─── Layout ─────────────────────────────── */
.app{display:grid;grid-template-columns:280px 1fr;height:100vh;}

/* ─── Sidebar ────────────────────────────── */
.sidebar{
  background:var(--surface);
  border-right:1px solid var(--border);
  display:flex;flex-direction:column;
  overflow:hidden;
}
.logo{
  padding:24px 20px 16px;
  border-bottom:1px solid var(--border);
}
.logo-mark{
  display:flex;align-items:center;gap:10px;
  font-size:1.25rem;font-weight:700;letter-spacing:-.5px;
  color:var(--accent);
}
.logo-mark svg{width:28px;height:28px;}
.logo-sub{font-size:.72rem;color:var(--muted);margin-top:4px;font-weight:400;}

/* status badge */
.status-bar{
  padding:12px 16px;
  display:flex;align-items:center;gap:8px;
  font-size:.72rem;font-weight:500;
  border-bottom:1px solid var(--border);
}
.dot{width:8px;height:8px;border-radius:50%;flex-shrink:0;}
.dot.loading{background:var(--amber);animation:pulse 1s infinite;}
.dot.ready   {background:var(--green);}
.dot.error   {background:var(--red);}
@keyframes pulse{0%,100%{opacity:1;}50%{opacity:.3;}}

/* nav tabs */
.nav{padding:12px 8px;display:flex;flex-direction:column;gap:4px;}
.nav-btn{
  display:flex;align-items:center;gap:10px;
  padding:10px 12px;border-radius:8px;
  font-size:.82rem;font-weight:500;
  background:none;border:none;color:var(--muted);
  cursor:pointer;text-align:left;transition:all .2s;
}
.nav-btn:hover{background:var(--surface2);color:var(--text);}
.nav-btn.active{background:var(--pill);color:var(--accent);border:1px solid var(--border);}
.nav-btn svg{width:16px;height:16px;flex-shrink:0;}

/* history */
.history-section{flex:1;overflow-y:auto;padding:8px;}
.history-label{font-size:.65rem;font-weight:600;color:var(--muted);
  padding:8px 12px 4px;letter-spacing:.08em;text-transform:uppercase;}
.history-item{
  padding:8px 12px;border-radius:6px;font-size:.75rem;
  color:var(--muted);cursor:pointer;transition:all .15s;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
}
.history-item:hover{background:var(--surface2);color:var(--text);}

/* lang badge */
.lang-badges{padding:12px 16px;border-top:1px solid var(--border);
  display:flex;gap:6px;flex-wrap:wrap;}
.badge{
  padding:3px 10px;border-radius:20px;font-size:.68rem;font-weight:600;
  background:var(--pill);color:var(--accent);border:1px solid var(--border);
}

/* ─── Main area ──────────────────────────── */
.main{display:flex;flex-direction:column;overflow:hidden;position:relative;}

/* ─── Tab panels ─────────────────────────── */
.panel{display:none;flex:1;flex-direction:column;overflow:hidden;}
.panel.active{display:flex;}

/* ── Chat panel ─────────── */
.chat-header{
  padding:16px 24px;border-bottom:1px solid var(--border);
  display:flex;align-items:center;justify-content:space-between;
  background:var(--surface);flex-shrink:0;
}
.chat-title{font-size:.95rem;font-weight:600;}
.clear-btn{
  padding:6px 14px;border-radius:6px;font-size:.72rem;font-weight:500;
  background:none;border:1px solid var(--border);color:var(--muted);
  cursor:pointer;transition:all .2s;
}
.clear-btn:hover{border-color:var(--red);color:var(--red);}

.messages{flex:1;overflow-y:auto;padding:20px 24px;display:flex;flex-direction:column;gap:16px;}

.msg{display:flex;gap:12px;animation:fadeUp .3s ease;}
@keyframes fadeUp{from{opacity:0;transform:translateY(8px);}to{opacity:1;transform:translateY(0);}}
.msg.user{flex-direction:row-reverse;}

.avatar{
  width:32px;height:32px;border-radius:8px;flex-shrink:0;
  display:flex;align-items:center;justify-content:center;
  font-size:.75rem;font-weight:700;
}
.avatar.bot{background:linear-gradient(135deg,#0ea5e9,#00d4ff);color:#fff;}
.avatar.user{background:linear-gradient(135deg,#6366f1,#8b5cf6);color:#fff;}

.bubble{
  max-width:72%;padding:12px 16px;border-radius:12px;
  font-size:.85rem;line-height:1.65;
}
.bubble.bot{
  background:var(--surface2);border:1px solid var(--border);
  border-top-left-radius:4px;
}
.bubble.user{
  background:linear-gradient(135deg,#1e3a5f,#0f2744);
  border:1px solid #1e3a5f;border-top-right-radius:4px;
}
.bubble strong{color:var(--accent);font-weight:600;}
.bubble .price-tag{
  display:inline-block;padding:1px 8px;border-radius:4px;
  background:#0a2e1a;color:var(--green);font-size:.75rem;
  border:1px solid #0d4a28;margin:0 2px;
}

/* typing indicator */
.typing-dots{display:flex;gap:4px;padding:4px 0;}
.typing-dots span{
  width:6px;height:6px;border-radius:50%;background:var(--accent);
  animation:bounce .8s infinite;
}
.typing-dots span:nth-child(2){animation-delay:.15s;}
.typing-dots span:nth-child(3){animation-delay:.3s;}
@keyframes bounce{0%,80%,100%{transform:translateY(0);}40%{transform:translateY(-6px);}}

/* input area */
.input-area{
  padding:16px 24px;border-top:1px solid var(--border);
  background:var(--surface);flex-shrink:0;
}
.input-row{
  display:flex;gap:10px;align-items:flex-end;
  background:var(--surface2);border:1px solid var(--border);
  border-radius:12px;padding:8px 8px 8px 16px;
  transition:border-color .2s;
}
.input-row:focus-within{border-color:var(--accent);}
#chatInput{
  flex:1;background:none;border:none;outline:none;
  color:var(--text);font-family:'Sora',sans-serif;
  font-size:.875rem;resize:none;max-height:120px;
  padding:4px 0;line-height:1.5;
}
#chatInput::placeholder{color:var(--muted);}
.send-btn{
  width:36px;height:36px;border-radius:8px;flex-shrink:0;
  background:linear-gradient(135deg,var(--accent2),var(--accent));
  border:none;cursor:pointer;
  display:flex;align-items:center;justify-content:center;
  transition:opacity .2s;
}
.send-btn:hover{opacity:.85;}
.send-btn:disabled{opacity:.4;cursor:not-allowed;}
.send-btn svg{width:16px;height:16px;color:#fff;}

.quick-btns{display:flex;gap:6px;margin-bottom:10px;flex-wrap:wrap;}
.quick-btn{
  padding:5px 12px;border-radius:20px;font-size:.72rem;
  background:var(--pill);border:1px solid var(--border);
  color:var(--muted);cursor:pointer;transition:all .15s;
  white-space:nowrap;
}
.quick-btn:hover{color:var(--accent);border-color:var(--accent);}

/* ── Search panel ───────── */
.search-panel-inner{padding:24px;overflow-y:auto;}
.search-box{
  display:flex;gap:10px;margin-bottom:24px;
}
.search-input-wrap{
  flex:1;display:flex;align-items:center;gap:10px;
  background:var(--surface2);border:1px solid var(--border);
  border-radius:10px;padding:10px 16px;transition:border-color .2s;
}
.search-input-wrap:focus-within{border-color:var(--accent);}
.search-input-wrap svg{width:18px;height:18px;color:var(--muted);flex-shrink:0;}
#searchInput{
  flex:1;background:none;border:none;outline:none;
  color:var(--text);font-size:.9rem;font-family:'Sora',sans-serif;
}
#searchInput::placeholder{color:var(--muted);}
.search-go{
  padding:10px 20px;border-radius:10px;font-size:.82rem;font-weight:600;
  background:linear-gradient(135deg,var(--accent2),var(--accent));
  border:none;color:#fff;cursor:pointer;transition:opacity .2s;
}
.search-go:hover{opacity:.85;}

/* result cards */
.result-card{
  background:var(--surface);border:1px solid var(--border);
  border-radius:12px;margin-bottom:16px;overflow:hidden;
  animation:fadeUp .3s ease;
}
.card-header{
  padding:16px 20px;border-bottom:1px solid var(--border);
  display:flex;align-items:flex-start;justify-content:space-between;gap:12px;
}
.card-generic{font-size:1rem;font-weight:700;color:var(--accent);}
.card-class{font-size:.72rem;color:var(--muted);margin-top:3px;}
.card-indication{
  font-size:.72rem;padding:3px 10px;border-radius:20px;
  background:var(--pill);color:var(--accent2);border:1px solid var(--border);
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:200px;
}
.card-body{padding:16px 20px;}
.section-label{
  font-size:.65rem;font-weight:600;color:var(--muted);
  text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px;
}
.cheapest-box{
  background:#0a2e1a;border:1px solid #0d4a28;border-radius:8px;
  padding:12px 16px;margin-bottom:16px;
  display:flex;align-items:center;justify-content:space-between;
}
.cheapest-name{font-weight:600;font-size:.9rem;}
.cheapest-price{font-size:1.1rem;font-weight:700;color:var(--green);}
.cheapest-mfr{font-size:.72rem;color:var(--muted);}
.brands-grid{display:flex;flex-direction:column;gap:6px;}
.brand-row{
  display:flex;align-items:center;justify-content:space-between;
  padding:7px 12px;border-radius:7px;background:var(--surface2);
  font-size:.8rem;
}
.brand-row-left{display:flex;flex-direction:column;gap:2px;}
.brand-name{font-weight:500;}
.brand-mfr{font-size:.7rem;color:var(--muted);}
.brand-price{
  font-weight:600;color:var(--green);font-size:.82rem;white-space:nowrap;
}
.no-price{color:var(--muted);}
.show-more{
  width:100%;padding:8px;border-radius:7px;font-size:.75rem;
  background:none;border:1px dashed var(--border);color:var(--muted);
  cursor:pointer;margin-top:6px;transition:all .15s;
}
.show-more:hover{border-color:var(--accent);color:var(--accent);}

/* ── About panel ─────────── */
.about-inner{padding:32px;max-width:640px;overflow-y:auto;}
.about-inner h2{font-size:1.4rem;font-weight:700;color:var(--accent);margin-bottom:8px;}
.about-inner p{font-size:.85rem;color:var(--muted);line-height:1.7;margin-bottom:16px;}
.feature-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:20px 0;}
.feature-card{
  background:var(--surface2);border:1px solid var(--border);
  border-radius:10px;padding:16px;
}
.feature-card h4{font-size:.85rem;font-weight:600;color:var(--text);margin-bottom:6px;}
.feature-card p{font-size:.75rem;color:var(--muted);line-height:1.6;}

/* scrollbar */
::-webkit-scrollbar{width:4px;}
::-webkit-scrollbar-track{background:transparent;}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:4px;}

/* loading overlay */
.loading-overlay{
  position:fixed;inset:0;background:var(--bg);
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  z-index:100;transition:opacity .5s;
}
.loading-overlay.hidden{opacity:0;pointer-events:none;}
.loader-logo{font-size:2rem;font-weight:700;color:var(--accent);margin-bottom:16px;}
.loader-bar{width:200px;height:3px;background:var(--surface2);border-radius:3px;overflow:hidden;}
.loader-fill{height:100%;background:linear-gradient(90deg,var(--accent2),var(--accent));
  width:0;animation:load 3s ease forwards;}
@keyframes load{to{width:100%;}}
.loader-msg{font-size:.8rem;color:var(--muted);margin-top:12px;}
</style>
</head>
<body>

<!-- Loading overlay -->
<div class="loading-overlay" id="loadingOverlay">
  <div class="loader-logo">💊 MedEase BD</div>
  <div class="loader-bar"><div class="loader-fill"></div></div>
  <div class="loader-msg" id="loaderMsg">Loading AI model…</div>
</div>

<div class="app">
  <!-- ── Sidebar ── -->
  <aside class="sidebar">
    <div class="logo">
      <div class="logo-mark">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M9 3H5a2 2 0 00-2 2v4m6-6h10a2 2 0 012 2v4M9 3v18m0 0h10a2 2 0 002-2v-4M9 21H5a2 2 0 01-2-2v-4m0 0h18"/>
        </svg>
        MedEase BD
      </div>
      <div class="logo-sub">Bilingual Medicine Assistant · Offline</div>
    </div>

    <div class="status-bar">
      <div class="dot loading" id="statusDot"></div>
      <span id="statusText">Loading model…</span>
    </div>

    <nav class="nav">
      <button class="nav-btn active" onclick="switchTab('chat')" id="tab-chat">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/>
        </svg>
        AI Chat
      </button>
      <button class="nav-btn" onclick="switchTab('search')" id="tab-search">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/>
        </svg>
        Medicine Search
      </button>
      <button class="nav-btn" onclick="switchTab('about')" id="tab-about">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/>
        </svg>
        About
      </button>
    </nav>

    <div style="padding:8px;">
      <div class="history-label">Recent Chats</div>
      <div id="historyList"></div>
    </div>

    <div class="lang-badges">
      <span class="badge">English</span>
      <span class="badge">বাংলা</span>
      <span class="badge">Banglish</span>
    </div>
  </aside>

  <!-- ── Main ── -->
  <main class="main">

    <!-- Chat Panel -->
    <div class="panel active" id="panel-chat">
      <div class="chat-header">
        <span class="chat-title">💬 Ask about any medicine</span>
        <button class="clear-btn" onclick="clearChat()">Clear Chat</button>
      </div>
      <div class="messages" id="messages">
        <div class="msg">
          <div class="avatar bot">AI</div>
          <div class="bubble bot">
            <strong>স্বাগতম! Welcome to MedEase BD 👋</strong><br><br>
            আমি আপনাকে ওষুধ সম্পর্কে সাহায্য করতে পারি।<br>
            I can help you with medicines — generic names, brands, cheapest alternatives, dosage, and side effects.<br><br>
            <em style="color:var(--muted);font-size:.8rem;">Type in English, বাংলা, or Banglish!</em>
          </div>
        </div>
      </div>
      <div class="input-area">
        <div class="quick-btns">
          <button class="quick-btn" onclick="quickAsk('What is Napa?')">What is Napa?</button>
          <button class="quick-btn" onclick="quickAsk('নাপা কি?')">নাপা কি?</button>
          <button class="quick-btn" onclick="quickAsk('Paracetamol er shosto brand ki?')">Shosto brand?</button>
          <button class="quick-btn" onclick="quickAsk('Amoxicillin side effects ki?')">Side effects?</button>
          <button class="quick-btn" onclick="quickAsk('Metformin এর বিকল্প কি?')">বিকল্প ওষুধ?</button>
        </div>
        <div class="input-row">
          <textarea id="chatInput" rows="1" placeholder="Ask about any medicine… (English / বাংলা / Banglish)"
            onkeydown="handleKey(event)" oninput="autoResize(this)"></textarea>
          <button class="send-btn" id="sendBtn" onclick="sendMessage()">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
              <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/>
            </svg>
          </button>
        </div>
      </div>
    </div>

    <!-- Search Panel -->
    <div class="panel" id="panel-search">
      <div class="search-panel-inner">
        <h2 style="font-size:1.1rem;font-weight:700;margin-bottom:6px;">🔍 Medicine Search</h2>
        <p style="font-size:.8rem;color:var(--muted);margin-bottom:20px;">
          Search by brand name or generic name — instant results from the database.
        </p>
        <div class="search-box">
          <div class="search-input-wrap">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/>
            </svg>
            <input id="searchInput" type="text" placeholder="e.g. Napa, Paracetamol, Azithromycin…"
              onkeydown="if(event.key==='Enter') doSearch()"/>
          </div>
          <button class="search-go" onclick="doSearch()">Search</button>
        </div>
        <div id="searchResults"></div>
      </div>
    </div>

    <!-- About Panel -->
    <div class="panel" id="panel-about">
      <div class="about-inner">
        <h2>About MedEase BD</h2>
        <p>A fully offline, bilingual AI medicine assistant trained on Bangladesh medicine data. No internet required after setup.</p>
        <div class="feature-grid">
          <div class="feature-card">
            <h4>🤖 Local AI</h4>
            <p>Runs 100% on your machine. No API calls, no data sent anywhere.</p>
          </div>
          <div class="feature-card">
            <h4>🌐 3 Languages</h4>
            <p>English, বাংলা, and Banglish — reply in the language you ask.</p>
          </div>
          <div class="feature-card">
            <h4>💰 Price Aware</h4>
            <p>Shows cheapest brand alternatives with ৳ prices from BD market.</p>
          </div>
          <div class="feature-card">
            <h4>📊 21,000+ Medicines</h4>
            <p>Covers 21,714 brands across 1,700+ generics from 241 manufacturers.</p>
          </div>
        </div>
        <p style="font-size:.75rem;color:var(--muted);margin-top:16px;padding:12px;background:var(--surface2);border-radius:8px;border-left:3px solid var(--amber);">
          ⚠️ <strong>Disclaimer:</strong> This tool is for informational purposes only. Always consult a licensed doctor or pharmacist before taking any medicine.
        </p>
      </div>
    </div>

  </main>
</div>

<script>
const messages   = [];
const historyLog = [];
let   isTyping   = false;

// ── Status polling ─────────────────────────────────────────────
async function pollStatus() {
  try {
    const r = await fetch('/api/status');
    const s = await r.json();
    const dot  = document.getElementById('statusDot');
    const txt  = document.getElementById('statusText');
    const overlay = document.getElementById('loadingOverlay');
    const msg  = document.getElementById('loaderMsg');
    if (s.loaded) {
      dot.className = 'dot ready';
      txt.textContent = 'Model ready';
      overlay.classList.add('hidden');
    } else if (s.error) {
      dot.className = 'dot error';
      txt.textContent = 'Model error';
      msg.textContent = s.error;
    } else {
      setTimeout(pollStatus, 1500);
    }
  } catch(e) { setTimeout(pollStatus, 2000); }
}
pollStatus();

// ── Tab switching ──────────────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('panel-'+name).classList.add('active');
  document.getElementById('tab-'+name).classList.add('active');
}

// ── Auto-resize textarea ───────────────────────────────────────
function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}

function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
}

function quickAsk(q) {
  document.getElementById('chatInput').value = q;
  sendMessage();
}

// ── Render message ─────────────────────────────────────────────
function appendMsg(role, text, streaming=false) {
  const wrap = document.getElementById('messages');
  const div  = document.createElement('div');
  div.className = `msg ${role}`;
  const av = document.createElement('div');
  av.className  = `avatar ${role === 'user' ? 'user' : 'bot'}`;
  av.textContent= role === 'user' ? 'আপনি' : 'AI';
  const bub = document.createElement('div');
  bub.className = `bubble ${role === 'user' ? 'user' : 'bot'}`;
  if (streaming) {
    bub.innerHTML = '<div class="typing-dots"><span></span><span></span><span></span></div>';
  } else {
    bub.innerHTML = formatText(text);
  }
  div.appendChild(av);
  div.appendChild(bub);
  wrap.appendChild(div);
  wrap.scrollTop = wrap.scrollHeight;
  return bub;
}

function formatText(t) {
  return t
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/৳([\d.]+)/g, '<span class="price-tag">৳$1</span>')
    .replace(/\n/g, '<br>');
}

// ── Send message ───────────────────────────────────────────────
async function sendMessage() {
  const input = document.getElementById('chatInput');
  const text  = input.value.trim();
  if (!text || isTyping) return;

  input.value = '';
  input.style.height = 'auto';
  isTyping = true;
  document.getElementById('sendBtn').disabled = true;

  appendMsg('user', text);
  messages.push({role:'user', content:text});

  // Add to history sidebar
  if (historyLog.length === 0 || historyLog[historyLog.length-1] !== text) {
    historyLog.push(text);
    const hl = document.getElementById('historyList');
    const item = document.createElement('div');
    item.className = 'history-item';
    item.textContent = text;
    item.onclick = () => quickAsk(text);
    hl.insertBefore(item, hl.firstChild);
    if (hl.children.length > 8) hl.removeChild(hl.lastChild);
  }

  const botBub = appendMsg('bot', '', true);

  // Streaming response
  let fullText = '';
  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({messages, stream: true})
    });
    const reader = res.body.getReader();
    const dec    = new TextDecoder();
    botBub.innerHTML = '';

    while(true) {
      const {done, value} = await reader.read();
      if (done) break;
      const lines = dec.decode(value).split('\n');
      for (const line of lines) {
        if (!line.startsWith('data:')) continue;
        const d = line.slice(5).trim();
        if (d === '[DONE]') break;
        try {
          const j = JSON.parse(d);
          if (j.token) { fullText += j.token; botBub.innerHTML = formatText(fullText); }
          if (j.error) { botBub.innerHTML = `<span style="color:var(--red)">${j.error}</span>`; }
        } catch(e) {}
      }
      document.getElementById('messages').scrollTop = 999999;
    }
  } catch(e) {
    botBub.innerHTML = `<span style="color:var(--red)">Connection error: ${e.message}</span>`;
  }

  if (fullText) messages.push({role:'assistant', content:fullText});
  // Keep last 10 turns
  while (messages.length > 20) messages.splice(0,2);

  isTyping = false;
  document.getElementById('sendBtn').disabled = false;
}

function clearChat() {
  messages.length = 0;
  const wrap = document.getElementById('messages');
  wrap.innerHTML = '';
  appendMsg('bot', 'Chat cleared. Ask me anything about medicines! 💊');
}

// ── Medicine Search ────────────────────────────────────────────
async function doSearch() {
  const q   = document.getElementById('searchInput').value.trim();
  const box = document.getElementById('searchResults');
  if (!q) return;

  box.innerHTML = '<div style="color:var(--muted);font-size:.82rem;padding:12px 0">Searching…</div>';
  try {
    const r = await fetch('/api/search', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({query: q})
    });
    const data = await r.json();
    if (!data.results || data.results.length === 0) {
      box.innerHTML = '<div style="color:var(--muted);font-size:.85rem;padding:20px 0;text-align:center">No results found for "<strong style=\'color:var(--text)\'>' + q + '</strong>"</div>';
      return;
    }
    box.innerHTML = data.results.map(renderCard).join('');
  } catch(e) {
    box.innerHTML = `<div style="color:var(--red)">Error: ${e.message}</div>`;
  }
}

function renderCard(r) {
  const cheapest = r.cheapest;
  const cheapBox = cheapest && cheapest.price ? `
    <div class="cheapest-box">
      <div>
        <div class="cheapest-name">🏷️ ${cheapest.brand}</div>
        <div class="cheapest-mfr">${cheapest.manufacturer} · ${cheapest.dosage_form} · ${cheapest.strength}</div>
      </div>
      <div class="cheapest-price">৳${cheapest.price.toFixed(2)}</div>
    </div>` : '';

  const brandRows = r.brands.slice(0,5).map(b => `
    <div class="brand-row">
      <div class="brand-row-left">
        <span class="brand-name">${b.brand}</span>
        <span class="brand-mfr">${b.manufacturer}</span>
      </div>
      ${b.price ? `<span class="brand-price">৳${b.price.toFixed(2)}</span>` : '<span class="no-price">—</span>'}
    </div>`).join('');

  const moreCount = r.brands.length - 5;
  const moreBtn   = moreCount > 0 ? `<button class="show-more" onclick="askMore('${r.generic}')">+ ${moreCount} more brands — Ask AI for details</button>` : '';

  return `
  <div class="result-card">
    <div class="card-header">
      <div>
        <div class="card-generic">${r.generic}</div>
        <div class="card-class">${r.drug_class || 'Drug'}</div>
      </div>
      ${r.indication ? `<div class="card-indication" title="${r.indication}">${r.indication.slice(0,40)}${r.indication.length>40?'…':''}</div>` : ''}
    </div>
    <div class="card-body">
      ${cheapBox ? `<div class="section-label">💚 Cheapest Option</div>${cheapBox}` : ''}
      <div class="section-label">All Brands</div>
      <div class="brands-grid">${brandRows}</div>
      ${moreBtn}
    </div>
  </div>`;
}

function askMore(generic) {
  switchTab('chat');
  document.getElementById('chatInput').value = `What are all the brands and cheapest alternatives for ${generic}?`;
  sendMessage();
}

// Enter key for search
document.getElementById('searchInput').addEventListener('keydown', e => {
  if (e.key === 'Enter') doSearch();
});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print(f"🌐 Open your browser → http://{args.host}:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
