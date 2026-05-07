#!/usr/bin/env python3
"""
MedEase BD — Dataset Generator
Reads the raw CSV files from MedData/ and produces qa_bilingual.json
Run: python generate_dataset.py --data-dir MedData --output qa_bilingual.json
"""

import pandas as pd, json, re, random, argparse, html
from pathlib import Path
from collections import defaultdict

try:
    from bs4 import BeautifulSoup
    def strip_html(text, limit=600):
        if not isinstance(text, str): return ''
        t = BeautifulSoup(text, 'html.parser').get_text(separator=' ').strip()
        return re.sub(r'\s+', ' ', t)[:limit]
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable,"-m","pip","install","beautifulsoup4","-q"])
    from bs4 import BeautifulSoup
    def strip_html(text, limit=600):
        if not isinstance(text, str): return ''
        t = BeautifulSoup(text, 'html.parser').get_text(separator=' ').strip()
        return re.sub(r'\s+', ' ', t)[:limit]

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default="MedData")
parser.add_argument("--output",   default="qa_bilingual.json")
parser.add_argument("--seed",     type=int, default=42)
args = parser.parse_args()

random.seed(args.seed)
DATA = Path(args.data_dir)

print("📂 Loading CSVs…")
med = pd.read_csv(DATA / "medicine.csv")
gen = pd.read_csv(DATA / "generic.csv")
ind = pd.read_csv(DATA / "indication.csv")
dc  = pd.read_csv(DATA / "drug class.csv")
print(f"   Medicines: {len(med):,} | Generics: {len(gen):,}")

# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_price(pkg):
    if not isinstance(pkg, str): return None
    m = re.search(r'৳\s*([\d.]+)', pkg)
    return float(m.group(1)) if m else None

def fmt_chat(q, a):
    """Gemma / Qwen / LLaMA chat template"""
    return {"text": f"<start_of_turn>user\n{q}<end_of_turn>\n<start_of_turn>model\n{a}<end_of_turn>"}

# ── Build generic → brand lookup ──────────────────────────────────────────────
generic_brands = defaultdict(list)
for _, row in med.iterrows():
    g = str(row['generic']).strip()
    price = extract_price(str(row.get('package container','')))
    generic_brands[g].append({
        'brand':        str(row['brand name']).strip(),
        'manufacturer': str(row['manufacturer']).strip(),
        'strength':     str(row['strength']).strip(),
        'dosage_form':  str(row['dosage form']).strip(),
        'price':        price,
        'package':      str(row['package container']).strip(),
    })

for g in generic_brands:
    generic_brands[g].sort(key=lambda x: (x['price'] is None, x['price'] or 9999))

# ── QA Generation ─────────────────────────────────────────────────────────────
qa_pairs = []

def add(en_q, en_a, bn_q, bn_a, banglish_q=None, banglish_a=None):
    if en_q and en_a:
        qa_pairs.append(fmt_chat(en_q, en_a))
    if bn_q and bn_a:
        qa_pairs.append(fmt_chat(bn_q, bn_a))
    if banglish_q and banglish_a:
        qa_pairs.append(fmt_chat(banglish_q, banglish_a))

print("⚙️  Generating generic-level QA pairs…")
for _, row in gen.iterrows():
    gname      = str(row['generic name']).strip()
    drug_class = str(row.get('drug class','')).strip()
    indication = str(row.get('indication','')).strip()
    dosage     = strip_html(str(row.get('dosage description','')))
    side_fx    = strip_html(str(row.get('side effects description','')))
    preg       = strip_html(str(row.get('pregnancy and lactation description','')))
    contra     = strip_html(str(row.get('contraindications description','')))
    storage    = strip_html(str(row.get('storage conditions description','')))
    interact   = strip_html(str(row.get('interaction description','')))
    precaution = strip_html(str(row.get('precautions description','')))

    brands = generic_brands.get(gname, [])
    if not brands:
        continue

    cheap      = [b for b in brands if b['price'] is not None][:5]
    top_brands = brands[:10]
    brand_list = ', '.join([b['brand'] for b in top_brands])
    cheap_str  = ', '.join([f"{b['brand']} (৳{b['price']}, {b['manufacturer']})" for b in cheap]) if cheap else 'N/A'

    # What is
    add(
        f"What is {gname}?",
        f"{gname} is a {drug_class} medication used for: {indication}. {dosage[:300] if dosage else ''}",
        f"{gname} কি?",
        f"{gname} হলো একটি {drug_class} ওষুধ যা {indication} এর চিকিৎসায় ব্যবহৃত হয়। {dosage[:200] if dosage else ''}",
        f"{gname} ki?",
        f"{gname} holo ekta {drug_class} oshudh, {indication} er jonno use kora hoy।"
    )

    # Brand names
    add(
        f"What are the brand names of {gname} available in Bangladesh?",
        f"Available brands of {gname} in Bangladesh: {brand_list}.",
        f"বাংলাদেশে {gname} এর কোন কোন ব্র্যান্ড পাওয়া যায়?",
        f"বাংলাদেশে {gname} এর পরিচিত ব্র্যান্ডগুলো হলো: {brand_list}।",
        f"{gname} er brand name ki ki Bangladesh e?",
        f"Bangladesh e {gname} er brand: {brand_list}।"
    )

    # Cheapest / affordable
    if cheap:
        add(
            f"What is the cheapest brand of {gname}?",
            f"Most affordable brands of {gname}: {cheap_str}. All contain the same active ingredient.",
            f"{gname} এর সবচেয়ে সস্তা ব্র্যান্ড কোনটি?",
            f"{gname} এর সাশ্রয়ী ব্র্যান্ডগুলো: {cheap_str}। সবগুলোতে একই সক্রিয় উপাদান আছে।",
            f"{gname} er shosto brand konta?",
            f"{gname} er shosto brand gulo: {cheap_str}।"
        )
        add(
            f"Suggest a low-cost alternative for {gname}.",
            f"Budget-friendly options for {gname}: {cheap_str}. These are therapeutically equivalent.",
            f"{gname} এর কম দামের বিকল্প কি?",
            f"{gname} এর কম দামের বিকল্প: {cheap_str}। এগুলো চিকিৎসাগতভাবে সমতুল্য।",
            f"{gname} er kom dame alternative ki?",
            f"{gname} er alternative: {cheap_str}।"
        )

    # Generic name of brand
    if brands:
        b0 = brands[0]['brand']
        add(
            f"What is the generic name of {b0}?",
            f"The generic name of {b0} is {gname} — a {drug_class} used for {indication}.",
            f"{b0} এর জেনেরিক নাম কি?",
            f"{b0} এর জেনেরিক নাম হলো {gname} — এটি {drug_class} শ্রেণির ওষুধ, {indication} এর জন্য।",
            f"{b0} er generic name ki?",
            f"{b0} er generic name holo {gname}।"
        )

    # Dosage
    if dosage:
        add(
            f"What is the dose of {gname}?",
            f"Dosage of {gname}: {dosage[:500]}",
            f"{gname} এর ডোজ কত?",
            f"{gname} এর ডোজ: {dosage[:400]}",
            f"{gname} koto dose nite hoy?",
            f"{gname} er dose: {dosage[:300]}"
        )

    # Side effects
    if side_fx:
        add(
            f"What are the side effects of {gname}?",
            f"Side effects of {gname}: {side_fx[:500]}",
            f"{gname} এর পার্শ্বপ্রতিক্রিয়া কী কী?",
            f"{gname} এর পার্শ্বপ্রতিক্রিয়া: {side_fx[:400]}",
            f"{gname} er side effects ki?",
            f"{gname} er side effects: {side_fx[:300]}"
        )

    # Pregnancy
    if preg:
        add(
            f"Is {gname} safe during pregnancy?",
            f"Pregnancy & lactation info for {gname}: {preg[:500]}",
            f"গর্ভাবস্থায় {gname} নিরাপদ কি?",
            f"গর্ভাবস্থায় {gname}: {preg[:400]}",
            f"Pregnancy te {gname} newa jabe?",
            f"Pregnancy te {gname}: {preg[:300]}"
        )

    # Contraindications
    if contra:
        add(
            f"Who should not take {gname}?",
            f"Contraindications for {gname}: {contra[:500]}",
            f"কারা {gname} খাবেন না?",
            f"{gname} এর contraindication: {contra[:400]}",
            f"Ke {gname} khabe na?",
            f"{gname} ja khawa jabena: {contra[:300]}"
        )

    # Drug interactions
    if interact:
        add(
            f"What drugs interact with {gname}?",
            f"Drug interactions of {gname}: {interact[:500]}",
            f"{gname} এর সাথে কোন ওষুধের মিথস্ক্রিয়া আছে?",
            f"{gname} এর drug interaction: {interact[:400]}",
            None, None
        )

    # Precautions
    if precaution:
        add(
            f"What precautions should I take with {gname}?",
            f"Precautions for {gname}: {precaution[:500]}",
            f"{gname} খাওয়ার সময় কি সতর্কতা নেওয়া উচিত?",
            f"{gname} এর সতর্কতা: {precaution[:400]}",
            None, None
        )

    # Storage
    if storage:
        add(
            f"How should {gname} be stored?",
            f"Storage for {gname}: {storage[:400]}",
            f"{gname} কিভাবে সংরক্ষণ করবেন?",
            f"{gname} সংরক্ষণ: {storage[:300]}",
            None, None
        )

print(f"   Generic QA: {len(qa_pairs):,} pairs")

# ── Brand-level QA (sampled 20%) ──────────────────────────────────────────────
print("⚙️  Generating brand-level QA pairs…")
for _, row in med.iterrows():
    if random.random() > 0.20:
        continue
    brand       = str(row['brand name']).strip()
    generic     = str(row['generic']).strip()
    manufacturer= str(row['manufacturer']).strip()
    strength    = str(row['strength']).strip()
    dosage_form = str(row['dosage form']).strip()
    price       = extract_price(str(row.get('package container','')))
    price_str   = f"৳{price:.2f}" if price else 'price not available'

    # Find alternatives (same generic, different brand)
    alts = [b for b in generic_brands.get(generic,[]) if b['brand'] != brand][:3]
    alt_str = ', '.join([f"{b['brand']} (৳{b['price']})" for b in alts if b['price']]) if alts else 'none listed'

    add(
        f"What is {brand}?",
        f"{brand} ({dosage_form}, {strength}) is a brand of {generic} made by {manufacturer}. Price: {price_str}. Alternatives: {alt_str}.",
        f"{brand} কি?",
        f"{brand} হলো {generic} এর একটি ব্র্যান্ড ({dosage_form}, {strength}), প্রস্তুতকারক: {manufacturer}। মূল্য: {price_str}। বিকল্প: {alt_str}।",
        f"{brand} ki oshudh?",
        f"{brand} holo {generic} er ekta brand ({dosage_form}, {strength}), {manufacturer} theke। Dam: {price_str}।"
    )

    if alts:
        add(
            f"What is the alternative to {brand}?",
            f"Alternatives to {brand} ({generic}): {alt_str}. All contain {generic}.",
            f"{brand} এর বিকল্প কি?",
            f"{brand} ({generic}) এর বিকল্প: {alt_str}। সবগুলোতে {generic} আছে।",
            f"{brand} er alternative ki?",
            f"{brand} er alternative: {alt_str}।"
        )

print(f"   Total QA pairs: {len(qa_pairs):,}")

# ── Shuffle and save ──────────────────────────────────────────────────────────
random.shuffle(qa_pairs)
with open(args.output, 'w', encoding='utf-8') as f:
    json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

print(f"\n✅ Dataset saved → {args.output}  ({len(qa_pairs):,} examples)")
