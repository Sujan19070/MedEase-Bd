#!/bin/bash
# ─────────────────────────────────────────────────────────────
#  MedEase BD — GitHub Repo Organizer
#  Run this on your Desktop to build the MVC folder structure
#  Usage: bash organize_repo.sh
# ─────────────────────────────────────────────────────────────

GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

SRC="/home/sujan/Desktop/medease_bd"
MOB="/home/sujan/Desktop/medease_mobile"
MVP_SRC="/home/sujan/Desktop/medease_mvp"
DEST="/home/sujan/Desktop/medease-bd-repo"

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║   MedEase BD — Repo Organizer            ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════╝${NC}"
echo ""

# ── Create MVC folder structure ────────────────────────────
echo -e "${YELLOW}Creating MVC folders...${NC}"
mkdir -p "$DEST/model"
mkdir -p "$DEST/controller"
mkdir -p "$DEST/view/templates"
mkdir -p "$DEST/mobile/screens"
mkdir -p "$DEST/mvp"
mkdir -p "$DEST/data/MedData"
mkdir -p "$DEST/docs/screenshots"
echo -e "${GREEN}✅ Folders created${NC}"

# ── MODEL layer ────────────────────────────────────────────
echo -e "${YELLOW}Copying Model layer...${NC}"
cp "$SRC/generate_dataset.py"  "$DEST/model/"
cp "$SRC/train.py"              "$DEST/model/"
cp "$SRC/export_gguf_v4.py"    "$DEST/model/"
cp "$SRC/chat.py"               "$DEST/model/"
echo -e "${GREEN}✅ Model layer done${NC}"

# ── CONTROLLER layer ───────────────────────────────────────
echo -e "${YELLOW}Copying Controller layer...${NC}"
cp "$SRC/app.py"    "$DEST/controller/"
cp "$SRC/start.sh"  "$DEST/controller/"
echo -e "${GREEN}✅ Controller layer done${NC}"

# ── MOBILE ────────────────────────────────────────────────
echo -e "${YELLOW}Copying Mobile app...${NC}"
cp "$MOB/App.js"           "$DEST/mobile/"
cp "$MOB/app.json"         "$DEST/mobile/"
cp "$MOB/babel.config.js"  "$DEST/mobile/"
cp "$MOB/package.json"     "$DEST/mobile/"
cp "$MOB/screens/"*.js     "$DEST/mobile/screens/"
echo -e "${GREEN}✅ Mobile done${NC}"

# ── MVP ────────────────────────────────────────────────────
echo -e "${YELLOW}Copying MVP...${NC}"
cp "$MVP_SRC/mvp.py"  "$DEST/mvp/"
echo -e "${GREEN}✅ MVP done${NC}"

# ── DATA ──────────────────────────────────────────────────
echo -e "${YELLOW}Copying MedData CSVs...${NC}"
cp "$SRC/MedData/"*.csv  "$DEST/data/MedData/"
echo -e "${GREEN}✅ Data done${NC}"

# ── ROOT files ─────────────────────────────────────────────
echo -e "${YELLOW}Copying root files...${NC}"
cp "$SRC/requirements.txt"  "$DEST/"
cp "$SRC/Modelfile"         "$DEST/"
echo -e "${GREEN}✅ Root files done${NC}"

# ── Summary ───────────────────────────────────────────────
echo ""
echo -e "${CYAN}════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ Repo organized at: $DEST${NC}"
echo ""
echo "Next steps:"
echo ""
echo "  1. cd $DEST"
echo "  2. git init"
echo "  3. git add ."
echo "  4. git commit -m 'Initial commit — MedEase BD'"
echo "  5. git remote add origin https://github.com/YOUR_USERNAME/medease-bd.git"
echo "  6. git push -u origin main"
echo ""
echo -e "${YELLOW}Then upload model to HuggingFace:${NC}"
echo "  pip install huggingface_hub"
echo "  huggingface-cli login"
echo "  huggingface-cli upload YOUR_USERNAME/medease-bd-gemma \\"
echo "    $SRC/medease-gemma-gguf/medease-gemma-q6k.gguf \\"
echo "    --repo-type model"
echo -e "${CYAN}════════════════════════════════════════${NC}"
