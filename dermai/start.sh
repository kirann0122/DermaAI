#!/bin/bash
# DermAI Pro — Start Script
echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║          DermAI Pro v3.0 — Startup           ║"
echo "║    Skin Disease Prediction System            ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

cd "$(dirname "$0")"

# Check dependencies
echo "[1/3] Checking dependencies..."
python3 -c "import flask, tensorflow, PIL, numpy" 2>/dev/null || {
  echo "Installing dependencies..."
  pip install flask tensorflow pillow numpy flask-cors --break-system-packages -q
}
echo "      ✓ All dependencies OK"

# Train model if not present
if [ ! -f "model/skin_model.h5" ]; then
  echo ""
  echo "[2/3] Training model (first run — takes ~2 minutes)..."
  python3 train_model.py
  echo "      ✓ Model trained and saved"
else
  echo "[2/3] ✓ Model already trained (model/skin_model.h5)"
fi

echo ""
echo "[3/3] Starting Flask server..."
echo ""
echo "  ➜  Open in browser: http://localhost:5000"
echo "  ➜  Press Ctrl+C to stop"
echo ""

python3 app.py
