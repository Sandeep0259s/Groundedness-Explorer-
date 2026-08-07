#!/usr/bin/env bash
# One-command setup for Linux/Mac: creates the venv, installs Python deps,
# and checks for the external tools the app needs.
set -euo pipefail

echo "== RAG project setup =="

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment (.venv)..."
    python3 -m venv .venv
else
    echo "Virtual environment already exists, skipping."
fi

echo "Installing Python dependencies..."
./.venv/bin/python -m pip install --upgrade pip >/dev/null
./.venv/bin/python -m pip install -r requirements.txt

echo ""
echo "== Checking external tools =="

if command -v ollama >/dev/null 2>&1; then
    echo "Ollama: found."
else
    echo "Ollama not found. Install it from https://ollama.com/download (required — this is the LLM backend)."
fi

if command -v ollama >/dev/null 2>&1; then
    if ! ollama list 2>/dev/null | grep -q "llama3.2:3b"; then
        echo "Pulling llama3.2:3b (default model, ~2GB, one-time download)..."
        ollama pull llama3.2:3b
    else
        echo "Default model already pulled."
    fi
fi

if command -v tesseract >/dev/null 2>&1; then
    echo "Tesseract OCR: found."
else
    echo "Tesseract OCR not found (needed only for scanned PDFs/images)."
    echo "  Debian/Ubuntu: sudo apt-get install tesseract-ocr"
    echo "  macOS:         brew install tesseract"
fi

if command -v ffmpeg >/dev/null 2>&1; then
    echo "ffmpeg: found."
else
    echo "ffmpeg not found (needed only for video/audio transcription)."
    echo "  Debian/Ubuntu: sudo apt-get install ffmpeg"
    echo "  macOS:         brew install ffmpeg"
fi

echo ""
echo "== Setup complete =="
echo "Start the app with:"
echo "  ./.venv/bin/python -m uvicorn src.api.main:app --reload"
echo "Then open http://127.0.0.1:8000"
