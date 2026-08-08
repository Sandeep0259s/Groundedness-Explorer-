FROM python:3.12-slim

# Tesseract (scanned-PDF/image OCR) and ffmpeg (video/audio transcription)
# are optional at the Python-dependency level but needed for those specific
# ingestion paths to work — installed here so the container supports every
# format out of the box.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# This container always runs CPU-only (RAG_DEVICE=cpu below — no GPU in a
# container, no TTY to ask permission for one anyway), but plain `pip
# install torch` pulls the CUDA-enabled build by default, dragging in
# several GB of nvidia-*/cuda-toolkit packages nothing here ever uses.
# Installing the CPU-only wheel first means requirements.txt's own torch
# dependency (via sentence-transformers) is already satisfied and skips
# that entirely — smaller image, faster build, and enough disk headroom
# to matter on constrained CI runners.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY frontend/ frontend/
COPY scripts/ scripts/
COPY data/eval/ data/eval/
COPY data/raw/.gitkeep data/raw/.gitkeep

ENV RAG_DEVICE=cpu
# No TTY in a container, so the GPU permission prompt would never resolve —
# force CPU explicitly rather than relying on the non-interactive default.

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
