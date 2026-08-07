#Requires -Version 5.1
<#
One-command setup for Windows: creates the venv, installs Python deps, and
checks for the external tools the app needs. Run from the project root:

    .\setup.ps1
#>

$ErrorActionPreference = "Stop"

function Test-CommandExists($name) {
    return [bool](Get-Command $name -ErrorAction SilentlyContinue)
}

Write-Host "== RAG project setup ==" -ForegroundColor Cyan

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment (.venv)..."
    py -m venv .venv
} else {
    Write-Host "Virtual environment already exists, skipping."
}

Write-Host "Installing Python dependencies..."
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip | Out-Null
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt

Write-Host ""
Write-Host "== Checking external tools ==" -ForegroundColor Cyan

if (Test-CommandExists "ollama") {
    Write-Host "Ollama: found." -ForegroundColor Green
} else {
    Write-Host "Ollama not found. Installing via winget (required — this is the LLM backend)..." -ForegroundColor Yellow
    winget install --id Ollama.Ollama -e --accept-package-agreements --accept-source-agreements
}

$models = & ollama list 2>$null
if (-not $models -or ($models | Select-String "llama3.2:3b") -eq $null) {
    Write-Host "Pulling llama3.2:3b (default model, ~2GB, one-time download)..." -ForegroundColor Yellow
    ollama pull llama3.2:3b
} else {
    Write-Host "Default model already pulled." -ForegroundColor Green
}

if (Test-CommandExists "tesseract") {
    Write-Host "Tesseract OCR: found." -ForegroundColor Green
} else {
    $answer = Read-Host "Tesseract OCR not found (needed only for scanned PDFs/images). Install now? [y/N]"
    if ($answer -eq "y") {
        winget install --id UB-Mannheim.TesseractOCR
    } else {
        Write-Host "Skipped — scanned PDFs and image OCR won't work until this is installed." -ForegroundColor DarkYellow
    }
}

if (Test-CommandExists "ffmpeg") {
    Write-Host "ffmpeg: found." -ForegroundColor Green
} else {
    $answer = Read-Host "ffmpeg not found (needed only for video/audio transcription). Install now? [y/N]"
    if ($answer -eq "y") {
        winget install --id Gyan.FFmpeg -e --accept-package-agreements --accept-source-agreements
    } else {
        Write-Host "Skipped — video/audio ingestion won't work until this is installed." -ForegroundColor DarkYellow
    }
}

Write-Host ""
Write-Host "== Setup complete ==" -ForegroundColor Cyan
Write-Host "Start the app with:"
Write-Host "  .\.venv\Scripts\python.exe -m uvicorn src.api.main:app --reload" -ForegroundColor White
Write-Host "Then open http://127.0.0.1:8000"
