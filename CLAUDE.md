# CLAUDE.md — Drawtex

## Markdown File Purpose
Allows Claude Code to understand codebase and find relevant context itself.

## Project Overview

Drawtex is a PyQt6 desktop application that converts handwritten drawings into LaTeX. Users draw text, math equations, or diagrams on a canvas (with tablet/pen support), and a pipeline of ML models classifies each stroke (text/math/diagram via a bidirectional LSTM), then routes them to the appropriate OCR engine (TrOCR for text, Pix2Text for math) to produce LaTeX output rendered in a live preview pane.

## Tech Stack

- **Python 3.12** (virtual env in `.venv/`)
- **PyQt6 6.10** — GUI framework
- **PyTorch 2.10** — LSTM stroke classifier
- **Transformers 4.57 (TrOCR)** — handwriting OCR
- **Pix2Text 1.1** — math expression OCR (ONNX-backed)
- **Scikit-Learn 1.8** — feature scaling (StandardScaler)
- **Matplotlib 3.10** — LaTeX rendering in the preview pane
- **OpenCV 4.13 / Pillow 12.1** — image processing
- **CUDA 12.8** — GPU acceleration (auto-fallback to CPU)

## Directory Guide

```
core/
  main.py          # Main GUI app (DrawingScene, DrawingView, MainWindow, AnalysisWorker) ~1400 lines
  analyze.py       # ML pipeline: feature extraction → LSTM classify → OCR routing → LaTeX output ~750 lines

models/
  lstm.py          # BiLSTM model definition (9 features → 3 classes, 2 layers, 64 hidden)
  feature_extraction.py  # 9-feature extraction per stroke + sequence padding utilities

scripts/
  train_lstm.py         # Train LSTM on data/collected_strokes.json
  verify_lstm.py        # Validate model on data/validation_data.json
  train_random_forest.py # Legacy classifier (superseded by LSTM)

weights/
  0_9163_best_model.pth  # Best LSTM (91.63% acc) — default used by analyze.py
  0_9147_best_model.pth  # Alternate model
  scaler.save            # StandardScaler fitted during training

data/
  collected_strokes.json   # Training dataset (~26 MB)
  validation_data.json     # Validation dataset (~1 MB)

debug_images/              # Intermediate images saved during analysis (for debugging)
```

## Commands

```bash
# Run the app (WSL → Windows Python for tablet support)
./run.sh

# Run the app directly
python core/main.py

# Train the LSTM classifier
python scripts/train_lstm.py

# Validate the trained model
python scripts/verify_lstm.py

# Install dependencies
pip install -r requirements.txt            # Linux/WSL
# Or on Windows: windows_setup_.bat
```

## Key Architecture Notes

- **Stroke features (9-dim):** dx_prev, dy_prev, dt_prev, width, height, duration, num_points, linearity, speed.
- **Classification labels:** 0=text, 1=math, 2=diagram. Diagram recognition is a placeholder — not yet implemented.
- **Threading:** Analysis runs on a `QThread` (`AnalysisWorker`) to keep the UI responsive.
- **Model loading:** Lazy singleton — all models load on first call to `analyze_vectors()`.
- **Data collection mode:** Toggle `data_collection_mode = True` in `core/main.py` to label strokes for training data.
- **Config constants** live at the top of `core/analyze.py` (model paths, dimensions, label map, render sizes).
