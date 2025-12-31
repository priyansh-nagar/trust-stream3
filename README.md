# 🛡️ DeepTrust – Explainable AI Deepfake Detection

## Overview
DeepTrust is an explainable AI system for detecting AI-generated images.
It focuses on accuracy, transparency, and long-term learning using
human feedback.

## Why DeepTrust?
- Uses verified research-grade pretrained models
- No black-box APIs
- Visual explanations (heatmaps)
- Trust Score for user clarity
- Designed for long-term deployment

## How It Works
1. Upload an image
2. Pretrained deepfake model analyzes artifacts
3. Trust Score (0–100) is generated
4. Heatmap shows suspicious regions
5. User feedback improves the model

## Model Loading (Important)
DeepTrust uses Torch Hub to automatically load a verified pretrained
deepfake detection model on first run.
- No manual model download required
- Model is cached locally after first use
- Fully legal and reproducible

## Tech Stack
- Python
- Streamlit
- PyTorch
- EfficientNet
- SQLite

## Future Scope
- Video deepfake detection
- Face-level analysis
- Browser extension

## Disclaimer
DeepTrust is a decision-support system and should not be used as the sole
authority for forensic conclusions.