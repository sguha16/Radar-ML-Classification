# 🎯 Radar-ML-Classification

A deep learning project to classify radar return signals using spectrograms and convolutional neural networks (CNNs).  

## 📡 Problem Overview

Radar systems generate time-varying signals from moving objects. Using signal processing and deep learning, we aim to classify targets such as pedestrians, vehicles, or gestures based on radar spectrograms.

---

## 🛠️ Features

- FFT/STFT-based preprocessing of FMCW radar data
- Spectrogram-based CNN classifier
- Evaluation with accuracy, confusion matrix
- Training on synthetic or real radar datasets

---

## 📁 Folder Structure

- `data/` – Raw or preprocessed radar samples
- `notebooks/` – Training and visualization in Jupyter
- `src/` – Source code for preprocessing, models, training
- `models/` – Saved PyTorch models
- `utils/` – Helper functions and metrics

---

## 🚀 Quickstart

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook notebooks/radar_classification.ipynb
