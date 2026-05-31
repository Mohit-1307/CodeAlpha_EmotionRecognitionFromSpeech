# 🎙️ Speech Emotion Recognition (SER)

> A deep learning system for classifying human emotions from raw speech audio using CNN with hand-engineered acoustic features — trained on the RAVDESS dataset and deployed via an interactive Streamlit web application.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset: RAVDESS](https://img.shields.io/badge/Dataset-RAVDESS-blueviolet)](https://zenodo.org/record/1188976)

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [System Architecture](#-system-architecture)
- [Feature Engineering](#-feature-engineering)
- [Model Architecture](#-model-architecture)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results & Visualizations](#-results--visualizations)
- [Limitations & Future Work](#-limitations--future-work)
- [Tech Stack](#-tech-stack)
- [Author](#-author)

---

## 🧠 Overview

Speech Emotion Recognition (SER) is the task of automatically identifying the emotional state of a speaker from audio signals. This project implements an end-to-end SER pipeline:

1. **Audio Preprocessing** — Load raw `.wav` files, trim silence, fix duration
2. **Feature Extraction** — Extract multi-dimensional acoustic feature vectors per sample
3. **Model Training** — Train a 1D CNN on the feature vectors with regularization and callbacks
4. **Evaluation** — Confusion matrix, classification report, and training curves
5. **Deployment** — Real-time prediction via a Streamlit web UI

---

## 🎯 Problem Statement

Given a raw speech audio file, classify it into one of **8 discrete emotion categories**:

| Code | Emotion   |
|------|-----------|
| 01   | Neutral   |
| 02   | Calm      |
| 03   | Happy     |
| 04   | Sad       |
| 05   | Angry     |
| 06   | Fearful   |
| 07   | Disgust   |
| 08   | Surprised |

This is a **multi-class classification** problem over a structured acoustic feature space.

---

## 🏗️ System Architecture

```
Raw Audio (.wav)
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│                     utils.py — Feature Extraction        │
│                                                          │
│  librosa.load()  →  duration=3s, offset=0.5s            │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │  MFCC (40)  →  mean + std  =  80 dims            │   │
│  │  Δ-MFCC (40) → mean + std  =  80 dims            │   │
│  │  Chroma STFT (12) → mean   =  12 dims            │   │
│  │  Mel Spectrogram (128) → mean = 128 dims         │   │
│  │  ZCR (1) → mean            =   1 dim             │   │
│  │  RMS Energy (1) → mean     =   1 dim             │   │
│  └────────────────────────────────────────────────── ┘  │
│                   Total: 302-dim feature vector          │
└──────────────────────────────────────────────────────────┘
       │
       ▼
StandardScaler  →  Normalized Feature Matrix  →  Reshape [N, 302, 1]
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│                  CNN Model (train.py)                    │
│                                                          │
│  Conv1D(64, k=3, relu) → BN → MaxPool(2) → Dropout(0.3) │
│  Conv1D(128, k=3, relu) → BN → MaxPool(2) → Dropout(0.3)│
│  Flatten → Dense(256, relu) → Dropout(0.4)              │
│  Dense(8, softmax)                                       │
└──────────────────────────────────────────────────────────┘
       │
       ▼
Predicted Emotion Class  →  Streamlit UI (app.py)
```

---

## 🔬 Feature Engineering

All features are extracted in `utils.py` using **librosa**. Each audio file is loaded at a fixed `duration=3s` with `offset=0.5s` to skip leading silence.

### Feature Breakdown

| Feature | Dimensions | Aggregation | Rationale |
|---|---|---|---|
| **MFCC** | 40 coefficients | mean + std | Captures timbre and vocal tract shape — core SER feature |
| **Δ-MFCC (Delta)** | 40 coefficients | mean + std | First-order temporal dynamics; models how MFCCs evolve |
| **Chroma STFT** | 12 bins | mean | Pitch-class energy; useful for happy/sad distinction |
| **Mel Spectrogram** | 128 bins | mean | Perceptual frequency representation of energy |
| **Zero Crossing Rate** | 1 | mean | Measures signal noisiness; correlates with anger/fear |
| **RMS Energy** | 1 | mean | Loudness proxy; distinguishes calm from intense emotions |

**Final feature vector dimension: `40×2 + 40×2 + 12 + 128 + 1 + 1 = 302`**

> **Note:** Aggregating spectral features via mean/std collapses the time axis. This discards temporal dynamics within a segment — a known limitation. Sequence models (LSTM, Transformer) or 2D CNN on spectrograms would preserve this information.

---

## 🧬 Model Architecture

The model is a **1D CNN** operating over the 302-dimensional feature vector reshaped as a 1D sequence of length 302 with 1 channel.

```
Input: (batch, 302, 1)

Conv1D(64, kernel=3, activation='relu')   → (batch, 300, 64)
BatchNormalization
MaxPooling1D(pool_size=2)                 → (batch, 150, 64)
Dropout(0.3)

Conv1D(128, kernel=3, activation='relu')  → (batch, 148, 128)
BatchNormalization
MaxPooling1D(pool_size=2)                 → (batch, 74, 128)
Dropout(0.3)

Flatten                                   → (batch, 9472)
Dense(256, activation='relu')
Dropout(0.4)

Dense(8, activation='softmax')            → (batch, 8)
```

**Loss:** Categorical Cross-Entropy  
**Optimizer:** Adam (default lr=0.001)  
**Training:** up to 50 epochs, batch size 32

### Callbacks

| Callback | Config | Purpose |
|---|---|---|
| `EarlyStopping` | patience=10, restore_best_weights=True | Prevents overfitting |
| `ReduceLROnPlateau` | patience=5, factor=0.5 | Adaptive learning rate decay |
| `ModelCheckpoint` | save_best_only=True | Saves `models/cnn_emotion_model.h5` |

---

## 📦 Dataset

**[RAVDESS](https://zenodo.org/record/1188976)** — Ryerson Audio-Visual Database of Emotional Speech and Song

- 24 professional actors (12 male, 12 female)
- 1440 audio files across 8 emotion classes
- Balanced gender representation
- 16-bit audio, 48kHz, stereo `.wav` files
- Filename encodes metadata: `03-01-06-01-02-01-12.wav`
  - Modality – Vocal Channel – **Emotion** – Intensity – Statement – Repetition – Actor

### Downloading the Dataset

```bash
# Option 1: Kaggle
kaggle datasets download -d uwrfkaggler/ravdess-emotional-speech-audio

# Option 2: Direct download from Zenodo
# https://zenodo.org/record/1188976

# Unzip into the data/ directory
unzip archive.zip -d data/
```

Expected structure:
```
data/
├── Actor_01/
│   ├── 03-01-01-01-01-01-01.wav
│   └── ...
├── Actor_02/
└── ...
```

---

## 📁 Project Structure

```
CodeAlpha_EmotionRecognitionFromSpeech/
│
├── data/                        # RAVDESS dataset (Actor_XX/ subdirs)
│
├── models/                      # Saved model artifacts (auto-created)
│   └── cnn_emotion_model.h5     # Best checkpoint from training
│
├── outputs/                     # Training artifacts (auto-created)
│   ├── confusion_matrix.png
│   ├── training_curves.png
│   ├── emotion_distribution.png
│   ├── mfcc_visualization.png
│   └── model_comparison.png
│
├── utils.py                     # Feature extraction pipeline
├── labels.py                    # Emotion label mappings
├── train.py                     # Model training + evaluation + visualization
├── predict.py                   # Inference on new audio files
├── app.py                       # Streamlit web application
├── requirements.txt             # Python dependencies
├── label_encoder.pkl            # Saved LabelEncoder (post-training)
├── scaler.pkl                   # Saved StandardScaler (post-training)
└── README.md
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.8+
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/Mohit-1307/CodeAlpha_EmotionRecognitionFromSpeech.git
cd CodeAlpha_EmotionRecognitionFromSpeech

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Key Dependencies

```
tensorflow>=2.10
librosa>=0.9
streamlit>=1.20
scikit-learn>=1.1
numpy
pandas
matplotlib
seaborn
joblib
tqdm
```

---

## 🚀 Usage

### 1. Train the Model

Ensure the RAVDESS data is placed in the `data/` directory first.

```bash
python train.py
```

This will:
- Extract features from all audio files
- Normalize features and save `scaler.pkl`
- Encode labels and save `label_encoder.pkl`
- Train the CNN with callbacks
- Save the best model to `models/cnn_emotion_model.h5`
- Generate and save evaluation plots to `outputs/`

### 2. Predict on a New Audio File

```bash
python predict.py --file path/to/audio.wav
```

### 3. Launch the Streamlit Web App

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser. Upload a `.wav` file and get real-time emotion predictions.

---

## 📊 Results & Visualizations

After training, the following outputs are generated in `outputs/`:

| File | Description |
|---|---|
| `confusion_matrix.png` | Per-class prediction accuracy heatmap |
| `training_curves.png` | Train vs. validation accuracy and loss over epochs |
| `emotion_distribution.png` | Class distribution bar chart in the dataset |
| `mfcc_visualization.png` | MFCC spectrogram of a sample audio |
| `model_comparison.png` | Model accuracy summary bar chart |

---

## ⚠️ Limitations & Future Work

### Current Limitations

- **Temporal collapse:** Mean/std aggregation of features discards within-clip dynamics. Emotions like fear or surprise often manifest in rapid transitions that a mean-pooled vector cannot capture.
- **Fixed duration:** Audio is truncated/padded to exactly 3 seconds. Longer or shorter utterances may lose or pad information.
- **Dataset scope:** RAVDESS uses acted (not naturalistic) speech. Real-world performance will degrade on spontaneous, noisy, or code-switched speech.
- **Single-dataset training:** No cross-corpus evaluation; generalization to EMODB, IEMOCAP, or CREMA-D is untested.

### Suggested Improvements

| Area | Improvement |
|---|---|
| **Features** | Replace mean-pooled features with raw 2D Mel spectrogram → 2D CNN or CNN-LSTM |
| **Model** | Add LSTM/GRU layers after CNN blocks to model temporal dynamics |
| **Data Augmentation** | Add noise injection, pitch shift, time stretch to reduce overfitting |
| **Cross-corpus eval** | Test on EMODB / IEMOCAP to measure generalization |
| **Explainability** | Add Grad-CAM or SHAP to visualize which time-frequency regions drive predictions |
| **Deployment** | Containerize with Docker; expose REST API via FastAPI |

---

## 🛠️ Tech Stack

| Component | Library/Tool |
|---|---|
| Audio Processing | `librosa` |
| Deep Learning | `TensorFlow / Keras` |
| ML Utilities | `scikit-learn` |
| Data Handling | `numpy`, `pandas` |
| Visualization | `matplotlib`, `seaborn` |
| Web App | `Streamlit` |
| Model Persistence | `joblib` |

---

## 👤 Author

**MOHIT SINGH RAJPUT** — AI / ML Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/mohitsingh1307)
[![GitHub](https://img.shields.io/badge/GitHub-121011?style=flat-square&logo=github&logoColor=white)](https://github.com/Mohit-1307)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/mohitsinghrajput1307)
[![Email](https://img.shields.io/badge/Email-D14836?style=flat-square&logo=gmail&logoColor=white)](mailto:mohitsinghdausa@gmail.com)

---

<div align="center">
<sub>If this helped you, a ⭐ on the repo is appreciated.</sub>
</div>
