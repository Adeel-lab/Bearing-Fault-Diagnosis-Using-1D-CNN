# 🔩 Bearing Fault Diagnosis Using 1D CNN on CWRU Dataset

A complete end-to-end pipeline for bearing fault diagnosis using vibration signals from the **Case Western Reserve University (CWRU)** benchmark dataset. Raw accelerometer signals are transformed into **envelope spectra** via classical signal processing, then classified by a **1D Convolutional Neural Network** into one of four bearing health states.

> **Test Accuracy: 96.22%** on held-out Drive-End data  
> **Cross-sensor generalisation** tested on Fan-End (FE) signals

***

## 📋 Table of Contents

- [Overview](#overview)
- [Pipeline](#pipeline)
- [Dataset](#dataset)
- [Signal Processing](#signal-processing)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Known Issues & Improvements](#known-issues--improvements)
- [References](#references)

***

## Overview

Bearing faults are among the most common failure modes in rotating machinery. This project implements a **supervised fault classification pipeline** that:

1. Loads raw `.mat` vibration files from the CWRU dataset
2. Applies bandpass filtering + Hilbert envelope demodulation to each signal segment
3. Computes the **envelope spectrum** (FFT of envelope) as the feature representation
4. Trains a **1D CNN** to classify segments into: `Normal`, `IR` (Inner Race), `OR` (Outer Race), or `B` (Ball) fault
5. Evaluates the trained model on unseen **Fan-End (FE) sensor** signals to test cross-sensor generalisation

***

## Pipeline

```
Raw .mat file
     │
     ▼
DE_time signal extracted
     │
     ▼
Segmentation (1024 samples, 50% overlap)
     │
     ▼
Bandpass Filter (500–3000 Hz, 4th-order Butterworth)
     │
     ▼
Hilbert Transform → Amplitude Envelope
     │
     ▼
FFT of Envelope → 513-point Envelope Spectrum
     │
     ▼
1D CNN (3× Conv1D + MaxPool + Dense)
     │
     ▼
Fault Class: Normal / IR / OR / B
```

***

## Dataset

**Case Western Reserve University (CWRU) Bearing Dataset**  
→ [https://engineering.case.edu/bearingdatacenter](https://engineering.case.edu/bearingdatacenter)

| Property | Value |
|---|---|
| Sampling rate | 12 000 Hz |
| Fault types | Normal, Inner Race (IR), Outer Race (OR), Ball (B) |
| Sensor used for training | Drive End (DE) accelerometer |
| Sensor used for inference | Fan End (FE) accelerometer |
| Total processed segments | 70 061 |
| Feature vector length | 513 (envelope spectrum bins) |
| Train / Test split | 80% / 20% (stratified) |

Download the dataset and place it in the following structure:

```
CWRU-dataset-main/
├── Normal/
├── IR/
├── OR/
└── B/
```

Update `root_path` in the notebook to point to your local copy.

***

## Signal Processing

### Bandpass Filter
A 4th-order zero-phase Butterworth filter (500–3 000 Hz) isolates the bearing housing resonance band, where fault impulses are amplified. `filtfilt` ensures no phase distortion.

### Hilbert Envelope
The Hilbert transform converts the filtered signal into its analytic representation. The magnitude gives the **amplitude envelope** — a slowly-varying signal that captures periodic modulations caused by fault impacts.

### Envelope Spectrum
The FFT of the envelope transforms periodic fault impulses into sharp spectral peaks at **Bearing Characteristic Frequencies** (BPFO, BPFI, BSF). These peaks are the primary discriminative features learned by the CNN.

***

## Model Architecture

```
Input: (513, 1)
│
├── Conv1D(32, k=5, ReLU) → (509, 32)
├── MaxPooling1D(2)        → (254, 32)
│
├── Conv1D(64, k=5, ReLU) → (250, 64)
├── MaxPooling1D(2)        → (125, 64)
│
├── Conv1D(128, k=5, ReLU) → (121, 128)
├── MaxPooling1D(2)         → (60, 128)
│
├── Flatten → (7680,)
├── Dense(128, ReLU)
└── Dense(4, Softmax)

Total parameters: 1,035,268 (~3.95 MB)
```

The three stacked Conv1D blocks with progressively increasing filter counts (32→64→128) learn hierarchical spectral features — from low-level frequency peaks in shallow layers to fault-class-specific patterns in deeper layers.

***

## Results

### Test Set Performance (DE signal)

| Metric | Value |
|---|---|
| Test Accuracy | **96.22%** |
| Test Loss | 0.130 |
| Epochs trained | 15 |
| Best validation accuracy | 96.06% (epoch 15) |

### Training Curve Summary

| Epoch | Train Acc | Val Acc | Train Loss | Val Loss |
|---|---|---|---|---|
| 1 | 82.8% | 89.9% | 0.407 | 0.253 |
| 3 | 94.4% | 94.9% | 0.140 | 0.131 |
| 6 | 96.6% | 95.8% | 0.089 | 0.106 |
| 10 | 98.2% | 95.8% | 0.049 | 0.121 |
| 15 | 98.6% | 96.1% | 0.042 | 0.127 |

The growing gap between training (~98.6%) and validation (~96.1%) accuracy after epoch 9 indicates **mild overfitting**. Adding `Dropout` or `EarlyStopping` (at epoch ~9) is recommended.

### Cross-Sensor Inference (FE signal)

Per-segment predictions on Fan-End signals show the model predominantly predicts `B` (Ball) for signals that are ground-truth `OR` (Outer Race). This is a classic **domain shift** effect: the DE-trained model learns resonance path features specific to the Drive-End mounting location. The FE sensor, mounted on the opposite side of the motor, produces a different spectral signature for the same physical fault — causing misclassification. Fine-tuning on a small set of FE samples is expected to resolve this.

***

## Project Structure

```
bearing-fault-diagnosis/
│
├── bearing_fault_cnn.ipynb     # Main notebook
├── README.md                   # This file
├── requirements.txt            # Python dependencies
│
└── figures/
    ├── output_24_0.png         # Filtered vs envelope overlay
    ├── output_28_0.png         # Training accuracy & loss curves
    ├── output_30_1.png         # Segment-wise predictions (color-coded)
    ├── output_28_1.png         # FE segment-wise predictions
    └── output_32_0.png         # FE 6-panel signal analysis
```

***

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/bearing-fault-diagnosis.git
cd bearing-fault-diagnosis

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

**`requirements.txt`**
```
numpy
scipy
scikit-learn
tensorflow>=2.10
matplotlib
```

> ⚠️ **Windows GPU note:** TensorFlow >= 2.11 does not support native Windows GPU. Use WSL2 or install the `tensorflow-directml` plugin for GPU acceleration on Windows.

***

## Usage

1. Download the CWRU dataset and update `root_path` in the notebook:
   ```python
   root_path = r"path/to/your/CWRU-dataset-main"
   ```

2. Launch the notebook:
   ```bash
   jupyter lab bearing_fault_cnn.ipynb
   ```

3. Run all cells sequentially. The full pipeline (data loading → training → evaluation → FE inference) runs end-to-end.

***

## Known Issues & Improvements

| Issue | Suggested Fix |
|---|---|
| Mild overfitting after epoch 9 | Add `Dropout(0.3)` after Dense(128); use `EarlyStopping(patience=3)` |
| Domain shift on FE sensor | Fine-tune last 2 layers on a small FE labelled set |
| No confusion matrix | Add `sklearn.metrics.confusion_matrix` after test evaluation |
| Dense layer parameter bottleneck (95% of total params) | Add `GlobalAveragePooling1D` before Dense to reduce to ~100K params |
| `input_shape` deprecation warning | Replace with `layers.Input(shape=(513,1))` as first layer |
| Fixed bandpass band | Make `band_low/band_high` tunable; try 1 000–5 000 Hz for FE |

***

## References

- Smith, W.A. & Randall, R.B. (2015). *Rolling element bearing diagnostics using the Case Western Reserve University data: A benchmark study.* Mechanical Systems and Signal Processing, 64–65, 100–131.
- Case Western Reserve University Bearing Data Center: [https://engineering.case.edu/bearingdatacenter](https://engineering.case.edu/bearingdatacenter)
- Jiang, G. et al. (2019). *Multiscale Convolutional Neural Networks for Fault Diagnosis of Wind Turbine Gearbox.* IEEE Transactions on Industrial Electronics.


*Developed as part of a thesis project on machine learning-based predictive maintenance.*
