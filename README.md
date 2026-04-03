# Unsupervised Clinical Anomaly Detection Pipeline

## Overview
This repository contains a state-of-the-art ensemble pipeline for unsupervised anomaly detection on multivariate surgical ICU time-series data (VitalDB). The system is engineered to handle aleatoric uncertainty and high-frequency sensor noise from clinical environments without the need for labeled training data.

The architecture combines deep geometric feature extraction with gradient-boosted decision refinement to identify physiological crises while maintaining a high Precision-Recall boundary.

## Core Components

### 1. Robust Preprocessing
- **MAD Z-Scoring**: Utilizes Median Absolute Deviation (MAD) rather than standard mean/std to establish patient-specific baselines, ensuring normalization is immune to initial sensor artifacts.
- **Inductive Design**: Implements a strict patient-wise data split to prevent manifold leakage between training and validation sets.

### 2. SQOB Autoencoder
- **Orthogonal Basis Encoder**: Projects raw vitals into a sinusoidal subspace to capture frequency-domain physiological features.
- **Exponential Recency Loss**: A custom Smooth L1 loss function that exponentially weights recent timestamps, prioritizing current patient status over historical data.
- **Stiefel Manifold Regularization**: Enforces orthogonality in the projection weights to ensure diverse feature extraction.

### 3. Knowledge Distillation & Ensembling
- **Isolation Forest**: Analyzes deep latent embeddings to identify structural density outliers.
- **Manifold Distillation**: Generates high-confidence pseudo-labels (Top 5% Anomaly / Bottom 80% Normal) based on the autoencoder's reconstruction error.
- **XGBoost Refinement**: Trains a supervised classifier on the distilled pseudo-labels to calibrate final probability scores and reduce false positives.

### 4. Inference Stability
- **Test-Time Augmentation (TTA)**: Employs Monte Carlo integration by injecting micro-noise during inference to smooth anomaly scores.
- **Gaussian Smoothing**: Deploys a temporal low-pass filter to the final ensemble output to stabilize the detection of sustained clinical deterioration.

## Performance
- **Validation Loss**: 0.8040 (Smooth L1)
- **Baseline Comparison**: Outperformed initial geometric baselines (0.94) by 14% through advanced scheduling and statistical normalization tweaks.

## Usage
1. Ensure `train_vitals.csv` and `test_vitals.csv` are in the correct directory.
2. Execute the pipeline in the Notebook
