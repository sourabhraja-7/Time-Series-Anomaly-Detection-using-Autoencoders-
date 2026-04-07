# Time-Series-Anomaly-Detection-using-Autoencoders-

# Time-Series Anomaly Detection using Autoencoders

This project implements and compares multiple **autoencoder-based anomaly detection models** in **PyTorch** on a real-world machine temperature time-series dataset. The goal is to detect abnormal behavior by learning how to reconstruct normal patterns and flagging points with high reconstruction error as anomalies.

## Project Overview

Anomaly detection is important in machine monitoring, predictive maintenance, and manufacturing-style operational settings where unusual sensor behavior may indicate failures or system issues. In this project, I used the **Numenta Anomaly Benchmark (NAB)** dataset, specifically the `machine_temperature_system_failure.csv` file, to build an end-to-end anomaly detection pipeline.

The workflow includes:

- data loading and exploration
- anomaly window labeling
- missing-value handling
- normalization
- train/validation/test splitting
- training multiple autoencoder architectures
- threshold-based anomaly classification
- evaluation using classification and reconstruction metrics

## Dataset

- **Dataset:** Numenta Anomaly Benchmark (NAB)
- **File used:** `machine_temperature_system_failure.csv`
- **Type:** univariate machine temperature time-series data
- **Use case:** anomaly detection in machine/system behavior

The dataset contains timestamped temperature values, and anomaly labels are created using the official NAB anomaly windows.

## Preprocessing

The following preprocessing steps were performed:

- converted timestamps to datetime format
- sorted records by time
- removed duplicate timestamps
- resampled the series to **5-minute intervals**
- handled missing values using:
  - linear interpolation
  - forward fill / backward fill
- normalized the temperature values using **MinMaxScaler**

## Data Split Strategy

To make anomaly detection more realistic:

- the model was trained only on **normal samples**
- validation and test sets contained both **normal** and **anomalous** samples

This helps the autoencoder learn normal reconstruction patterns and detect deviations during evaluation.

## Models Implemented

Three different autoencoder architectures were built and compared:

### 1. Shallow Dense Autoencoder
A simple fully connected baseline model with a small bottleneck.

### 2. Deep Dense Autoencoder with Dropout
A deeper fully connected architecture with dropout for regularization.

### 3. Conv1D Autoencoder
A convolutional 1D autoencoder designed to better capture local patterns in sequential data.

## Training Setup

- **Framework:** PyTorch
- **Loss Function:** Mean Squared Error (MSE)
- **Optimizer:** Adam
- **Epochs:** 100
- **Hyperparameters tuned:** learning rate, batch size
- **Model comparison criteria:**
  - F1-score
  - ROC-AUC
  - reconstruction loss
  - training time per epoch
  - number of trainable parameters

## Final Model

The **Conv1D Autoencoder** was selected as the best-performing architecture.

### Final Architecture Summary
- encoder:
  - Conv1D: 1 → 16 filters
  - ReLU
  - Conv1D: 16 → 32 filters
  - ReLU
- decoder:
  - Conv1D: 32 → 16 filters
  - ReLU
  - Conv1D: 16 → 1 filter
  - Sigmoid

## Thresholding Strategy

After training, reconstruction errors were computed for each sample.  
Anomalies were identified by applying thresholds on reconstruction error.

Different threshold strategies were explored, including:

- mean + 2.0 std
- mean + 2.1 std
- mean + 2.5 std
- 70th percentile
- 75th percentile
- 80th percentile
- 85th percentile
- 90th percentile

The final model used an **F1-optimized threshold**.

## Results

### Best Model: Conv1D Autoencoder

**Reconstruction Loss**
- Train Loss: **0.000007**
- Validation Loss: **0.000027**
- Test Loss: **0.000016**

**Classification Metrics**
- Accuracy: **0.7539**
- Precision: **0.5527**
- Recall: **0.4674**
- F1-score: **0.5065**
- ROC-AUC: **0.6792**

### Threshold Analysis
Several thresholding methods were compared to study the precision-recall tradeoff.  
Among the tested models and threshold strategies, the Conv1D autoencoder produced the best overall anomaly detection performance.

## Key Learnings

- autoencoders can learn normal system behavior without direct anomaly supervision during training
- threshold selection has a strong effect on precision and recall
- reconstruction-error overlap between normal and anomalous points makes anomaly detection challenging
- Conv1D architectures can better capture local sequential patterns than simple dense baselines

## Files / Contents

This notebook includes:

- dataset loading and labeling
- exploratory data analysis and visualizations
- preprocessing pipeline
- implementation of 3 autoencoder architectures
- model training and validation
- threshold-based anomaly detection
- metric comparison and final analysis

## Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn
- torchinfo
- KaggleHub

## How to Run

1. Install the required libraries.
2. Open the notebook.
3. Run all cells in order.
4. Make sure the NAB dataset file is accessible.
5. Review the training outputs, plots, and final evaluation section.

## Future Improvements

Some possible improvements for this project:

- train on windowed sequences instead of single-point inputs
- use LSTM or Transformer-based autoencoders
- improve threshold calibration using validation-based search
- try weighted losses or hybrid anomaly scoring
- test on additional machine or manufacturing-related datasets

## Author

**Sourabh Raja**
