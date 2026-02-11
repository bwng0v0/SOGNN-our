# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SOGNN (Self-Organized Graph Neural Network) is a PyTorch-based implementation for emotion recognition using EEG data. The model combines CNN layers with self-organized graph construction modules to classify emotional states from the SEED-IV dataset.

## Architecture

### Core Components

1. **main.py** - Training and evaluation pipeline
   - Implements Leave-One-Subject-Out (LOSO) cross-validation with 15 subjects
   - Trains for up to 200 epochs with early stopping when train_AUC > 0.99 and train_acc > 0.90
   - Uses CrossEntropyLoss with Adam optimizer (lr=0.00001, weight_decay=0.0001)
   - Batch size: 16 for both training and testing
   - Outputs results to `./result/{Network}_LOG_{version}.csv`

2. **Net.py** - Neural network architecture
   - **SOGC (Self-Organized Graph Construction)**: Custom module that learns graph structure from features
     - Uses a bottleneck linear layer followed by adjacency matrix computation via matrix multiplication
     - Applies top-k sparsification to keep only the strongest connections (topk=10 by default)
     - Processes 62 EEG channels
   - **SOGNN**: Main network architecture
     - Three-stage CNN processing with corresponding SOGC modules
     - Input shape: (batch, 1, 5, 265) representing frequency bands and features
     - Three parallel branches feeding into SOGC at different hierarchical levels
     - Final concatenation of all SOGC outputs followed by linear classifier
     - Output: 3-class emotion classification

3. **datapipe.py** - Data loading and preprocessing
   - Expects SEED-IV dataset at `../SEED4/eeg_feature_smooth/`
   - Processes differential entropy (DE) features with moving average
   - Data shape per sample: (62 channels, 5 frequency bands, 265 features)
   - Loads data from 3 sessions with 24 trials each (72 trials per subject)
   - Implements z-score normalization per subject
   - Uses PyTorch Geometric's InMemoryDataset for efficient loading
   - Stores processed datasets in `./processed/` directory with versioning

### Data Flow

1. Raw SEED-IV .mat files → `get_data()` → normalized DE features (62, 5, 265)
2. Features reshaped to (62, 1325) for graph processing
3. LOSO cross-validation: 14 subjects train, 1 subject test
4. Labels are one-hot encoded for 4 emotion classes (neutral=0, sad=1, fear=2, happy=3)

## Commands

### Running Training
```bash
python main.py
```
This will:
- Build datasets for all 15 subjects if not already processed
- Run LOSO cross-validation
- Save results to `./result/SOGNN_LOG_{version}.csv`
- Print final average validation accuracy and AUC

### Data Requirements
- Place SEED-IV dataset in `../SEED4/eeg_feature_smooth/` directory
- Required structure:
  - `../SEED4/eeg_feature_smooth/1/` (session 1)
  - `../SEED4/eeg_feature_smooth/2/` (session 2)
  - `../SEED4/eeg_feature_smooth/3/` (session 3)
- Each session folder contains 15 subject .mat files with DE features
- Labels are hardcoded in datapipe.py (24 trials per session)
- The dataset is automatically processed and cached in `./processed/` on first run

## Important Implementation Details

### Model Architecture Constants
- **Channels**: 62 EEG electrodes (hardcoded in SOGC and SOGNN)
- **Classes**: 4 emotion categories (neutral, sad, fear, happy)
- **Trials**: 24 per session, 72 per subject (3 sessions)
- **Input dimensions**: The network expects specific intermediate dimensions:
  - After conv1 + pool1: (batch*62, 32, 1, 65) → SOGC1 processes 65*32 features
  - After conv2 + pool2: (batch*62, 64, 1, 15) → SOGC2 processes 15*64 features
  - After conv3 + pool3: (batch*62, 128, 1, 2) → SOGC3 processes 2*128 features

### Training Behavior
- The model uses two output tensors: raw logits for loss computation and softmax predictions for evaluation
- Early stopping triggers when train_AUC > 0.99 AND train_acc > 0.90
- Results are saved incrementally after each fold
- Automatic versioning prevents overwriting previous experiment logs

### Dataset Processing
- First run will process all data (can take time for 15 subjects)
- Subsequent runs load pre-processed .dataset files from `./processed/`
- Each CV fold has separate train/test .dataset files
- Version number in datapipe.py controls cache invalidation

## Modifying the Code

### Changing Number of Subjects
Update `subjects = 15` in both main.py and datapipe.py (must match)

### Changing Number of Classes
Update `classes = 4` in both main.py and datapipe.py, and adjust the final Linear layer in SOGNN (currently set for SEED-IV's 4 classes)

### Changing Model Architecture
- CNN kernel sizes and pooling in Net.py lines 59-73
- Ensure the calculated feature dimensions match the SOGC input dimensions
- The three SOGC modules extract features at different temporal resolutions

### GPU Configuration
- Default device is `torch.device('cuda', 0)` (first GPU)
- Change in main.py line 99 and Net.py line 14 if needed
