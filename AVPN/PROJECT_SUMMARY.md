# Project Summary - Speaker Recognition System

## Quick Overview

**What This Project Does:**
- Recognizes 20 different speakers from audio recordings of isolated names
- Uses deep learning (CNN) to learn voice patterns
- Tests noise robustness with different signal-to-noise ratios
- Provides command-line testing interface for new recordings

---

## Project Structure at a Glance

```
Audio/          → 800 training files (40 per speaker × 20 speakers)
Audio_test/     → Your test recordings (for demo_test.py)
features/       → Preprocessed MFCC features (120-dimensional)
models/         → 3 trained models (baseline, baseline noise, matched noise)
results/        → Confusion matrices, reports, training plots
tasks/          → 7 Python scripts (pipeline stages)
helpers/        → 5 utility modules (signal processing, noise mixing)
```

---

## How to Use

### Run Everything:
```powershell
cd c:/Users/buhum/5lh/ss/TestingAOu-locase
python run_project.py
```

### Test Your Voice:
```powershell
# Record yourself saying names → save in Audio_test/
# Name files: ahmed_000.wav, charlie_001.wav, etc.

python tasks/demo_test.py --pattern "Audio_test/*.wav" --true-label ahmed
```

### Clean Everything:
```powershell
python reset_project.py
```

---

## Key Files Explained

| File | What It Does |
|------|-------------|
| **run_project.py** | Runs all 6 tasks in order, skips what's already done |
| **task2.py** | Extracts MFCC features from audio (120-dim vectors) |
| **task3.py** | Trains baseline CNN on clean audio |
| **task4_create_noise.py** | Adds noise at SNR 0/10/20 dB |
| **task4_train.py** | Trains 2 noise strategies (baseline vs matched) |
| **task5.py** | Compares all models across conditions |
| **demo_test.py** | Tests with your own recordings |

---

## Technical Specs (Quick Reference)

**Audio**: 16kHz mono WAV files  
**Features**: 120-dim MFCCs (40 + Δ + ΔΔ), 99 frames  
**Model**: CNN (Conv2D → MaxPool → Dense → Softmax)  
**Training**: 80/10/10 split, Adam optimizer, 20 epochs  
**Evaluation**: Full dataset (800 samples, 40 per speaker)  

---

## Results Summary

| Model | Clean | SNR 20dB | SNR 10dB | SNR 0dB |
|-------|-------|----------|----------|---------|
| Task 3 Baseline | 95% | 79% | 33% | 16% |
| Task 4 Baseline | 91% | 75% | 39% | 16% |
| Task 4 Matched | 100% | 100% | 100% | 95% |

**Conclusion**: Matched training (clean + noisy) dramatically improves noise robustness!

---

## Where to Find Stuff

**Confusion Matrices**:
- `results/task3/confusion_matrix.png`
- `results/task4/baseline/confusion_clean.png`
- `results/task4/matched/confusion_clean_matched.png`

**Classification Reports**:
- `results/task4/matched/classification_report_clean_matched.txt`

**Training History**:
- `results/task3/training_history.png`
- `results/task4/baseline/training_history_baseline.png`

**Models**:
- `models/task3_baseline.keras`
- `models/task4_baseline.keras`
- `models/task4_matched.keras`

---

## Common Commands

```powershell
# Full pipeline
python run_project.py

# Individual tasks
python tasks/task2.py                # Extract features
python tasks/task3.py                # Train baseline
python tasks/task4_train.py          # Train noise models

# Testing
python tasks/demo_test.py --audio Audio_test/ahmed_000.wav --true-label ahmed
python tasks/demo_test.py --pattern "Audio_test/*.wav" --true-label ahmed

# Cleanup
python reset_project.py              # Delete all outputs
```

---

## For More Details

See **PROJECT_DOCUMENTATION.md** for:
- Complete technical specifications
- Detailed explanations of each file
- How the algorithms work
- Data flow diagrams
- Troubleshooting guide

---

**Status**: ✅ Project Complete  
**Date**: November 2025  
**Course**: CMP-6026A/CMP-7016A
