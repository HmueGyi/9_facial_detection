# 🎭 Facial Emotion Detection

A deep learning project for real-time facial emotion detection using **PyTorch** and **ResNet-50** transfer learning.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Emotion Classes](#emotion-classes)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Training](#training)
- [Inference](#inference)
- [Results](#results)
- [Documentation](#documentation)
- [Author](#author)

---

## Overview

This project implements a **9-class facial emotion classifier** that can:
- Train on custom facial expression datasets
- Perform real-time emotion detection via webcam
- Predict emotions from image files

**Key Technologies:**
- **ResNet-50** pretrained on ImageNet (transfer learning)
- **MTCNN** for face detection (in webcam mode)
- **PyTorch** for deep learning
- **OpenCV** for video processing

---

## Features

| Feature | Description |
|---------|-------------|
| 🧠 Transfer Learning | Fine-tuned ResNet-50 with frozen early layers |
| ⚖️ Class Balancing | Weighted loss function for imbalanced datasets |
| 📈 Early Stopping | Prevents overfitting by monitoring validation loss |
| 🎛️ LR Scheduling | `ReduceLROnPlateau` for adaptive learning rate |
| 📹 Real-time Detection | Webcam-based emotion detection with MTCNN |
| 💾 Best Model Saving | Automatically saves the best performing model |
| 📊 Visualization | Confusion matrix, loss plots, prediction samples |

---

## Emotion Classes

| ID | Emotion | ID | Emotion |
|----|---------|----|---------|
| 0 | Angry | 5 | Natural |
| 1 | Contempt | 6 | Sad |
| 2 | Disgust | 7 | Sleepy |
| 3 | Fear | 8 | Surprised |
| 4 | Happy | | |

---

## Project Structure

```
9_facial_detection/
├── README.md                      # This file
├── INSTALL.md                     # Installation guide
├── ARCHITECTURE.md                # Model architecture details
├── Emotion_detection_Train.ipynb  # Training notebook
├── Webcam_detect.py               # Real-time webcam detection script
├── best_model.pth                 # Trained model weights
└── data/                          # Dataset (not included)
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
```

---

## Quick Start

### 1. Clone and Install

```bash
cd /home/mr_robot/Desktop/RestNet/9_facial_detection
pip install -r requirements.txt  # or see INSTALL.md
```

### 2. Run Webcam Detection (with pretrained model)

```bash
python3 Webcam_detect.py --model best_model.pth --source 0
```

### 3. Train Your Own Model

```bash
jupyter notebook Emotion_detection_Train.ipynb
```

---

## Training

### Using Jupyter Notebook (Interactive)

```bash
jupyter notebook Emotion_detection_Train.ipynb
```

### Using nbconvert (Headless)

```bash
jupyter nbconvert --to notebook --execute Emotion_detection_Train.ipynb --output executed.ipynb
```

### Key Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_epochs` | 15 | Number of training epochs |
| `batch_size` | 32 | Images per batch |
| `lr` | 0.0001 | Learning rate |
| `patience` | 3 | Early stopping patience |
| `image_size` | 224 | Input image size |

---

## Inference

### Webcam Detection

```bash
# Default webcam (index 0)
python3 Webcam_detect.py

# Specify camera and device
python3 Webcam_detect.py --source 1 --device cuda

# Process a video file and save output
python3 Webcam_detect.py --source video.mp4 --save-output output.mp4

# Adjust confidence threshold
python3 Webcam_detect.py --conf 0.7
```

### CLI Options for Webcam_detect.py

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `best_model.pth` | Path to model weights |
| `--source` | `0` | Camera index or video path |
| `--device` | `auto` | `auto`, `cpu`, or `cuda` |
| `--conf` | `0.5` | Min face detection confidence |
| `--save-output` | - | Save annotated video to path |
| `--width` | - | Capture width |
| `--height` | - | Capture height |

---

## Results

After training, you'll get:

1. **`best_model.pth`** - Saved model weights
2. **Training/Validation Loss Plot** - Overfitting detection
3. **Confusion Matrix** - Per-class accuracy visualization
4. **Sample Predictions** - Visual inspection of results

---

## Documentation

| Document | Description |
|----------|-------------|
| [INSTALL.md](INSTALL.md) | Installation guide and dependencies |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Model architecture and design details |
| [Emotion_detection_Train.ipynb](Emotion_detection_Train.ipynb) | Annotated training notebook |

---

## 🛠️ Quick Commands

```bash
# Check GPU availability
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Count dataset images
ls data/train/images | wc -l

# Preview a label file
head -n 1 data/train/labels/$(ls data/train/labels | head -n1)

# Run webcam detection
python3 Webcam_detect.py --source 0 --device auto
```

---

## Author

**Hmue_Gyi**  
Launch Date: April 2025  
Framework: PyTorch + torchvision

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [PyTorch](https://pytorch.org/) - Deep learning framework
- [torchvision](https://pytorch.org/vision/) - Pretrained ResNet models
- [facenet-pytorch](https://github.com/timesler/facenet-pytorch) - MTCNN face detection
- [OpenCV](https://opencv.org/) - Computer vision library
