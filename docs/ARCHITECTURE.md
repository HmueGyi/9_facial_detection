# Architecture

## Pipeline Overview

Each input frame goes through the following steps:

1. **Capture** from webcam or video file (OpenCV).
2. **Face detection** with MediaPipe (`FaceDetection`).
3. **Crop and clamp** bounding box to frame boundaries.
4. **Preprocess** face image:
   - resize to `224x224`
   - normalize with ImageNet mean/std
5. **Emotion classification** with EfficientNet-V2-S custom head.
6. **Render output** (label + confidence + face box) and optionally save video.

## Model

- Backbone: `torchvision.models.efficientnet_v2_s`
- Head:
  - `Linear(in_features -> 128)`
  - `ReLU`
  - `Dropout(0.3)`
  - `Linear(128 -> num_classes)`

Implementation: `src/models/emotion_model.py`

## Checkpoint Loading Compatibility

`get_model()` supports common checkpoint layouts:

- raw `state_dict`
- wrapped dict with `model_state_dict` or `state_dict`
- key prefixes from different saves:
  - `backbone.*`
  - raw backbone keys (`features.*`, `classifier.*`)
  - optional `module.*` (DataParallel)

This avoids load failures when training/export format differs slightly.

## Class Labels

Current inference labels in `src/webcam_detect.py`:

`['angry', 'happy', 'neutral', 'sad', 'suprised', 'tired']`

These must match training dataset folder names and order.

## Main Files

- `src/webcam_detect.py`: CLI inference, MediaPipe + OpenCV loop
- `src/models/emotion_model.py`: model definition and robust loader
- `notebooks/facial-training-efficientnet.ipynb`: training/export notebook
