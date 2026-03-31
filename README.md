# Facial Emotion Detection

Real-time facial emotion detection using:
- **MediaPipe** for face detection
- **EfficientNet-V2-S (PyTorch)** for emotion classification

This repository is built for two tasks:
1. **Train** a model on Kaggle (`notebooks/facial-training-efficientnet.ipynb`)
2. **Run inference** locally (`src/webcam_detect.py`)

Training result images: [results/](results/)

## 1. Project Structure

```text
src/
  webcam_detect.py
  models/emotion_model.py
notebooks/
  facial-training-efficientnet.ipynb
results/
  loss_and_accuracy.png
  loss_curve_and_accuracy_curve.png
  confusion_matixes.png
  random_validation_prediction.png
weights/
  .gitkeep
  best_emotion_model.pth
scripts/
  run_webcam.sh
  run_webcam.bat
setup.sh
setup.bat
environment.yml
requirements.txt
docs/
  ARCHITECTURE.md
  INSTALL.md
```

## 2. Local Inference (Step by Step)

### Step 1: Create environment

Linux/macOS:

```bash
./setup.sh
conda activate facial-emotion
```

Windows:

```bat
setup.bat
conda activate facial-emotion
```

### Step 2: Put model weights in the correct path

```text
weights/best_emotion_model.pth
```

### Step 3: Run webcam inference

```bash
python src/webcam_detect.py --source 0
```

Press `q` to close the preview window.

## 3. Training on Kaggle (Step by Step)

### Step 1: Open notebook

Open:

```text
notebooks/facial-training-efficientnet.ipynb
```

in Kaggle Notebook.

### Step 2: Enable GPU

In Kaggle notebook settings, enable **GPU** accelerator.

### Step 3: Add Roboflow secrets

In **Add-ons -> Secrets**, add:
- `ROBOFLOW_API_KEY`
- `ROBOFLOW_WORKSPACE`
- `ROBOFLOW_PROJECT`
- `ROBOFLOW_VERSION` (use `1` if you are using version 1)

### Step 4: Run all cells

After training finishes:
- Save/export training visual outputs to `results/` (loss curve, accuracy curve, confusion matrix, sample predictions).
- Download model weights:

```text
best_emotion_model.pth
```

### Step 5: Move the model to local project

Place downloaded file here:

```text
weights/best_emotion_model.pth
```

Then run local inference (Section 2).

Training output images are kept in:

```text
results/
```

## 4. Roboflow Setup (Simple)

Use this rule: **set values in one place only**.
- Kaggle training -> set in **Kaggle Secrets**
- Local training -> set in **`.env`**

Required keys:
- `ROBOFLOW_API_KEY`
- `ROBOFLOW_WORKSPACE`
- `ROBOFLOW_PROJECT`
- `ROBOFLOW_VERSION` (default is `1`)

The notebook already reads them using `os.getenv(...)`, so you usually do **not** need to edit notebook code.

How to find each value:
1. `ROBOFLOW_API_KEY`
   - Go to Roboflow -> profile menu -> **Settings** -> **API**.
   - Copy your **Private API Key**.
2. `ROBOFLOW_WORKSPACE`
   - Open your project page in browser.
   - In URL `https://app.roboflow.com/<workspace>/<project>/...`, the first part is workspace.
3. `ROBOFLOW_PROJECT`
   - In the same URL, the second part is project.
4. `ROBOFLOW_VERSION`
   - Open the dataset **Versions** page in Roboflow.
   - Use the version number you exported/trained on (for example `1`, `2`, `3`).

Example URL:

```text
https://app.roboflow.com/hmue/face-emotion-classification-bg4ho/1
```

Then:
- `ROBOFLOW_WORKSPACE=hmue`
- `ROBOFLOW_PROJECT=face-emotion-classification-bg4ho`
- `ROBOFLOW_VERSION=1`

Example `.env`:

```bash
ROBOFLOW_API_KEY=your_api_key
ROBOFLOW_WORKSPACE=your_workspace
ROBOFLOW_PROJECT=your_project
ROBOFLOW_VERSION=1
```

Do not commit `.env`.

Quick inline option (works for training, but **not recommended for commit**):

```python
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "your-api-key")
ROBOFLOW_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE", "hmue")
ROBOFLOW_PROJECT = os.getenv("ROBOFLOW_PROJECT", "face-emotion-classification-bg4ho")
ROBOFLOW_VERSION = int(os.getenv("ROBOFLOW_VERSION", "1"))
EXPORT_FORMAT = "folder"
```

## 5. CLI Options

| Flag | Meaning | Default |
|---|---|---|
| `--model-path` | Path to model checkpoint | `weights/best_emotion_model.pth` |
| `--source` | Camera index (`0`, `1`, ...) or video path | `0` |
| `--device` | `auto`, `cpu`, `cuda` | `auto` |
| `--conf` | Face detection confidence | `0.5` |
| `--save-to` | Save annotated output video | disabled |

## 6. Troubleshooting

### `Error loading model` (missing/unexpected keys)

The loader in `src/models/emotion_model.py` already adapts common checkpoint key formats (`backbone.*`, raw keys, `module.*`).

### `AttributeError: module 'mediapipe' has no attribute 'solutions'`

Reinstall MediaPipe in active environment:

```bash
pip uninstall -y mediapipe mediapipe-nightly
pip install mediapipe==0.10.14
```

### Webcam does not open

- Try `--source 1` (or `2`)
- Close other camera apps
- Check OS camera permissions

### Missing packages

```bash
conda activate facial-emotion
```

or:

```bash
pip install -r requirements.txt
```

## 7. Current Class Labels

Current model labels are:

`['angry', 'happy', 'neutral', 'sad', 'suprised', 'tired']`

`suprised` is intentionally kept to match dataset folder naming.

## 8. Notes

- Recommended Python version: `3.10` to `<3.13`.
- `weights/*.pth` is git-ignored by design.
