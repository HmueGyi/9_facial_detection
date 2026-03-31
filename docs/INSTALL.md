# Installation Guide

For a quick path, see `README.md`. This file provides extra platform detail.

## Recommended: Conda Environment

From project root:

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

Manual alternative:

```bash
conda env create -f environment.yml
conda activate facial-emotion
```

Update existing env:

```bash
conda env update -f environment.yml --prune
```

## Pip-Only Alternative

```bash
pip install -r requirements.txt
```

## Verify Installation

```bash
python -c "import torch, torchvision, cv2, mediapipe; print('ok')"
```

## Run Inference

```bash
python src/webcam_detect.py --source 0
```

## Common Fixes

### `conda: command not found`

- Open Miniforge/Anaconda prompt, or
- run `conda init`, restart terminal.

### MediaPipe API error (`no attribute solutions`)

```bash
pip uninstall -y mediapipe mediapipe-nightly
pip install mediapipe==0.10.14
```

### Webcam not detected

- Try `--source 1` (or `2`)
- Close other camera apps
- Check OS camera permissions
