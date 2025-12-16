# 📦 Installation Guide

This guide covers installation of all dependencies for the Facial Emotion Detection project.

---

## 📋 Requirements

| Component | Version | Required |
|-----------|---------|----------|
| Python | 3.8+ | ✅ |
| PyTorch | 2.0+ | ✅ |
| CUDA | 11.8+ | ⚡ (GPU only) |
| RAM | 8GB+ | ✅ |
| GPU VRAM | 4GB+ | ⚡ (recommended) |

---

## 🚀 Quick Install

### Option 1: Using pip (Recommended)

```bash
# Navigate to project directory
cd /home/mr_robot/Desktop/RestNet/9_facial_detection

# Install all dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install facenet-pytorch opencv-python pillow tqdm scikit-learn seaborn matplotlib numpy
```

### Option 2: Using requirements.txt

```bash
# Create requirements.txt
cat > requirements.txt << 'EOF'
torch>=2.0.0
torchvision>=0.15.0
facenet-pytorch>=2.5.2
opencv-python>=4.8.0
Pillow>=9.0.0
tqdm>=4.65.0
scikit-learn>=1.2.0
seaborn>=0.12.0
matplotlib>=3.7.0
numpy>=1.24.0
jupyter>=1.0.0
nbconvert>=7.0.0
EOF

# Install from requirements.txt
pip install -r requirements.txt
```

### Option 3: Using Conda

```bash
# Create new conda environment
conda create -n emotion_detection python=3.10 -y
conda activate emotion_detection

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other packages
pip install facenet-pytorch opencv-python tqdm scikit-learn seaborn matplotlib
```

---

## 📦 Dependency Details

### Core Dependencies

| Package | Purpose | Install Command |
|---------|---------|-----------------|
| `torch` | Deep learning framework | `pip install torch` |
| `torchvision` | Pretrained models, transforms | `pip install torchvision` |
| `facenet-pytorch` | MTCNN face detection | `pip install facenet-pytorch` |
| `opencv-python` | Video/image processing | `pip install opencv-python` |
| `Pillow` | Image loading | `pip install Pillow` |

### Training Dependencies

| Package | Purpose | Install Command |
|---------|---------|-----------------|
| `tqdm` | Progress bars | `pip install tqdm` |
| `scikit-learn` | Class weights, metrics | `pip install scikit-learn` |
| `seaborn` | Confusion matrix heatmap | `pip install seaborn` |
| `matplotlib` | Plotting | `pip install matplotlib` |
| `numpy` | Array operations | `pip install numpy` |

### Development Dependencies

| Package | Purpose | Install Command |
|---------|---------|-----------------|
| `jupyter` | Interactive notebooks | `pip install jupyter` |
| `nbconvert` | Notebook execution | `pip install nbconvert` |
| `papermill` | Parameterized notebooks | `pip install papermill` |

---

## ⚡ GPU Setup (CUDA)

### Check CUDA Availability

```bash
# Check if CUDA is available
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Check CUDA version
python3 -c "import torch; print('CUDA version:', torch.version.cuda)"

# Check GPU name
python3 -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

### Install PyTorch with CUDA Support

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### NVIDIA Driver Requirements

| CUDA Version | Minimum Driver |
|--------------|----------------|
| CUDA 11.8 | 450.80.02+ |
| CUDA 12.1 | 525.60.13+ |

```bash
# Check NVIDIA driver version
nvidia-smi
```

---

## 🐧 Linux-Specific Setup

### Ubuntu/Debian

```bash
# Update package list
sudo apt update

# Install Python and pip
sudo apt install python3 python3-pip python3-venv -y

# Install OpenCV dependencies
sudo apt install libgl1-mesa-glx libglib2.0-0 -y

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install packages
pip install --upgrade pip
pip install torch torchvision facenet-pytorch opencv-python pillow tqdm scikit-learn seaborn matplotlib
```

### Webcam Permissions

```bash
# Add user to video group (for webcam access)
sudo usermod -aG video $USER

# Verify webcam is detected
ls /dev/video*

# Test webcam with OpenCV
python3 -c "import cv2; cap=cv2.VideoCapture(0); print('Webcam OK' if cap.isOpened() else 'Webcam FAIL')"
```

---

## 🪟 Windows-Specific Setup

### Using PowerShell

```powershell
# Install Python from python.org or Microsoft Store

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install packages
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install facenet-pytorch opencv-python pillow tqdm scikit-learn seaborn matplotlib jupyter
```

### Common Windows Issues

| Issue | Solution |
|-------|----------|
| `DLL load failed` | Install Visual C++ Redistributable |
| `No module named cv2` | `pip install opencv-python-headless` |
| CUDA not found | Install CUDA Toolkit from NVIDIA |

---

## 🍎 macOS-Specific Setup

### Using Homebrew

```bash
# Install Python
brew install python@3.10

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch (MPS for Apple Silicon)
pip install torch torchvision

# Install other packages
pip install facenet-pytorch opencv-python pillow tqdm scikit-learn seaborn matplotlib
```

### Apple Silicon (M1/M2) Notes

```python
# Use MPS (Metal Performance Shaders) instead of CUDA
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

---

## ✅ Verify Installation

Run this script to verify all dependencies:

```bash
python3 << 'EOF'
import sys
print(f"Python: {sys.version}")

try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
except ImportError:
    print("❌ PyTorch not installed")

try:
    import torchvision
    print(f"✅ torchvision: {torchvision.__version__}")
except ImportError:
    print("❌ torchvision not installed")

try:
    from facenet_pytorch import MTCNN
    print("✅ facenet-pytorch installed")
except ImportError:
    print("❌ facenet-pytorch not installed")

try:
    import cv2
    print(f"✅ OpenCV: {cv2.__version__}")
except ImportError:
    print("❌ OpenCV not installed")

try:
    from PIL import Image
    print("✅ Pillow installed")
except ImportError:
    print("❌ Pillow not installed")

try:
    from sklearn.metrics import confusion_matrix
    print("✅ scikit-learn installed")
except ImportError:
    print("❌ scikit-learn not installed")

try:
    import seaborn
    print(f"✅ seaborn: {seaborn.__version__}")
except ImportError:
    print("❌ seaborn not installed")

try:
    import matplotlib
    print(f"✅ matplotlib: {matplotlib.__version__}")
except ImportError:
    print("❌ matplotlib not installed")

print("\n🎉 Installation check complete!")
EOF
```

---

## 🔧 Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError` | Package not installed | `pip install <package>` |
| `CUDA out of memory` | GPU memory full | Reduce `batch_size` |
| `Webcam not found` | Permissions or driver | Check `/dev/video*` |
| `libGL error` | Missing OpenGL | `apt install libgl1-mesa-glx` |

### Reset Environment

```bash
# Remove and recreate virtual environment
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 📞 Support

If you encounter issues:

1. Check the error message carefully
2. Verify all dependencies are installed: `pip list`
3. Check GPU drivers: `nvidia-smi`
4. Try CPU-only mode: `--device cpu`

---

**Happy Training! 🎭**
