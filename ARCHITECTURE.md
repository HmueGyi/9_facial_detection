# 🧠 Model Architecture

This document details the neural network architecture used for facial emotion detection.

---

## 📋 Overview

| Component | Description |
|-----------|-------------|
| **Base Model** | ResNet-50 (pretrained on ImageNet) |
| **Strategy** | Transfer Learning with partial fine-tuning |
| **Input Size** | 224 × 224 × 3 (RGB) |
| **Output** | 9 emotion classes |
| **Face Detection** | MTCNN (for webcam inference) |

---

## 🏗️ ResNet-50 Architecture

### What is ResNet?

**ResNet** (Residual Network) introduced **skip connections** that allow gradients to flow directly through the network, enabling training of very deep networks.

```
Input → [Conv Block] → [Residual Blocks] → [Global Avg Pool] → [FC] → Output
              ↓              ↓                                    ↓
         conv1, bn1      layer1-4                              classifier
```

### ResNet-50 Layer Structure

| Layer | Output Size | Description |
|-------|-------------|-------------|
| `conv1` | 112 × 112 × 64 | 7×7 conv, stride 2 |
| `bn1` | 112 × 112 × 64 | Batch normalization |
| `maxpool` | 56 × 56 × 64 | 3×3 max pool, stride 2 |
| `layer1` | 56 × 56 × 256 | 3 bottleneck blocks |
| `layer2` | 28 × 28 × 512 | 4 bottleneck blocks |
| `layer3` | 14 × 14 × 1024 | 6 bottleneck blocks |
| `layer4` | 7 × 7 × 2048 | 3 bottleneck blocks |
| `avgpool` | 1 × 1 × 2048 | Global average pooling |
| `fc` | 9 | Fully connected (custom) |

### Bottleneck Block (ResNet-50)

```
Input (256 channels)
    │
    ├──────────────────────────┐
    │                          │ (skip connection)
    ▼                          │
[1×1 Conv, 64] → BN → ReLU     │
    │                          │
    ▼                          │
[3×3 Conv, 64] → BN → ReLU     │
    │                          │
    ▼                          │
[1×1 Conv, 256] → BN           │
    │                          │
    ▼                          │
   (+) ←───────────────────────┘
    │
    ▼
   ReLU
    │
    ▼
Output (256 channels)
```

---

## 🎯 Transfer Learning Strategy

### Frozen vs Trainable Layers

```python
# Freeze early layers (feature extractors)
for name, param in model.named_parameters():
    if "layer4" in name or "fc" in name:
        param.requires_grad = True   # ✅ Trainable
    else:
        param.requires_grad = False  # ❄️ Frozen
```

| Layer | Status | Reason |
|-------|--------|--------|
| `conv1`, `bn1` | ❄️ Frozen | Low-level features (edges, colors) |
| `layer1` | ❄️ Frozen | Basic patterns |
| `layer2` | ❄️ Frozen | Textures, shapes |
| `layer3` | ❄️ Frozen | Mid-level features |
| `layer4` | ✅ Trainable | High-level, task-specific features |
| `fc` | ✅ Trainable | Custom classifier for 9 emotions |

### Why Freeze Layers?

| Benefit | Explanation |
|---------|-------------|
| **Faster Training** | Fewer parameters to update |
| **Less Overfitting** | Pretrained features generalize well |
| **Lower Memory** | Frozen layers don't store gradients |
| **Better Features** | ImageNet features transfer to faces |

---

## 🔧 Custom Classifier Head

The original ResNet-50 `fc` layer outputs 1000 classes (ImageNet). We replace it with:

```python
model.fc = nn.Sequential(
    nn.Linear(2048, 128),      # Reduce dimensions
    nn.ReLU(),                 # Non-linearity
    nn.Dropout(p=0.3),         # Regularization
    nn.Linear(128, 9)          # 9 emotion classes
)
```

### Classifier Architecture

```
Input: 2048 features (from avgpool)
         │
         ▼
┌─────────────────────┐
│  Linear(2048, 128)  │  ← Dimensionality reduction
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│       ReLU()        │  ← Non-linear activation
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   Dropout(p=0.3)    │  ← Prevents overfitting
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   Linear(128, 9)    │  ← Output: 9 emotion logits
└─────────────────────┘
         │
         ▼
Output: [Angry, Contempt, Disgust, Fear, Happy, Natural, Sad, Sleepy, Surprised]
```

### Parameter Count

| Component | Parameters | Trainable |
|-----------|------------|-----------|
| conv1 - layer3 | ~11.2M | ❄️ No |
| layer4 | ~7.1M | ✅ Yes |
| fc (custom) | ~263K | ✅ Yes |
| **Total** | ~23.5M | ~7.4M |

---

## 📊 Data Augmentation Pipeline

### Training Transforms

```python
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomGrayscale(p=0.1),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))
])
```

### Transform Effects

| Transform | Effect | Purpose |
|-----------|--------|---------|
| `RandomResizedCrop` | Random crop & resize | Scale invariance |
| `RandomHorizontalFlip` | Mirror image | Left/right invariance |
| `RandomRotation(15)` | ±15° rotation | Rotation invariance |
| `RandomGrayscale` | 10% → grayscale | Color robustness |
| `ColorJitter` | Adjust colors | Lighting robustness |
| `RandomAffine` | Translate, scale, shear | Spatial robustness |
| `RandomErasing` | Erase random patches | Occlusion robustness |
| `Normalize` | ImageNet mean/std | Match pretrained stats |

### Validation Transforms

```python
val_transforms = transforms.Compose([
    transforms.Resize(int(224 * 1.1)),   # Resize to 246
    transforms.CenterCrop(224),          # Crop center 224×224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

---

## ⚖️ Loss Function

### Weighted Cross-Entropy Loss

```python
# Compute class weights for imbalanced dataset
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
```

### Why Weighted Loss?

| Problem | Solution |
|---------|----------|
| Some emotions have fewer samples | Weight rare classes higher |
| Model biased toward majority class | Balanced weights equalize importance |

### Cross-Entropy Formula

```
Loss = -Σ (weight_i × y_i × log(p_i))

where:
  y_i = true label (one-hot)
  p_i = predicted probability
  weight_i = class weight
```

---

## 🎛️ Optimizer & Scheduler

### Adam Optimizer

```python
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.0001,
    weight_decay=1e-4
)
```

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `lr` | 0.0001 | Learning rate (small for fine-tuning) |
| `weight_decay` | 1e-4 | L2 regularization |
| `betas` | (0.9, 0.999) | Momentum parameters (default) |

### Learning Rate Scheduler

```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.1,
    patience=3
)
```

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `mode` | 'max' | Reduce LR when metric stops improving |
| `factor` | 0.1 | Multiply LR by 0.1 on plateau |
| `patience` | 3 | Wait 3 epochs before reducing |

---

## 👁️ Face Detection (MTCNN)

For real-time webcam detection, we use **MTCNN** (Multi-task Cascaded Convolutional Networks):

```python
from facenet_pytorch import MTCNN
mtcnn = MTCNN(keep_all=True, device=device)
```

### MTCNN Pipeline

```
Input Image
     │
     ▼
┌─────────────┐
│   P-Net     │  ← Proposal Network (face candidates)
└─────────────┘
     │
     ▼
┌─────────────┐
│   R-Net     │  ← Refine Network (filter false positives)
└─────────────┘
     │
     ▼
┌─────────────┐
│   O-Net     │  ← Output Network (final bounding boxes)
└─────────────┘
     │
     ▼
Face Boxes + Confidence
```

---

## 📈 Training Pipeline

### Flow Diagram

```
                    ┌─────────────────────────────────────┐
                    │         Training Loop               │
                    └─────────────────────────────────────┘
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        │                            │                            │
        ▼                            ▼                            ▼
┌───────────────┐          ┌───────────────┐          ┌───────────────┐
│  Load Batch   │          │   Forward     │          │   Backward    │
│  (DataLoader) │    →     │   Pass        │    →     │   Pass        │
└───────────────┘          └───────────────┘          └───────────────┘
        │                            │                            │
        │                            ▼                            ▼
        │                  ┌───────────────┐          ┌───────────────┐
        │                  │  Compute Loss │          │ Update Weights│
        │                  │ (CrossEntropy)│          │   (Adam)      │
        │                  └───────────────┘          └───────────────┘
        │                                                         │
        │                                                         │
        └───────────────────────── Repeat ────────────────────────┘
                                     │
                                     ▼
                          ┌───────────────────┐
                          │   Validation      │
                          │   (every epoch)   │
                          └───────────────────┘
                                     │
                          ┌──────────┴──────────┐
                          │                     │
                          ▼                     ▼
                   ┌─────────────┐       ┌─────────────┐
                   │ Save Best   │       │Early Stop?  │
                   │   Model     │       │             │
                   └─────────────┘       └─────────────┘
```

---

## 🔢 Model Summary

```
ResNet-50 (Modified for Emotion Detection)
===========================================================================
Layer (type)                 Output Shape              Param #
===========================================================================
Conv2d-1                     [1, 64, 112, 112]         9,408
BatchNorm2d-2                [1, 64, 112, 112]         128
ReLU-3                       [1, 64, 112, 112]         0
MaxPool2d-4                  [1, 64, 56, 56]           0
... (layer1-3: frozen)
... (layer4: trainable)
AdaptiveAvgPool2d            [1, 2048, 1, 1]           0
Flatten                      [1, 2048]                 0
---------------------------------------------------------------------------
Linear-fc1                   [1, 128]                  262,272
ReLU                         [1, 128]                  0
Dropout                      [1, 128]                  0
Linear-fc2                   [1, 9]                    1,161
===========================================================================
Total params: ~23.5M
Trainable params: ~7.4M
Non-trainable params: ~16.1M
===========================================================================
```

---

## 🎭 Output Interpretation

### Softmax Output

```python
# Model outputs logits (raw scores)
logits = model(input)  # shape: [batch, 9]

# Convert to probabilities
probabilities = F.softmax(logits, dim=1)

# Get predicted class
predicted_class = torch.argmax(probabilities, dim=1)
emotion_label = class_names[predicted_class]
```

### Example Output

```
Input: Face image (224×224×3)
       │
       ▼
Model Output (logits): [-1.2, 0.5, -0.3, 0.1, 2.8, 0.2, -0.8, -0.5, 0.3]
       │
       ▼
Softmax (probabilities): [0.02, 0.05, 0.03, 0.04, 0.65, 0.05, 0.02, 0.02, 0.05]
       │
       ▼
Prediction: class 4 → "Happy" (65% confidence)
```

---

## 📚 References

1. **ResNet**: He, K., et al. "Deep Residual Learning for Image Recognition." CVPR 2016.
2. **MTCNN**: Zhang, K., et al. "Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks." IEEE SPL 2016.
3. **Transfer Learning**: Yosinski, J., et al. "How transferable are features in deep neural networks?" NeurIPS 2014.
4. **Adam Optimizer**: Kingma, D. P., & Ba, J. "Adam: A Method for Stochastic Optimization." ICLR 2015.

---

**Happy Learning! 🧠**
