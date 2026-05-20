# brain-tumor-mri
Brain tumor classification from MRI scans using transfer learning. Multi-class baseline on the Br35H dataset (glioma, meningioma, pituitary, no tumor).

# brain-tumor-mri 

**Multi-Class Brain Tumor Classification from MRI Scans**

[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-yellow.svg)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Status](https://img.shields.io/badge/Status-Active_Development-orange)](https://github.com/mfhrm31/brain-tumor-mri)

Transfer-learning baseline for 4-class brain tumor classification from MRI scans (glioma, meningioma, pituitary tumor, no tumor) using ImageNet-pretrained CNN backbones.

---

##  Overview

Brain tumors are among the most aggressive cancers, with glioblastomas alone accounting for ~15,000 deaths annually in the US. Early and accurate classification of tumor type guides treatment decisions — surgical, radiation, chemotherapy, or watchful waiting. MRI is the imaging modality of choice, but expert neuroradiologists are scarce in low-resource settings.

This project implements clean multi-class classification baselines on a standard public MRI dataset, with comparison across CNN backbones (ResNet50, DenseNet121, InceptionV3) commonly used in neuroimaging literature.

Companion work to my broader medical AI research:
- **[Lung nodule prediction with hybrid feature fusion](https://github.com/mfhrm31/lungnet-hybrid)** (Wiley, 2026)
- **[Chest X-ray pneumonia classification](https://github.com/mfhrm31/chest-xray-pneumonia)**
- **[Medical image preprocessing toolkit](https://github.com/mfhrm31/medical-image-utils)**

##  Goals

- Establish multi-class classification baselines for brain tumor MRI
- Compare three pretrained backbones (ResNet50, DenseNet121, InceptionV3)
- Quantify per-class performance with confusion matrices
- Apply Grad-CAM to visualize attention on tumor regions
- Maintain reproducible pipeline aligned with my research interests

##  Classes

| Label | Class | Description |
|---|---|---|
| 0 | **Glioma** | Aggressive tumor arising from glial cells |
| 1 | **Meningioma** | Tumor of the meninges (typically benign) |
| 2 | **Pituitary** | Tumor in the pituitary gland |
| 3 | **No Tumor** | Normal MRI scan |

##  Quickstart

### Installation

```bash
git clone https://github.com/mfhrm31/brain-tumor-mri.git
cd brain-tumor-mri
pip install -r requirements.txt
```
### Train a model

```bash
python scripts/train.py --config configs/resnet50.yaml
```

### Evaluate

```bash
python scripts/evaluate.py --checkpoint outputs/best_model.pt
```

##  Project Structure

brain-tumor-mri/
├── src/
│   ├── data/           # Dataset loading and MRI-specific augmentation
│   ├── models/         # ResNet50, DenseNet121, InceptionV3 wrappers
│   ├── training/       # Train loop with logging
│   ├── evaluation/     # Multi-class metrics + confusion matrices
│   └── interpretability/  # Grad-CAM heatmaps
├── configs/            # YAML configurations per backbone
├── scripts/            # Train and evaluate runners
└── notebooks/          # Exploration and analysis

##  Methodology

1. **Preprocessing**: Resize to 224×224 (299×299 for InceptionV3), histogram equalization, ImageNet normalization
2. **Augmentation**: Small rotations (±15°), brightness/contrast jitter — no horizontal flip (brain asymmetry matters clinically)
3. **Training**: Frozen-then-unfrozen schedule, weighted cross-entropy for class balance, cosine LR schedule
4. **Evaluation**: Per-class precision/recall/F1, macro and weighted averages, confusion matrices

##  Dataset

**Brain Tumor MRI Dataset** (Kaggle combined dataset)

- ~7,000 MRI images total across training and testing splits
- 4 classes: glioma, meningioma, pituitary tumor, no tumor
- T1-weighted contrast-enhanced MRI
- License: CC0 (Public Domain)

## 🛠️ Development Status

| Component | Status |
|---|---|
| Dataset loader | ✅ Implemented |
| ResNet50 baseline | 🚧 In development |
| DenseNet121 baseline | 🚧 Planned |
| InceptionV3 baseline | 🚧 Planned |
| Multi-class metrics | ✅ Implemented |
| Grad-CAM visualizations | 🚧 Planned |

Results will be populated as experiments complete.

## 📖 Why Brain Tumor MRI?

This task is directly aligned with my stated research interests in deep learning for oncology imaging. While my published work focuses on lung nodule prediction (CT) and biometric-conditioned activity recognition, brain tumor MRI represents the natural next extension:

- **Same architectural family** (CNN backbones with transfer learning)
- **Different modality** (MRI vs CT) — demonstrates cross-modality generalization
- **Multi-class** vs binary — practical step toward clinically realistic systems
- **High-stakes diagnosis** — strong motivation for calibration and interpretability work I plan to extend in future projects
