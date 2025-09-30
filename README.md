# Wildfire Scar Mapping - Multi-Model Experimental Repository

This repository contains a comprehensive collection of state-of-the-art deep learning models for wildfire scar mapping and change detection experiments using satellite imagery. The project focuses on wildfire detection and damage assessment using various architectures including CNNs, Transformers, and modern State Space Models (SSMs) like Mamba.

## Project Overview

Wildfire monitoring and burnt area mapping are crucial for disaster response, environmental monitoring, and climate change studies. This repository implements and compares multiple deep learning approaches for:

- **Binary Change Detection**: Detecting whether areas have been burnt
- **Temporal Image Analysis**: Using before/after satellite imagery pairs

## Repository Structure

### [FLOGA](./FLOGA/) - Main Dataset & BAM-CD Model
**Primary focus: The FLOGA dataset and Burnt Area Mapping with Change Detection**

- [Download the FLOGA dataset here](https://www.dropbox.com/scl/fo/3sqbs3tioox7s5vb4jmwl/h?rlkey=5p3e7wa5al4cy9x34pmtp9g6d&e=1&dl=0)
- Based on the [FLOGA](https://github.com/Orion-AI-Lab/FLOGA) repository

- **Dataset**: All experiments use the 60m sentinel resolution version of the dataset with 256 x 256 generated patches.
- **Models**: UNet and other change detection variants(UNet, SNUNet, FC-EF, ADHR, ChangeFormer, Mamba, BAM-CD)

**Data Structure**:
```
FLOGA/
├── FLOGA-github/          # Main implementation
├── data/                  # Preprocessed datasets and results
├── data_exploration/      # Data analysis tools
└── notebooks/            # Jupyter notebooks for exploration
```

### [M-CD](./M-CD/) - Mamba-based Siamese Network
**Focus: Remote Sensing Change Detection with Mamba Architecture**

- **Architecture**: Implements a Vision Mamba (VMamba) based Siamese network for change detection, as described in [JayParanjape/M-CD](https://github.com/JayParanjape/M-CD)
- **Features**:
  - Siamese encoder structure for bi-temporal image pairs
  - Vision Mamba blocks for efficient long-range dependency modeling
  - Designed for remote sensing change detection tasks
  - Supports training and evaluation on standard change detection datasets
  - Custom config and dataset generation/processing code added for using the FLOGA dataset

### [MambaCD](./MambaCD/) - ChangeMamba Suite
**Focus: Comprehensive Mamba-based Change Detection Framework**

**ChangeMamba** is a modern, efficient framework for change detection in remote sensing imagery, based on the [ChangeMamba repository](https://github.com/ChenHongruixuan/ChangeMamba).

- **Architecture**: Utilizes the Mamba state space model for capturing long-range dependencies in bi-temporal satellite images.
- **Models Included**: MambaBCD (binary change detection), MambaSCD (semantic change detection), and other Mamba-based variants. MambaBCD is the primary model for binary change detection in this suite, leveraging the Mamba architecture for efficient and accurate identification of burnt areas in satellite imagery.
- **Features**:
  - End-to-end training and evaluation scripts
  - Flexible configuration for datasets and model parameters
  - Support for multiple remote sensing change detection benchmarks
  - Modular codebase for easy extension and experimentation
  - Custom config and dataset generation/processing code added for using the FLOGA dataset

### [VM-UNet](./VM-UNet/) - Vision Mamba UNet
**This directory contains an implementation of [VM-UNet](https://github.com/JCruan519/VM-UNet), a U-shaped segmentation network that leverages Vision State Space (VSS) blocks for efficient long-range dependency modeling.**

- **Original Source**: [JCruan519/VM-UNet](https://github.com/JCruan519/VM-UNet)
- **Purpose**: Designed for medical image segmentation, but the architecture and codebase are adaptable for remote sensing and other segmentation tasks.
- **Key Features**:
  - U-shaped encoder-decoder structure with skip connections
  - Vision State Space (VSS) blocks for global context modeling
  - Linear computational complexity with respect to input size
  - Outperforms many transformer-based models on medical benchmarks
- **Modifications**: This repo includes custom configuration and dataset processing code to facilitate adaptation to remote sensing datasets such as FLOGA.

**Usage**:
- See the [VM-UNet README](./VM-UNet/README.md) for setup, training, and evaluation instructions.
- Example datasets: ISIC17, ISIC18, Synapse (medical); FLOGA (remote sensing, with adaptation).

**References**:
- [VM-UNet: Vision State Space Model for Medical Image Segmentation](https://arxiv.org/abs/2311.16477)

### [notebooks](./notebooks/) - Analysis & Visualization
**Focus: Metrics visualization and model analysis**

- `visualize_metrics.ipynb`: Comprehensive model performance analysis based on training logs
- `vssblock.ipynb`: Explores the Vision State Space block(from VMamba)

## Quick Start Guide

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- conda/mamba package manager

### FLOGA Dataset

```bash
cd FLOGA/FLOGA-github
pip install -r requirements.txt  # Install dependencies

# Download FLOGA dataset
# Follow instructions in FLOGA/FLOGA-github/README.md

# Create analysis-ready dataset
python create_dataset.py --floga_path path/to/hdf/files --out_path data/ --out_size 256 256 --sample 1

# Train models
python run_experiment.py
```

### 2. M-CD (Mamba Change Detection)

```bash
cd M-CD
cd models/encoders/selective_scan && pip install . && cd ../../..
```

### 3. ChangeMamba (MambaCD)

```bash
cd MambaCD
conda create -n changemamba
conda activate changemamba
pip install -r requirements.txt
cd kernels/selective_scan && pip install .

# Download pre-trained weights
# Configure dataset paths

# Train ChangeMamba (MambaCD) model

# Edit the config file as needed (e.g., `MambaCD/changedetection/configs/vssm1/vssm_base_224.yaml`)
# and set dataset paths in the arguments or config.

cd MambaCD/changedetection/script

# Example: Train with default config and GPU 0
python train_MambaBCD.py --gpu_id 0 --mode train

# For evaluation (after training):
python train_MambaBCD.py --gpu_id 0 --mode test

```

## Datasets

### Data Formats
- **Input**: Sentinel-2 or MODIS, high-resolution optical/multispectral imagery
- **Labels**: Binary masks
- **Resolution**: 10m-500m depending on sensor and application

## Model Performance Comparison

| Model | Architecture | Key Strength | Primary Use Case |
|-------|-------------|--------------|------------------|
| BAM-CD | CNN-based | Proven architecture | Burnt area mapping |
| M-CD | Mamba Siamese | Linear complexity | General change detection |
| ChangeMamba | Mamba variants | SOTA performance | Multi-task CD |
| VM-UNet | Mamba U-Net | Medical segmentation | Segmentation tasks |

### Custom Dataset Integration
1. Organize data following the expected structure (see individual READMEs)
2. Create configuration files
3. Modify dataset classes if needed
4. Train with custom parameters

Pre-trained weights and detailed results available in each model's directory.

## Contributing

This repository combines multiple research efforts in change detection and segmentation. Each model has its own maintainers and citation requirements. Please refer to individual READMEs for specific contribution guidelines.

## Citations

If you use any models from this repository, please cite the respective papers:

**FLOGA & BAM-CD**:
```bibtex
@ARTICLE{10479972,
  author={Sdraka, Maria and Dimakos, Alkinoos and Malounis, Alexandros and others},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={FLOGA: A Machine-Learning-Ready Dataset, a Benchmark, and a Novel Deep Learning Model for Burnt Area Mapping With Sentinel-2}, 
  year={2024}
}
```

**ChangeMamba**:
```bibtex
@article{chen2024changemamba,
  author={Hongruixuan Chen and Jian Song and Chengxi Han and Junshi Xia and Naoto Yokoya},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={ChangeMamba: Remote Sensing Change Detection with Spatiotemporal State Space Model}, 
  year={2024}
}
```

**VM-UNet**:
```bibtex
@article{ruan2024vmunet,
  title={VM-UNet: Vision Mamba UNet for Medical Image Segmentation},
  author={Ruan, Jiacheng and Xiang, Suncheng},
  journal={arXiv preprint arXiv:2402.02491},
  year={2024}
}
```

## Implementation Details

- **GPU Memory**: Models were trained on 2 x 24GB NVIDIA-L4 GPUs
- **Storage**: 50GB+ for datasets and models
- **OS**: Linux

## Support

For technical issues:
- Check individual model READMEs
- Review common issues in each repository
- Open issues in the respective model repositories

---

This repository serves as a comprehensive benchmark and starting point for burnt area mapping and change detection research. Each model offers unique advantages, and the choice depends on your specific requirements for accuracy, computational efficiency, and task type.
