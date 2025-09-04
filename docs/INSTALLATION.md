# Installation Guide

## System Requirements

### Hardware Requirements
- **CPU**: Multi-core processor (Intel i5/AMD Ryzen 5 or better)
- **RAM**: Minimum 8GB, Recommended 16GB+
- **GPU**: Optional but recommended (NVIDIA with CUDA support)
- **Storage**: 10GB free space for models and data

### Software Requirements
- **Python**: 3.8 or higher
- **Operating System**: Windows 10+, macOS 10.15+, or Linux
- **Git**: For repository management

## Installation Steps

### 1. Clone the Repository
```bash
git clone https://github.com/dappalumbo91/FSOT-Neuromorphic-AI-System.git
cd FSOT-Neuromorphic-AI-System
```

### 2. Create Virtual Environment
```bash
# Using venv
python -m venv neuromorphic_env
source neuromorphic_env/bin/activate  # Linux/Mac
# or
neuromorphic_env\Scripts\activate  # Windows

# Using conda
conda create -n neuromorphic_env python=3.9
conda activate neuromorphic_env
```

### 3. Install Dependencies
```bash
# Basic installation
pip install -r requirements.txt

# Development installation
pip install -e .[dev]

# Full installation with GPU support
pip install -e .[all]
```

### 4. Download Required Models
```bash
python scripts/download_models.py
```

### 5. Verify Installation
```bash
python -m pytest tests/
python examples/basic_usage.py
```

## GPU Setup (Optional)

### NVIDIA CUDA
```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Verify GPU Support
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
```

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure virtual environment is activated
2. **Memory Issues**: Reduce batch sizes or use CPU mode
3. **GPU Issues**: Verify CUDA installation and compatibility

### Getting Help
- Check [GitHub Issues](https://github.com/dappalumbo91/FSOT-Neuromorphic-AI-System/issues)
- Review documentation in `docs/`
- Run diagnostic script: `python scripts/system_check.py`
