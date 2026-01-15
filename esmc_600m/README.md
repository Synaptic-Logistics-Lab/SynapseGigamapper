# ESM-C 600M Fine-tuning for Synapse Prediction

This directory contains the code to fine-tune ESM-C 600M (EvolutionaryScale, Dec 2024) for predicting protein localization to neuronal synapses.

## Model Overview

| Property | Value |
|----------|-------|
| **Base Model** | ESM-C 600M (Cambrian) |
| **Parameters** | ~600 million |
| **Embedding Dimension** | 1152 |
| **Transformer Layers** | 36 |
| **Loss Function** | Synapse-Weighted BCE (A2) |
| **Synapse Weights** | 10x Inhibitory, 8x Excitatory |

## Performance Results

### Comparison: ESM2-8M (Baseline) vs ESM-C 600M + Weighted Loss

| Compartment | Baseline F1 | ESM-C A2 F1 | Improvement |
|-------------|-------------|-------------|-------------|
| **Inhibitory Synapses** | 0.027 | **0.348** | **+1189%** |
| **Excitatory Synapse** | 0.241 | **0.400** | **+66%** |
| Mitochondrion | 0.676 | 0.796 | +18% |
| Nucleus | 0.615 | 0.749 | +22% |
| ER | 0.557 | 0.669 | +20% |
| Cytosol | 0.647 | 0.713 | +10% |
| **Mean** | **0.461** | **0.612** | **+33%** |

### ROC-AUC Scores (ESM-C A2)

| Compartment | ROC-AUC |
|-------------|---------|
| Mitochondrion | 0.970 |
| Nucleus | 0.926 |
| ER | 0.923 |
| Cytosol | 0.853 |
| Excitatory Synapse | 0.808 |
| Inhibitory Synapses | 0.774 |
| **Mean** | **0.875** |

## Quick Start

### 1. Create Environment

```bash
# Requires Python 3.10+ for ESM package
conda env create -f environment_esmc.yml
conda activate syna_esmc
```

### 2. Run Training

```bash
# From SynapseNavigator root directory
cd /path/to/SynapseNavigator
./esmc_600m/scripts/train_esmc.sh
```

### 3. (Optional) Download Pre-trained Weights

Pre-trained model weights are available on HuggingFace:

```bash
# Coming soon - weights will be available at:
# https://huggingface.co/Synaptic-Logistics-Lab/SynapseNavigator-ESMC-A2
```

## File Structure

```
esmc_600m/
├── README.md                    # This file
├── environment_esmc.yml         # Conda environment (Python 3.10)
├── configs/
│   └── esmc_weight_tuning_A2.json    # Training hyperparameters
├── models/
│   └── esmc_encoder.py          # ESM-C 600M encoder
├── losses/
│   └── synapse_weighted_bce.py  # A2 weighted loss (10x/8x)
├── scripts/
│   └── train_esmc.sh            # Training script
└── data/
    └── dataset.json             # Training data (6,627 proteins)
```

## Key Components

### ESM-C Encoder (`models/esmc_encoder.py`)

The ESM-C encoder wraps the EvolutionaryScale ESMC model:
- Auto-downloads weights from EvolutionaryScale
- Outputs 1152-dimensional embeddings
- Supports fine-tuning or freezing encoder
- Mean-pools per-residue embeddings to sequence-level

### Weighted Loss (`losses/synapse_weighted_bce.py`)

The A2 weighted BCE loss addresses class imbalance:

```python
# Class weights (order: cytosol, ER, mitochondrion, nucleus, Excitatory, Inhibitory)
weights = [2.67, 8.75, 8.15, 4.02, 120.37, 119.43]

# Multipliers applied:
# - Inhibitory Synapses: 10x (12.0 → 119.4)
# - Excitatory Synapse: 8x (15.0 → 120.4)
# - Other compartments: 1x (baseline class imbalance)
```

### Training Configuration (`configs/esmc_weight_tuning_A2.json`)

Key hyperparameters:
- **Batch size**: 8 (requires 40GB+ GPU)
- **Learning rate**: 1e-4
- **Epochs**: 30
- **Precision**: fp16 (mixed precision)
- **Loss**: `synapse_weighted_bce_a2`

## Dataset

The `data/dataset.json` contains 6,627 human proteins with multi-label annotations:

| Compartment | Proteins | Percentage |
|-------------|----------|------------|
| Cytosol | 1,806 | 27.3% |
| Nucleus | 1,319 | 19.9% |
| Mitochondrion | 724 | 10.9% |
| ER | 680 | 10.3% |
| Inhibitory Synapses | 512 | 7.7% |
| Excitatory Synapse | 413 | 6.2% |

**Data splits**: 70% train / 15% dev / 15% test

## Hardware Requirements

| GPU | VRAM | Status |
|-----|------|--------|
| A100 80GB | 80GB | Recommended |
| A100 40GB | 40GB | Supported |
| V100 32GB | 32GB | Supported (batch_size=4) |
| RTX 4090 | 24GB | Supported (batch_size=2) |

**Estimated training time**: 2-4 hours on A100 (30 epochs)

## Citations

If you use this code, please cite:

### SynapseNavigator
```bibtex
@misc{brandt2025synapsenavigator,
  author = {Brandt, Lars Boye},
  title = {SynapseNavigator: Synapse-associated protein localization prediction},
  year = {2025},
  url = {https://github.com/Synaptic-Logistics-Lab/SynapseNavigator}
}
```

### ESM-C (Cambrian)
```bibtex
@misc{evolutionaryscale2024esmc,
  author = {EvolutionaryScale},
  title = {ESM Cambrian: A New Foundation Model for Biology},
  year = {2024},
  url = {https://www.evolutionaryscale.ai/blog/esm-cambrian}
}
```

### protGPS (Original Framework)
```bibtex
@article{mikhael2023protgps,
  title = {protGPS: Protein Group Prediction System},
  author = {Mikhael, Peter G. and others},
  journal = {Bioinformatics},
  year = {2023}
}
```

## Contact

- **Creator**: Lars Boye Brandt (larsboyebrandt@gmail.com)
- **Collaborator**: Chao Sun (chaosun@dandrite.au.dk)
- **Lab**: Synaptic Logistics Lab

## License

MIT License - see LICENSE file in repository root.
