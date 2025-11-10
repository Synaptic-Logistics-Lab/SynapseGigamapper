# Synapse Navigation

Synapse Navigation is a fork and extension of the [protGPS](https://github.com/pgmikhael/protgps) model, designed to explore and predict synapse-associated protein interactions using large-scale language model embeddings (ESM2) and fine-tuned transformer architectures (DR-BERT).  
This repository includes scripts for training, evaluation, and visualization of model predictions.

---

## 🧬 Features

- Integration with [Facebook’s ESM](https://github.com/facebookresearch/esm) (as a submodule)
- DR-BERT model for protein sequence-based prediction
- Support for batch prediction and feature extraction
- Configurable training pipeline (`configs/`)
- Notebook examples for inference and visualization
- Modular code structure for extending new models or datasets
- 
## ⚙️ Installation

Make sure you’ve installed [git](https://git-scm.com) and [conda](https://docs.conda.io/en/latest/).

```bash
# clone the repo including the ESM submodule
git clone https://github.com/larsduved/SynapseNavigation.git
cd "Synapse Navigation"
git submodule update --init --recursive

# create environment
conda env create -f environment.yml
conda activate synapse-nav
