# Synapse Navigator

Synapse Navigation is a fork and extension of the [protGPS](https://github.com/pgmikhael/protgps) model, designed to explore and predict synapse-associated protein interactions using large-scale language model embeddings (ESM2) and fine-tuned transformer architectures (DR-BERT).  
This repository includes scripts for training, evaluation, and visualization of model predictions.

---

# Setup
## This code is optimized to run on a linux-based system or an Ubuntu WSL.
1. Install mamba or conda 

1.1 Download installer from: https://github.com/conda-forge/miniforge

1.2 Install and init

```
bash Miniforge-pypy3-Linux-x86_64.sh
```

2. Create environment
```
mamba env create -f environment.yml
```

3. Activate
```
mamba activate syna
```

# Setting up checkpoints folder
## SynapseNavigator
1. Download model checkpoints by contacting chaosun@dandrite.au.dk (O:\Nat_sun-lab\PROTGPS\Lars\SyNa\SynapseNavigator\checkpoints\protgps)
2. Unzip file (Should end up named "checkponts" and contain a folder called "SyNa")

## [ESM2](https://github.com/facebookresearch/esm/) & [DR-BERT](https://github.com/maslov-group/DR-BERT)
1. Find "ESM2&BertDownload.ipynb" and run the 2 chunks (Use the syna kernel)


# Predictions
For predictions gene name and sequence is needed.

To make predictions, edit and run either.
* Predict-SyNa(2Synapse).ipynb 
* Predict-SyNa(1Synapse).ipynb

### For large list of proteins
"BatchPredictionConverter.ipynb" can be used - Converts excel files of protein lists* to format suitable for prediction.


*Fetched from UniProt ID mapping (Any Excel list can be used, but should have column name 'Entry Name' and 'Sequence')

# Analysis 
Analysis using the SynapseNavigator model has not yet been fully implemented,
but the underlying code has been left in place for those interested in extending it.

# citation
If you use SynapseNavigator in your work, please cite both this fork and the original protGPS publication. 

## SynapseNavigator
Brandt, L. B. (2025). SynapseNavigator: extension of protGPS for synapse-associated protein prediction. 

```
@misc{brandt2025synapsenavigator,
  author       = {Lars Boye Brandt},
  title        = {SynapseNavigator: synapse-associated protein localisation prediction},
  year         = {2025},
  howpublished = {\url{https://github.com/larsduved/SynapseNavigator}},
  note         = {Fork of protGPS (Mikhael et al., 2023)}
}
```

## Original protGPS
Mikhael, P. G., et al. (2023). protGPS: Protein Group Prediction System using ESM embeddings. Bioinformatics
```
@article{mikhael2023protGPS,
  title   = {protGPS: Protein Group Prediction System using ESM embeddings},
  author  = {Mikhael, Peter G. and others},
  journal = {Bioinformatics},
  year    = {2023}
}
```
