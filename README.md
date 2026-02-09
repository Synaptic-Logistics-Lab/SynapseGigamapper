# Synapse Gigamapper (Sygi)

Synapse Gigamapper is custom-built based on the [ProtGPS] model (https://github.com/pgmikhael/protgps), designed to explore and predict protein localisations at excitatory and inhibitory synapses in addition to standard subcellular compartments using large-scale language model embeddings (ESM2 or ESM-C) and fine-tuned transformer architectures (DR-BERT).  
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
## SynapseGigamapper
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
Analyses using the SynapseGigamapper model have not yet been fully updated here.
The underlying code has been left in place for those interested in extending it.

# citation
If you use SynapseGigamapper in your work, please cite this work. 

## SynapseGigamapper
Chao Sun Lab (2025). SynapseGigamapper: a Protein Language Model for protein localization prediction at Neuronal Synapses. 

```
@misc{brandt2025synapsegigamapper,
  author       = {Sun lab, DANDRITE & Department of Molecular Biology and Genetics, Aarhus University},
  title        = {SynapseGigamapper: Protein localisation prediction for neuronal synapses},
  year         = {2025},
  howpublished = {\url{https://github.com/Synaptic-Logistics-lab/SynapseGigamapper}},
  note         = {see also ProtGPS (Mikhael et al., 2023)}
}
```

