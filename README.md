# Synapse Gigamapper (SyGi)

Synapse Gigamapper is custom-built based on the [ProtGPS] model (https://github.com/pgmikhael/protgps), designed to explore and predict protein localisations at excitatory and inhibitory synapses in addition to standard subcellular compartments using large-scale language model embeddings (ESM2 or ESM-C) and fine-tuned transformer architectures (DR-BERT).  
This repository includes scripts for training, evaluation, and visualization of model predictions.

---

## Installation Guide

This guide explains how to install **SynapseGigamapper (SyGi)** using
Ubuntu (native Linux or Windows Subsystem for Linux).

------------------------------------------------------------------------

## 1. Install Ubuntu (Windows Only)

If you are using Windows, install **Windows Subsystem for Linux (WSL)**:

1.  Open **PowerShell as Administrator**
2.  Run:

``` bash
wsl --install
```

3.  Follow the installation prompts.
4.  Restart your PC when prompted.

After restarting, open your Ubuntu terminal.

> If you are already using Linux, you can skip this step.

------------------------------------------------------------------------

## 2. Install Conda/Mamba (Miniforge)

We recommend using **Miniforge (conda-forge)** for managing
dependencies.

### Download Miniforge

Open your Ubuntu terminal and run:

``` bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh
```

### Install Miniforge

``` bash
bash Miniforge3-$(uname)-$(uname -m).sh
```

Follow the installation prompts and allow it to initialize conda.

After installation, restart your terminal.

------------------------------------------------------------------------

## 3. Download SynapseGigamapper

### Option A --- Download ZIP

1.  Download the repository as a ZIP file from GitHub.
2.  Extract `SynapseGigamapper-main.zip`.
3.  Move the extracted folder into your Ubuntu home directory.

### Option B --- Clone via Git

``` bash
git clone https://github.com/Synaptic-Logistics-Lab/SynapseGigamapper
```

------------------------------------------------------------------------

## 4. Create the Conda Environment

Navigate into the project directory:

``` bash
cd SynapseGigamapper-main
```

Create the environment:

``` bash
mamba env create -f environment.yml
```

> This step may take several minutes.

------------------------------------------------------------------------

## 5. Activate the Environment

Once installation is complete, activate the environment:

``` bash
mamba activate syna
```

Your terminal prompt should now start with:

    (syna)

Install the environment as a Jupyter kernel:

``` bash
python -m ipykernel install --user --name=syna --display-name "Python (syna)"
```

------------------------------------------------------------------------

## 6. Setting Up Checkpoints Folder
Checkpoints from DR-BERT (https://github.com/maslov-group/DR-BERT) and ESM2 (https://github.com/facebookresearch/esm?tab=readme-ov-file#esmfold)

1.  Download model checkpoints by contacting:

```
    chaosun@dandrite.au.dk
    Lab Storage: PROTGPS\Lars\SyNa\SynapseNavigator\checkpoints\protgps
```
2.  Unzip the file
    -   The folder should be named `checkpoints`
    -   It should contain:
        -   `SyNa`
        -   `drbert`
3.  Move the `checkpoints` folder into:

```{=html}
<!-- -->
```
    SynapseGigamapper-main

4.  Open `ESM2&BertDownload.ipynb`
    -   Ensure you are using the **syna kernel**
    -   Run both code cells

------------------------------------------------------------------------

## Troubleshooting

### Environment Creation Fails Due to pip Dependency Conflicts

If you encounter dependency conflicts (for example with `jupyterlab`):

-   Try disabling any VPNs
-   Install missing dependencies in batches

------------------------------------------------------------------------

## Predictions

Predictions require:

-   Gene name
-   Protein sequence

To make predictions, edit and run one of the following notebooks:

-   `Predict-SyNa(2Synapse).ipynb`
-   `Predict-SyNa(1Synapse).ipynb`

### Batch Predictions

For large protein lists, use:

    BatchPredictionConverter.ipynb

This notebook converts Excel protein lists into prediction-ready format.

#### Excel File Requirements

-   Typically fetched from UniProt ID mapping
-   Any Excel list can be used if it contains:
    -   `Entry Name`
    -   `Sequence`

------------------------------------------------------------------------

## Analysis

Analysis tools using the SynapseGigamapper model are not yet fully
updated.\
The underlying code remains available for users interested in extending
functionality.

------------------------------------------------------------------------

## Citation

If you use SynapseGigamapper in your work, please cite:

**SynapseGigamapper**\
Chao Sun Lab (2025). SynapseGigamapper: A Protein Language Model for
Protein Localization Prediction at Neuronal Synapses.

``` bibtex
@misc{brandt2025synapsegigamapper,
  author       = {Sun lab, DANDRITE, Aarhus University},
  title        = {SynapseGigamapper: Protein localisation prediction for synapses},
  year         = {2025},
  howpublished = {\url{https://github.com/Synaptic-Logistics-lab/SynapseGigamapper}},
  note         = {see also ProtGPS (Mikhael et al., 2023)}
}
```

