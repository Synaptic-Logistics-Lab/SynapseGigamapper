#!/bin/bash
#
# Train ESM-C 600M for Synapse Protein Localization (A2 Weighted Loss)
#
# This script fine-tunes the ESM-C 600M protein language model for predicting
# protein localization to 6 cellular compartments, with special focus on
# synapse prediction using weighted BCE loss.
#
# Requirements:
#   - GPU with 40GB+ VRAM (e.g., A100, RTX 4090, V100)
#   - Python 3.10+ (required for ESM package)
#   - Conda environment with PyTorch and ESM package
#
# Usage:
#   ./train_esmc.sh
#
# Expected training time: ~2-4 hours for 30 epochs on A100

set -e  # Exit on error

# Configuration
CONFIG_FILE="${1:-configs/esmc_weight_tuning_A2.json}"
CONDA_ENV="${2:-syna_esmc}"

echo "=============================================="
echo "ESM-C 600M Synapse Prediction Training"
echo "=============================================="
echo ""
echo "Configuration: ${CONFIG_FILE}"
echo "Conda Environment: ${CONDA_ENV}"
echo ""

# Check if running from correct directory
if [ ! -f "scripts/dispatcher.py" ]; then
    echo "ERROR: Please run this script from the SynapseNavigator root directory"
    echo "  cd /path/to/SynapseNavigator"
    echo "  ./esmc_600m/scripts/train_esmc.sh"
    exit 1
fi

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
    echo ""
else
    echo "WARNING: nvidia-smi not found. Training will be VERY slow on CPU."
    echo ""
fi

# Activate conda environment
echo "Activating conda environment: ${CONDA_ENV}"
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate ${CONDA_ENV}
elif command -v mamba &> /dev/null; then
    eval "$(mamba shell.bash hook)"
    mamba activate ${CONDA_ENV}
else
    echo "ERROR: Neither conda nor mamba found. Please install miniforge."
    exit 1
fi

# Verify ESM package is installed
echo "Checking ESM package..."
python -c "from esm.models.esmc import ESMC; print('ESM package OK')" 2>/dev/null || {
    echo "Installing ESM package..."
    pip install esm --quiet
}

# Verify weighted loss is available
echo "Checking weighted loss implementation..."
python -c "
from protgps.utils.registry import register_object
from protgps.learning.losses.basic import SynapseWeightedBCE_A2
print('Weighted BCE loss OK')
" || {
    echo "ERROR: Weighted loss not found. Check protgps/learning/losses/basic.py"
    exit 1
}

# Run training
echo ""
echo "=============================================="
echo "Starting Training..."
echo "=============================================="
echo ""

python scripts/dispatcher.py --config_path "${CONFIG_FILE}"

echo ""
echo "=============================================="
echo "Training Complete!"
echo "=============================================="
echo ""
echo "Checkpoints saved to: checkpoints/esmc_600m_a2/"
echo ""
echo "To analyze results:"
echo "  python comprehensive_analysis.py \\"
echo "    --checkpoint_dir checkpoints/esmc_600m_a2 \\"
echo "    --output_dir figures_esmc_a2"
