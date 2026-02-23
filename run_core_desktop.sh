#!/bin/bash
#================================================================
# Visual Grounding – CORE Desktop Training Script
# Run this from a terminal in your CORE Desktop session
#================================================================

set -euo pipefail

echo "=========================================="
echo "CORE Desktop Training - Optimized for 2x RTX 8000"
echo "=========================================="
echo "Start Time: $(date)"
echo ""

# Check GPU availability
echo "Checking GPUs..."
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

# Load conda environment (if not already activated)
echo "Loading environment..."
if [[ "$CONDA_DEFAULT_ENV" != "vg_env" ]]; then
    module load anaconda 2>/dev/null || true
    source activate vg_env || conda activate vg_env || true
else
    echo "  Environment 'vg_env' already active ✓"
fi

# Navigate to project directory (if not already there)
PROJECT_DIR="/projects/${USER}/Visual-Grounding"
if [[ "$PWD" != "$PROJECT_DIR" ]]; then
    cd "$PROJECT_DIR"
fi
echo "  Working directory: $PWD"

# Run training with optimized settings
echo "Starting training..."
echo "  - Batch size: 64 (optimized for 46GB VRAM)"
echo "  - Workers: 12 (for fast data loading)"
echo "  - Mixed precision: Enabled (AMP)"
echo "  - TF32: Enabled (fast mode for RTX GPUs)"
echo "  - Multi-GPU: Automatic (using both RTX 8000s)"
echo ""

python main.py \
    --data_dir ./data \
    --output_dir ./outputs \
    --epochs 40 \
    --batch_size 64 \
    --num_workers 12 \
    --lr 3e-4 \
    --patience 8 \
    --fast \
    --amp

echo ""
echo "=========================================="
echo "Training Complete!"
echo "End Time: $(date)"
echo "Model saved to: ./outputs/best_model.pt"
echo "=========================================="
