#!/bin/bash
#================================================================
# Visual Grounding – SLURM batch script
# Cluster : CU Alpine  (aa100 partition, NVIDIA A100)
# Submit  : sbatch run_alpine.sbatch
# Monitor : squeue -u $USER
# Cancel  : scancel <job_id>
# Logs    : cat slurm-<job_id>.out
#================================================================

#SBATCH --job-name=vg-train
#SBATCH --partition=aa100
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=shku2971@colorado.edu

set -euo pipefail

# -------------------------------------------------------------------
# 0. Job info
# -------------------------------------------------------------------
echo "=========================================="
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Job Name     : ${SLURM_JOB_NAME}"
echo "Node         : $(hostname)"
echo "CPUs         : ${SLURM_CPUS_PER_TASK}"
echo "GPUs         : ${CUDA_VISIBLE_DEVICES:-not set}"
echo "Start Time   : $(date)"
echo "=========================================="
nvidia-smi

# -------------------------------------------------------------------
# 1. Load software stack
# -------------------------------------------------------------------
module purge
module load anaconda

ENV_NAME="vg_env"
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "ERROR: Conda environment '${ENV_NAME}' not found!" >&2
    echo "       Create it on a login/compile node first." >&2
    exit 1
fi
conda activate "${ENV_NAME}"
echo ">>> Python: $(python --version) | PyTorch: $(python -c 'import torch; print(torch.__version__)')"

# -------------------------------------------------------------------
# 2. Verify data is present (no internet on compute nodes)
# -------------------------------------------------------------------
SRC_DATA="${SLURM_SUBMIT_DIR}/data"

if [[ ! -d "${SRC_DATA}/train/images" ]] || [[ ! -f "${SRC_DATA}/train/annotations_1.json" ]]; then
    echo "ERROR: Data not found at ${SRC_DATA}" >&2
    echo "       Run prepare_data.py on a login/compile node first." >&2
    exit 1
else
    echo ">>> Data present at ${SRC_DATA}"
fi

# -------------------------------------------------------------------
# 3. Copy dataset into RAM (/dev/shm) for zero-disk-I/O training
# -------------------------------------------------------------------
RAM_DATA="/dev/shm/${USER}_vg_data"

if [[ ! -d "${RAM_DATA}" ]]; then
    echo ">>> Copying dataset into RAM (${RAM_DATA}) ..."
    mkdir -p "${RAM_DATA}"
    cp -a -L "${SRC_DATA}/." "${RAM_DATA}/"
    echo ">>> Dataset in RAM: $(du -sh "${RAM_DATA}" | cut -f1)"
fi

# -------------------------------------------------------------------
# 4. HuggingFace cache on scratch (avoids $HOME quota)
# -------------------------------------------------------------------
export HF_HOME="/scratch/alpine/${USER}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"
export TORCH_HOME="/scratch/alpine/${USER}/torch_cache"
mkdir -p "${HF_HOME}" "${TORCH_HOME}"

# -------------------------------------------------------------------
# 5. Performance flags for A100
# -------------------------------------------------------------------
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1

# -------------------------------------------------------------------
# 6. Training hyper-parameters
# -------------------------------------------------------------------
OUTPUT_DIR="${SLURM_SUBMIT_DIR}/outputs"
EPOCHS=40
BATCH_SIZE=48          # A100 40/80 GB — safe default
LR=3e-4
PATIENCE=8
NUM_WORKERS=8
IMAGE_SIZE=512
SEED=42

mkdir -p "${OUTPUT_DIR}"

# -------------------------------------------------------------------
# 7. Launch training
# -------------------------------------------------------------------
echo ">>> Starting training ..."

python "${SLURM_SUBMIT_DIR}/main.py" \
    --data_dir    "${RAM_DATA}" \
    --output_dir  "${OUTPUT_DIR}" \
    --epochs      "${EPOCHS}" \
    --batch_size  "${BATCH_SIZE}" \
    --lr          "${LR}" \
    --patience    "${PATIENCE}" \
    --num_workers "${NUM_WORKERS}" \
    --image_size  "${IMAGE_SIZE}" \
    --seed        "${SEED}" \
    --amp \
    --fast

# -------------------------------------------------------------------
# 8. Cleanup RAM disk
# -------------------------------------------------------------------
echo ">>> Cleaning up RAM disk ..."
rm -rf "${RAM_DATA}"

echo "=========================================="
echo "End Time : $(date)"
echo "=========================================="
