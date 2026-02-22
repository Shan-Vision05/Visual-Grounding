#!/bin/bash
#================================================================
# Visual Grounding – SLURM batch script
# Cluster : CU Blanca  (blanca-bortz, NVIDIA H100)
# Submit  : module load slurm/blanca && sbatch run_training.sbatch
# Monitor : squeue -u $USER
# Cancel  : scancel <job_id>
# Logs    : cat slurm-<job_id>.out
#================================================================

#SBATCH --job-name=vg-train
#SBATCH --account=blanca-bortz
#SBATCH --qos=blanca-bortz
#SBATCH --partition=blanca-bortz
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=0                       # request ALL available node memory
#SBATCH --time=1-00:00:00             # max wall-time: 7 days (high-priority QoS)
#SBATCH --output=slurm-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=$USER@colorado.edu

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

# Conda environment (created once, reused afterwards)
ENV_NAME="vg_env"
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo ">>> Creating conda environment '${ENV_NAME}' ..."
    conda create -y -n "${ENV_NAME}" python=3.11
fi
conda activate "${ENV_NAME}"

pip install --quiet --upgrade pip
pip install --quiet -r "${SLURM_SUBMIT_DIR}/requirements.txt"

# -------------------------------------------------------------------
# 2. Download & prepare data if not already present
#    (RefCOCOg + COCO 2014 images — runs once, cached in .cache/)
# -------------------------------------------------------------------
SRC_DATA="${SLURM_SUBMIT_DIR}/data"
CACHE_DIR="${SLURM_SUBMIT_DIR}/.cache"

if [[ ! -d "${SRC_DATA}/train/images" ]] || [[ ! -f "${SRC_DATA}/train/annotations_1.json" ]]; then
    echo ">>> Data not found — running prepare_data.py ..."
    python "${SLURM_SUBMIT_DIR}/prepare_data.py" \
        --data_dir  "${SRC_DATA}" \
        --cache_dir "${CACHE_DIR}"
else
    echo ">>> Data already present at ${SRC_DATA}"
fi

# -------------------------------------------------------------------
# 3. Copy dataset into RAM  (/dev/shm = tmpfs, purely in-memory)
#    This eliminates ALL disk I/O during training.
# -------------------------------------------------------------------
RAM_DATA="/dev/shm/${USER}_vg_data"

if [[ ! -d "${RAM_DATA}" ]]; then
    echo ">>> Copying dataset into RAM (${RAM_DATA}) ..."
    mkdir -p "${RAM_DATA}"
    # Resolve symlinks during copy so RAM has real files
    cp -a -L "${SRC_DATA}/." "${RAM_DATA}/"
    echo ">>> Dataset copied.  Size in RAM: $(du -sh "${RAM_DATA}" | cut -f1)"
fi

# -------------------------------------------------------------------
# 4. HuggingFace cache on scratch (fast NVMe, avoids $HOME quota)
# -------------------------------------------------------------------
export HF_HOME="/scratch/alpine/${USER}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"
mkdir -p "${HF_HOME}"

# -------------------------------------------------------------------
# 5. Performance flags for H100
# -------------------------------------------------------------------
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
# Let NCCL use all available bandwidth (single-node, still helpful)
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0

# -------------------------------------------------------------------
# 6. Training hyper-parameters  (edit as needed)
# -------------------------------------------------------------------
OUTPUT_DIR="${SLURM_SUBMIT_DIR}/outputs"
EPOCHS=40
BATCH_SIZE=64          # H100 80 GB VRAM – push batch size for speed
LR=3e-4
PATIENCE=5
NUM_WORKERS=16         # half of --cpus-per-task → overlap I/O and compute
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
