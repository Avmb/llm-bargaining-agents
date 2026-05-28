#!/bin/bash
#SBATCH --job-name=bargain-train
#SBATCH --partition=PARTITION   # <-- set to your Slurm partition
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=slurm_train_%j.out
#SBATCH --error=slurm_train_%j.err

set -euo pipefail

# --- Environment setup ---
module load cudatoolkit
module load brics/nccl
module load gcc-native/13.2

# HF_TOKEN is taken from the environment (export it before submitting).
# WANDB_API_KEY is taken from the environment (export it before submitting).
export CC=gcc
export CXX=g++
export CUDAHOSTCXX=g++

# Redirect temp/cache dirs to writable locations.
export TMPDIR="${TMPDIR:-/tmp/bargain_cache/tmp}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/bargain_cache/inductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/bargain_cache/triton}"

# Two separate venvs: vLLM (inference) and training (Unsloth)
VLLM_VENV="${VLLM_VENV:-$HOME/llm-bargaining-agents/.venv}"
TRAIN_VENV="${TRAIN_VENV:-$HOME/llm-bargaining-agents/rl/.train_venv}"

# --- Configuration ---
OPPONENT_MODEL="${OPPONENT_MODEL:-Qwen/Qwen3-8B}"
CONFIG="${CONFIG:-configs/train_default.yaml}"
MODE="${MODE:-train}"

# --- Node allocation ---
NODES=($(scontrol show hostnames $SLURM_JOB_NODELIST))
VLLM_NODE=${NODES[0]}
TRAIN_NODE=${NODES[1]}
PORT=8000

# Set up LD_LIBRARY_PATH using vLLM venv (has CUDA/NCCL libs)
source "${VLLM_VENV}/bin/activate"

# Resolve model to local cache path for Unsloth
TRAIN_MODEL_PATH=$(python3 -c "
from huggingface_hub import scan_cache_dir
model = '${OPPONENT_MODEL}'
for info in scan_cache_dir().repos:
    if info.repo_id == model:
        for rev in info.revisions:
            print(rev.snapshot_path)
            break
        break
")
if [ -n "${TRAIN_MODEL_PATH}" ]; then
    echo "Resolved local model cache: ${TRAIN_MODEL_PATH}"
else
    echo "WARNING: Model not found in local cache, will use HuggingFace Hub name"
    TRAIN_MODEL_PATH="${OPPONENT_MODEL}"
fi
SITE_PKGS=$(python3 -c 'import site; print(site.getsitepackages()[0])')
export LD_LIBRARY_PATH="${SITE_PKGS}/nvidia/nvjitlink/lib:${SITE_PKGS}/nvidia/cusparse/lib:${SITE_PKGS}/nvidia/cublas/lib:${SITE_PKGS}/nvidia/cuda_runtime/lib:${SITE_PKGS}/nvidia/cudnn/lib:${SITE_PKGS}/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"
export VLLM_NCCL_SO_PATH="${SITE_PKGS}/nvidia/nccl/lib/libnccl.so.2"
deactivate 2>/dev/null || true

echo "=== Bargaining RL Training (2-node) ==="
echo "vLLM node: ${VLLM_NODE} (4 GPUs), Training node: ${TRAIN_NODE} (4 GPUs)"
echo "Opponent model (vLLM): ${OPPONENT_MODEL}"
echo "Config: ${CONFIG}"
echo "Mode: ${MODE}"
echo "Start: $(date)"

# --- Start vLLM server on node 0 (all 4 GPUs) ---
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
export VLLM_SERVER_DEV_MODE=1
VLLM_LOG="vllm_${SLURM_JOB_ID}.log"
echo "Starting vLLM server on ${VLLM_NODE}, port ${PORT} (LoRA enabled, TP=4)..."

srun --nodes=1 --ntasks=1 --nodelist="${VLLM_NODE}" --gres=gpu:4 \
    --export=ALL \
    /usr/bin/bash -c "
        source '${VLLM_VENV}/bin/activate'
        export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
        export VLLM_SERVER_DEV_MODE=1
        vllm serve \
            --model '${OPPONENT_MODEL}' \
            --enable-lora \
            --max-loras 2 \
            --max-lora-rank 32 \
            --tensor-parallel-size 4 \
            --port ${PORT} \
            --max-model-len 32768 \
            --max-num-seqs 1024 \
            --reasoning-parser qwen3 \
            --enable-prefix-caching \
            --language-model-only \
            --host 0.0.0.0
    " > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!

# Wait for vLLM to be ready (up to 10 minutes — TP=4 + torch.compile is slow)
echo "Waiting for vLLM to be ready on ${VLLM_NODE}:${PORT}..."
for i in $(seq 1 120); do
    if srun --nodes=1 --ntasks=1 --nodelist="${TRAIN_NODE}" --gres=gpu:0 \
        /usr/bin/bash -c "curl -s 'http://${VLLM_NODE}:${PORT}/v1/models'" > /dev/null 2>&1; then
        echo "vLLM ready after $((i * 5))s"
        break
    fi
    if ! kill -0 ${VLLM_PID} 2>/dev/null; then
        echo "ERROR: vLLM process died"
        exit 1
    fi
    sleep 5
done

if ! srun --nodes=1 --ntasks=1 --nodelist="${TRAIN_NODE}" --gres=gpu:0 \
    /usr/bin/bash -c "curl -s 'http://${VLLM_NODE}:${PORT}/v1/models'" > /dev/null 2>&1; then
    echo "ERROR: vLLM failed to start within 10 minutes"
    kill ${VLLM_PID} 2>/dev/null || true
    exit 1
fi

# --- Run training on node 1 (all 4 GPUs, DDP via torchrun) ---
echo "Running training on ${TRAIN_NODE}..."
echo "Local model path: ${TRAIN_MODEL_PATH}"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
echo "Launching torchrun with --nproc_per_node=${NPROC_PER_NODE}"

srun --nodes=1 --ntasks=1 --nodelist="${TRAIN_NODE}" --gres=gpu:4 \
    --export=ALL \
    /usr/bin/bash -c "
        source '${TRAIN_VENV}/bin/activate'
        torchrun --standalone --nproc_per_node=${NPROC_PER_NODE} bargaining_rl.py \
            --config '${CONFIG}' \
            --mode '${MODE}' \
            --buyer_model.base_url 'http://${VLLM_NODE}:${PORT}/v1' \
            --seller_model.base_url 'http://${VLLM_NODE}:${PORT}/v1' \
            --train_model_path '${TRAIN_MODEL_PATH}'
    "

EXIT_CODE=$?

# --- Cleanup ---
echo "Shutting down vLLM..."
kill ${VLLM_PID} 2>/dev/null || true
wait ${VLLM_PID} 2>/dev/null || true

echo "Finished at $(date) with exit code ${EXIT_CODE}"
exit ${EXIT_CODE}
