#!/bin/bash
#SBATCH --job-name=vllm-qwen3-8b
#SBATCH --partition=PARTITION   # <-- set to your Slurm partition
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --output=vllm_serve_%j.out
#SBATCH --error=vllm_serve_%j.err

module load cudatoolkit
module load brics/nccl
module load gcc-native/13.2

source "${VLLM_VENV:-$HOME/llm-bargaining-agents/.venv}/bin/activate"
# HF_TOKEN is taken from the environment (export it before submitting).

export CC=gcc
export CXX=g++
export CUDAHOSTCXX=g++

SITE_PKGS=$(python3 -c 'import site; print(site.getsitepackages()[0])')
export LD_LIBRARY_PATH="${SITE_PKGS}/nvidia/nvjitlink/lib:${SITE_PKGS}/nvidia/cusparse/lib:${SITE_PKGS}/nvidia/cublas/lib:${SITE_PKGS}/nvidia/cuda_runtime/lib:${SITE_PKGS}/nvidia/cudnn/lib:${SITE_PKGS}/nvidia/nccl/lib:${LD_LIBRARY_PATH}"
export VLLM_NCCL_SO_PATH="${SITE_PKGS}/nvidia/nccl/lib/libnccl.so.2"

export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
export VLLM_SERVER_DEV_MODE=1

PORT=8000
HOST_IP=$(hostname -i)

echo "Starting vLLM server on $(hostname), address: ${HOST_IP}:${PORT} at $(date)"
echo "GPUs:"
nvidia-smi -L

vllm serve \
    --model Qwen/Qwen3-8B \
    --enable-lora \
    --max-loras 2 \
    --max-lora-rank 32 \
    --tensor-parallel-size 4 \
    --port $PORT \
    --max-model-len 32768 \
    --max-num-seqs 1024 \
    --reasoning-parser qwen3 \
    --enable-prefix-caching \
    --language-model-only \
    --host 0.0.0.0
