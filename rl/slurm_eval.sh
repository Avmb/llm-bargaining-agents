#!/bin/bash
#SBATCH --job-name=bargain-eval
#SBATCH --partition=PARTITION   # <-- set to your Slurm partition
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --output=slurm_eval_%j.out
#SBATCH --error=slurm_eval_%j.err

set -euo pipefail

# --- Environment setup ---
module load cudatoolkit
module load brics/nccl
module load gcc-native/13.2

source "${VLLM_VENV:-$HOME/llm-bargaining-agents/.venv}/bin/activate"

# HF_TOKEN is taken from the environment (export it before submitting).
# WANDB_API_KEY is taken from the environment (export it before submitting).
export CC=gcc
export CXX=g++
export CUDAHOSTCXX=g++

SITE_PKGS=$(python3 -c 'import site; print(site.getsitepackages()[0])')
export LD_LIBRARY_PATH="${SITE_PKGS}/nvidia/nvjitlink/lib:${SITE_PKGS}/nvidia/cusparse/lib:${SITE_PKGS}/nvidia/cublas/lib:${SITE_PKGS}/nvidia/cuda_runtime/lib:${SITE_PKGS}/nvidia/cudnn/lib:${SITE_PKGS}/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"
export VLLM_NCCL_SO_PATH="${SITE_PKGS}/nvidia/nccl/lib/libnccl.so.2"

PORT=8000
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
CONFIG="${CONFIG:-configs/eval_default.yaml}"
LORA_PATH="${LORA_PATH:-}"          # path to LoRA checkpoint directory
LORA_NAME="${LORA_NAME:-trained}"   # model name to use in API calls for the LoRA model

echo "=== Bargaining RL Evaluation ==="
echo "Node: $(hostname), GPUs: $(nvidia-smi -L | wc -l)"
echo "Base model: ${MODEL}"
echo "Config: ${CONFIG}"
if [ -n "${LORA_PATH}" ]; then
    echo "LoRA adapter: ${LORA_PATH} (name: ${LORA_NAME})"
else
    echo "LoRA adapter: none (base model only)"
fi
echo "Start: $(date)"

# --- Build vLLM args ---
VLLM_ARGS=(
    --model "${MODEL}"
    --tensor-parallel-size 4
    --port "${PORT}"
    --max-model-len 32768
    --reasoning-parser qwen3
    --enable-prefix-caching
    --language-model-only
)
if [ -n "${LORA_PATH}" ]; then
    VLLM_ARGS+=(--enable-lora --lora-modules "${LORA_NAME}=${LORA_PATH}")
fi

# --- Start vLLM server in background ---
VLLM_LOG="vllm_${SLURM_JOB_ID}.log"
echo "Starting vLLM server on port ${PORT}..."
echo "vLLM logs: ${VLLM_LOG}"
vllm serve "${VLLM_ARGS[@]}" > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!

# Wait for vLLM to be ready (up to 5 minutes)
echo "Waiting for vLLM to be ready..."
for i in $(seq 1 60); do
    if curl -s "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; then
        echo "vLLM ready after $((i * 5))s"
        break
    fi
    if ! kill -0 ${VLLM_PID} 2>/dev/null; then
        echo "ERROR: vLLM process died"
        exit 1
    fi
    sleep 5
done

if ! curl -s "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; then
    echo "ERROR: vLLM failed to start within 5 minutes"
    kill ${VLLM_PID} 2>/dev/null || true
    exit 1
fi

# --- Determine model name for API calls ---
# If LoRA is loaded, the trained role uses the LoRA model name;
# the opponent uses the base model name.
EVAL_MODEL="${MODEL}"
EVAL_ARGS=()
if [ -n "${LORA_PATH}" ]; then
    # Read train_role from config to know which agent uses the LoRA model
    TRAIN_ROLE=$(python3 -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c.get('train',{}).get('train_role','buyer'))")
    if [ "${TRAIN_ROLE}" = "buyer" ]; then
        EVAL_ARGS+=(--buyer_model.model_name "${LORA_NAME}")
        EVAL_ARGS+=(--seller_model.model_name "${MODEL}")
    else
        EVAL_ARGS+=(--seller_model.model_name "${LORA_NAME}")
        EVAL_ARGS+=(--buyer_model.model_name "${MODEL}")
    fi
    echo "Train role: ${TRAIN_ROLE} uses LoRA model '${LORA_NAME}', opponent uses base '${MODEL}'"
else
    EVAL_ARGS+=(--buyer_model.model_name "${MODEL}")
    EVAL_ARGS+=(--seller_model.model_name "${MODEL}")
fi

# --- Run evaluation ---
echo "Running evaluation..."
python3 bargaining_rl.py \
    --config "${CONFIG}" \
    --mode eval \
    --buyer_model.base_url "http://localhost:${PORT}/v1" \
    --seller_model.base_url "http://localhost:${PORT}/v1" \
    "${EVAL_ARGS[@]}"

EXIT_CODE=$?

# --- Cleanup ---
echo "Shutting down vLLM..."
kill ${VLLM_PID} 2>/dev/null || true
wait ${VLLM_PID} 2>/dev/null || true

echo "Finished at $(date) with exit code ${EXIT_CODE}"
exit ${EXIT_CODE}
