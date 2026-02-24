#!/bin/bash
# Script to start vLLM server for QA4RE
# Usage: bash start_vllm_server.sh [model_name] [gpu_id] [port] [dtype] [max_len] [enable_tools]

MODEL=${1:-"google/gemma-2-2b"}
GPU_ID=${2:-0}
PORT=${3:-8000}
DTYPE=${4:-"bfloat16"}
MAX_LEN=${5:-2048}

# Load and export env vars (HF_TOKEN, HF_HOME, etc.)
if [ -f /mnt/12T/huy/QA4RE/.env ]; then
    set -a
    source /mnt/12T/huy/QA4RE/.env
    set +a
fi

# Normalize token env var for HF
export HF_TOKEN="$HUGGING_FACE_HUB_TOKEN"

echo "Starting vLLM server..."
echo "Model: $MODEL"
echo "GPU: $GPU_ID"
echo "Port: $PORT"
echo "dtype: $DTYPE"
echo "max len: $MAX_LEN"

export CUDA_VISIBLE_DEVICES=$GPU_ID

vllm serve $MODEL \
    --dtype $DTYPE \
    --port $PORT \
    --max-model-len $MAX_LEN \
    --gpu-memory-utilization 0.4

echo "vLLM server started at http://localhost:$PORT"
echo "To stop the server, press Ctrl+C"
