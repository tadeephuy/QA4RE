#!/bin/bash
# Run QA4RE with vLLM offline engine (batch inference, no server needed)

DATASET="TACRED"
SPLIT="test"
MODEL="google/gemma-2-2b-it"
NUM_GPUS=1
CUDA_DEVICE="1"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export OPENAI_KEY=""

python qa4re_hf_llm.py \
    --dataset "$DATASET" \
    --mode "$SPLIT" \
    --model "$MODEL" \
    --use_vllm_offline \
    --num_gpus "$NUM_GPUS" \
    --run_setting zero_shot \
    --type_constrained \
    --prompt_config_name qa4re_prompt_config.yaml
