#!/bin/bash
# Run vanilla RE with vLLM server


export OPENAI_KEY=""
export CUDA_VISIBLE_DEVICES="0"
DATASET="TACRED"
mode="test"  # or dev
model="google/gemma-2-2b"

python vanilla_re_hf_llm.py \
        --mode "$mode" \
        --dataset "$DATASET" \
        --run_setting zero_shot \
        --type_constrained \
        --prompt_config_name vanilla_prompt_config.yaml \
        --model "$model" \
        --use_vllm \
        --vllm_base_url http://localhost:8001/v1

# Outputs saved under ../../outputs/{dataset}/{ex_name}/{engine}/
