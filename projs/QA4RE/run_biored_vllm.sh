#!/bin/bash
# Run QA4RE with BIORED dataset using vLLM server
# Make sure to start the vLLM server first

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate QA4RE

export OPENAI_KEY=""
export CUDA_VISIBLE_DEVICES="0"

DATA=BIORED
mode=${1:-dev}  # dev or test, default is dev
model=${2:-google/gemma-2-2b}
vllm_port=${3:-8001}

echo "Running QA4RE on BIORED dataset with vLLM"
echo "Mode: $mode"
echo "Model: $model"
echo "vLLM URL: http://localhost:${vllm_port}/v1"
echo ""

python qa4re_hf_llm.py \
    --mode $mode \
    --dataset ${DATA} \
    --run_setting zero_shot \
    --type_constrained \
    --prompt_config_name qa4re_prompt_config.yaml \
    --model $model \
    --use_vllm \
    --vllm_base_url http://localhost:${vllm_port}/v1 \
    --dev_subset_name dev.csv \
    --test_subset_name test.csv \
    --train_subset_name train.csv \
    --dev_subset_samples 1162 \
    --test_subset_samples 1163 \
    --train_subset_samples 4178

# Output will be saved to: ../../outputs/BIORED/multi_choice_qa4re/{model}/zero_shot/
