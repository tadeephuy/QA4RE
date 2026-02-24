#!/bin/bash
# Run vanilla RE on BIORED dataset with vLLM offline engine (batch inference, no server needed)

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate QA4RE

export OPENAI_KEY=""
export CUDA_VISIBLE_DEVICES="0"

DATA=BIORED
mode=${1:-dev}  # dev or test, default is dev
model=${2:-google/gemma-2-2b-it}
seed=${3:-42}  # random seed, default is 42

echo "Running Vanilla RE on BIORED dataset with vLLM offline"
echo "Mode: $mode"
echo "Model: $model"
echo "Seed: $seed"
echo ""

python vanilla_re_hf_llm.py \
    --mode $mode \
    --dataset ${DATA} \
    --run_setting zero_shot \
    --type_constrained \
    --prompt_config_name vanilla_prompt_config.yaml \
    --model $model \
    --use_vllm_offline \
    --random_seed $seed \
    --dev_subset_name dev.csv \
    --test_subset_name test.csv \
    --train_subset_name train.csv \
    --dev_subset_samples 1162 \
    --test_subset_samples 1163 \
    --train_subset_samples 4178

# Output will be saved to: ../../outputs/BIORED/vanilla_re/{model}/zero_shot/
