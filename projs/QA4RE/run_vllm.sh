#!/bin/bash
# Run QA4RE with vLLM server
# Make sure to start the vLLM server first with: bash start_vllm_server.sh

export OPENAI_KEY=""
export CUDA_VISIBLE_DEVICES="0"
DATA=TACRED
mode=test # or dev
model=google/gemma-2-2b

# Run with vLLM server (requires vLLM server running on port 8000)
# Start server in another terminal first:
#   bash start_vllm_server.sh $model 0 8000

python qa4re_hf_llm.py \
    --mode $mode \
    --dataset ${DATA} \
    --run_setting zero_shot \
    --type_constrained \
    --prompt_config_name qa4re_prompt_config.yaml \
    --model $model \
    --use_vllm \
    --vllm_base_url http://localhost:8001/v1

# saved file in '../../outputs/{}/{}/{}/{}'.format(args.dataset, args.ex_name, args.engine.replace('/', '-'), args.run_setting)
