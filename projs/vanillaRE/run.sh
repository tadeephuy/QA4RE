# In the vanillaRE folder


export OPENAI_KEY=""
export CUDA_VISIBLE_DEVICES="0"
DATA=TACRED
mode=test # or dev
model=google/gemma-2-2b

# remove --debug to run full split
python vanilla_re_hf_llm.py \
  --mode "$mode" \
  --dataset "$DATA" \
  --run_setting zero_shot \
  --type_constrained \
  --prompt_config_name vanilla_prompt_config.yaml \
  --model "$model" \

# Outputs saved under ../../outputs/{dataset}/{ex_name}/{engine}/
