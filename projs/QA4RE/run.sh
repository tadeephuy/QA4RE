# in the QA4RE folder
export OPENAI_KEY=""
export CUDA_VISIBLE_DEVICES="1"
DATA=TACRED
mode=test # or dev
# model=meta-llama/Llama-3.2-1B
model=google/gemma-2-2b-it
# remove --debug for entire run
python qa4re_hf_llm.py --mode $mode --dataset ${DATA} --run_setting zero_shot --type_constrained --prompt_config_name qa4re_prompt_config.yaml --model $model --debug

# saved file in '../../outputs/{}/{}/{}/{}'.format(args.dataset, args.ex_name, args.engine.replace('/', '-'), args.run_setting)