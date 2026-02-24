import sys
import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '../../.env'))
import random
from argparse import ArgumentParser

sys.path.append(os.path.join(sys.path[0], '..'))
sys.path.append(os.path.join(sys.path[0], '../..'))
sys.path.append(os.path.join(sys.path[0], '../../..'))


import warnings
warnings.filterwarnings("ignore")

from copy import deepcopy
import numpy as np
import pandas as pd
from lime.lime_text import *
pd.set_option('display.max_columns', 40)
pd.set_option('display.max_colwidth', 400)
pd.set_option('display.max_rows', 100)

from utils.gpt3_utils import *
from utils.data_utils import *
from utils.eval_utils import *
from utils.hf_model_utils import *
from utils.general_utils import *

from projs.re_templates import *
from projs.re_utils import *
from projs.QA4RE.qa4re_utils import *


def call_vllm_df(args, input_ready_df, output_dir):
    """Call vLLM server with examples in the input_ready_df."""
    predictions = []
    model_generations = []
    time_list = []
    
    # Create vLLM client
    client = create_vllm_client(base_url=args.vllm_base_url, api_key="dummy")
    
    for i, row in tqdm(input_ready_df.iterrows()):
        time_start = time.time()
        final_input_prompt = row['final_input_prompts']
        all_choices = row['all_choices']
        index2rel = row['index2rels']
        
        # Call vLLM server with guided_choice
        generation = call_vllm_server(
            client, 
            args.model, 
            final_input_prompt, 
            all_choices,
            temperature=0.0,
            max_tokens=2
        )
        model_generations.append(generation)
        prediction = index2rel[generation]
        predictions.append(prediction)

        time_list.append(time.time() - time_start)

    input_ready_df['model_generations'] = model_generations
    input_ready_df['predictions'] = predictions
    input_ready_df['time'] = time_list
    input_ready_df.to_csv(os.path.join(output_dir, f'subset.{args.mode}.output.csv'), index=False)

    return input_ready_df


def call_vllm_offline_df(args, input_ready_df, output_dir):
    """Call vLLM offline engine (no server) with batch inference."""
    from utils.hf_model_utils import load_vllm_offline, call_vllm_offline_batch
    import time
    
    print("Loading vLLM offline engine...")
    llm = load_vllm_offline(
        args.model, 
        dtype="bfloat16",
        gpu_memory_utilization=0.4,
        max_model_len=2048,
        seed=args.random_seed
    )
    
    # Prepare batch data
    prompts = input_ready_df['final_input_prompts'].tolist()
    all_choices_list = input_ready_df['all_choices'].tolist()
    
    print(f"Running batch inference on {len(prompts)} examples...")
    time_start = time.time()
    
    # Batch generate with guided decoding
    result = call_vllm_offline_batch(
        llm, 
        prompts, 
        all_choices_list,
        temperature=0.0,
        max_tokens=2,
        logprobs=args.logprobs
    )
    
    # Handle both return types (backward compatibility)
    if args.logprobs is not None:
        generations, logprobs_list = result
    else:
        generations = result
        logprobs_list = None
    
    total_time = time.time() - time_start
    avg_time = total_time / len(prompts)
    
    print(f"Batch inference completed in {total_time:.2f}s ({len(prompts)/total_time:.2f} it/s)")
    
    # Map generations to predictions
    predictions = [
        input_ready_df.iloc[i]['index2rels'][gen] 
        for i, gen in enumerate(generations)
    ]
    
    input_ready_df['model_generations'] = generations
    input_ready_df['predictions'] = predictions
    input_ready_df['time'] = [avg_time] * len(prompts)  # avg per sample
    
    # Add logprobs if requested
    if logprobs_list is not None:
        # Extract relevant information from logprobs
        # logprobs_list[i] is a list of dicts, one per generated token
        # Each dict maps token to Logprob object with logprob, rank, and decoded_token
        logprobs_data = []
        for lp in logprobs_list:
            if lp:
                # Get logprob of the first generated token (the answer choice)
                token_logprobs = lp[0] if lp else {}
                # Convert to dict with token info: {token: (token_id, logprob, rank)}
                token_info = {}
                for token_id, logprob_obj in token_logprobs.items():
                    decoded = logprob_obj.decoded_token if hasattr(logprob_obj, 'decoded_token') else str(token_id)
                    token_info[decoded] = {
                        'token_id': token_id,
                        'logprob': logprob_obj.logprob,
                        'rank': logprob_obj.rank
                    }
                logprobs_data.append(str(token_info))
            else:
                logprobs_data.append(None)
        input_ready_df['logprobs'] = logprobs_data
    
    input_ready_df.to_csv(os.path.join(output_dir, f'subset.{args.mode}.output.csv'), index=False)
    
    return input_ready_df


def call_hf_llm_df(args, input_ready_df, output_dir):
    """Call huggingface LLMs with examples in the input_ready_df."""
    predictions = []
    model_generations = []
    time_list = []
    # load model
    tokenizer, model = load_huggingface_model(args.model, args.num_gpus, args.use_t5)

    for i, row in tqdm(input_ready_df.iterrows()):
        time_start = time.time()
        final_input_prompt = row['final_input_prompts']
        all_choices = row['all_choices']
        # correct_choices = row['correct_choices']
        index2rel = row['index2rels']
        # first_token_of_each_verb_label = row['first_token_of_each_verb_label']
        # call Huggingface model
        if args.use_t5:
            generation = call_hf_llm_enc_dec(model, tokenizer, final_input_prompt, all_choices)
        else:
            generation = call_hf_llm_causal(model, tokenizer, final_input_prompt, all_choices)
        model_generations.append(generation)
        prediction = index2rel[generation]
        predictions.append(prediction)

        time_list.append(time.time() - time_start)

    input_ready_df['model_generations'] = model_generations
    input_ready_df['predictions'] = predictions
    input_ready_df['time'] = time_list
    input_ready_df.to_csv(os.path.join(output_dir, f'subset.{args.mode}.output.csv'), index=False)

    return input_ready_df

def run_qa4re(args):
    if args.mode == 'dev':
        subset_test = pd.read_csv(os.path.join(args.data_root, args.dataset, args.dev_subset_name), sep='\t')
    elif args.mode == 'test':
        subset_test = pd.read_csv(os.path.join(args.data_root, args.dataset, args.test_subset_name), sep='\t')

    prompt_params, output_dir, experiment_num = exp_dir_setup(args)  # every run, except debug mode, we will create a new experiment dir.s

    print("prompt_params: ", prompt_params)

    if args.run_setting.startswith('retrieval_few_shot'):  # train set as retrieval pool for retrieval few shot setting
        subset_train = pd.read_csv(os.path.join(args.data_root, args.dataset, args.train_subset_name), sep='\t')
    elif args.run_setting == 'zero_shot' or args.run_setting == 'fixed_few_shot':
        subset_train = None
    else:
        raise NotImplementedError

    if args.use_relation_definitions:
        from projs.QA4RE.qa4re_utils import prompt_preparation_with_relation_definitions
        input_ready_df = prompt_preparation_with_relation_definitions(args, prompt_params, subset_train, subset_test, tokenizer='google/gemma-2-2b-it')
    elif args.shuffle_answers:
        from projs.QA4RE.qa4re_utils import prompt_preparation_shuffle_answers
        input_ready_df = prompt_preparation_shuffle_answers(args, prompt_params, subset_train, subset_test, tokenizer='google/gemma-2-2b-it')
    else:
        input_ready_df = prompt_preparation(args, prompt_params, subset_train, subset_test, tokenizer='google/gemma-2-2b-it')
    
    input_ready_df.to_pickle('input_ready_df.pkl')
    # import pdb; pdb.set_trace()
    check_example(input_ready_df, ['final_input_prompts', 'all_choices', 'correct_choices', 'index2rels'])
    
    # Choose inference method based on flags
    if args.use_vllm_offline:
        output_df = call_vllm_offline_df(args, input_ready_df, output_dir)
    elif args.use_vllm:
        output_df = call_vllm_df(args, input_ready_df, output_dir)
    else:
        output_df = call_hf_llm_df(args, input_ready_df, output_dir)
    
    check_example(input_ready_df, ['final_input_prompts', 'all_choices', 'correct_choices', 'index2rels', 'model_generations', 'predictions'])
    metric_dict = evaluate_re(output_df, args.LABEL_VERBALIZER, args.POS_LABELS,)
    # sum up the time and cost
    metric_dict['time'] = output_df['time'].sum()
    metric_dict['cost'] = output_df['cost']
    metric_dict.update(prompt_params)
    metric_dict.update(vars(args))

    pd.DataFrame([metric_dict]).to_csv(output_dir + '/' + args.mode + '.metrics')

    batch_re_eval_print(output_df, 'predictions', args.LABEL_VERBALIZER, args.POS_LABELS, return_results=False)
    output_df.to_csv(output_dir + '/' + args.mode + '.gpt3.output.csv', sep='\t')

    return


def main():
    parser = ArgumentParser()
    # general args
    parser.add_argument('--debug', action='store_true')  # for debug.
    parser.add_argument('--task', default='RE')
    parser.add_argument('--method', default='multi_choice_qa')  # default, no change.
    parser.add_argument('--data_root', type=str, default='../../data/')
    # parser.add_argument('--engine', type=str, default='text-davinci-003')
    parser.add_argument('--model', type=str, default='google/flan-t5-small')
    parser.add_argument('--use_t5', action='store_true')
    parser.add_argument('--use_vllm', action='store_true', help='Use vLLM server for inference')
    parser.add_argument('--use_vllm_offline', action='store_true', help='Use vLLM offline engine (batch, no server)')
    parser.add_argument('--vllm_base_url', type=str, default='http://localhost:8000/v1', help='vLLM server URL')
    parser.add_argument('--logprobs', type=int, default=None, help='Number of top logprobs to return per token (None = disabled)')
    parser.add_argument('--random_seed', type=int, default=42)  # how demonstration will be sampled.
    parser.add_argument('--ex_name', default='multi_choice_qa4re')  # folder name to save.

    # dataset args
    parser.add_argument('--dataset', type=str, default='TACRED')
    parser.add_argument('--train_subset_name', type=str, default='train_subset.csv')
    parser.add_argument('--dev_subset_name', type=str, default='dev_subset.csv')
    parser.add_argument('--test_subset_name', type=str, default='test_subset.csv')
    parser.add_argument('--train_subset_samples', type=int, default=100)
    parser.add_argument('--dev_subset_samples', type=int, default=250)
    parser.add_argument('--test_subset_samples', type=int, default=1000)

    # experiment args
    parser.add_argument('--mode', type=str, default='dev', choices=['dev', 'test'])
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--prompt_config_name', type=str, default='qa4re_prompt_config.yaml')
    parser.add_argument('--type_constrained', action='store_true')
    parser.add_argument('--use_relation_definitions', action='store_true', help='Include relation definitions in the prompt')
    parser.add_argument('--shuffle_answers', action='store_true', help='Shuffle answer choices to avoid position bias')
    parser.add_argument('--in_context_size', type=int, default=0)  # number of demonstrations, 0 for zero-shot setting
    parser.add_argument('--run_setting', choices=['retrieval_few_shot_zero_seed', 'fixed_few_shot', 'zero_shot'], default='zero_shot')  # default true few shot setting. use train set to cross validate hyperparameters
    parser.add_argument('--sampling_strategy', default='random', choices=['roberta-large', 'random'])  # sentence embedding model for retrieval few shot setting
    parser.add_argument('--template', type=str, default='sure')
    parser.add_argument('--train_subset_seed', type=int, default=42)  # 0, 21, 42 How train subset is sampled.

    args = parser.parse_args()
    # for using GPT-3 series code.
    args.engine = args.model
    args = process_arguments(args)

    if args.mode == 'dev':
        print(f"....Running {args.dev_subset_samples} Dev set evaluation....")
    else:
        print(f"....Running {args.test_subset_samples} Test set evaluation....")

    run_qa4re(args)

if __name__ == '__main__':
    main()
