import torch
import random
import numpy as np
from tqdm import tqdm
import ipdb
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList, MinNewTokensLengthLogitsProcessor, MinLengthLogitsProcessor
from transformers import T5Tokenizer, T5ForConditionalGeneration
from openai import OpenAI


def load_t5_model(model_name: str, **kwargs):
    """
    Local T5 Model."""
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(
        model_name, 
        offload_folder="offload", 
        torch_dtype=torch.float16, **kwargs)
        
    return tokenizer, model


def load_casual_model(model_name: str, **kwargs):
    """
    Local Casual Model like Alpaca."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        offload_folder="offload", 
        torch_dtype=torch.float16, **kwargs)
    return tokenizer, model


def load_huggingface_model(model_name: str, num_gpus: int, t5_model: bool):
    """
    Load a model from the HuggingFace library.
    """
    if num_gpus == 1:
        kwargs = {}
    else:
        kwargs = {
            "device_map": "auto",
            "max_memory": {i: "45GiB" for i in range(num_gpus)},
        }

    if t5_model:
        tokenizer, model = load_t5_model(model_name, **kwargs)
    else:
        tokenizer, model = load_casual_model(model_name, **kwargs)
    if num_gpus == 1:
        model.cuda()
        
    return tokenizer, model


def add_to_lattice(sequence, lattice={}):
    if len(sequence) == 0:
        return {}
    else:
        element = sequence[0]

        lattice[element] = add_to_lattice(sequence[1:], lattice.get(element,{}))

        return lattice


def call_hf_llm_enc_dec(model, tokenizer, prompt, label_space):

    inputs = tokenizer([prompt])
    
    label_prefix_dict = {}
    label_lengths = []
    
    for label_ind, label in enumerate(label_space):
        label_inp = tokenizer(label,add_special_tokens=False).input_ids
        label_lengths.append(len(label_inp))
        label_inp += [tokenizer.eos_token_id]

        label_prefix_dict = add_to_lattice(label_inp, label_prefix_dict)

    def constrain_fnc(batch, input_ids):

        rel_prefix = []

        for tok in input_ids:
            if tok not in tokenizer.all_special_ids:
                rel_prefix.append(tok)        
        
        options = label_prefix_dict
        
        for tok in rel_prefix:
            options = options.get(int(tok),{})
            
        return list(options.keys())
    
    max_length = int(np.max(label_lengths))
    min_length = int(np.min(label_lengths))
    
    logits_processor = LogitsProcessorList(
        [
            MinLengthLogitsProcessor(min_length, eos_token_id=tokenizer.eos_token_id),
        ]
    )

    output_ids = model.generate(torch.as_tensor(inputs.input_ids).cuda(),
                                max_new_tokens=max_length,
                                logits_processor=logits_processor,
                                prefix_allowed_tokens_fn=constrain_fnc,
                                eos_token_id=tokenizer.eos_token_id
                               )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
#     if outputs_unc in labels and outputs_unc != outputs:
#         ipdb.set_trace()
            
    if outputs not in label_space:
        ipdb.set_trace()
        print('random choice {}'.format(outputs))
        outputs = random.choice(label_space)
        
    return outputs

def call_hf_llm_causal(model, tokenizer, prompt, label_space):

    inputs = tokenizer([prompt])
    
    label_prefix_dict = {}
    label_lengths = []
    
    for label_ind, label in enumerate(label_space):
        label_inp = tokenizer(label,add_special_tokens=False).input_ids
        label_lengths.append(len(label_inp))
        label_inp += [tokenizer.eos_token_id]
        
        label_prefix_dict = add_to_lattice(label_inp, label_prefix_dict)
    
    prefix_to_ignore = len(inputs.input_ids[0])
     
    def constrain_fnc(batch, input_ids):
        
        rel_prefix = input_ids[prefix_to_ignore:]
        
        options = label_prefix_dict
        
        for tok in rel_prefix:
            options = options.get(int(tok),{})
            
        return list(options.keys())
    
    max_length = int(np.max(label_lengths))
    min_length = int(np.min(label_lengths))
    
    device = next(model.parameters()).device
    eos_token_id_tensor = torch.tensor([tokenizer.eos_token_id], device=device)

    logits_processor = LogitsProcessorList(
        [
            MinNewTokensLengthLogitsProcessor(prefix_to_ignore, min_length, eos_token_id=eos_token_id_tensor),
        ]
    )

    output_ids = model.generate(torch.as_tensor(inputs.input_ids).cuda(),
                                max_new_tokens=max_length,
                                logits_processor=logits_processor,
                                prefix_allowed_tokens_fn=constrain_fnc,
                                eos_token_id=tokenizer.eos_token_id
                               )

    outputs = tokenizer.batch_decode(output_ids[:,len(inputs.input_ids[0]):], skip_special_tokens=True)[0].strip()
    
    output_ids_unc = model.generate(torch.as_tensor(inputs.input_ids).cuda(),
                                max_new_tokens=max_length,
                                logits_processor=logits_processor,
                                eos_token_id=tokenizer.eos_token_id
                               )

    outputs_unc = tokenizer.batch_decode(output_ids_unc[:,len(inputs.input_ids[0]):], skip_special_tokens=True)[0].strip()
        
#     if outputs_unc in labels and outputs_unc != outputs:
#         ipdb.set_trace()
            
    if outputs not in label_space:
        ipdb.set_trace()
        print('random choice {}'.format(outputs))
        outputs = random.choice(label_space)
        
    return outputs


def create_vllm_client(base_url: str = "http://localhost:8000/v1", api_key: str = "dummy"):
    """
    Create an OpenAI client connected to vLLM server.
    """
    client = OpenAI(base_url=base_url, api_key=api_key)
    return client


def call_vllm_server(client, model_name: str, prompt: str, label_space: list, temperature: float = 0.0, max_tokens: int = 2):
    """
    Call vLLM server with guided_choice constraint for multiple choice QA.
    
    Args:
        client: OpenAI client pointing to vLLM server
        model_name: Name of the model being served
        prompt: Input prompt string
        label_space: List of valid choices (e.g., ['A', 'B', 'C'])
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
    
    Returns:
        Generated text (should be one of the labels in label_space)
    """
    try:
        response = client.completions.create(
            model=model_name,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body={"guided_choice": label_space}  # vLLM's guided decoding
        )
        
        outputs = response.choices[0].text.strip()
        
        # Fallback to random choice if output not in label_space
        if outputs not in label_space:
            print(f'Warning: Generated "{outputs}" not in label_space {label_space}, using random choice')
            outputs = random.choice(label_space)
            
        return outputs
        
    except Exception as e:
        print(f'Error calling vLLM server: {e}')
        return "Error"


def load_vllm_offline(model_name: str, dtype: str = "bfloat16", gpu_memory_utilization: float = 0.8, max_model_len: int = 2048, seed: int = 42, **kwargs):
    """
    Load vLLM offline engine for batch inference (no server needed).
    
    Args:
        model_name: HuggingFace model name
        dtype: Model dtype (bfloat16, float16, etc.)
        gpu_memory_utilization: Fraction of GPU memory to use
        max_model_len: Maximum context length
        seed: Random seed for reproducibility
        **kwargs: Additional vLLM LLM arguments
    
    Returns:
        vLLM LLM engine
    """
    from vllm import LLM
    
    llm = LLM(
        model=model_name,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        seed=seed,
        **kwargs
    )
    return llm


def call_vllm_offline_batch(llm, prompts: list, all_choices_list: list, temperature: float = 0.0, max_tokens: int = 2, logprobs: int = None):
    """
    Batch inference with vLLM offline engine with optimized guided decoding.
    Groups prompts by choice pattern to minimize FSM compilation overhead.
    
    Args:
        llm: vLLM LLM engine
        prompts: List of prompt strings
        all_choices_list: List of label_space lists (one per prompt)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        logprobs: Number of top logprobs to return per token (None = don't return logprobs)
    
    Returns:
        If logprobs is None:
            List of generated texts (one per prompt)
        If logprobs is not None:
            Tuple of (generations, logprobs_list) where logprobs_list contains detailed probability info
    """
    from vllm import SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
    from collections import defaultdict
    
    # Group prompts by choice pattern (minimize FSM compilations)
    choice_groups = defaultdict(list)
    for i, choices in enumerate(all_choices_list):
        key = tuple(sorted(choices))  # Normalize order for caching
        choice_groups[key].append(i)
    
    # Process each group with same guided_choice (single FSM compile per group)
    all_generations = [None] * len(prompts)
    all_logprobs = [None] * len(prompts) if logprobs is not None else None
    
    for choices_key, indices in choice_groups.items():
        # Batch for this choice pattern
        batch_prompts = [prompts[i] for i in indices]
        
        # Single sampling params for this group
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=logprobs,
            guided_decoding=GuidedDecodingParams(
                backend="outlines",
                choice=list(choices_key)
            )
        )
        
        # Generate for this batch
        outputs = llm.generate(batch_prompts, [sampling_params] * len(batch_prompts))
        
        # Map back to original indices
        for idx, out in zip(indices, outputs):
            all_generations[idx] = out.outputs[0].text.strip()
            if logprobs is not None:
                all_logprobs[idx] = out.outputs[0].logprobs
    
    if logprobs is not None:
        return all_generations, all_logprobs
    return all_generations
