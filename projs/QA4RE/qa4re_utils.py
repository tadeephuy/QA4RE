import numpy as np
import pandas as pd

from copy import deepcopy
from tqdm import tqdm
from transformers import GPT2Tokenizer, AutoTokenizer

from projs.re_templates import *
from projs.re_utils import *
from utils.gpt3_utils import *
from utils.eval_utils import *
from utils.general_utils import *


def fill_prompt_format_re_df(args, df, prompt_config, sent_col='masked_sents'):
    """
    prompts are full, with sentence and label.
    empty_prompts with only sentences.
    So prompts are used for demonstrations while empty_prompts are used for test examples
    """
    prompts = []
    empty_prompts = []
    index2rels = []
    all_choices = []
    correct_choices = []

    for i, row in df.iterrows():
        # sent = row[sent_col]
        ent1 = row['ent1']
        ent2 = row['ent2']
        ent1_type = row['ent1_type']
        ent2_type = row['ent2_type']
        sent = row[sent_col].replace('ENT1', f"[{ent1}]").replace('ENT2', f"[{ent2}]")
        # label = all_label_verbalizer[str(label)]
        label = row['label']
        type_cons = row['type_cons']
        prompt_sample_structure = prompt_config['example_format'] + ' {}'
        empty_prompt_sample_structure = prompt_config['example_format']
        example = sent + "\nOptions:\n"
        correct_templates = args.LABEL_TEMPLATES[label]
        correct_template_index = []
        prediction_range = []
        valid_relations = args.VALID_CONDITIONS_REV[type_cons]
        if args.NOTA_RELATION is not None:
            valid_relations = valid_relations + [args.NOTA_RELATION]
        start_chr = 'A'
        index2rel = {}
        for valid_relation in valid_relations:
            for template in args.LABEL_TEMPLATES[valid_relation]:
                # filled_template = template.format(subj=ent1, obj=ent2)
                filled_template = template.format(subj=f"[{ent1}]", obj=f"[{ent2}]")
                if template in correct_templates:
                    correct_template_index.append(start_chr)
                prediction_range.append(start_chr)
                example += f"{start_chr}. {filled_template}\n"
                index2rel[start_chr] = valid_relation
                start_chr = chr(ord(start_chr) + 1)

        if correct_template_index == []:  # some type constraint cases not included
            print("Warning: no correct template found!")
            print("sent: ", sent)
            print("ent1: ", ent1)
            print("ent2: ", ent2)
            print("ent1_type: ", ent1_type)
            print("ent2_type: ", ent2_type)
            print("label: ", label)
            print("valid_relations: ", valid_relations)
            print("correct_templates: ", correct_templates)

            print("#" * 50)
            correct_template_index.append(chr(ord(start_chr) - 1))
        correct_index = correct_template_index[0]

        prompt = prompt_sample_structure.format(example, correct_index)
        empty_prompt = empty_prompt_sample_structure.format(example)

        prompts.append(prompt)
        empty_prompts.append(empty_prompt)

        correct_choices.append(correct_template_index)
        index2rels.append(index2rel)
        all_choices.append(prediction_range)

    df['prompts'] = prompts
    df['empty_prompts'] = empty_prompts
    df['correct_choices'] = correct_choices
    df['index2rels'] = index2rels
    df['all_choices'] = all_choices

    return df


def fill_prompt_format_re_df_biored(args, df, prompt_config, possible_label_dict, sent_col='masked_sents'):
    """
    Modified version for BIORED that uses entity type combinations to constrain possible labels.
    Instead of using type_cons from VALID_CONDITIONS_REV, we use the actual possible labels
    from the training data for each entity type pair.
    
    Args:
        args: Arguments containing LABEL_TEMPLATES and NOTA_RELATION
        df: DataFrame with the data
        prompt_config: Prompt configuration
        possible_label_dict: Dictionary mapping entity type pairs to possible labels
        sent_col: Column name for sentences
    
    Returns:
        DataFrame with prompt columns added
    """
    prompts = []
    empty_prompts = []
    index2rels = []
    all_choices = []
    correct_choices = []

    for i, row in df.iterrows():
        ent1 = row['ent1']
        ent2 = row['ent2']
        ent1_type = row['ent1_type']
        ent2_type = row['ent2_type']
        sent = row[sent_col].replace('ENT1', f"[{ent1}]").replace('ENT2', f"[{ent2}]")
        label = row['label']
        
        # Create entity type pair key
        ent_type_pair = f"{ent1_type} -> {ent2_type}"
        
        prompt_sample_structure = prompt_config['example_format'] + ' {}'
        empty_prompt_sample_structure = prompt_config['example_format']
        example = sent + "\nOptions:\n"
        
        correct_templates = args.LABEL_TEMPLATES[label]
        correct_template_index = []
        prediction_range = []
        
        # Get valid relations from the possible_label_dict based on entity type pair
        if ent_type_pair in possible_label_dict:
            valid_relations = possible_label_dict[ent_type_pair]
        else:
            # Fallback to the original method if entity type pair not found
            print(f"Warning: entity type pair '{ent_type_pair}' not found in possible_label_dict, using fallback")
            type_cons = row['type_cons']
            valid_relations = args.VALID_CONDITIONS_REV[type_cons]
            if args.NOTA_RELATION is not None:
                valid_relations = valid_relations + [args.NOTA_RELATION]
        
        start_chr = 'A'
        index2rel = {}
        
        for valid_relation in valid_relations:
            for template in args.LABEL_TEMPLATES[valid_relation]:
                filled_template = template.format(subj=f"[{ent1}]", obj=f"[{ent2}]")
                if template in correct_templates:
                    correct_template_index.append(start_chr)
                prediction_range.append(start_chr)
                example += f"{start_chr}. {filled_template}\n"
                index2rel[start_chr] = valid_relation
                start_chr = chr(ord(start_chr) + 1)

        if correct_template_index == []:  # some type constraint cases not included
            print("Warning: no correct template found!")
            print("sent: ", sent)
            print("ent1: ", ent1)
            print("ent2: ", ent2)
            print("ent1_type: ", ent1_type)
            print("ent2_type: ", ent2_type)
            print("label: ", label)
            print("valid_relations: ", valid_relations)
            print("correct_templates: ", correct_templates)
            print("#" * 50)
            correct_template_index.append(chr(ord(start_chr) - 1))
            
        correct_index = correct_template_index[0]

        prompt = prompt_sample_structure.format(example, correct_index)
        empty_prompt = empty_prompt_sample_structure.format(example)

        prompts.append(prompt)
        empty_prompts.append(empty_prompt)
        correct_choices.append(correct_template_index)
        index2rels.append(index2rel)
        all_choices.append(prediction_range)

    df['prompts'] = prompts
    df['empty_prompts'] = empty_prompts
    df['correct_choices'] = correct_choices
    df['index2rels'] = index2rels
    df['all_choices'] = all_choices

    return df


def fill_prompt_format_re_df_biored_shuffle_answer(args, df, prompt_config, possible_label_dict, sent_col='masked_sents', random_seed=42):
    """
    Modified version for BIORED that shuffles answer choices to avoid position bias.
    Uses entity type combinations to constrain possible labels.
    
    Args:
        args: Arguments containing LABEL_TEMPLATES and NOTA_RELATION
        df: DataFrame with the data
        prompt_config: Prompt configuration
        possible_label_dict: Dictionary mapping entity type pairs to possible labels
        sent_col: Column name for sentences
        random_seed: Random seed for shuffling (default: 42)
    
    Returns:
        DataFrame with prompt columns added
    """
    import random
    rng = random.Random(random_seed)
    
    prompts = []
    empty_prompts = []
    index2rels = []
    all_choices = []
    correct_choices = []

    for i, row in df.iterrows():
        ent1 = row['ent1']
        ent2 = row['ent2']
        ent1_type = row['ent1_type']
        ent2_type = row['ent2_type']
        sent = row[sent_col].replace('ENT1', f"[{ent1}]").replace('ENT2', f"[{ent2}]")
        label = row['label']
        
        # Create entity type pair key
        ent_type_pair = f"{ent1_type} -> {ent2_type}"
        
        prompt_sample_structure = prompt_config['example_format'] + ' {}'
        empty_prompt_sample_structure = prompt_config['example_format']
        
        correct_templates = args.LABEL_TEMPLATES[label]
        
        # Get valid relations from the possible_label_dict based on entity type pair
        if ent_type_pair in possible_label_dict:
            valid_relations = possible_label_dict[ent_type_pair]
        else:
            # Fallback to the original method if entity type pair not found
            print(f"Warning: entity type pair '{ent_type_pair}' not found in possible_label_dict, using fallback")
            type_cons = row['type_cons']
            valid_relations = args.VALID_CONDITIONS_REV[type_cons]
            if args.NOTA_RELATION is not None:
                valid_relations = valid_relations + [args.NOTA_RELATION]
        
        # Create list of (relation, template) tuples and shuffle them
        relation_template_pairs = []
        for valid_relation in valid_relations:
            for template in args.LABEL_TEMPLATES[valid_relation]:
                relation_template_pairs.append((valid_relation, template))
        
        # Shuffle the order of answer choices
        rng.shuffle(relation_template_pairs)
        
        # Now build the example with shuffled choices
        example = sent + "\nOptions:\n"
        correct_template_index = []
        prediction_range = []
        start_chr = 'A'
        index2rel = {}
        
        for valid_relation, template in relation_template_pairs:
            filled_template = template.format(subj=f"[{ent1}]", obj=f"[{ent2}]")
            # Check if this relation matches the correct label (more robust than template matching)
            if valid_relation == label:
                correct_template_index.append(start_chr)
            prediction_range.append(start_chr)
            example += f"{start_chr}. {filled_template}\n"
            index2rel[start_chr] = valid_relation
            start_chr = chr(ord(start_chr) + 1)

        if correct_template_index == []:  # some type constraint cases not included
            print("Warning: no correct template found!")
            print("sent: ", sent)
            print("ent1: ", ent1)
            print("ent2: ", ent2)
            print("ent1_type: ", ent1_type)
            print("ent2_type: ", ent2_type)
            print("label: ", label)
            print("valid_relations: ", valid_relations)
            print("correct_templates: ", correct_templates)
            print("#" * 50)
            correct_template_index.append(chr(ord(start_chr) - 1))
            
        correct_index = correct_template_index[0]

        prompt = prompt_sample_structure.format(example, correct_index)
        empty_prompt = empty_prompt_sample_structure.format(example)

        prompts.append(prompt)
        empty_prompts.append(empty_prompt)
        correct_choices.append(correct_template_index)
        index2rels.append(index2rel)
        all_choices.append(prediction_range)

    df['prompts'] = prompts
    df['empty_prompts'] = empty_prompts
    df['correct_choices'] = correct_choices
    df['index2rels'] = index2rels
    df['all_choices'] = all_choices

    return df


def fill_prompt_format_re_df_biored_with_definitions(args, df, prompt_config, possible_label_dict, relation_definitions, sent_col='masked_sents'):
    """
    Modified version for BIORED that includes relation definitions in the prompt.
    Only includes definitions for relations that are possible for each sample based on entity type combinations.
    
    Args:
        args: Arguments containing LABEL_TEMPLATES and NOTA_RELATION
        df: DataFrame with the data
        prompt_config: Prompt configuration
        possible_label_dict: Dictionary mapping entity type pairs to possible labels
        relation_definitions: Dictionary with relation definitions (loaded from JSON)
        sent_col: Column name for sentences
    
    Returns:
        DataFrame with prompt columns added
    """
    prompts = []
    empty_prompts = []
    index2rels = []
    all_choices = []
    correct_choices = []

    for i, row in df.iterrows():
        ent1 = row['ent1']
        ent2 = row['ent2']
        ent1_type = row['ent1_type']
        ent2_type = row['ent2_type']
        sent = row[sent_col].replace('ENT1', f"[{ent1}]").replace('ENT2', f"[{ent2}]")
        label = row['label']
        
        # Create entity type pair key
        ent_type_pair = f"{ent1_type} -> {ent2_type}"
        
        prompt_sample_structure = prompt_config['example_format'] + ' {}'
        empty_prompt_sample_structure = prompt_config['example_format']
        
        # Get valid relations from the possible_label_dict based on entity type pair
        if ent_type_pair in possible_label_dict:
            valid_relations = possible_label_dict[ent_type_pair]
        else:
            # Fallback to the original method if entity type pair not found
            print(f"Warning: entity type pair '{ent_type_pair}' not found in possible_label_dict, using fallback")
            type_cons = row['type_cons']
            valid_relations = args.VALID_CONDITIONS_REV[type_cons]
            if args.NOTA_RELATION is not None:
                valid_relations = valid_relations + [args.NOTA_RELATION]
        
        # Build the example with sentence and options
        example = sent + "\nOptions:\n"
        
        correct_templates = args.LABEL_TEMPLATES[label]
        correct_template_index = []
        prediction_range = []
        
        start_chr = 'A'
        index2rel = {}
        
        for valid_relation in valid_relations:
            # Get relation definition if available
            if valid_relation in relation_definitions:
                rel_def = relation_definitions[valid_relation]['definition']
                # Create filled template using definition: [ENT1] definition [ENT2]
                filled_template = f"[{ent1}] {rel_def} [{ent2}]"
            else:
                # Fallback to original template format
                for template in args.LABEL_TEMPLATES[valid_relation]:
                    filled_template = template.format(subj=f"[{ent1}]", obj=f"[{ent2}]")
                    break  # Just use the first template
            
            # Check if this matches the correct answer
            if valid_relation == label:
                correct_template_index.append(start_chr)
            
            prediction_range.append(start_chr)
            example += f"{start_chr}. {filled_template}\n"
            index2rel[start_chr] = valid_relation
            start_chr = chr(ord(start_chr) + 1)

        if correct_template_index == []:  # some type constraint cases not included
            print("Warning: no correct template found!")
            print("sent: ", sent)
            print("ent1: ", ent1)
            print("ent2: ", ent2)
            print("ent1_type: ", ent1_type)
            print("ent2_type: ", ent2_type)
            print("label: ", label)
            print("valid_relations: ", valid_relations)
            print("correct_templates: ", correct_templates)
            print("#" * 50)
            correct_template_index.append(chr(ord(start_chr) - 1))
        
        correct_index = correct_template_index[0]

        prompt = prompt_sample_structure.format(example, correct_index)
        empty_prompt = empty_prompt_sample_structure.format(example)

        prompts.append(prompt)
        empty_prompts.append(empty_prompt)

        correct_choices.append(correct_template_index)
        index2rels.append(index2rel)
        all_choices.append(prediction_range)

    df['prompts'] = prompts
    df['empty_prompts'] = empty_prompts
    df['correct_choices'] = correct_choices
    df['index2rels'] = index2rels
    df['all_choices'] = all_choices

    return df


def build_final_input(df, params, engine, tokenizer):
    """build the test ready prompts for the API call."""
    max_tokens = params['max_tokens']
    build_final_input = []
    if engine in GPT_MODEL_MAX_TOKEN_DICT:
            max_allowed_token_num = GPT_MODEL_MAX_TOKEN_DICT[engine]
    else:
        print(f"Warning: unknown engine {engine}, make sure you are using huggingface LLMs or update the GPT_MODEL_MAX_TOKEN_DICT.")
        max_allowed_token_num = 2000 #  token num for FLAN-T5

    for i, row in tqdm(df.iterrows()):
        task_instructions = params['task_instructions'].strip()
        final_input_prompt = task_instructions + '\n\n' + row['data_prompts']

        tokens = tokenizer.encode(final_input_prompt)
        if len(tokens) + max_tokens > max_allowed_token_num:  # only necessary for few-shot setting.
            print(f"Warning: the prompt is too long for {engine}, will remove all the demonstrations.")
            while len(tokens) + max_tokens > max_allowed_token_num:  # remove all the demos.
                final_input_prompt = task_instructions + '\n\n' + row['empty_prompts']
                tokens = tokenizer.encode(final_input_prompt)
        build_final_input.append(final_input_prompt.strip())  # make sure not ends with ' '.
    return build_final_input


def build_logit_biases_df(df, params, tokenizer):
    """build the logit biases for each row in the dataframe.
    return: all_logit_biases, all_first_token_of_each_verb_label
    """
    all_logit_biases = []
    all_first_token_of_each_verb_label = []
    for i, row in df.iterrows():
        verbalized_labels = row['verbalized_labels']
        logit_biases = build_logit_biases(verbalized_labels, params['max_tokens'], tokenizer)
        all_logit_biases.append(logit_biases)
        first_token_of_each_verb_label = [tokenizer.decode(tokenizer.encode(" " + label)[0]) for label in verbalized_labels]
        all_first_token_of_each_verb_label.append(first_token_of_each_verb_label)
    return all_logit_biases, all_first_token_of_each_verb_label


def get_prediction(args, params, openai_api_response, all_choices, correct_choices, index2rel, engine, first_token_of_each_verb_label):
    # correct_template_indexes
    """get the prediction from the openai api response.
    return generated_content, and parsed prediction.
    """
    def get_pred_from_generated_content(generated_content, all_choices):
        """get the prediction from the generated content, for multiple token generation and chatCompletion"""
        candidates = []
        for choice in all_choices:
            if choice in generated_content:
                candidates.append(choice)
            if len(candidates) == 0:  # not generated, randomly choose one.
                pred = random.choice(all_choices)
            elif len(candidates) > 1:  # get the first one in generation order
                start_indexes = [generated_content.index(can) for can in candidates]
                pred = candidates[np.argmin(start_indexes)]
            else:
                pred = candidates[0]
        return pred

    if len(correct_choices) == 0:  # answer is not in the choices.  # type constrains wrong, < 0.3%
        return args.nota_relation

    if GPT_MODEL_TYPE_DICT[engine] == 'text':  # able to get prob and some logit.
        text = openai_api_response['choices'][0]['text']
        if params['max_tokens'] == 1:
            probs = []
            top_logprobs = openai_api_response['choices'][0]['logprobs']['top_logprobs'][0]
            for token in first_token_of_each_verb_label:
                if token in top_logprobs:
                    probs.append(top_logprobs[token])
                else:
                    probs.append(-100)
            pred_index = all_choices[np.argmax(probs)]
        else:
            generated_content = openai_api_response['choices'][0]['text']
            pred_index = get_pred_from_generated_content(generated_content, all_choices)
    elif GPT_MODEL_TYPE_DICT[engine] == 'chat':
        generated_content = openai_api_response['choices'][0]['message']['content']
        text = generated_content
        pred_index = get_pred_from_generated_content(generated_content, all_choices)
    else:
        raise ValueError('GPT_MODEL_TYPE_DICT[engine] should be text or chat')

    return text, index2rel[pred_index]



def prompt_preparation(args, params, subset_train_df=None, test_df=None, tokenizer='gpt2'):
    """prepare the prompt for the API query."""
    # 3. fill the template for final task.
    test_df = deepcopy(test_df)
    test_df = test_df[:args.run_subset_num]
    # reindex
    test_df = test_df.reset_index(drop=True)

    # add constrains to both dfs
    test_df = add_type_constraints(test_df)
    if subset_train_df is not None:
        subset_train_df = add_type_constraints(subset_train_df)

    # final_input_df = fill_prompt_format_re_df(args, test_df, params, 'masked_sents')
    import json
    with open("../../data/BIORED/configs/possible_labels.json", 'r') as f:
        possible_labels_biored = json.load(f)
    final_input_df = fill_prompt_format_re_df_biored(args, test_df, params, possible_labels_biored, 'masked_sents')
    
    if args.setting == 'few_shot':
        subset_train_df = fill_prompt_format_re_df(args, subset_train_df, params, 'masked_sents')
        # 4. retrieval in-context examples for retrieval few-shot.
        # for each example, get demonstrations with retrieval / random.
        if args.type_constrained and args.sampling_strategy == 'random':  # TODO test
            final_input_df = get_type_constrained_random_demos(subset_train_df, final_input_df, args.in_context_size, args.random_seed)
        else:
            final_input_df = get_demonstrations(subset_train_df, final_input_df, args.sampling_strategy, args.random_seed)
    elif args.setting == 'zero_shot':  # directly input for zero-shot!
        assert subset_train_df is None
        final_input_df['data_prompts'] = final_input_df['empty_prompts']
    else:
        raise ValueError('setting should be either few_shot or zero_shot')

    if tokenizer=='gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    final_input_df['final_input_prompts'] = build_final_input(final_input_df, params, args.engine, tokenizer)
    # labels are answer indexes in QA4RE
    final_input_df['verbalized_labels'] = final_input_df['all_choices']
    logit_biases, all_first_token_of_each_verb_label = build_logit_biases_df(final_input_df, params, tokenizer)
    final_input_df['logit_biases'] = logit_biases
    final_input_df['first_token_of_each_verb_label'] = all_first_token_of_each_verb_label
    final_input_df['cost'] = estimate_cost_df(final_input_df, args.engine, tokenizer)

    return final_input_df


def prompt_preparation_shuffle_answers(args, params, subset_train_df=None, test_df=None, tokenizer='gpt2'):
    """
    Prepare the prompt for the API query with shuffled answer choices.
    Uses the same random seed as args.random_seed for reproducibility.
    """
    # 3. fill the template for final task.
    test_df = deepcopy(test_df)
    test_df = test_df[:args.run_subset_num]
    # reindex
    test_df = test_df.reset_index(drop=True)

    # add constrains to both dfs
    test_df = add_type_constraints(test_df)
    if subset_train_df is not None:
        subset_train_df = add_type_constraints(subset_train_df)

    # Load possible labels
    import json
    with open("../../data/BIORED/configs/possible_labels.json", 'r') as f:
        possible_labels_biored = json.load(f)
    
    # Use the shuffle function with the random seed from args
    final_input_df = fill_prompt_format_re_df_biored_shuffle_answer(
        args, test_df, params, possible_labels_biored, 'masked_sents', random_seed=args.random_seed
    )
    
    if args.setting == 'few_shot':
        subset_train_df = fill_prompt_format_re_df(args, subset_train_df, params, 'masked_sents')
        # 4. retrieval in-context examples for retrieval few-shot.
        # for each example, get demonstrations with retrieval / random.
        if args.type_constrained and args.sampling_strategy == 'random':  # TODO test
            final_input_df = get_type_constrained_random_demos(subset_train_df, final_input_df, args.in_context_size, args.random_seed)
        else:
            final_input_df = get_demonstrations(subset_train_df, final_input_df, args.sampling_strategy, args.random_seed)
    elif args.setting == 'zero_shot':  # directly input for zero-shot!
        assert subset_train_df is None
        final_input_df['data_prompts'] = final_input_df['empty_prompts']
    else:
        raise ValueError('setting should be either few_shot or zero_shot')

    if tokenizer=='gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    final_input_df['final_input_prompts'] = build_final_input(final_input_df, params, args.engine, tokenizer)
    # labels are answer indexes in QA4RE
    final_input_df['verbalized_labels'] = final_input_df['all_choices']
    logit_biases, all_first_token_of_each_verb_label = build_logit_biases_df(final_input_df, params, tokenizer)
    final_input_df['logit_biases'] = logit_biases
    final_input_df['first_token_of_each_verb_label'] = all_first_token_of_each_verb_label
    final_input_df['cost'] = estimate_cost_df(final_input_df, args.engine, tokenizer)

    return final_input_df
def prompt_preparation_with_relation_definitions(args, params, subset_train_df=None, test_df=None, tokenizer='gpt2'):
    """
    Prepare the prompt for the API query with relation definitions included.
    Only includes definitions for relations that are possible based on entity type combinations.
    """
    # 3. fill the template for final task.
    test_df = deepcopy(test_df)
    test_df = test_df[:args.run_subset_num]
    # reindex
    test_df = test_df.reset_index(drop=True)

    # add constrains to both dfs
    test_df = add_type_constraints(test_df)
    if subset_train_df is not None:
        subset_train_df = add_type_constraints(subset_train_df)

    # Load possible labels and relation definitions
    import json
    with open("/home/huy/12T/QA4RE/data/BIORED/configs/possible_labels.json", 'r') as f:
        possible_labels_biored = json.load(f)
    
    with open("/home/huy/12T/QA4RE/data/BIORED/configs/relation_definitions.json", 'r') as f:
        relation_definitions = json.load(f)
    
    # Use the new function with relation definitions
    final_input_df = fill_prompt_format_re_df_biored_with_definitions(
        args, test_df, params, possible_labels_biored, relation_definitions, 'masked_sents'
    )
    
    if args.setting == 'few_shot':
        subset_train_df = fill_prompt_format_re_df(args, subset_train_df, params, 'masked_sents')
        # 4. retrieval in-context examples for retrieval few-shot.
        # for each example, get demonstrations with retrieval / random.
        if args.type_constrained and args.sampling_strategy == 'random':  # TODO test
            final_input_df = get_type_constrained_random_demos(subset_train_df, final_input_df, args.in_context_size, args.random_seed)
        else:
            final_input_df = get_demonstrations(subset_train_df, final_input_df, args.sampling_strategy, args.random_seed)
    elif args.setting == 'zero_shot':  # directly input for zero-shot!
        assert subset_train_df is None
        final_input_df['data_prompts'] = final_input_df['empty_prompts']
    else:
        raise ValueError('setting should be either few_shot or zero_shot')

    if tokenizer=='gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    final_input_df['final_input_prompts'] = build_final_input(final_input_df, params, args.engine, tokenizer)
    # labels are answer indexes in QA4RE
    final_input_df['verbalized_labels'] = final_input_df['all_choices']
    logit_biases, all_first_token_of_each_verb_label = build_logit_biases_df(final_input_df, params, tokenizer)
    final_input_df['logit_biases'] = logit_biases
    final_input_df['first_token_of_each_verb_label'] = all_first_token_of_each_verb_label
    final_input_df['cost'] = estimate_cost_df(final_input_df, args.engine, tokenizer)

    return final_input_df

