import pickle
import pandas as pd
import os
import openai
import numpy as np
import ipdb
import re
import tqdm

from utils.data_utils import *
from scipy import special
import json
from sklearn.metrics import precision_recall_fscore_support, f1_score, precision_score, recall_score

def filter_by_dict(df, filter_v):
    for k, v in filter_v.items():
        df = df[df[k] == v]
    return df

def evaluate_re(df, label_verbalizer, pos_label_list, prediction_name='predictions', nota_eval=True, nota_eval_average='micro'):
    # Verbalize both predictions and labels for consistent comparison
    # Use .get() fallback to keep original value if not in verbalizer (for identity mappings)
    y_pred_verbalized = df[prediction_name].apply(lambda x: label_verbalizer.get(x, x))
    y_true_verbalized = df['verbalized_label'].apply(lambda x: label_verbalizer.get(x, x))
    
    if nota_eval:
        precision, recall, f1, support = precision_recall_fscore_support(y_pred=y_pred_verbalized, y_true=y_true_verbalized, labels=list(label_verbalizer.values()), average=nota_eval_average)
    else:
        precision, recall, f1, support = precision_recall_fscore_support(y_pred=y_pred_verbalized, y_true=y_true_verbalized,
                                                 labels=[label_verbalizer[l] for l in pos_label_list],
                                                # average='weighted')
                                                 average='micro')
    return {'f1': f1, 'precision': precision, 'recall': recall}


def batch_re_eval_print(df, prediction_name, label_verbalizer, pos_label_list, return_results=False):
    """Evaluate the RE task with different metrics on the given dataframe."""

    closed_micro_metric_dict = evaluate_re(df, label_verbalizer, pos_label_list, prediction_name, nota_eval=False)
    print('Closed Set Performance (Micro):')
    print(closed_micro_metric_dict)

    print('-'*100)

    open_macro_metric_dict = evaluate_re(df, label_verbalizer, pos_label_list, prediction_name, nota_eval=True, nota_eval_average='macro') # todo
    print('Open Set Performance (Macro):')
    print(open_macro_metric_dict)

    print('-'*100)

    open_micro_metric_dict = evaluate_re(df, label_verbalizer, pos_label_list, prediction_name, nota_eval=True) # todo
    print('Open Set Performance (Micro):')
    print(open_micro_metric_dict)

    print('-'*100)

    verbalized_pos_lables = [label_verbalizer[l] for l in pos_label_list]
    df['binary_preds'] = df[prediction_name].apply(lambda x: 1 if x in verbalized_pos_lables else 0)
    binary_preds = df['binary_preds'].to_list()
    binary_labels = df['verbalized_label'].apply(lambda x: 1 if x in verbalized_pos_lables else 0)

    f1 = f1_score(binary_labels, binary_preds, labels=range(2), average="micro")
    print("binary P vs N micro F1 score:", f1)

    f1 = f1_score(binary_labels, binary_preds, labels=range(2), average="macro")
    print("binary P vs N macro F1 score:", f1)

    if return_results:
        return closed_micro_metric_dict, open_macro_metric_dict, open_micro_metric_dict


def evaluate_re_by_entity_type_combinations(df, label_verbalizer, prediction_name='predictions', 
                                             entity_type_abbrev=None, return_dataframe=False):
    """
    Evaluate relation extraction performance broken down by entity type combinations and relation types.
    Returns a table similar to Table S3 showing F1-scores for each relation type across entity type pairs.
    
    Args:
        df: DataFrame with predictions, labels, and entity type information
        label_verbalizer: Dictionary mapping labels to their verbalized forms
        prediction_name: Column name for predictions
        entity_type_abbrev: Dictionary for abbreviating entity types (e.g., {'GeneOrGeneProduct': 'G'})
        return_dataframe: If True, return results as DataFrame; if False, print formatted table
    
    Returns:
        DataFrame with F1-scores if return_dataframe=True, otherwise None
    """
    # Verbalize predictions and labels
    df = df.copy()
    df['pred_verbalized'] = df[prediction_name].apply(lambda x: label_verbalizer.get(x, x))
    df['true_verbalized'] = df['verbalized_label'].apply(lambda x: label_verbalizer.get(x, x))
    
    # Create entity type pair column
    df['ent_type_pair'] = df['ent1_type'] + ' -> ' + df['ent2_type']
    
    # Create abbreviated version if mapping provided
    if entity_type_abbrev:
        df['ent_type_pair_abbrev'] = df['ent1_type'].map(entity_type_abbrev) + ',' + df['ent2_type'].map(entity_type_abbrev)
        type_pair_col = 'ent_type_pair_abbrev'
    else:
        type_pair_col = 'ent_type_pair'
    
    # Get all unique relation types and entity type pairs
    all_relations = sorted(df['true_verbalized'].unique())
    all_type_pairs = sorted(df[type_pair_col].unique())
    
    # Initialize results dictionary
    results = {rel: {} for rel in all_relations}
    results['Overall'] = {}
    
    # Calculate F1 for each relation type and entity type pair
    for type_pair in all_type_pairs:
        subset = df[df[type_pair_col] == type_pair]
        
        # Overall F1 for this entity type pair
        if len(subset) > 0:
            y_true = subset['true_verbalized']
            y_pred = subset['pred_verbalized']
            
            # Micro F1 across all relations for this type pair
            f1 = f1_score(y_true, y_pred, labels=all_relations, average='micro', zero_division=0)
            results['Overall'][type_pair] = f1 * 100  # Convert to percentage
        
        # F1 for each specific relation type
        for rel in all_relations:
            # Count of true instances for this relation in this entity type pair
            true_count = (subset['true_verbalized'] == rel).sum()
            
            if true_count == 0:
                results[rel][type_pair] = None  # No samples for this combination
            else:
                # Binary classification for this relation across ALL instances in this entity type pair
                y_true_binary = (subset['true_verbalized'] == rel).astype(int)
                y_pred_binary = (subset['pred_verbalized'] == rel).astype(int)
                
                f1 = f1_score(y_true_binary, y_pred_binary, average='binary', zero_division=0)
                results[rel][type_pair] = f1 * 100  # Convert to percentage
    
    # Create DataFrame
    results_df = pd.DataFrame(results).T
    results_df = results_df[sorted(results_df.columns)]  # Sort columns
    
    # Move 'Overall' row to the bottom
    if 'Overall' in results_df.index:
        overall_row = results_df.loc['Overall']
        results_df = results_df.drop('Overall')
        results_df = pd.concat([results_df, overall_row.to_frame().T])
    
    if return_dataframe:
        return results_df
    else:
        # Print formatted table
        print("\nPerformance by Entity Type Combinations (F1-scores)")
        print("="*100)
        print(results_df.round(1).to_string())
        print("="*100)
        print("\nNote: '-' or 'NaN' indicates no samples for that combination")
        return results_df


def evaluate_re_by_relation_type(df, label_verbalizer, prediction_name='predictions', return_dataframe=False):
    """
    Evaluate relation extraction performance for each relation type (overall, not broken down by entity types).
    
    Args:
        df: DataFrame with predictions and labels
        label_verbalizer: Dictionary mapping labels to their verbalized forms
        prediction_name: Column name for predictions
        return_dataframe: If True, return results as DataFrame; if False, print table
    
    Returns:
        DataFrame with metrics if return_dataframe=True, otherwise None
    """
    df = df.copy()
    df['pred_verbalized'] = df[prediction_name].apply(lambda x: label_verbalizer.get(x, x))
    df['true_verbalized'] = df['verbalized_label'].apply(lambda x: label_verbalizer.get(x, x))
    
    # Get all relations from both predictions and ground truth
    all_relations = sorted(set(df['true_verbalized'].unique()) | set(df['pred_verbalized'].unique()))
    
    # Use entire dataset for binary classification (not just true positives)
    y_true = df['true_verbalized']
    y_pred = df['pred_verbalized']
    
    results = []
    for rel in all_relations:
        # Count of true instances for this relation
        true_count = (y_true == rel).sum()
        
        if true_count == 0:
            continue
            
        # Binary classification for this relation across ALL instances
        y_true_binary = (y_true == rel).astype(int)
        y_pred_binary = (y_pred == rel).astype(int)
        
        precision = precision_score(y_true_binary, y_pred_binary, average='binary', zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, average='binary', zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_binary, average='binary', zero_division=0)
        
        results.append({
            'Relation Type': rel,
            'Count': true_count,
            'Precision': precision * 100,
            'Recall': recall * 100,
            'F1': f1 * 100
        })
    
    results_df = pd.DataFrame(results)
    
    if return_dataframe:
        return results_df
    else:
        print("\nPerformance by Relation Type")
        print("="*100)
        print(results_df.round(1).to_string(index=False))
        print("="*100)
        return results_df
