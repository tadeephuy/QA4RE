#!/usr/bin/env python3
"""
Convert BIORED JSON format to TACRED-style CSV format.

Usage:
    python convert_biored_to_csv.py Dev.BioC.JSON dev_subset.csv
    python convert_biored_to_csv.py Train.BioC.JSON train_subset.csv
    python convert_biored_to_csv.py Test.BioC.JSON test_subset.csv
"""

import json
import pandas as pd
import argparse
from pathlib import Path


def extract_entity_text_and_type(annotations, entity_id):
    """Extract entity text, type, and location from annotations by identifier.
    
    Handles both exact matches and composite identifiers (e.g., '22083,54624,76246').
    """
    for ann in annotations:
        ann_id = ann['infons'].get('identifier', '')
        # Check for exact match or if entity_id is part of a composite identifier
        if ann_id == entity_id or (entity_id in ann_id.split(',')):
            # Return text, type, and first location (offset, length)
            if ann['locations']:
                loc = ann['locations'][0]
                return ann['text'], ann['infons']['type'], loc['offset'], loc['length']
            return ann['text'], ann['infons']['type'], None, None
    return None, None, None, None


def convert_biored_to_csv(input_json_path, output_csv_path, max_samples=None):
    """
    Convert BIORED JSON to TACRED-style CSV format.
    
    Args:
        input_json_path: Path to BIORED .BioC.JSON file
        output_csv_path: Path to output CSV file
        max_samples: Maximum number of samples to convert (None for all)
    """
    print(f"Loading {input_json_path}...")
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    
    records = []
    doc_count = 0
    relation_count = 0
    
    for doc in data['documents']:
        doc_id = doc['id']
        doc_count += 1
        
        # Collect all annotations and text from all passages
        # In BioC format, passages are already positioned correctly (title + abstract)
        # and annotations use absolute offsets
        all_annotations = []
        passage_texts = []
        
        for passage in doc.get('passages', []):
            passage_texts.append(passage.get('text', ''))
            
            # Collect annotations - offsets are already absolute in BioC format
            for ann in passage.get('annotations', []):
                all_annotations.append(ann)
        
        # Concatenate passages with space (typically title + abstract)
        full_text = ' '.join(passage_texts)
        
        # Process relations
        relations = doc.get('relations', [])
        if not relations:
            # If no relations, we might want to skip or create negative examples
            continue
        
        for rel in relations:
            relation_count += 1
            infons = rel['infons']
            entity1_id = infons.get('entity1', '')
            entity2_id = infons.get('entity2', '')
            relation_type = infons.get('type', 'no_relation')
            
            # Find entity mentions in annotations
            entity1_text, entity1_type, entity1_offset, entity1_length = extract_entity_text_and_type(all_annotations, entity1_id)
            entity2_text, entity2_type, entity2_offset, entity2_length = extract_entity_text_and_type(all_annotations, entity2_id)
            
            # Skip if we can't find both entities
            if entity1_text is None or entity2_text is None:
                continue
            
            # Create masked sentence using offset information
            # Skip if either entity lacks offset info (entity not in text)
            if entity1_offset is None or entity2_offset is None:
                continue
            
            # Sort entities by offset (replace from end to start to maintain offsets)
            entities_to_mask = [
                (entity1_offset, entity1_length, 'ENT1'),
                (entity2_offset, entity2_length, 'ENT2')
            ]
            entities_to_mask.sort(key=lambda x: x[0], reverse=True)
            
            masked_text = full_text
            for offset, length, placeholder in entities_to_mask:
                masked_text = masked_text[:offset] + placeholder + masked_text[offset + length:]
            
            # Create a unique ID
            unique_id = f"biored_{doc_id}_{rel['id']}"
            
            record = {
                'Unnamed: 0': len(records),
                'id': unique_id,
                'label': relation_type,
                'ent1_type': entity1_type or 'Unknown',
                'ent2_type': entity2_type or 'Unknown',
                'ent1': entity1_text,
                'ent2': entity2_text,
                'sents': full_text,
                'masked_sents': masked_text,
                'verbalized_label': relation_type,  # Could be mapped to more readable labels
            }
            
            records.append(record)
            
            if max_samples and len(records) >= max_samples:
                break
        
        if max_samples and len(records) >= max_samples:
            break
        
        if doc_count % 100 == 0:
            print(f"Processed {doc_count} documents, {len(records)} relations...")
    
    print(f"\nTotal documents processed: {doc_count}")
    print(f"Total relations extracted: {len(records)}")
    
    # Create DataFrame and save
    df = pd.DataFrame(records)
    
    # Ensure column order matches TACRED format
    column_order = [
        'Unnamed: 0', 'id', 'label', 'ent1_type', 'ent2_type',
        'ent1', 'ent2', 'sents', 'masked_sents', 'verbalized_label'
    ]
    
    # Handle empty dataframe case
    if len(df) == 0:
        # Create empty dataframe with correct columns
        df = pd.DataFrame(columns=column_order)
    else:
        df = df[column_order]
    
    print(f"\nSaving to {output_csv_path}...")
    df.to_csv(output_csv_path, sep='\t', index=False)
    
    print(f"Done! Saved {len(df)} records.")
    
    if len(df) > 0:
        print(f"\nRelation type distribution:")
        print(df['label'].value_counts())
        print(f"\nEntity type pairs (top 10):")
        entity_pairs = df['ent1_type'] + ' -> ' + df['ent2_type']
        print(entity_pairs.value_counts().head(10))
    else:
        print("\nNo relations found in the dataset.")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Convert BIORED JSON to TACRED-style CSV format'
    )
    parser.add_argument('input_json', type=str, help='Input BIORED .BioC.JSON file')
    parser.add_argument('output_csv', type=str, help='Output CSV file')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to convert (default: all)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_json)
    output_path = Path(args.output_csv)
    
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist!")
        return 1
    
    convert_biored_to_csv(input_path, output_path, args.max_samples)
    return 0


if __name__ == '__main__':
    exit(main())
