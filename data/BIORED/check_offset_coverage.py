#!/usr/bin/env python3
"""
Check if all entities in BIORED have location/offset information.
This helps determine if the fallback masking code is necessary.
"""

import json
import sys

def check_offset_coverage(json_file):
    """Check what percentage of entities have offset information."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    total_relations = 0
    relations_with_both_offsets = 0
    relations_with_missing_offsets = 0
    
    missing_examples = []
    
    for doc in data['documents']:
        doc_id = doc['id']
        
        # Collect all annotations
        all_annotations = []
        for passage in doc.get('passages', []):
            all_annotations.extend(passage.get('annotations', []))
        
        # Check each relation
        for rel in doc.get('relations', []):
            total_relations += 1
            infons = rel['infons']
            entity1_id = infons.get('entity1', '')
            entity2_id = infons.get('entity2', '')
            
            # Find entities
            entity1_has_offset = False
            entity2_has_offset = False
            entity1_text = None
            entity2_text = None
            
            for ann in all_annotations:
                if ann['infons'].get('identifier') == entity1_id:
                    entity1_text = ann['text']
                    if ann.get('locations') and len(ann['locations']) > 0:
                        entity1_has_offset = True
                
                if ann['infons'].get('identifier') == entity2_id:
                    entity2_text = ann['text']
                    if ann.get('locations') and len(ann['locations']) > 0:
                        entity2_has_offset = True
            
            if entity1_has_offset and entity2_has_offset:
                relations_with_both_offsets += 1
            else:
                relations_with_missing_offsets += 1
                if len(missing_examples) < 5:  # Keep first 5 examples
                    missing_examples.append({
                        'doc_id': doc_id,
                        'relation_id': rel['id'],
                        'entity1_id': entity1_id,
                        'entity1_text': entity1_text,
                        'entity1_has_offset': entity1_has_offset,
                        'entity2_id': entity2_id,
                        'entity2_text': entity2_text,
                        'entity2_has_offset': entity2_has_offset
                    })
    
    print("=" * 80)
    print("BIORED OFFSET COVERAGE CHECK")
    print("=" * 80)
    print(f"\nTotal relations: {total_relations}")
    print(f"Relations with both entity offsets: {relations_with_both_offsets} ({relations_with_both_offsets/total_relations*100:.1f}%)")
    print(f"Relations with missing offsets: {relations_with_missing_offsets} ({relations_with_missing_offsets/total_relations*100:.1f}%)")
    
    if missing_examples:
        print(f"\n{'=' * 80}")
        print(f"Examples of relations with missing offsets (first {len(missing_examples)}):")
        print("=" * 80)
        for i, ex in enumerate(missing_examples, 1):
            print(f"\n{i}. Document {ex['doc_id']}, Relation {ex['relation_id']}:")
            print(f"   Entity1: {ex['entity1_id']}")
            print(f"     Text: '{ex['entity1_text']}'")
            print(f"     Has offset: {ex['entity1_has_offset']}")
            print(f"   Entity2: {ex['entity2_id']}")
            print(f"     Text: '{ex['entity2_text']}'")
            print(f"     Has offset: {ex['entity2_has_offset']}")
    
    print("\n" + "=" * 80)
    if relations_with_missing_offsets == 0:
        print("✓ CONCLUSION: All entities have offsets - fallback code is NOT needed")
    else:
        print("✗ CONCLUSION: Some entities lack offsets - fallback code IS needed")
    print("=" * 80)
    
    return relations_with_missing_offsets == 0

if __name__ == '__main__':
    files_to_check = ['Dev.BioC.JSON', 'Train.BioC.JSON', 'Test.BioC.JSON']
    
    all_have_offsets = True
    for filename in files_to_check:
        try:
            print(f"\nChecking {filename}...")
            has_all_offsets = check_offset_coverage(filename)
            all_have_offsets = all_have_offsets and has_all_offsets
        except FileNotFoundError:
            print(f"File {filename} not found, skipping...")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    print("\n" + "=" * 80)
    print("OVERALL RESULT:")
    if all_have_offsets:
        print("✓ All checked files have complete offset information")
        print("  → The fallback masking code can be safely removed")
    else:
        print("✗ Some files have missing offset information")
        print("  → The fallback masking code should be kept for safety")
    print("=" * 80)
