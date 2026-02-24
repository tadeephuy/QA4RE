#!/usr/bin/env python3
"""
Unit tests for convert_biored_to_csv.py

Tests cover:
- Entity extraction from annotations
- Full conversion pipeline (JSON -> CSV)
- Relation type and entity type extraction
- Sentence masking with ENT1/ENT2
- Unique ID generation
- CSV format validation
- Edge cases (empty data, missing entities)

Run with:
    python -m pytest test_convert_biored_to_csv.py -v
    # or
    python test_convert_biored_to_csv.py
"""

import json
import pytest
import pandas as pd
from pathlib import Path
import tempfile
import sys

# Import the module to test
from convert_biored_to_csv import (
    extract_entity_text_and_type,
    convert_biored_to_csv
)


class TestExtractEntityTextAndType:
    """Test the extract_entity_text_and_type function."""
    
    def test_extract_valid_entity(self):
        """Test extracting a valid entity from annotations."""
        annotations = [
            {
                'id': '0',
                'infons': {
                    'identifier': 'D003409',
                    'type': 'DiseaseOrPhenotypicFeature'
                },
                'text': 'Congenital hypothyroidism',
                'locations': [{'offset': 0, 'length': 25}]
            },
            {
                'id': '1',
                'infons': {
                    'identifier': '6528',
                    'type': 'GeneOrGeneProduct'
                },
                'text': 'sodium/iodide symporter',
                'locations': [{'offset': 55, 'length': 23}]
            }
        ]
        
        text, entity_type, offset, length = extract_entity_text_and_type(annotations, 'D003409')
        assert text == 'Congenital hypothyroidism'
        assert entity_type == 'DiseaseOrPhenotypicFeature'
        assert offset == 0
        assert length == 25
        
        text, entity_type, offset, length = extract_entity_text_and_type(annotations, '6528')
        assert text == 'sodium/iodide symporter'
        assert entity_type == 'GeneOrGeneProduct'
        assert offset == 55
        assert length == 23
    
    def test_extract_nonexistent_entity(self):
        """Test extracting an entity that doesn't exist."""
        annotations = [
            {
                'id': '0',
                'infons': {
                    'identifier': 'D003409',
                    'type': 'DiseaseOrPhenotypicFeature'
                },
                'text': 'Congenital hypothyroidism',
                'locations': [{'offset': 0, 'length': 25}]
            }
        ]
        
        text, entity_type, offset, length = extract_entity_text_and_type(annotations, 'NONEXISTENT')
        assert text is None
        assert entity_type is None
        assert offset is None
        assert length is None
    
    def test_extract_from_empty_annotations(self):
        """Test extracting from empty annotations list."""
        text, entity_type, offset, length = extract_entity_text_and_type([], 'D003409')
        assert text is None
        assert entity_type is None
        assert offset is None
        assert length is None
    
    def test_extract_composite_identifier(self):
        """Test extracting entity with composite identifier (e.g., protein complex)."""
        annotations = [
            {
                'id': '0',
                'infons': {
                    'identifier': '22083,54624,76246',  # Composite ID
                    'type': 'GeneOrGeneProduct'
                },
                'text': 'PAF1 complex',
                'locations': [{'offset': 100, 'length': 12}]
            },
            {
                'id': '1',
                'infons': {
                    'identifier': 'D003409',
                    'type': 'DiseaseOrPhenotypicFeature'
                },
                'text': 'hypothyroidism',
                'locations': [{'offset': 200, 'length': 14}]
            }
        ]
        
        # Should find entity by any component of composite ID
        text, entity_type, offset, length = extract_entity_text_and_type(annotations, '54624')
        assert text == 'PAF1 complex'
        assert entity_type == 'GeneOrGeneProduct'
        assert offset == 100
        assert length == 12
        
        # Should also work with first component
        text, entity_type, offset, length = extract_entity_text_and_type(annotations, '22083')
        assert text == 'PAF1 complex'
        assert entity_type == 'GeneOrGeneProduct'
        
        # Should also work with last component
        text, entity_type, offset, length = extract_entity_text_and_type(annotations, '76246')
        assert text == 'PAF1 complex'
        assert entity_type == 'GeneOrGeneProduct'
        
        # Regular single ID should still work
        text, entity_type, offset, length = extract_entity_text_and_type(annotations, 'D003409')
        assert text == 'hypothyroidism'
        assert entity_type == 'DiseaseOrPhenotypicFeature'


class TestConvertBioredToCsv:
    """Test the main conversion function."""
    
    @pytest.fixture
    def sample_biored_data(self):
        """Create sample BIORED data for testing."""
        return {
            "source": "PubTator",
            "date": "2021-11-30",
            "key": "BioC.key",
            "documents": [
                {
                    "id": "12345678",
                    "passages": [
                        {
                            "offset": 0,
                            "text": "Vitamin D deficiency is associated with diabetes.",
                            "annotations": [
                                {
                                    "id": "0",
                                    "infons": {
                                        "identifier": "D014807",
                                        "type": "ChemicalEntity"
                                    },
                                    "text": "Vitamin D",
                                    "locations": [{"offset": 0, "length": 9}]
                                },
                                {
                                    "id": "1",
                                    "infons": {
                                        "identifier": "D003920",
                                        "type": "DiseaseOrPhenotypicFeature"
                                    },
                                    "text": "diabetes",
                                    "locations": [{"offset": 40, "length": 8}]
                                }
                            ]
                        }
                    ],
                    "relations": [
                        {
                            "id": "R0",
                            "infons": {
                                "entity1": "D014807",
                                "entity2": "D003920",
                                "type": "Association",
                                "novel": "No"
                            }
                        }
                    ]
                },
                {
                    "id": "87654321",
                    "passages": [
                        {
                            "offset": 0,
                            "text": "Aspirin reduces inflammation in patients.",
                            "annotations": [
                                {
                                    "id": "0",
                                    "infons": {
                                        "identifier": "D001241",
                                        "type": "ChemicalEntity"
                                    },
                                    "text": "Aspirin",
                                    "locations": [{"offset": 0, "length": 7}]
                                },
                                {
                                    "id": "1",
                                    "infons": {
                                        "identifier": "D007249",
                                        "type": "DiseaseOrPhenotypicFeature"
                                    },
                                    "text": "inflammation",
                                    "locations": [{"offset": 16, "length": 12}]
                                }
                            ]
                        }
                    ],
                    "relations": [
                        {
                            "id": "R0",
                            "infons": {
                                "entity1": "D001241",
                                "entity2": "D007249",
                                "type": "Negative_Correlation",
                                "novel": "Novel"
                            }
                        }
                    ]
                }
            ]
        }
    
    @pytest.fixture
    def sample_json_file(self, sample_biored_data, tmp_path):
        """Create a temporary JSON file with sample data."""
        json_file = tmp_path / "test_biored.json"
        with open(json_file, 'w') as f:
            json.dump(sample_biored_data, f)
        return json_file
    
    def test_convert_basic(self, sample_json_file, tmp_path):
        """Test basic conversion functionality."""
        output_csv = tmp_path / "output.csv"
        
        df = convert_biored_to_csv(sample_json_file, output_csv)
        
        # Check that output file exists
        assert output_csv.exists()
        
        # Check DataFrame structure
        assert len(df) == 2  # Two relations in sample data
        assert list(df.columns) == [
            'Unnamed: 0', 'id', 'label', 'ent1_type', 'ent2_type',
            'ent1', 'ent2', 'sents', 'masked_sents', 'verbalized_label'
        ]
    
    def test_convert_relation_types(self, sample_json_file, tmp_path):
        """Test that relation types are correctly extracted."""
        output_csv = tmp_path / "output.csv"
        df = convert_biored_to_csv(sample_json_file, output_csv)
        
        assert df.iloc[0]['label'] == 'Association'
        assert df.iloc[1]['label'] == 'Negative_Correlation'
    
    def test_convert_entity_types(self, sample_json_file, tmp_path):
        """Test that entity types are correctly extracted."""
        output_csv = tmp_path / "output.csv"
        df = convert_biored_to_csv(sample_json_file, output_csv)
        
        # First relation: Vitamin D (Chemical) -> diabetes (Disease)
        assert df.iloc[0]['ent1_type'] == 'ChemicalEntity'
        assert df.iloc[0]['ent2_type'] == 'DiseaseOrPhenotypicFeature'
        
        # Second relation: Aspirin (Chemical) -> inflammation (Disease)
        assert df.iloc[1]['ent1_type'] == 'ChemicalEntity'
        assert df.iloc[1]['ent2_type'] == 'DiseaseOrPhenotypicFeature'
    
    def test_convert_entity_texts(self, sample_json_file, tmp_path):
        """Test that entity texts are correctly extracted."""
        output_csv = tmp_path / "output.csv"
        df = convert_biored_to_csv(sample_json_file, output_csv)
        
        assert df.iloc[0]['ent1'] == 'Vitamin D'
        assert df.iloc[0]['ent2'] == 'diabetes'
        assert df.iloc[1]['ent1'] == 'Aspirin'
        assert df.iloc[1]['ent2'] == 'inflammation'
    
    def test_convert_masked_sentences(self, sample_json_file, tmp_path):
        """Test that sentences are correctly masked."""
        output_csv = tmp_path / "output.csv"
        df = convert_biored_to_csv(sample_json_file, output_csv)
        
        # Check that entities are replaced with ENT1 and ENT2
        masked_0 = df.iloc[0]['masked_sents']
        assert 'ENT1' in masked_0
        assert 'ENT2' in masked_0
        assert 'Vitamin D' not in masked_0
        assert 'diabetes' not in masked_0
        
        masked_1 = df.iloc[1]['masked_sents']
        assert 'ENT1' in masked_1
        assert 'ENT2' in masked_1
        assert 'Aspirin' not in masked_1
        assert 'inflammation' not in masked_1
    
    def test_convert_multiple_mentions(self, tmp_path):
        """Test that only the specific mention is masked when entity appears multiple times."""
        data = {
            "source": "PubTator",
            "documents": [
                {
                    "id": "test_multi",
                    "passages": [
                        {
                            "offset": 0,
                            "text": "Aspirin treats pain. Aspirin is a drug.",
                            "annotations": [
                                {
                                    "id": "0",
                                    "infons": {
                                        "identifier": "D001241",
                                        "type": "ChemicalEntity"
                                    },
                                    "text": "Aspirin",
                                    "locations": [{"offset": 0, "length": 7}]  # First mention
                                },
                                {
                                    "id": "1",
                                    "infons": {
                                        "identifier": "D010146",
                                        "type": "DiseaseOrPhenotypicFeature"
                                    },
                                    "text": "pain",
                                    "locations": [{"offset": 15, "length": 4}]
                                }
                            ]
                        }
                    ],
                    "relations": [
                        {
                            "id": "R0",
                            "infons": {
                                "entity1": "D001241",  # First Aspirin at offset 0
                                "entity2": "D010146",
                                "type": "Negative_Correlation"
                            }
                        }
                    ]
                }
            ]
        }
        
        json_file = tmp_path / "multi_mention.json"
        with open(json_file, 'w') as f:
            json.dump(data, f)
        
        output_csv = tmp_path / "output.csv"
        df = convert_biored_to_csv(json_file, output_csv)
        
        masked = df.iloc[0]['masked_sents']
        # Only first "Aspirin" should be replaced
        # Result should be: "ENT1 treats ENT2. Aspirin is a drug."
        assert 'ENT1 treats ENT2' in masked
        # Second "Aspirin" should remain (this validates offset-based masking)
        assert masked.count('Aspirin') == 1  # Second mention still there
    
    def test_convert_unique_ids(self, sample_json_file, tmp_path):
        """Test that unique IDs are generated correctly."""
        output_csv = tmp_path / "output.csv"
        df = convert_biored_to_csv(sample_json_file, output_csv)
        
        # Check ID format: biored_{doc_id}_{rel_id}
        assert df.iloc[0]['id'] == 'biored_12345678_R0'
        assert df.iloc[1]['id'] == 'biored_87654321_R0'
        
        # All IDs should be unique
        assert len(df['id'].unique()) == len(df)
    
    def test_convert_max_samples(self, sample_json_file, tmp_path):
        """Test that max_samples parameter limits output."""
        output_csv = tmp_path / "output.csv"
        df = convert_biored_to_csv(sample_json_file, output_csv, max_samples=1)
        
        assert len(df) == 1
    
    def test_convert_no_relations(self, tmp_path):
        """Test conversion when document has no relations."""
        data = {
            "source": "PubTator",
            "documents": [
                {
                    "id": "99999999",
                    "passages": [
                        {
                            "offset": 0,
                            "text": "Some text without relations.",
                            "annotations": []
                        }
                    ],
                    "relations": []  # No relations
                }
            ]
        }
        
        json_file = tmp_path / "no_relations.json"
        with open(json_file, 'w') as f:
            json.dump(data, f)
        
        output_csv = tmp_path / "output.csv"
        df = convert_biored_to_csv(json_file, output_csv)
        
        # Should return empty DataFrame
        assert len(df) == 0
    
    def test_csv_format(self, sample_json_file, tmp_path):
        """Test that output CSV has correct format (tab-separated)."""
        output_csv = tmp_path / "output.csv"
        convert_biored_to_csv(sample_json_file, output_csv)
        
        # Read the file and check it's tab-separated
        with open(output_csv, 'r') as f:
            header = f.readline()
            assert '\t' in header
            
            # Check column count
            columns = header.strip().split('\t')
            assert len(columns) == 10
    
    def test_original_sentences_preserved(self, sample_json_file, tmp_path):
        """Test that original sentences are preserved."""
        output_csv = tmp_path / "output.csv"
        df = convert_biored_to_csv(sample_json_file, output_csv)
        
        # Original sentences should be intact
        assert 'Vitamin D deficiency is associated with diabetes.' in df.iloc[0]['sents']
        assert 'Aspirin reduces inflammation in patients.' in df.iloc[1]['sents']
    
    def test_multi_passage_handling(self, tmp_path):
        """Test correct handling of multiple passages (title + abstract)."""
        data = {
            "source": "PubTator",
            "documents": [
                {
                    "id": "14510914",
                    "passages": [
                        {
                            "offset": 0,
                            "text": "Congenital hypothyroidism due to deletion.",
                            "annotations": [
                                {
                                    "id": "0",
                                    "infons": {
                                        "identifier": "D003409",
                                        "type": "DiseaseOrPhenotypicFeature"
                                    },
                                    "text": "Congenital hypothyroidism",
                                    "locations": [{"offset": 0, "length": 25}]
                                }
                            ]
                        },
                        {
                            "offset": 44,  # Length of passage 0 + 1 space
                            "text": "The patient shows symptoms of hypothyroidism.",
                            "annotations": [
                                {
                                    "id": "1",
                                    "infons": {
                                        "identifier": "D007037",
                                        "type": "DiseaseOrPhenotypicFeature"
                                    },
                                    "text": "hypothyroidism",
                                    "locations": [{"offset": 78, "length": 14}]  # Absolute offset
                                }
                            ]
                        }
                    ],
                    "relations": [
                        {
                            "id": "R0",
                            "infons": {
                                "entity1": "D003409",
                                "entity2": "D007037",
                                "type": "Association"
                            }
                        }
                    ]
                }
            ]
        }
        
        json_file = tmp_path / "multi_passage.json"
        with open(json_file, 'w') as f:
            json.dump(data, f)
        
        output_csv = tmp_path / "output.csv"
        df = convert_biored_to_csv(json_file, output_csv)
        
        # Should have combined both passages
        full_text = df.iloc[0]['sents']
        assert 'Congenital hypothyroidism due to deletion.' in full_text
        assert 'The patient shows symptoms of hypothyroidism.' in full_text
        
        # Check masking is correct with proper offsets
        masked = df.iloc[0]['masked_sents']
        # First entity at offset 0 should be masked
        assert masked.startswith('ENT1')
        # Second entity at offset 78 should be masked
        assert 'ENT2' in masked


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_document_list(self, tmp_path):
        """Test with empty document list."""
        data = {
            "source": "PubTator",
            "documents": []
        }
        
        json_file = tmp_path / "empty.json"
        with open(json_file, 'w') as f:
            json.dump(data, f)
        
        output_csv = tmp_path / "output.csv"
        df = convert_biored_to_csv(json_file, output_csv)
        
        assert len(df) == 0
    
    def test_missing_entity_annotations(self, tmp_path):
        """Test when relation references entities not in annotations."""
        data = {
            "source": "PubTator",
            "documents": [
                {
                    "id": "12345",
                    "passages": [
                        {
                            "offset": 0,
                            "text": "Some text.",
                            "annotations": []  # Empty annotations
                        }
                    ],
                    "relations": [
                        {
                            "id": "R0",
                            "infons": {
                                "entity1": "MISSING1",
                                "entity2": "MISSING2",
                                "type": "Association"
                            }
                        }
                    ]
                }
            ]
        }
        
        json_file = tmp_path / "missing.json"
        with open(json_file, 'w') as f:
            json.dump(data, f)
        
        output_csv = tmp_path / "output.csv"
        df = convert_biored_to_csv(json_file, output_csv)
        
        # Should skip relations with missing entities
        assert len(df) == 0


def run_tests():
    """Run tests without pytest."""
    import sys
    
    # Create test instances
    test_extract = TestExtractEntityTextAndType()
    test_convert = TestConvertBioredToCsv()
    test_edge = TestEdgeCases()
    
    print("Running unit tests for convert_biored_to_csv.py\n")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    # Test extract_entity_text_and_type
    tests = [
        ("Extract valid entity", test_extract.test_extract_valid_entity),
        ("Extract nonexistent entity", test_extract.test_extract_nonexistent_entity),
        ("Extract from empty annotations", test_extract.test_extract_from_empty_annotations),
    ]
    
    for name, test_func in tests:
        try:
            test_func()
            print(f"✓ {name}")
            passed += 1
        except Exception as e:
            print(f"✗ {name}: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed > 0:
        sys.exit(1)
    else:
        print("\nAll tests passed!")
        sys.exit(0)


if __name__ == '__main__':
    # Check if pytest is available
    try:
        import pytest
        # Run with pytest
        sys.exit(pytest.main([__file__, '-v']))
    except ImportError:
        print("pytest not found, running basic tests...\n")
        run_tests()
