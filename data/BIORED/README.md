# BIORED to CSV Conversion

This directory contains tools to convert BIORED dataset from BioC JSON format to TACRED-style CSV format for relation extraction.

## Files
- `convert_biored_to_csv.py` - Conversion script
- `Dev.BioC.JSON` - Development set (original format)
- `Train.BioC.JSON` - Training set (original format)  
- `Test.BioC.JSON` - Test set (original format)

## Usage

### Basic conversion
```bash
# Activate the QA4RE environment
conda activate QA4RE

# Convert development set
python convert_biored_to_csv.py Dev.BioC.JSON dev_subset.csv

# Convert training set
python convert_biored_to_csv.py Train.BioC.JSON train_subset.csv

# Convert test set
python convert_biored_to_csv.py Test.BioC.JSON test_subset.csv
```

### Limited samples (for debugging)
```bash
python convert_biored_to_csv.py Dev.BioC.JSON dev_subset.csv --max-samples 250
```

## Output Format

The converted CSV files follow the TACRED format with tab-separated columns:

| Column | Description |
|--------|-------------|
| `Unnamed: 0` | Row index |
| `id` | Unique identifier (format: `biored_{doc_id}_{relation_id}`) |
| `label` | Relation type (e.g., Association, Positive_Correlation) |
| `ent1_type` | Entity 1 type (e.g., GeneOrGeneProduct, DiseaseOrPhenotypicFeature) |
| `ent2_type` | Entity 2 type |
| `ent1` | Entity 1 text |
| `ent2` | Entity 2 text |
| `sents` | Original sentence(s) |
| `masked_sents` | Sentence with entities replaced by ENT1/ENT2 |
| `verbalized_label` | Human-readable label (same as label for now) |

## BIORED Relation Types

The dataset includes the following biomedical relation types:
- **Association** - General association between entities
- **Positive_Correlation** - Positive correlation
- **Negative_Correlation** - Negative/inverse correlation  
- **Bind** - Binding interaction
- **Cotreatment** - Co-treatment relationship
- **Comparison** - Comparative relationship
- **Conversion** - Conversion/transformation
- **Drug_Interaction** - Drug interaction

## Entity Types

Common entity types in BIORED:
- `GeneOrGeneProduct` - Genes and gene products
- `DiseaseOrPhenotypicFeature` - Diseases and phenotypes
- `ChemicalEntity` - Chemical compounds and drugs
- `SequenceVariant` - Genetic variants
- `OrganismTaxon` - Organism/species
- `CellLine` - Cell lines

## Example

Input (BIORED JSON relation):
```json
{
  "id": "R0",
  "infons": {
    "entity1": "6528",
    "entity2": "D003409", 
    "type": "Association"
  }
}
```

Output (CSV row):
```
0	biored_14510914_R0	Association	GeneOrGeneProduct	DiseaseOrPhenotypicFeature	sodium/iodide symporter	Congenital hypothyroidism	...
```

## Notes

- The script combines text from all passages in a document
- Entity mentions are extracted from annotations using their identifiers
- Relations without valid entity pairs are skipped
- Tab-separated format is used to match TACRED conventions
