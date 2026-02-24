# QA4RE with vLLM Support

Enhanced implementation of **LLM-QA4RE** from ACL 2023 Findings: [Aligning Instruction Tasks Unlocks Large Language Models as Zero-Shot Relation Extractors](https://arxiv.org/pdf/2305.11159.pdf).

This version adds:
- Gemma and LLama model support
- BIORED data support
- fast batch inference using vLLM with guided decoding
- relation definitions integration
- answer shuffling for position bias testing



## Installation

### Setup the environment variables
Set up the HF token in .env file

### Setup Environment

1. Create a conda environment:
```bash
conda create -n QA4RE python=3.9 pip
conda activate QA4RE
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Data Setup

The BIORED dataset should be organized as follows:
```
QA4RE/
├── data/
│   └── BIORED/
│       ├── train.csv
│       ├── dev.csv
│       ├── test.csv
│       └── configs/
│           └── relation_definitions.json
|           └── possible_labels.json
├── outputs/
│   └── BIORED/
└── projs/
    └── QA4RE/
```

BIORED contains biomedical relations between entities:
- **Relations**: Association, Positive_Correlation, Negative_Correlation, Bind, Cotreatment, Drug_Interaction, Conversion, Comparison
- **Entity Types**: Gene (G), Disease (D), Chemical (C), Variant (V)

Added files for BIORED data:
- `relation_definitions.json`: contains the definition for the relations in BIORED, used for Relation Definitions experiment
- `possible_labels.json`: contains the combination entity and their possible relation label, used to constrain the label space for QA4RE prompting method

### Data Preparation:

If you need to convert BIORED JSON format to CSV format used by QA4RE:

```bash
cd data/BIORED

# Convert train/dev/test sets
python convert_biored_to_csv.py Train.BioC.JSON train.csv
python convert_biored_to_csv.py Dev.BioC.JSON dev.csv
python convert_biored_to_csv.py Test.BioC.JSON test.csv
```

The conversion script extracts entities and relations from BioC JSON format and creates TACRED-style CSV files with masked sentences (entities replaced with ENT1/ENT2 placeholders).


## Running

### Basic QA4RE Inference

```bash
cd projs/QA4RE


# Run on test set
bash run_biored_vllm_offline.sh test google/gemma-2-2b-it 42
```

### With Relation Definitions

Integrate relation definitions into answer choices:
```bash
bash run_biored_vllm_offline_with_definitions.sh test google/gemma-2-2b-it 42
```

### With Answer Shuffling

Test position bias by shuffling answer choices then we can aggregate the answer later for voting evaluation:
```bash
# Run with different seeds
bash run_biored_vllm_offline_shuffle.sh test google/gemma-2-2b-it 5
bash run_biored_vllm_offline_shuffle.sh test google/gemma-2-2b-it 1
bash run_biored_vllm_offline_shuffle.sh test google/gemma-2-2b-it 2
bash run_biored_vllm_offline_shuffle.sh test google/gemma-2-2b-it 9
bash run_biored_vllm_offline_shuffle.sh test google/gemma-2-2b-it 4
```


## Output Structure

Results are saved to:
```
outputs/BIORED/multi_choice_qa4re/{model}/zero_shot/seed_{seed}/
├── subset.test.output.csv  # Predictions with prompts
```


## Supported Models

Tested with instruction-tuned models:
- `google/gemma-2-2b-it`
- `google/gemma-2-2b`
- `meta-llama/Llama-3.2-1B-Instruct`
- `meta-llama/Llama-3.2-1B`
