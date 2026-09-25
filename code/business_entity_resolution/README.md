# Business Entity Resolution Pipeline

ML solution for the Amazon ML Challenge 2026: Business Entity Resolution.

## Directory Structure
```
code/business_entity_resolution/
├── src/
│   └── pipeline.ipynb        # Primary end-to-end pipeline notebook
├── README.md                 # Execution and reproduction instructions
└── requirements.txt          # Pinned dependency environment
```

## Setup & Environment
Run with Python 3.11+:
```bash
# Using uv (recommended)
uv venv
uv pip install -r requirements.txt

# Or standard pip
pip install -r requirements.txt
```

## Execution
Run the pipeline notebook end-to-end to generate output files:
```bash
jupyter execute src/pipeline.ipynb
```
Outputs are written to:
- `output/matching_results.tsv` (Leaderboard submission)
- `output/candidate_pairs.tsv` (Blocking candidate pairs)

## Validation
Verify submission compliance:
```bash
python student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
