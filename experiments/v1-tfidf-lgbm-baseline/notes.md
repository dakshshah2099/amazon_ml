# Experiment Notes: v1-tfidf-lgbm-baseline

## Summary
Baseline implementation adhering strictly to Amazon ML Challenge specifications:
1. Universal country-agnostic preprocessing (Unicode NFKD, universal postal code regex, universal landmark stripping, legal suffix canonicalization).
2. Open-set country-partitioned dual-channel sparse TF-IDF blocking (char 3-4 grams on clean name and clean address dynamically grouped by whatever country labels appear in data).
3. Fast vectorized pairwise feature extraction via RapidFuzz and DataFrame merge.
4. 5-Fold GroupKFold LightGBM classifier with threshold optimization for macro F0.5.
5. Official submission generation at `output/` and validation check (`validate_submission.py`) passing with exit code 0.

## Directory Structure Alignment
- Source code in `code/business_entity_resolution/src/pipeline.ipynb`
- Execution README in `code/business_entity_resolution/README.md`
- Pinned dependencies in `code/business_entity_resolution/requirements.txt`
- Root `Documentation_template.md`
- Root `dataset/` junction to `student_resource/dataset/`

## Key Validation Metrics
- Validation Blocking Recall: 99.60%
- Macro F0.5 Score: 0.1214 (Precision: 0.1616, Recall: 0.0851)
- Optimal Threshold: 0.46
- Official Validator: PASS (exit code 0) on full test set (1,732,544 entities)
