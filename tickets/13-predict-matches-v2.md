# 13 — Create `predict_matches_v2.py` (inference script)

**What to build:** The new inference script that loads both models + calibrator + threshold, computes 30-feature vectors per candidate pair, scores with averaged LGB+CB probabilities, applies isotonic calibration, thresholds, and writes the contest-format output TSV.

**Blocked by:** 11 (Models trained and saved), 09 (load_full_record_map), 10 (adapted compute_pair_features), 08 (all_features.py)

**Status:** ready-for-agent

## Detailed instructions

### Create file: `code/business_entity_resolution/src/predict_matches_v2.py`

### Critical constraint: NO inference-time-only filtering

**Do NOT add any `min_cand_score` filter or score-based pre-filter that was not also applied during training.** The old `predict_matches.py` had `min_cand_score=0.60` which filtered candidates at inference only — this is a known bug that caused train/test distribution mismatch. The ONLY score filter is `min_score` in `blocking_v2.py` (applied identically at blocking time for both training and inference candidates).

### Structure

```python
import os, time, csv, pickle, argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from multiprocessing import Pool

from data_loading import load_full_record_map
from all_features import ALL_FEATURE_NAMES
from feature_utils import compute_pair_features, compute_embedding_features

def predict_matches_v2(
    split='test',
    cand_dir='output_v2',
    model_lgb_path='models/lgb_matcher_v2.txt',
    model_cb_path='models/cb_matcher_v2.cbm',
    meta_path='models/matcher_v2_metadata.pkl',
    out_path='output_v2/matching_results.tsv',
    num_workers=8,
    chunksize=500_000
):
```

### Loading models

```python
lgb_model = lgb.Booster(model_file=model_lgb_path)
cb_model = CatBoostClassifier()
cb_model.load_model(model_cb_path)

with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
threshold = meta['threshold']
calibrator = meta.get('calibrator', None)
assert meta['features'] == ALL_FEATURE_NAMES, (
    f"Feature mismatch between saved model ({len(meta['features'])}) "
    f"and current ALL_FEATURE_NAMES ({len(ALL_FEATURE_NAMES)})"
)
```

### Loading data

```python
prefix = f"dataset/{split}/{split}"
s1_map = load_full_record_map(f"{prefix}_source1_clean.parquet",
                               f"{prefix}_source1_embeddings.npy",
                               f"{prefix}_source1_embed_ids.npy")
s2_map = load_full_record_map(f"{prefix}_source2_clean.parquet",
                               f"{prefix}_source2_embeddings.npy",
                               f"{prefix}_source2_embed_ids.npy")
s3_map = load_full_record_map(f"{prefix}_source3_clean.parquet",
                               f"{prefix}_source3_embeddings.npy",
                               f"{prefix}_source3_embed_ids.npy")
combined_map = {**s2_map, **s3_map}  # candidates come from S2 and S3
```

### Feature extraction per pair

For each row in the detailed candidates TSV (4-col: `s1_id, cand_id, source, embed_score`):

```python
s1_rec = s1_map[s1_id]         # 7-tuple
cand_rec = combined_map[cand_id]  # 7-tuple

lexical = compute_pair_features(
    s1_rec[0], s1_rec[1], s1_rec[2],
    cand_rec[0], cand_rec[1], cand_rec[2],
    embed_score=float(embed_score),
    s1_has_state=s1_rec[3], s2_has_state=cand_rec[3],
    s1_needs_trans=s1_rec[4], s2_needs_trans=cand_rec[4]
)
emb_feats = compute_embedding_features(s1_rec[6], cand_rec[6])
features = lexical + emb_feats  # 30 features
```

### Scoring

```python
lgb_probs = lgb_model.predict(X)  # raw LGB probabilities
cb_probs = cb_model.predict_proba(X)[:, 1]  # CatBoost probabilities
avg_probs = (lgb_probs + cb_probs) / 2.0

if calibrator is not None:
    avg_probs = calibrator.transform(avg_probs)

matches = avg_probs >= threshold
```

### One threshold, no country-specific logic

Use the single `threshold` from training. No per-country thresholds.

### Output format

Write contest-format TSV:
```
source1_entity_id\tmatched_entity_ids
```
- Every S1 entity must appear (even if matched_entity_ids is empty).
- Matched IDs are comma-separated, deduplicated.
- Preserve the ordering of S1 entity IDs from the original test source1 parquet.

### Multiprocessing

Keep the old `predict_matches.py`'s `Pool`-based parallel feature extraction pattern — it's reasonable engineering. Use `num_workers=8`, chunk candidate pairs into batches for parallel extraction, then score sequentially (GBDT scoring is fast enough).

### CLI

```python
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='test', choices=['train', 'test'])
    parser.add_argument('--cand-dir', default='output_v2')
    parser.add_argument('--out', default='output_v2/matching_results.tsv')
    args = parser.parse_args()
    predict_matches_v2(split=args.split, cand_dir=args.cand_dir, out_path=args.out)
```

- [ ] `predict_matches_v2.py` created
- [ ] NO `min_cand_score` or inference-only filter
- [ ] Uses `load_full_record_map` for all 3 sources
- [ ] Imports `ALL_FEATURE_NAMES` from `all_features.py` and asserts match with saved model
- [ ] Computes 30-feature vector (27 lexical + 3 embedding) per pair
- [ ] Scores with averaged LGB + CB probabilities
- [ ] Applies isotonic calibration if calibrator exists in metadata
- [ ] Uses single threshold from training metadata, no country-specific logic
- [ ] Output TSV has all S1 entities, comma-separated matched IDs
- [ ] `--split` flag supports both 'train' and 'test'
