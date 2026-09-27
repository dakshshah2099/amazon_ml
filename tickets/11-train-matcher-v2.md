# 11 — Create `train_matcher_v2.py` with LightGBM + CatBoost 5-fold CV

**What to build:** The new training script that uses 30-feature vectors (27 lexical + 3 embedding), trains LightGBM + CatBoost via 5-fold stratified CV, selects a threshold on out-of-fold predictions, applies isotonic calibration, and saves 2 final models + calibrator + threshold.

**Blocked by:** 07 (Tuned blocking run on train), 09 (load_full_record_map), 10 (adapted compute_pair_features), 08 (all_features.py)

**Status:** ready-for-agent

## Detailed instructions

### Create file: `code/business_entity_resolution/src/train_matcher_v2.py`

### Data loading

Use `load_full_record_map()` from `data_loading.py` (ticket 09) to load record maps for S1, S2, S3 with embeddings.

```python
from data_loading import load_full_record_map
from all_features import ALL_FEATURE_NAMES  # 30 features
from feature_utils import compute_pair_features, compute_embedding_features

s1_map = load_full_record_map(
    'dataset/train/train_source1_clean.parquet',
    'dataset/train/train_source1_embeddings.npy',
    'dataset/train/train_source1_embed_ids.npy'
)
# same for s2_map, s3_map
```

### Ground truth and candidate loading

Reuse the logic from the old `train_matcher.py`:
1. Load GT from `dataset/train/train_ground_truth.tsv` (TSV: `source1_entity_id\tmatched_entity_ids`).
2. Load candidates from `output_v2/train_candidate_pairs_detailed.tsv` (4-col format from blocking_v2: `source1_entity_id\tcandidate_entity_id\tsource\tembed_score`).

### Negative mining (adapted for v2)

Old logic used `signal_type == 'both'` for hard negatives. New definition:

- **Hard negatives:** candidates with `embed_score` in the **top-3** for that S1 entity that are NOT the true match. These are the hardest cases the model needs to learn to reject.
- **Soft negatives:** random other candidates for that S1 entity that are not true matches.
- Keep 50/50 hard/soft split, `neg_ratio=3` (3 negatives per positive).

### Feature computation per pair

For each (s1_id, candidate_id) pair:
```python
s1_rec = s1_map[s1_id]        # 7-tuple from load_full_record_map
s2_rec = s2_map[candidate_id]  # or s3_map

# 27 lexical features
lexical = compute_pair_features(
    s1_rec[0], s1_rec[1], s1_rec[2],  # name, addr, postal
    s2_rec[0], s2_rec[1], s2_rec[2],  # name, addr, postal
    embed_score=embed_score,           # from blocking detailed TSV
    s1_has_state=s1_rec[3], s2_has_state=s2_rec[3],
    s1_needs_trans=s1_rec[4], s2_needs_trans=s2_rec[4]
)

# 3 embedding features
emb_feats = compute_embedding_features(s1_rec[6], s2_rec[6])

# Full 30-feature vector
features = lexical + emb_feats
```

### 5-fold stratified CV

```python
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from catboost import CatBoostClassifier
import numpy as np

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_probs = np.zeros(len(y))

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_va, y_va = X[val_idx], y[val_idx]

    lgb_model = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=31, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    lgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])

    cb_model = CatBoostClassifier(
        iterations=400, learning_rate=0.05, depth=6, random_seed=42,
        verbose=False, early_stopping_rounds=50
    )
    cb_model.fit(X_tr, y_tr, eval_set=(X_va, y_va))

    fold_probs = (lgb_model.predict_proba(X_va)[:, 1] +
                  cb_model.predict_proba(X_va)[:, 1]) / 2.0
    oof_probs[val_idx] = fold_probs
    print(f"Fold {fold_idx+1}/5 done.")
```

### Isotonic calibration

```python
from sklearn.isotonic import IsotonicRegression

calibrator = IsotonicRegression(out_of_bounds='clip')
calibrated_oof = calibrator.fit_transform(oof_probs, y)
```

Compare F0.5 before and after calibration. If calibration hurts by > 0.3pp, skip it and note why. Otherwise use calibrated probabilities for threshold search.

### Threshold search on OOF predictions

```python
from sklearn.metrics import precision_score, recall_score

best_thresh, best_f05 = 0.5, 0.0
probs_for_thresh = calibrated_oof  # or oof_probs if calibration hurts
for t in np.arange(0.30, 0.991, 0.01):
    preds = (probs_for_thresh >= t).astype(int)
    p = precision_score(y, preds, zero_division=0)
    r = recall_score(y, preds, zero_division=0)
    denom = (0.25 * p + r)
    f05 = (1.25 * p * r / denom) if denom > 0 else 0.0
    if f05 > best_f05:
        best_f05 = f05
        best_thresh = t

print(f"OUT-OF-FOLD Optimal Threshold: {best_thresh:.2f}, OOF F0.5: {best_f05*100:.2f}%")
print("NOTE: This is OOF on training data, NOT the real test-set F0.5.")
```

### Final models (trained on ALL data)

After CV validates the approach:
```python
final_lgb = lgb.LGBMClassifier(
    n_estimators=400, learning_rate=0.05, num_leaves=31, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, random_state=42
)
final_lgb.fit(X, y)

final_cb = CatBoostClassifier(
    iterations=400, learning_rate=0.05, depth=6, random_seed=42, verbose=False
)
final_cb.fit(X, y)
```

### Save artifacts

```python
import pickle, os
os.makedirs('models', exist_ok=True)

final_lgb.booster_.save_model('models/lgb_matcher_v2.txt')
final_cb.save_model('models/cb_matcher_v2.cbm')

metadata = {
    'threshold': best_thresh,
    'oof_f05': best_f05,
    'features': ALL_FEATURE_NAMES,
    'calibrator': calibrator,  # or None if skipped
    'note': 'OOF F0.5 is on training distribution, not real test set'
}
with open('models/matcher_v2_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)
```

### Validation

1. Run on a 5,000-entity sample first (pass `--num-entities 5000`).
2. Confirm 5 folds complete, OOF F0.5 prints.
3. If OOF F0.5 comes out HIGHER than old pipeline's 97.96%, treat with suspicion — re-check for label leak.
4. Then run full training.

- [ ] `train_matcher_v2.py` created
- [ ] Uses `load_full_record_map` (7-tuple with embeddings)
- [ ] Uses `ALL_FEATURE_NAMES` from `all_features.py` (30 features)
- [ ] Hard negatives redefined as top-3 embed_score non-matches
- [ ] 5-fold stratified CV with both LightGBM and CatBoost
- [ ] OOF probabilities computed across all folds
- [ ] Isotonic calibration applied (or skipped with documented reason)
- [ ] Threshold search on OOF predictions, not single split
- [ ] OOF F0.5 printed with explicit "this is OOF, not test-set" label
- [ ] Final LGBMClassifier + CatBoostClassifier trained on full data
- [ ] Saved: `lgb_matcher_v2.txt`, `cb_matcher_v2.cbm`, `matcher_v2_metadata.pkl`
- [ ] Metadata contains threshold, oof_f05, feature names, calibrator
