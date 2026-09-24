# Competition Strategies: F₀.₅ Optimization, Singletons & Precision

## Overview

This guide covers battle-tested strategies from Kaggle, Amazon KDD, and WDC competitions
specifically tuned for precision-heavy F₀.₅ evaluation.

---

## 1. F₀.₅ Mathematics & Implications

### Formula
```
F₀.₅ = (1.25 × P × R) / (0.25 × P + R) = (5 × P × R) / (P + 4R)
```

### Key Properties
- **Precision weighted 2× over recall**
- **1 False Positive ≈ 4× worse than 1 False Negative**
- Optimal threshold shifts from ~0.50 (F₁) to **~0.72–0.85** (F₀.₅)
- Macro-averaged per S1 entity → each entity contributes equally

### Score Impact Examples

| Scenario | Precision | Recall | F₀.₅ |
|---|---|---|---|
| Perfect match | 1.0 | 1.0 | **1.000** |
| Correct singleton (empty prediction) | 1.0 | 1.0 | **1.000** |
| 1 extra false match on singleton | 0.0 | — | **0.000** |
| Predict 3, only 2 correct (of 2 true) | 0.667 | 1.0 | **0.714** |
| Predict 1, correct (of 2 true) | 1.0 | 0.5 | **0.833** |
| Miss all matches | — | 0.0 | **0.000** |

> [!IMPORTANT]
> Predicting 1 correct out of 2 true matches (F₀.₅ = 0.833) scores HIGHER than
> predicting all 2 correct + 1 wrong (F₀.₅ = 0.714). **When in doubt, DON'T match.**

---

## 2. Threshold Optimization

### Grid Search on OOF Predictions

```python
import numpy as np

def compute_macro_f05(y_true_groups, y_probs_groups, threshold):
    """
    Compute macro-averaged F₀.₅ across S1 entity groups.
    y_true_groups: list of arrays (one per S1 entity)
    y_probs_groups: list of arrays (one per S1 entity)
    """
    f05_scores = []
    for y_true, y_prob in zip(y_true_groups, y_probs_groups):
        y_pred = (y_prob >= threshold).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        
        if tp + fp == 0 and tp + fn == 0:
            f05_scores.append(1.0)  # True singleton, predicted singleton
        elif tp == 0:
            f05_scores.append(0.0)
        else:
            p = tp / (tp + fp)
            r = tp / (tp + fn)
            f05 = (1.25 * p * r) / (0.25 * p + r)
            f05_scores.append(f05)
    
    return np.mean(f05_scores)

def find_optimal_threshold(y_true_groups, y_probs_groups):
    thresholds = np.linspace(0.40, 0.95, 551)
    scores = [compute_macro_f05(y_true_groups, y_probs_groups, t) for t in thresholds]
    best_idx = np.argmax(scores)
    return thresholds[best_idx], scores[best_idx]
```

### Advanced Threshold Practices
1. **OOF ensembling**: Use 5-fold CV, take median of 5 fold-optimal thresholds
2. **Smoothing**: Fit LOESS/spline over threshold-score curve to avoid noisy spikes
3. **Country-specific thresholds**: Different optimal τ for US vs India vs France

---

## 3. Singleton Handling

### Why Singletons Are Critical
- Singletons may be **40–70%** of all S1 entities
- Correct singleton = 1.0 score. **Free points.**
- False merge on singleton = 0.0 score. **Catastrophic.**

### Strategies

#### A. Strict Singleton Gate
```python
def apply_singleton_gate(s1_id, candidate_probs, threshold):
    """If no candidate exceeds threshold, emit empty match list."""
    max_prob = max(candidate_probs) if candidate_probs else 0.0
    if max_prob < threshold:
        return []  # Singleton
    return [cid for cid, p in zip(candidate_ids, candidate_probs) if p >= threshold]
```

#### B. Top-1 to Top-2 Margin Filter
```python
def margin_filter(sorted_probs, margin_threshold=0.15):
    """Reject if top-1 and top-2 are too close (ambiguous)."""
    if len(sorted_probs) < 2:
        return True  # Allow single candidate if above threshold
    s1, s2 = sorted_probs[0], sorted_probs[1]
    return (s1 - s2) >= margin_threshold
```

#### C. Reciprocal Nearest Neighbor (MNN)
```python
def mutual_nearest_neighbor(s1_to_candidates, s2s3_to_candidates):
    """Only allow match if both sides agree."""
    verified = []
    for s1_id, cands in s1_to_candidates.items():
        for cand_id in cands:
            if s1_id in s2s3_to_candidates.get(cand_id, []):
                verified.append((s1_id, cand_id))
    return verified
```

---

## 4. Precision Maximization Techniques

### A. Hard Veto Rules (Deterministic Overrides)
```python
def veto_match(rec1, rec2) -> bool:
    """Return True to REJECT the match regardless of model score."""
    # Different countries → reject
    if rec1['country'].lower() != rec2['country'].lower():
        return True
    
    # Postal codes exist but completely different → reject
    p1, p2 = rec1.get('postal', ''), rec2.get('postal', '')
    if p1 and p2 and p1[:3] != p2[:3]:
        return True
    
    # Name is extremely different → reject
    if name_jaro_winkler(rec1, rec2) < 0.5:
        return True
    
    return False
```

### B. Complete Linkage Clustering (Not Single Linkage!)

> [!CAUTION]
> **NEVER use connected components / single linkage for final clustering.**
> One false edge bridges two clusters → precision collapse.

```python
# Complete linkage: new member must match ALL existing cluster members
def complete_linkage_cluster(match_edges, threshold):
    """
    Only form cluster if every pair within cluster exceeds threshold.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    # Use complete linkage (maximum distance between cluster members)
    Z = linkage(distance_matrix, method='complete')
    clusters = fcluster(Z, t=1-threshold, criterion='distance')
    return clusters
```

### C. Ensemble Consensus (Intersection)
```python
# For max precision: require ALL models to agree
matches_model1 = set(predict_matches(model1, candidates))
matches_model2 = set(predict_matches(model2, candidates))
matches_model3 = set(predict_matches(model3, candidates))

# Intersection = highest precision
final_matches = matches_model1 & matches_model2 & matches_model3
```

### D. Probability Calibration
```python
from sklearn.calibration import CalibratedClassifierCV

# Calibrate on validation set so probabilities = true confidence
calibrated = CalibratedClassifierCV(base_model, method='isotonic', cv=5)
calibrated.fit(X_train, y_train)
```

### E. Hard Negative Mining in Training
Mine negatives with high lexical overlap but different entities:
```python
# Find hard negatives: TF-IDF cosine > 0.7 but y=0
hard_negs = candidates[(candidates['tfidf_cosine'] > 0.7) & (candidates['label'] == 0)]
# Upsample these in training data
```

---

## 5. Common Pitfalls

### ❌ Pitfall 1: Cluster Collapse (Single Linkage)
A single false positive edge bridges two clusters → giant corrupted blob → precision destroyed.
**Fix**: Use complete linkage or skip clustering entirely (this challenge is per-S1-entity, not clustering).

### ❌ Pitfall 2: CV Leakage
Random K-Fold on pairs → same entity in train and val → memorization.
**Fix**: `GroupKFold` by S1 entity ID.

### ❌ Pitfall 3: Class Imbalance
99.9% of candidate pairs are non-matches. Default threshold = terrible.
**Fix**: `scale_pos_weight`, hard negative mining, threshold tuning on F₀.₅.

### ❌ Pitfall 4: Ignoring Blocking Recall
If blocking misses a true match, it's gone forever.
**Fix**: Measure Pairs Completeness at blocking stage. Target > 98%.

### ❌ Pitfall 5: Optimizing Wrong Metric
Training on binary cross-entropy / accuracy ≠ F₀.₅.
**Fix**: Tune threshold on macro-averaged F₀.₅ on validation set.

### ❌ Pitfall 6: Hardcoding Countries
Train has US, India. Test adds France.
**Fix**: Never `if country in ['US', 'India']`. Treat country as open string.

---

## 6. Past Competition Winning Patterns

### Shopee Product Matching (Kaggle 2021)
- ArcFace contrastive embeddings + iterative neighborhood blending
- Dual threshold on text + image cosine
- Complete linkage clustering

### Foursquare Location Matching (Kaggle 2022)  
- Multi-channel blocking (spatial kNN + TF-IDF + exact categorical)
- 100+ pairwise features → LightGBM/CatBoost
- Graph refinement with Dijkstra shortest-path thresholding

### Amazon KDD Cup 2022 (ESCI)
- DeBERTa-v3-large cross-encoders
- Dense bi-encoders for retrieval
- Multi-task loss functions

### Common Pattern Across Winners
```
1. Multi-channel blocking (union of methods)
2. Rich pairwise features (30-100+ features)
3. GBDT classifier (LightGBM/CatBoost)
4. Aggressive threshold tuning on competition metric
5. Post-processing: reciprocal NN, margin filter, veto rules
```

---

## 7. Validation Strategy for This Challenge

```python
import pandas as pd
from sklearn.model_selection import GroupKFold

# Load ground truth
gt = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t")

# Create validation split using GroupKFold on S1 entity
gkf = GroupKFold(n_splits=5)
for fold, (train_idx, val_idx) in enumerate(gkf.split(gt, groups=gt['source1_entity_id'])):
    train_gt = gt.iloc[train_idx]
    val_gt = gt.iloc[val_idx]
    
    # Run full pipeline on val_gt
    # Compute macro F₀.₅ per S1 entity
    # Tune threshold on this fold
```

### Always Validate Output Format
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

---

## 8. Quick Decision Framework

```
For each S1 entity:
  1. Run blocking → get candidates from S2/S3
  2. If 0 candidates → singleton (empty match list)
  3. Score each candidate with classifier
  4. Apply veto rules (country mismatch, postal mismatch, extreme name diff)
  5. Apply singleton gate (max_prob < τ → empty)
  6. Apply margin filter (top1 - top2 < δ → singleton)
  7. Apply reciprocal NN check
  8. Emit surviving matches
```

> [!TIP]
> **When in doubt about a match, DON'T include it.**
> F₀.₅ rewards restraint. Missing a real match costs ~0.25× as much as adding a false one.
