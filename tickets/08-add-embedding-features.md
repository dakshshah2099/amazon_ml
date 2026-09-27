# 08 — Add embedding similarity features to `feature_utils.py`

**What to build:** Add 3 embedding-based features (`embed_cosine`, `embed_l2_dist`, `embed_high_conf`) alongside the existing 27 lexical features, and create a single canonical `ALL_FEATURE_NAMES` list that training and inference both import.

**Blocked by:** None — can start immediately (no dependency on embeddings being generated).

**Status:** ready-for-agent

## Detailed instructions

### Step 1: Add `compute_embedding_features()` to `feature_utils.py`

Add this function at the bottom of `code/business_entity_resolution/src/feature_utils.py`:

```python
def compute_embedding_features(emb1, emb2):
    """
    emb1, emb2: L2-normalized 384-dim float32 vectors.
    Returns 3 features.
    """
    cos_sim = float(np.dot(emb1, emb2))  # L2-normalized -> dot = cosine
    l2_dist = float(np.linalg.norm(emb1 - emb2))
    high_conf_semantic = 1.0 if cos_sim >= 0.75 else 0.0
    return [cos_sim, l2_dist, high_conf_semantic]

EMBEDDING_FEATURE_NAMES = ['embed_cosine', 'embed_l2_dist', 'embed_high_conf']
```

Make sure `import numpy as np` is at the top of the file (it likely already is for other uses — verify).

### Step 2: Create `all_features.py` as single source of truth

Create `code/business_entity_resolution/src/all_features.py`:

```python
"""
Single source of truth for the combined feature vector.
Both train_matcher_v2.py and predict_matches_v2.py must import from here.
This prevents train/inference feature-order drift — a bug class that has
bitten this project before.
"""
from feature_utils import FEATURE_NAMES, EMBEDDING_FEATURE_NAMES

ALL_FEATURE_NAMES = FEATURE_NAMES + EMBEDDING_FEATURE_NAMES
# Total: 27 lexical + 3 embedding = 30 features
```

### Step 3: Verify existing `FEATURE_NAMES`

The existing `FEATURE_NAMES` in `feature_utils.py` has 27 items:
```python
FEATURE_NAMES = [
    'name_ratio', 'name_partial_ratio', 'name_token_sort', 'name_token_set',
    'name_jw', 'addr_jw', 'addr_ratio', 'addr_partial_ratio',
    'addr_token_sort', 'addr_token_set', 'name_jaccard', 'addr_jaccard',
    'postal_match', 'num_overlap', 'len_diff_name', 'len_diff_addr',
    'sig_both', 'sig_name', 'name_sim', 'addr_sim', 'score_gap',
    'postal_and_num_match', 'name_containment', 'name_acronym_match',
    'both_has_state', 'state_presence_mismatch', 'either_needs_translit'
]
```

**Important:** Features 16-20 (`sig_both`, `sig_name`, `name_sim`, `addr_sim`, `score_gap`) reference the old blocking's `signal_type`, `name_score`, `addr_score` columns. The new blocking_v2 only has `embed_score`. These features need to be adapted in ticket 10 (feature computation update). For now, just define the feature lists — the actual computation changes come later.

### Do NOT modify `compute_pair_features()` in this ticket

That function's signature and internals will be updated in ticket 10. This ticket only adds the new function and the canonical feature list.

- [x] `compute_embedding_features()` added to `feature_utils.py`
- [x] `EMBEDDING_FEATURE_NAMES` defined in `feature_utils.py`
- [x] `all_features.py` created importing from `feature_utils` and exporting `ALL_FEATURE_NAMES`
- [x] `ALL_FEATURE_NAMES` has exactly 30 items (27 + 3)
- [x] `numpy` imported in `feature_utils.py`
