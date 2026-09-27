# 09 — Create `load_full_record_map()` with embedding alignment assertions

**What to build:** A shared data-loading function that returns a dict mapping `entity_id → (clean_name, clean_addr, postal_code, has_state, needs_trans_name, needs_trans_addr, embedding_vector)` with hard assertions on row-order alignment between parquet and .npy files. Used by training and inference.

**Blocked by:** 05 (Embeddings exist as .npy files)

**Status:** ready-for-agent

## Detailed instructions

### Create or update: `code/business_entity_resolution/src/data_loading.py`

Create a new shared module (do NOT modify the existing `load_record_map` in `train_matcher.py` or `predict_matches.py` — those are part of the old pipeline):

```python
"""
Shared data loading for the v2 pipeline.
Both train_matcher_v2.py and predict_matches_v2.py import from here.
"""
import numpy as np
import pandas as pd

def load_full_record_map(parquet_path, embeddings_path, embed_ids_path):
    """
    Returns dict: entity_id -> 7-tuple:
        (clean_name, clean_addr, postal_code, has_state,
         needs_trans_name, needs_trans_addr, embedding_384dim)

    The assertions on ID order alignment are NOT optional — they catch the
    single easiest silent catastrophic bug in this pipeline (computing
    features against a mismatched entity's embedding). A failure here means
    embeddings were regenerated without re-running preprocessing, or vice versa.
    """
    cols = [
        'entity_id', 'business_name_clean', 'business_address_clean',
        'postal_code', 'has_state',
        'needs_transliteration_name', 'needs_transliteration_address'
    ]
    df = pd.read_parquet(parquet_path, columns=cols)
    embeddings = np.load(embeddings_path)
    embed_ids = np.load(embed_ids_path)

    assert len(df) == len(embeddings) == len(embed_ids), (
        f"Row count mismatch: parquet={len(df)}, embeddings={len(embeddings)}, "
        f"embed_ids={len(embed_ids)}. Regenerate embeddings."
    )
    assert (df['entity_id'].to_numpy() == embed_ids).all(), (
        "ID order mismatch between parquet and embeddings — features would be "
        "computed against the WRONG entity's embedding. Regenerate embeddings."
    )

    rec_map = {}
    for i, r in enumerate(df.itertuples(index=False)):
        rec_map[r.entity_id] = (
            r.business_name_clean,
            r.business_address_clean,
            r.postal_code,
            bool(r.has_state),
            bool(r.needs_transliteration_name),
            bool(r.needs_transliteration_address),
            embeddings[i]  # 384-dim float32 vector
        )
    del df
    return rec_map
```

### Tuple position reference (critical for downstream)

When consuming a record from this map:
```python
rec = rec_map[entity_id]
clean_name       = rec[0]
clean_addr        = rec[1]
postal_code       = rec[2]
has_state         = rec[3]
needs_trans_name  = rec[4]
needs_trans_addr  = rec[5]
embedding         = rec[6]  # np.ndarray shape (384,)
```

The old `load_record_map` returned a 6-tuple (no embedding). The new one returns 7-tuple. All downstream code that indexes into this tuple must use the 7-tuple layout.

### Validation

```python
from data_loading import load_full_record_map

rm = load_full_record_map(
    'dataset/train/train_source1_clean.parquet',
    'dataset/train/train_source1_embeddings.npy',
    'dataset/train/train_source1_embed_ids.npy'
)
sample_id = list(rm.keys())[0]
rec = rm[sample_id]
print(f"Record tuple length: {len(rec)}")  # 7
print(f"Embedding shape: {rec[6].shape}")   # (384,)
print(f"Embedding L2 norm: {np.linalg.norm(rec[6]):.4f}")  # ~1.0
```

- [ ] `data_loading.py` created with `load_full_record_map()`
- [ ] Assertions on row count AND ID order present
- [ ] Returns 7-tuple with embedding as position 6
- [ ] Validation passes for at least one train source file
- [ ] No modification to old `load_record_map` in existing files
