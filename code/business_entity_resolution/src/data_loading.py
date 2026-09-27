"""
Shared data loading for the entity resolution pipeline.
Both train_matcher.py and predict_matches.py import from here.
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
    embed_ids = np.load(embed_ids_path, allow_pickle=True)

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

def load_record_map_text_only(parquet_path):
    """
    Memory-safe record loader for inference.
    Loads only string/bool metadata into 6-tuples:
        (clean_name, clean_addr, postal_code, has_state,
         needs_trans_name, needs_trans_addr)
    Omits 384-dim float32 vectors, saving ~16 GB of system RAM.
    """
    cols = [
        'entity_id', 'business_name_clean', 'business_address_clean',
        'postal_code', 'has_state',
        'needs_transliteration_name', 'needs_transliteration_address'
    ]
    df = pd.read_parquet(parquet_path, columns=cols)
    rec_map = {}
    for r in df.itertuples(index=False):
        rec_map[r.entity_id] = (
            r.business_name_clean,
            r.business_address_clean,
            r.postal_code,
            bool(r.has_state),
            bool(r.needs_transliteration_name),
            bool(r.needs_transliteration_address)
        )
    del df
    return rec_map
