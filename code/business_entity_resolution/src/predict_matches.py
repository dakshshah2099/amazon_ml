import os
import sys
import gc
import time
import pickle
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from multiprocessing import Pool

from data_loading import load_record_map_text_only
from feature_utils import ALL_FEATURE_NAMES, extract_pair_features

def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

# Global variables for worker processes
G_S1_MAP = None
G_CAND_MAP = None

def _init_worker(s1_map, cand_map):
    global G_S1_MAP, G_CAND_MAP
    G_S1_MAP = s1_map
    G_CAND_MAP = cand_map

def _worker_extract_features(pairs_chunk):
    """
    Worker process: extracts 30-feature vector for a chunk of pairs.
    pairs_chunk: list of (s1_id, cand_id, embed_score)
    returns: (valid_pairs, features_array)
    """
    valid_pairs = []
    features_list = []

    for s1_id, cand_id, embed_score in pairs_chunk:
        if s1_id not in G_S1_MAP or cand_id not in G_CAND_MAP:
            continue
        s1_rec = G_S1_MAP[s1_id]
        cand_rec = G_CAND_MAP[cand_id]
        feats = extract_pair_features(s1_rec, cand_rec, embed_score)
        features_list.append(feats)
        valid_pairs.append((s1_id, cand_id))

    if features_list:
        X = np.array(features_list, dtype=np.float32)
    else:
        X = np.empty((0, len(ALL_FEATURE_NAMES)), dtype=np.float32)

    return valid_pairs, X

def predict_matches(
    split='test',
    cand_dir='output',
    model_lgb_path='models/lgb_matcher.txt',
    model_cb_path='models/cb_matcher.cbm',
    meta_path='models/matcher_metadata.pkl',
    out_path='output/matching_results.tsv',
    chunksize=200_000,
    batch_size=10_000,
    num_workers=8,
    sample_s1=None,
    data_dir=None,
    prefix=None,
    eval_gt_path=None,
    threshold=None
):
    log("=" * 70)
    log(f"STAGE 5: INFERENCE ({split.upper()}) WITH NO INFERENCE-TIME PRE-FILTERING")
    log("=" * 70)

    # 1. Load Models and Metadata
    log(f"Loading LightGBM model from {model_lgb_path}...")
    lgb_model = lgb.Booster(model_file=model_lgb_path)

    log(f"Loading CatBoost model from {model_cb_path}...")
    cb_model = CatBoostClassifier()
    cb_model.load_model(model_cb_path)

    log(f"Loading metadata from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    if threshold is None:
        threshold = float(meta['threshold'])
    calibrator = meta.get('calibrator', None)
    log(f"Decision Threshold: {threshold:.4f} (Calibrator present: {calibrator is not None})")
    assert meta['features'] == ALL_FEATURE_NAMES, "Model features do not match ALL_FEATURE_NAMES!"

    # 2. Load Full Record Maps with Assertions
    if data_dir is None:
        data_dir = f"dataset/{split}"
    if prefix is None:
        file_prefix = f"{data_dir}/{split}"
    else:
        file_prefix = f"{data_dir}/{prefix}"

    log(f"Loading record maps from {file_prefix} (text-only, memory safe)...")
    s1_map = load_record_map_text_only(f"{file_prefix}_source1_clean.parquet")
    s2_map = load_record_map_text_only(f"{file_prefix}_source2_clean.parquet")
    s3_map = load_record_map_text_only(f"{file_prefix}_source3_clean.parquet")
    cand_map = {**s2_map, **s3_map}
    del s2_map, s3_map
    gc.collect()
    log(f"Loaded {len(s1_map):,} S1 entities and {len(cand_map):,} candidates (S2+S3) into memory.")

    # Order of S1 IDs to guarantee contest output preserves order
    df_s1_order = pd.read_parquet(f"{file_prefix}_source1_clean.parquet", columns=['entity_id'])
    s1_order_list = df_s1_order['entity_id'].tolist()
    del df_s1_order
    gc.collect()

    if sample_s1 and sample_s1 < len(s1_order_list):
        import random
        random.seed(42)
        s1_order_list = sorted(random.sample(s1_order_list, sample_s1))
        s1_filter_set = set(s1_order_list)
        log(f"Subsampled {len(s1_order_list):,} S1 entities for evaluation.")
    else:
        s1_filter_set = None

    # 3. Stream Candidates and Predict
    cand_detailed_path = os.path.join(cand_dir, f"{split}_candidate_pairs_detailed.tsv")
    log(f"Streaming candidate pairs from {cand_detailed_path}...")
    
    total_file_lines = 0
    if os.path.exists(cand_detailed_path):
        try:
            with open(cand_detailed_path, 'rb') as f_cnt:
                total_file_lines = max(0, sum(chunk.count(b'\n') for chunk in iter(lambda: f_cnt.read(4 * 1024 * 1024), b'')) - 1)
        except Exception:
            total_file_lines = 0
    
    total_chunks = (total_file_lines + chunksize - 1) // chunksize if total_file_lines > 0 else 0
    chunk_str = f" across {total_chunks:,} chunks" if total_chunks > 0 else ""
    log(f"Total candidate pairs to score: {total_file_lines:,}{chunk_str}")

    matched_results = defaultdict(set)
    total_pairs_processed = 0
    total_matches_found = 0
    t0 = time.time()

    with Pool(processes=num_workers, initializer=_init_worker, initargs=(s1_map, cand_map)) as pool:
        for chunk_idx, chunk in enumerate(pd.read_csv(
            cand_detailed_path,
            sep='\t',
            usecols=['source1_entity_id', 'candidate_entity_id', 'embed_score'],
            dtype={'source1_entity_id': str, 'candidate_entity_id': str, 'embed_score': float},
            chunksize=chunksize
        )):
            if s1_filter_set is not None:
                chunk = chunk[chunk['source1_entity_id'].isin(s1_filter_set)]
                if len(chunk) == 0:
                    continue

            pairs_to_score = list(zip(
                chunk['source1_entity_id'],
                chunk['candidate_entity_id'],
                chunk['embed_score']
            ))

            if not pairs_to_score:
                continue

            # Split into sub-batches for pool
            sub_batches = [pairs_to_score[i:i + batch_size] for i in range(0, len(pairs_to_score), batch_size)]
            results = pool.map(_worker_extract_features, sub_batches)

            # Collect features and valid pairs
            chunk_valid_pairs = []
            chunk_X_list = []
            for valid_pairs, X_sub in results:
                if len(valid_pairs) > 0:
                    chunk_valid_pairs.extend(valid_pairs)
                    chunk_X_list.append(X_sub)

            if not chunk_valid_pairs:
                continue

            X_chunk = np.vstack(chunk_X_list)

            # Score with LightGBM and CatBoost
            lgb_probs = lgb_model.predict(X_chunk)
            cb_probs = cb_model.predict_proba(X_chunk)[:, 1]
            avg_probs = (lgb_probs + cb_probs) / 2.0

            if calibrator is not None:
                avg_probs = calibrator.transform(avg_probs)

            # Precision safety guard: reject empty candidate addresses unless name is >= 95% identical
            idx_cand_empty = ALL_FEATURE_NAMES.index('cand_addr_empty')
            idx_name_ratio = ALL_FEATURE_NAMES.index('name_ratio')
            empty_guard = ~((X_chunk[:, idx_cand_empty] == 1.0) & (X_chunk[:, idx_name_ratio] < 0.95))

            # Filter by threshold with precision guard
            match_mask = (avg_probs >= threshold) & empty_guard
            matched_indices = np.where(match_mask)[0]

            for idx in matched_indices:
                s1_id, cand_id = chunk_valid_pairs[idx]
                matched_results[s1_id].add(cand_id)
                total_matches_found += 1

            total_pairs_processed += len(pairs_to_score)
            elapsed = time.time() - t0
            rate = total_pairs_processed / elapsed if elapsed > 0 else 0
            eta_s = (total_file_lines - total_pairs_processed) / rate if (rate > 0 and total_file_lines > total_pairs_processed) else 0
            pct = (total_pairs_processed / total_file_lines) * 100 if total_file_lines > 0 else 0
            chunk_num_str = f"{chunk_idx + 1}/{total_chunks}" if total_chunks > 0 else f"{chunk_idx + 1}"
            pct_str = f" ({pct:.1f}%)" if total_file_lines > 0 else ""
            eta_str = f" | ETA: {eta_s/60:.1f} min" if eta_s > 0 else ""

            if (chunk_idx + 1) % 2 == 1 or len(pairs_to_score) < chunksize or (total_file_lines > 0 and total_pairs_processed >= total_file_lines):
                log(f"  [Predict Chunk {chunk_num_str}]{pct_str} Processed {total_pairs_processed:,} pairs ({rate:,.0f} pairs/s) | {total_matches_found:,} matches found{eta_str}")

    del s1_map, cand_map
    gc.collect()

    # 4. Write Output in Contest Format
    log(f"Writing final submission TSV to {out_path}...")
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        buf = []
        for s1_id in s1_order_list:
            cands = matched_results.get(s1_id, set())
            matched_str = ','.join(sorted(cands))
            buf.append(f"{s1_id}\t{matched_str}\n")
            if len(buf) >= 100_000:
                f.writelines(buf)
                f.flush()
                buf = []
        if buf:
            f.writelines(buf)
            f.flush()

    log(f"Saved {out_path} with {len(s1_order_list):,} total rows.")

    if eval_gt_path:
        evaluate(out_path, eval_gt_path, split_name=split)

def load_id_map(path):
    import csv
    m = {}
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader, None)
        for row in reader:
            if len(row) >= 2 and row[1].strip():
                m[row[0].strip()] = {x.strip() for x in row[1].split(',') if x.strip()}
            elif len(row) >= 1:
                m[row[0].strip()] = set()
    return m

def evaluate(pred_path, gt_path, split_name='UNSPECIFIED'):
    print("=" * 60)
    print(f"EVALUATION ON SPLIT: [{split_name.upper()}]")
    print("=" * 60)

    gt = load_id_map(gt_path)
    pred = load_id_map(pred_path)

    if len(pred) < len(gt) * 0.9:
        print(f"Sample evaluation detected: evaluating across {len(pred):,} predicted S1 entities.")
        eval_s1_ids = set(pred.keys())
    else:
        eval_s1_ids = set(gt.keys()) | set(pred.keys())

    tp, fp, fn = 0, 0, 0
    for s1_id in eval_s1_ids:
        true_set = gt.get(s1_id, set())
        pred_set = pred.get(s1_id, set())

        tp += len(true_set & pred_set)
        fp += len(pred_set - true_set)
        fn += len(true_set - pred_set)

    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    denom = 0.25 * p + r
    f05 = (1.25 * p * r / denom) if denom > 0 else 0.0

    print(f"Split:               {split_name}")
    print(f"Ground Truth Items:  {len(gt):,}")
    print(f"Prediction Items:    {len(pred):,}")
    print(f"True Positives (TP): {tp:,}")
    print(f"False Positives (FP):{fp:,}")
    print(f"False Negatives (FN):{fn:,}")
    print(f"Precision:           {p*100:.2f}%")
    print(f"Recall:              {r*100:.2f}%")
    print(f"F0.5 Score:          {f05*100:.2f}%")
    print("=" * 60)
    return p, r, f05

# Backward-compat alias
predict_matches_v2 = predict_matches

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='test')
    parser.add_argument('--data-dir', default=None)
    parser.add_argument('--prefix', default=None)
    parser.add_argument('--cand-dir', default='output')
    parser.add_argument('--model-lgb', default='models/lgb_matcher.txt')
    parser.add_argument('--model-cb', default='models/cb_matcher.cbm')
    parser.add_argument('--meta', default='models/matcher_metadata.pkl')
    parser.add_argument('--out', default='output/matching_results.tsv')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--sample-s1', type=int, default=None, help='Sample N S1 entities for rapid evaluation')
    parser.add_argument('--eval-gt', default=None, help='Ground truth TSV path to evaluate against')
    parser.add_argument('--threshold', type=float, default=None, help='Decision threshold override')
    args = parser.parse_args()
    predict_matches(
        split=args.split,
        cand_dir=args.cand_dir,
        model_lgb_path=args.model_lgb,
        model_cb_path=args.model_cb,
        meta_path=args.meta,
        out_path=args.out,
        num_workers=args.workers,
        sample_s1=args.sample_s1,
        eval_gt_path=args.eval_gt,
        threshold=args.threshold,
        data_dir=args.data_dir,
        prefix=args.prefix
    )
