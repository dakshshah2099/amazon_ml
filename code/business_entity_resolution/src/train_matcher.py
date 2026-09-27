import os
import sys
import gc
import csv
import random
import pickle
import argparse
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score
from sklearn.isotonic import IsotonicRegression

from data_loading import load_full_record_map
from feature_utils import ALL_FEATURE_NAMES, extract_pair_features

def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def train_matcher(
    gt_path='dataset/train/train_ground_truth.tsv',
    cand_detailed_path='output/train_candidate_pairs_detailed.tsv',
    num_entities_sample=50_000,
    neg_ratio=3,
    chunksize=5_000_000,
    n_folds=5,
    data_dir='dataset/train',
    prefix='train',
    models_dir='models',
    random_seed=None
):
    log("=" * 70)
    log("STAGE 4: TRAINING LightGBM + CatBoost ENSEMBLE WITH 5-FOLD CV")
    log("=" * 70)
    os.makedirs(models_dir, exist_ok=True)

    # 1. Load Ground Truth
    log(f"Loading ground truth from {gt_path}...")
    gt_map = {}
    with open(gt_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            if len(row) >= 2 and row[1].strip():
                gt_map[row[0].strip()] = {m.strip() for m in row[1].split(',') if m.strip()}

    all_s1_with_gt = list(gt_map.keys())
    log(f"Total S1 entities with ground truth: {len(all_s1_with_gt):,}")

    if random_seed is None:
        random_seed = int.from_bytes(os.urandom(4), 'little')
    log(f"Sampling {num_entities_sample:,} S1 entities for training (seed={random_seed})...")
    random.seed(random_seed)
    np.random.seed(random_seed)
    sampled_s1 = set(random.sample(all_s1_with_gt, min(num_entities_sample, len(all_s1_with_gt))))

    # 2. Collect Candidate Pairs for Sampled S1
    # Hard negative: top-3 embed_score non-matches for that S1
    # Soft negative: other non-matches
    log(f"Scanning candidate file: {cand_detailed_path}...")
    s1_candidates = {s1: [] for s1 in sampled_s1}

    for chunk in pd.read_csv(
        cand_detailed_path,
        sep='\t',
        usecols=['source1_entity_id', 'candidate_entity_id', 'source', 'embed_score'],
        dtype={'source1_entity_id': str, 'candidate_entity_id': str, 'source': str, 'embed_score': float},
        chunksize=chunksize
    ):
        chunk_sub = chunk[chunk['source1_entity_id'].isin(sampled_s1)]
        for row in chunk_sub.itertuples(index=False):
            s1_candidates[row.source1_entity_id].append((row.candidate_entity_id, row.embed_score))

    log("Mining positive and negative pairs...")
    sampled_pairs = [] # (s1_id, cand_id, embed_score, label)
    pos_count = 0
    neg_count = 0

    for s1_id in sampled_s1:
        cands = s1_candidates[s1_id]
        if not cands:
            continue
        true_matches = gt_map.get(s1_id, set())

        # Sort candidates descending by embed_score
        cands.sort(key=lambda x: x[1], reverse=True)

        s1_pos = []
        s1_hard_neg = []
        s1_soft_neg = []

        non_match_rank = 0
        for c_id, sc in cands:
            if c_id in true_matches:
                s1_pos.append((s1_id, c_id, sc, 1))
            else:
                if non_match_rank < 3:
                    s1_hard_neg.append((s1_id, c_id, sc, 0))
                else:
                    s1_soft_neg.append((s1_id, c_id, sc, 0))
                non_match_rank += 1

        n_pos = len(s1_pos)
        if n_pos == 0:
            continue

        sampled_pairs.extend(s1_pos)
        pos_count += n_pos

        target_negs = n_pos * neg_ratio
        half_negs = target_negs // 2
        other_half = target_negs - half_negs

        chosen_hard = s1_hard_neg[:half_negs]
        chosen_soft = random.sample(s1_soft_neg, min(len(s1_soft_neg), other_half)) if s1_soft_neg else []
        
        # If not enough hard or soft, supplement from the other
        if len(chosen_hard) < half_negs and len(s1_soft_neg) > len(chosen_soft):
            need = half_negs - len(chosen_hard)
            rem = [x for x in s1_soft_neg if x not in chosen_soft]
            chosen_soft.extend(random.sample(rem, min(len(rem), need)))
        elif len(chosen_soft) < other_half and len(s1_hard_neg) > len(chosen_hard):
            need = other_half - len(chosen_soft)
            chosen_hard.extend(s1_hard_neg[len(chosen_hard):len(chosen_hard) + need])

        sampled_pairs.extend(chosen_hard)
        sampled_pairs.extend(chosen_soft)
        neg_count += (len(chosen_hard) + len(chosen_soft))

    del s1_candidates
    gc.collect()
    log(f"Pairs collected: {len(sampled_pairs):,} (positives={pos_count:,}, negatives={neg_count:,})")

    # 3. Load full record maps (with embeddings and assertions)
    log("Loading full record maps (parquets + embeddings)...")
    s1_map = load_full_record_map(
        f'{data_dir}/{prefix}_source1_clean.parquet',
        f'{data_dir}/{prefix}_source1_embeddings.npy',
        f'{data_dir}/{prefix}_source1_embed_ids.npy'
    )
    s2_map = load_full_record_map(
        f'{data_dir}/{prefix}_source2_clean.parquet',
        f'{data_dir}/{prefix}_source2_embeddings.npy',
        f'{data_dir}/{prefix}_source2_embed_ids.npy'
    )
    s3_map = load_full_record_map(
        f'{data_dir}/{prefix}_source3_clean.parquet',
        f'{data_dir}/{prefix}_source3_embeddings.npy',
        f'{data_dir}/{prefix}_source3_embed_ids.npy'
    )
    combined_cand_map = {**s2_map, **s3_map}
    del s2_map, s3_map
    gc.collect()

    # 4. Feature Extraction
    log(f"Extracting {len(ALL_FEATURE_NAMES)} features per pair...")
    X_list = []
    y_list = []

    for s1_id, cand_id, sc, label in sampled_pairs:
        if s1_id not in s1_map or cand_id not in combined_cand_map:
            continue
        s1_rec = s1_map[s1_id]
        cand_rec = combined_cand_map[cand_id]

        feats = extract_pair_features(s1_rec, cand_rec, sc)
        X_list.append(feats)
        y_list.append(label)

    del s1_map, combined_cand_map, sampled_pairs
    gc.collect()

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    assert X.shape[1] == len(ALL_FEATURE_NAMES), f"Expected {len(ALL_FEATURE_NAMES)} features, got {X.shape[1]}"
    log(f"Dataset matrix shape: X={X.shape}, y={y.shape}")

    # 5. Stratified 5-Fold Cross Validation
    log(f"Starting {n_folds}-fold Stratified CV...")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(y), dtype=np.float32)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]

        log(f"--- Fold {fold_idx + 1}/{n_folds}: Train={len(y_tr):,}, Val={len(y_va):,} ---")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=31, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
        )
        lgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )

        cb_model = CatBoostClassifier(
            iterations=400, learning_rate=0.05, depth=6, random_seed=42,
            verbose=False, early_stopping_rounds=50, thread_count=-1
        )
        cb_model.fit(X_tr, y_tr, eval_set=(X_va, y_va))

        lgb_va_prob = lgb_model.predict_proba(X_va)[:, 1]
        cb_va_prob = cb_model.predict_proba(X_va)[:, 1]
        oof_probs[val_idx] = (lgb_va_prob + cb_va_prob) / 2.0
        log(f"Fold {fold_idx + 1} complete.")

    # 6. Isotonic Calibration
    log("Fitting Isotonic Calibration on OOF predictions...")
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrated_oof = calibrator.fit_transform(oof_probs, y)

    # Compute uncalibrated vs calibrated best F0.5
    def eval_threshold(probs, labels):
        best_t, best_f = 0.5, 0.0
        for t in np.arange(0.30, 0.991, 0.01):
            preds = (probs >= t).astype(int)
            p = precision_score(labels, preds, zero_division=0)
            r = recall_score(labels, preds, zero_division=0)
            denom = (0.25 * p + r)
            f05 = (1.25 * p * r / denom) if denom > 0 else 0.0
            if f05 > best_f:
                best_f = f05
                best_t = t
        return best_t, best_f

    raw_t, raw_f05 = eval_threshold(oof_probs, y)
    cal_t, cal_f05 = eval_threshold(calibrated_oof, y)

    log(f"Raw OOF Optimal Threshold: {raw_t:.2f}, Raw OOF F0.5: {raw_f05*100:.2f}%")
    log(f"Calibrated OOF Optimal Threshold: {cal_t:.2f}, Calibrated OOF F0.5: {cal_f05*100:.2f}%")

    if (raw_f05 - cal_f05) > 0.003:
        log("Calibration dropped F0.5 by > 0.3pp; using raw probabilities.")
        use_calibrator = None
        best_thresh = raw_t
        best_f05 = raw_f05
    else:
        use_calibrator = calibrator
        best_thresh = cal_t
        best_f05 = cal_f05

    print("\n" + "=" * 70)
    print(f"OUT-OF-FOLD Optimal Threshold: {best_thresh:.2f}, OOF F0.5: {best_f05*100:.2f}%")
    print("NOTE: this OOF F0.5 is a better estimate of true generalization than a single")
    print("80/20 split would give, but it is STILL computed on training-distribution data.")
    print("It is not a substitute for measuring real test-set F0.5 after submission.")
    print("=" * 70 + "\n")

    # 7. Train Final Models on 100% of Data
    log("Training final LightGBM on 100% of data...")
    final_lgb = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=31, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
    )
    final_lgb.fit(X, y)

    log("Training final CatBoost on 100% of data...")
    final_cb = CatBoostClassifier(
        iterations=400, learning_rate=0.05, depth=6, random_seed=42, verbose=False, thread_count=-1
    )
    final_cb.fit(X, y)

    # 8. Save Models & Metadata
    lgb_save_path = os.path.join(models_dir, 'lgb_matcher.txt')
    cb_save_path = os.path.join(models_dir, 'cb_matcher.cbm')
    meta_save_path = os.path.join(models_dir, 'matcher_metadata.pkl')

    final_lgb.booster_.save_model(lgb_save_path)
    final_cb.save_model(cb_save_path)

    metadata = {
        'threshold': float(best_thresh),
        'oof_f05': float(best_f05),
        'features': ALL_FEATURE_NAMES,
        'calibrator': use_calibrator,
        'note': 'OOF F0.5 is on training distribution, not real test set'
    }
    with open(meta_save_path, 'wb') as f:
        pickle.dump(metadata, f)

    log(f"Saved: {lgb_save_path}, {cb_save_path}, and {meta_save_path}")

# Backward-compat alias
train_matcher_v2 = train_matcher

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', default='dataset/train/train_ground_truth.tsv')
    parser.add_argument('--cands', default='output/train_candidate_pairs_detailed.tsv')
    parser.add_argument('--num-entities', type=int, default=50_000)
    parser.add_argument('--neg-ratio', type=int, default=3)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--data-dir', default='dataset/train')
    parser.add_argument('--prefix', default='train')
    parser.add_argument('--models-dir', default='models')
    args = parser.parse_args()
    train_matcher(
        gt_path=args.gt,
        cand_detailed_path=args.cands,
        num_entities_sample=args.num_entities,
        neg_ratio=args.neg_ratio,
        n_folds=args.folds,
        data_dir=args.data_dir,
        prefix=args.prefix,
        models_dir=args.models_dir
    )
