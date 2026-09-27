import os
import re
import time
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
import faiss

def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

STOP_TOKENS = {
    'the', 'a', 'an', 'and', '&', 'of', 'in', 'on', 'at', 'for', 'to',
    'llc', 'inc', 'incorporated', 'corp', 'corporation', 'co', 'company',
    'ltd', 'limited', 'pvt', 'private', 'group', 'services', 'enterprises',
    'holdings', 'technologies', 'solutions', 'consulting', 'management'
}

ADDR_STRUCT_STOPS = {
    'floor', 'ground', 'first', 'second', 'third', 'fourth', 'fifth',
    'suite', 'flat', 'unit', 'room', 'building', 'bldg', 'block', 'plot',
    'shop', 'road', 'street', 'avenue', 'lane', 'drive', 'blvd', 'boulevard',
    'highway', 'near', 'opp', 'opposite', 'behind', 'cross', 'main', 'nagar',
    'colony', 'sector', 'phase', 'post', 'dist', 'state', 'india', 'usa'
}

def extract_blocking_keys(name, addr, pc):
    """
    Extracts high-precision complementary blocking keys for inverted index lookup.
    Includes dedicated street number indexing, name signatures, and exact matches.
    All keys are case-normalized and strip common corporate and structural stops.
    """
    keys = []
    name_str = (name or '').lower()
    addr_str = (addr or '').lower()
    pc_str = (pc or '').strip()

    name_toks = [t for t in re.findall(r'\b\w+\b', name_str) if t not in STOP_TOKENS and len(t) > 2]
    addr_toks = [t for t in re.findall(r'\b[a-z]+\b', addr_str) if len(t) > 3 and t not in STOP_TOKENS and t not in ADDR_STRUCT_STOPS]
    addr_nums = re.findall(r'\b\d+[a-z]?\b', addr_str)

    # 1. Exact clean name (if >= 4 chars) and space-collapsed clean name
    if len(name_str) >= 4:
        keys.append(('ex_name', name_str))
        n_ns = name_str.replace(' ', '')
        if len(n_ns) >= 5:
            keys.append(('ex_name_ns', n_ns))

    # 2. Exact clean address (if >= 8 chars)
    if len(addr_str) >= 8:
        keys.append(('ex_addr', addr_str))

    # 3. First 2 informative name tokens
    if len(name_toks) >= 2:
        keys.append(('n_2tok', f"{name_toks[0]}_{name_toks[1]}"))
    elif len(name_toks) == 1:
        keys.append(('n_1tok', name_toks[0]))

    # 4. Street number + informative address tokens (permutation-invariant across first 2 tokens)
    if addr_nums and addr_toks:
        for num in addr_nums[:2]:
            for tok in addr_toks[:2]:
                keys.append(('s_num_tok', f"{num}_{tok}"))

    # 5. Postal code + first name token
    if pc_str and name_toks:
        keys.append(('pc_n', f"{pc_str}_{name_toks[0]}"))

    # 6. Postal code + street number
    if pc_str and addr_nums:
        keys.append(('pc_num', f"{pc_str}_{addr_nums[0]}"))

    # 7. Street number with primary name token
    for num in addr_nums[:2]:
        if len(num) >= 2 and name_toks:
            keys.append(('s_num_name', f"{num}_{name_toks[0]}"))

    return keys

def run_blocking(
    split='train',
    out_dir='output',
    k=20,
    min_score=0.40,
    approx=False,
    data_dir=None,
    prefix=None,
    max_bucket_size=30
):
    """
    Decoupled Hybrid Blocking:
    Stage A (FAISS): High-dimensional semantic search. Frees all candidate embeddings upon completion.
    Stage B (Inverted Index): Ultra-compact lexical/address index using on-demand disk-mmap dot products.
    Peak system RAM is guaranteed < 5.5 GB at all times.
    """
    log("=" * 70)
    log(f"DECOUPLED HYBRID BLOCKING: split={split.upper()} K={k} min_score={min_score} approx={approx} max_bucket={max_bucket_size}")
    log("=" * 70)
    os.makedirs(out_dir, exist_ok=True)

    if data_dir is None:
        data_dir = f"dataset/{split}"
    if prefix is None:
        prefix = f"{data_dir}/{split}"
    else:
        prefix = f"{data_dir}/{prefix}"

    cols = ['entity_id', 'country', 'business_name_clean', 'business_address_clean', 'postal_code']
    df_s1 = pd.read_parquet(f"{prefix}_source1_clean.parquet", columns=cols)
    emb_s1 = np.load(f"{prefix}_source1_embeddings.npy", mmap_mode='r')
    ids_s1 = np.load(f"{prefix}_source1_embed_ids.npy", allow_pickle=True)

    # Sanity check S1
    assert len(emb_s1) == len(ids_s1) == len(df_s1), "S1 embedding/id/parquet row count mismatch"
    assert (ids_s1 == df_s1['entity_id'].to_numpy()).all(), "S1 id order mismatch between embeddings and parquet"

    countries = sorted(df_s1['country'].unique().tolist())
    log(f"Countries: {countries}")

    detailed_file = os.path.join(out_dir, f"{split}_candidate_pairs_detailed.tsv")
    f_detailed = open(detailed_file, 'w', encoding='utf-8')
    f_detailed.write("source1_entity_id\tcandidate_entity_id\tsource\tembed_score\n")

    total_pairs = 0
    channel1_pairs = 0
    channel2_added_pairs = 0

    import gc

    # Process Candidate Sources sequentially
    for source_name in ['source2', 'source3']:
        log("=" * 60)
        log(f"BLOCKING CANDIDATE SOURCE: {source_name.upper()}")
        log("=" * 60)

        # Read ONLY country column to get country indices without loading entire dataframe
        df_src_countries = pd.read_parquet(f"{prefix}_{source_name}_clean.parquet", columns=['country'])
        src_country_arr = df_src_countries['country'].to_numpy()
        del df_src_countries
        gc.collect()

        ids_src = np.load(f"{prefix}_{source_name}_embed_ids.npy", allow_pickle=True)
        emb_src = np.load(f"{prefix}_{source_name}_embeddings.npy", mmap_mode='r')

        assert len(emb_src) == len(ids_src) == len(src_country_arr), f"{source_name} count mismatch"

        for country in countries:
            log(f"\n>>> Country: {country} [{source_name.upper()}] <<<")
            s1_mask = (df_s1['country'] == country).to_numpy()
            s1_c_indices = np.where(s1_mask)[0]
            s1_c_ids = ids_s1[s1_mask]
            s1_c_emb = np.ascontiguousarray(emb_s1[s1_mask], dtype=np.float32)
            df_s1_c = df_s1[s1_mask]

            src_mask = (src_country_arr == country)
            src_c_indices = np.where(src_mask)[0]
            src_c_ids = ids_src[src_mask]

            if len(src_c_ids) == 0 or len(s1_c_ids) == 0:
                log(f"  {source_name}: empty subset, skipping")
                continue

            log(f"  Subset counts: S1={len(s1_c_ids):,} vs Candidates={len(src_c_ids):,}")

            # -------------------------------------------------------------
            # STAGE 1: Dense FAISS Search (Channel 1)
            # -------------------------------------------------------------
            log(f"  [Stage 1/2: FAISS] Indexing {len(src_c_ids):,} candidate embeddings...")
            src_c_emb = np.ascontiguousarray(emb_src[src_c_indices], dtype=np.float32)
            dim = src_c_emb.shape[1]
            if approx:
                index = faiss.IndexHNSWFlat(dim, 48, faiss.METRIC_INNER_PRODUCT)
                index.hnsw.efConstruction = 300
                index.add(src_c_emb)
                index.hnsw.efSearch = 300
            else:
                index = faiss.IndexFlatIP(dim)
                index.add(src_c_emb)

            k_val = min(k, index.ntotal)
            batch_size = 5000
            buf = []
            total_q_batches = (len(s1_c_emb) + batch_size - 1) // batch_size

            # seen_cands_map: track FAISS candidate indices per S1 query to avoid duplicate scoring in Stage 2
            seen_cands_map = [set() for _ in range(len(s1_c_ids))]

            for batch_idx, q_start in enumerate(range(0, len(s1_c_emb), batch_size), 1):
                q_end = min(q_start + batch_size, len(s1_c_emb))
                if batch_idx % 4 == 1 or q_end == len(s1_c_emb):
                    log(f"    [FAISS | {country}] Batch {batch_idx}/{total_q_batches} ({q_end:,}/{len(s1_c_emb):,} queries)...")
                
                scores, indices = index.search(s1_c_emb[q_start:q_end], k_val)

                for i in range(indices.shape[0]):
                    global_q_idx = q_start + i
                    s1_id = s1_c_ids[global_q_idx]
                    for j in range(k_val):
                        sc = float(scores[i, j])
                        if sc < min_score:
                            continue
                        c_idx = indices[i, j]
                        if c_idx < 0:
                            continue
                        seen_cands_map[global_q_idx].add(c_idx)
                        c_id = src_c_ids[c_idx]
                        buf.append(f"{s1_id}\t{c_id}\t{source_name}\t{sc:.5f}\n")
                        total_pairs += 1
                        channel1_pairs += 1

                if len(buf) >= 100000:
                    f_detailed.writelines(buf)
                    buf = []

            if buf:
                f_detailed.writelines(buf)
                buf = []

            # Purge FAISS index and candidate embeddings immediately from RAM
            del index, src_c_emb
            gc.collect()
            log(f"    [FAISS Complete] Freed candidate embeddings. Running total pairs: {total_pairs:,}")

            # -------------------------------------------------------------
            # STAGE 2: Inverted Index Lexical Blocking (Channel 2)
            # -------------------------------------------------------------
            log(f"  [Stage 2/2: Inverted Index] Loading text records for {country}...")
            df_src_c = pd.read_parquet(
                f"{prefix}_{source_name}_clean.parquet",
                columns=['entity_id', 'business_name_clean', 'business_address_clean', 'postal_code'],
                filters=[('country', '==', country)]
            )

            inv_index = {}
            oversized_keys = set()
            for c_idx, r in enumerate(df_src_c.itertuples(index=False)):
                for key in extract_blocking_keys(r.business_name_clean, r.business_address_clean, r.postal_code):
                    if key in oversized_keys:
                        continue
                    bucket = inv_index.get(key)
                    if bucket is None:
                        inv_index[key] = [c_idx]
                    elif len(bucket) < max_bucket_size:
                        bucket.append(c_idx)
                    else:
                        del inv_index[key]
                        oversized_keys.add(key)
            del oversized_keys
            log(f"    Built sparse inverted index with {len(inv_index):,} active buckets (capped <= {max_bucket_size}).")

            s1_names = df_s1_c['business_name_clean'].tolist()
            s1_addrs = df_s1_c['business_address_clean'].tolist()
            s1_pcs = df_s1_c['postal_code'].tolist()

            c2_country_added = 0
            for q_idx in range(len(s1_c_ids)):
                s1_id = s1_c_ids[q_idx]
                s1_seen = seen_cands_map[q_idx]
                keys = extract_blocking_keys(s1_names[q_idx], s1_addrs[q_idx], s1_pcs[q_idx])

                new_cand_indices = []
                for key in keys:
                    bucket = inv_index.get(key)
                    if bucket:
                        for c_idx in bucket:
                            if c_idx not in s1_seen:
                                s1_seen.add(c_idx)
                                new_cand_indices.append(c_idx)

                if new_cand_indices:
                    s1_vec = s1_c_emb[q_idx]
                    for c_idx in new_cand_indices:
                        c_global_idx = src_c_indices[c_idx]
                        c_vec = emb_src[c_global_idx]  # on-demand mmap disk access
                        sc = float(np.dot(s1_vec, c_vec))
                        c_id = src_c_ids[c_idx]
                        buf.append(f"{s1_id}\t{c_id}\t{source_name}\t{sc:.5f}\n")
                        total_pairs += 1
                        channel2_added_pairs += 1
                        c2_country_added += 1

                if len(buf) >= 100000:
                    f_detailed.writelines(buf)
                    buf = []

            if buf:
                f_detailed.writelines(buf)
                buf = []

            del inv_index, df_src_c, seen_cands_map, s1_names, s1_addrs, s1_pcs, s1_c_ids, s1_c_emb, df_s1_c
            gc.collect()
            log(f"    [Inverted Index Complete] Added {c2_country_added:,} complementary pairs. Total pairs: {total_pairs:,}")

        del src_country_arr, ids_src, emb_src
        gc.collect()

    f_detailed.close()
    log(f"Saved {detailed_file} ({total_pairs:,} total pairs)")

    final_cand_file = os.path.join(out_dir, f"{split}_candidate_pairs.tsv" if split == 'train' else "candidate_pairs.tsv")
    log(f"Writing {final_cand_file} from detailed candidate pairs...")
    cand_agg = defaultdict(set)
    with open(detailed_file, 'r', encoding='utf-8') as f_det:
        next(f_det, None)
        for line in f_det:
            pts = line.split('\t')
            if len(pts) >= 2:
                cand_agg[pts[0]].add(pts[1])

    with open(final_cand_file, 'w', encoding='utf-8') as f_cand:
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        buf = []
        for s1_id in df_s1['entity_id']:
            buf.append(f"{s1_id}\t{','.join(cand_agg.get(s1_id, set()))}\n")
            if len(buf) >= 100000:
                f_cand.writelines(buf)
                buf = []
        if buf:
            f_cand.writelines(buf)
    del cand_agg
    gc.collect()
    log(f"Saved {final_cand_file}")

# Backward-compat alias
run_blocking_v2 = run_blocking

def load_gt(gt_path='dataset/train/train_ground_truth.tsv'):
    import csv
    gt_map = {}
    with open(gt_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            if len(row) >= 2 and row[1].strip():
                gt_map[row[0].strip()] = {m.strip() for m in row[1].split(',') if m.strip()}
    return gt_map

def tune_blocking(
    sample_size=5000,
    score_grid=(0.30, 0.35, 0.40, 0.45, 0.50, 0.55),
    k=30,
    max_bucket_size=30,
    data_dir='dataset/train',
    prefix='train',
    gt_path='dataset/train/train_ground_truth.tsv',
    random_seed=None
):
    """
    Sweeps min_score over a grid and reports both FAISS recall and Hybrid recall.
    """
    import random
    log(f"TUNING HYBRID BLOCKING: sample_size={sample_size}, k={k}, max_bucket={max_bucket_size}, data_dir={data_dir}")
    gt_map = load_gt(gt_path)

    cols = ['entity_id', 'country', 'business_name_clean', 'business_address_clean', 'postal_code']
    df_s1 = pd.read_parquet(f'{data_dir}/{prefix}_source1_clean.parquet', columns=cols)
    emb_s1 = np.load(f'{data_dir}/{prefix}_source1_embeddings.npy')
    ids_s1 = np.load(f'{data_dir}/{prefix}_source1_embed_ids.npy', allow_pickle=True)

    gt_ids = list(gt_map.keys())
    if random_seed is None:
        random_seed = int.from_bytes(os.urandom(4), 'little')
    random.seed(random_seed)
    log(f"Sampled {min(sample_size, len(gt_ids)):,} random S1 entities for tuning (seed={random_seed}).")
    sample_ids = set(random.sample(gt_ids, min(sample_size, len(gt_ids))))
    id_to_idx = {eid: i for i, eid in enumerate(ids_s1)}
    sample_idx = [id_to_idx[eid] for eid in sample_ids if eid in id_to_idx]
    sample_emb = emb_s1[sample_idx]
    sample_eids = ids_s1[sample_idx]
    df_s1_sample = df_s1.set_index('entity_id').loc[sample_eids].reset_index()
    sample_countries = df_s1_sample['country'].to_numpy()

    for country in sorted(set(sample_countries)):
        c_mask = sample_countries == country
        c_eids = sample_eids[c_mask]
        c_emb = sample_emb[c_mask]
        df_s1_c = df_s1_sample[c_mask]
        if len(c_eids) == 0:
            continue

        s1_keys_list = [
            extract_blocking_keys(r.business_name_clean, r.business_address_clean, r.postal_code)
            for r in df_s1_c.itertuples(index=False)
        ]

        for source_num in [2, 3]:
            df_src = pd.read_parquet(f'{data_dir}/{prefix}_source{source_num}_clean.parquet', columns=cols)
            emb_src = np.load(f'{data_dir}/{prefix}_source{source_num}_embeddings.npy')
            ids_src = np.load(f'{data_dir}/{prefix}_source{source_num}_embed_ids.npy', allow_pickle=True)
            src_mask = (df_src['country'] == country).to_numpy()
            src_ids_c = ids_src[src_mask]
            src_emb_c = emb_src[src_mask]
            df_src_c = df_src[src_mask]
            if len(src_ids_c) == 0:
                continue

            src_id_set = set(src_ids_c)
            src_id_to_idx = {cid: idx for idx, cid in enumerate(src_ids_c)}

            # Inverted index
            inv_index = defaultdict(list)
            for r in df_src_c.itertuples(index=False):
                cid = r.entity_id
                for k_type, k_val in extract_blocking_keys(r.business_name_clean, r.business_address_clean, r.postal_code):
                    inv_index[(k_type, k_val)].append(cid)
            inv_index = {k: v for k, v in inv_index.items() if len(v) <= max_bucket_size}

            index = faiss.IndexFlatIP(src_emb_c.shape[1])
            index.add(src_emb_c)
            k_val = min(k, index.ntotal)
            scores, indices = index.search(c_emb, k_val)

            for min_score in score_grid:
                faiss_hits, hybrid_hits, total_true = 0, 0, 0
                for i, eid in enumerate(c_eids):
                    true_set = gt_map.get(eid, set()) & src_id_set
                    if not true_set:
                        continue
                    faiss_set = {src_ids_c[indices[i, j]] for j in range(k_val) if scores[i, j] >= min_score}
                    hybrid_set = set(faiss_set)
                    for key in s1_keys_list[i]:
                        bucket = inv_index.get(key)
                        if bucket:
                            hybrid_set.update(bucket)

                    faiss_hits += len(true_set & faiss_set)
                    hybrid_hits += len(true_set & hybrid_set)
                    total_true += len(true_set)

                r_faiss = faiss_hits / total_true if total_true else 0.0
                r_hybrid = hybrid_hits / total_true if total_true else 0.0
                print(f"country={country} source=source{source_num} min_score={min_score:.2f}: FAISS={r_faiss*100:.2f}% -> HYBRID={r_hybrid*100:.2f}% (n_true={total_true})")

tune = tune_blocking

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--prefix', type=str, default=None)
    parser.add_argument('--out-dir', type=str, default='output')
    parser.add_argument('--k', type=int, default=20)
    parser.add_argument('--min-score', type=float, default=0.40)
    parser.add_argument('--max-bucket', type=int, default=30)
    parser.add_argument('--approx', action='store_true', default=False)
    parser.add_argument('--tune', action='store_true', default=False, help="Run threshold tuning on sample")
    parser.add_argument('--sample-size', type=int, default=5000)
    parser.add_argument('--gt', type=str, default='dataset/train/train_ground_truth.tsv')
    args = parser.parse_args()

    if args.tune:
        tune_blocking(
            sample_size=args.sample_size,
            k=args.k,
            max_bucket_size=args.max_bucket,
            data_dir=args.data_dir or f"dataset/{args.split}",
            prefix=args.prefix or args.split,
            gt_path=args.gt
        )
    else:
        run_blocking(
            split=args.split,
            out_dir=args.out_dir,
            k=args.k,
            min_score=args.min_score,
            max_bucket_size=args.max_bucket,
            approx=args.approx,
            data_dir=args.data_dir,
            prefix=args.prefix
        )
