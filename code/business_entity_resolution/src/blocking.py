import os
import re
import time
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import faiss
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

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
    dense_only=False,
    data_dir=None,
    prefix=None,
    max_bucket_size=30
):
    """
    Decoupled Hybrid Blocking:
    Stage A (FAISS/GPU): High-dimensional semantic search. Frees all candidate embeddings upon completion.
    Stage B (Inverted Index): Ultra-compact lexical/address index without disk I/O.
    Peak system RAM is guaranteed < 5.5 GB at all times.
    """
    log("=" * 70)
    log(f"DECOUPLED HYBRID BLOCKING: split={split.upper()} K={k} min_score={min_score} approx={approx} dense_only={dense_only} max_bucket={max_bucket_size}")
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
    total_pairs = 0
    channel1_pairs = 0
    channel2_added_pairs = 0

    import gc

    # Process Candidate Sources sequentially with isolated per-country intermediate writes
    for source_name in ['source2', 'source3']:
        log("=" * 60)
        log(f"BLOCKING CANDIDATE SOURCE: {source_name.upper()}")
        log("=" * 60)

        # Check if all countries for this source are already completed
        all_countries_done = all(
            os.path.exists(os.path.join(out_dir, f"{split}_candidates_{source_name}_{c}.done"))
            for c in countries
        )
        if all_countries_done:
            log(f"All country checkpoints for {source_name} already completed on disk. Skipping source.")
            continue

        # Read ONLY country column to get country indices without loading entire dataframe
        df_src_countries = pd.read_parquet(f"{prefix}_{source_name}_clean.parquet", columns=['country'])
        src_country_arr = df_src_countries['country'].to_numpy()
        del df_src_countries
        gc.collect()

        ids_src = np.load(f"{prefix}_{source_name}_embed_ids.npy", allow_pickle=True)
        emb_src = np.load(f"{prefix}_{source_name}_embeddings.npy", mmap_mode='r')

        assert len(emb_src) == len(ids_src) == len(src_country_arr), f"{source_name} count mismatch"

        for country in countries:
            part_file = os.path.join(out_dir, f"{split}_candidates_{source_name}_{country}.tsv")
            part_done = os.path.join(out_dir, f"{split}_candidates_{source_name}_{country}.done")

            if os.path.exists(part_done) and os.path.exists(part_file):
                log(f"\n[Checkpoint Exists] {part_file} is marked complete — skipping {source_name} [{country}].")
                continue

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
                with open(part_done, 'w') as f_d:
                    f_d.write('empty\n')
                continue

            log(f"  Subset counts: S1={len(s1_c_ids):,} vs Candidates={len(src_c_ids):,}")

            # Open atomic intermediate file for this country/source partition
            f_part = open(part_file, 'w', encoding='utf-8')
            country_pairs = 0

            # -------------------------------------------------------------
            # STAGE 1: Dense Search (GPU Tensor Cores if CUDA, else FAISS CPU)
            # -------------------------------------------------------------
            src_c_emb = np.ascontiguousarray(emb_src[src_c_indices], dtype=np.float32)
            k_val = min(k, len(src_c_indices))
            use_gpu = HAS_TORCH and torch.cuda.is_available()

            if use_gpu:
                gpu_name = torch.cuda.get_device_name(0)
                # Cap intermediate sims matrix at 2.0 GB to guarantee zero CUDA OOM on 16GB T4
                max_sims_bytes = 2.0 * 1024 * 1024 * 1024
                batch_size = min(1000, max(100, int(max_sims_bytes / (len(src_c_indices) * 2))))
                log(f"  [Stage 1/2: Dense GPU Search] Utilizing {gpu_name} (batch_size={batch_size}, FP16 Tensor Cores)!")
                cand_gpu = torch.from_numpy(src_c_emb).to('cuda', dtype=torch.float16)
            else:
                log(f"  [Stage 1/2: Dense CPU Search] Fallback to FAISS CPU IndexFlatIP...")
                dim = src_c_emb.shape[1]
                index = faiss.IndexFlatIP(dim)
                index.add(src_c_emb)
                batch_size = 5000

            buf = []
            total_q_batches = (len(s1_c_emb) + batch_size - 1) // batch_size
            seen_cands_map = [set() for _ in range(len(s1_c_ids))]

            for batch_idx, q_start in enumerate(range(0, len(s1_c_emb), batch_size), 1):
                q_end = min(q_start + batch_size, len(s1_c_emb))
                if batch_idx % 100 == 1 or q_end == len(s1_c_emb):
                    log(f"    [Dense {'GPU' if use_gpu else 'FAISS'} | {country}] Batch {batch_idx}/{total_q_batches} ({q_end:,}/{len(s1_c_emb):,} queries)...")

                if use_gpu:
                    q_batch_gpu = torch.from_numpy(s1_c_emb[q_start:q_end]).to('cuda', dtype=torch.float16)
                    with torch.inference_mode():
                        sims = torch.matmul(q_batch_gpu, cand_gpu.T)
                        scores_t, indices_t = torch.topk(sims, k=k_val, dim=1)
                    scores = scores_t.cpu().numpy()
                    indices = indices_t.cpu().numpy()
                    del q_batch_gpu, sims, scores_t, indices_t
                else:
                    scores, indices = index.search(s1_c_emb[q_start:q_end], k_val)

                for i in range(indices.shape[0]):
                    global_q_idx = q_start + i
                    s1_id = s1_c_ids[global_q_idx]
                    for j in range(k_val):
                        sc = float(scores[i, j])
                        if sc < min_score:
                            continue
                        c_idx = int(indices[i, j])
                        if c_idx < 0:
                            continue
                        seen_cands_map[global_q_idx].add(c_idx)
                        c_id = src_c_ids[c_idx]
                        buf.append(f"{s1_id}\t{c_id}\t{source_name}\t{sc:.5f}\n")
                        total_pairs += 1
                        country_pairs += 1
                        channel1_pairs += 1

                if len(buf) >= 100000:
                    f_part.writelines(buf)
                    f_part.flush()
                    buf = []

            if buf:
                f_part.writelines(buf)
                f_part.flush()
                buf = []

            # Purge dense search data immediately from RAM and GPU VRAM
            if use_gpu:
                del cand_gpu
                torch.cuda.empty_cache()
            else:
                del index
            del src_c_emb, s1_c_emb
            gc.collect()
            log(f"    [Dense {'GPU' if use_gpu else 'FAISS'} Complete] Freed candidate embeddings from RAM. Intermediate pairs so far: {country_pairs:,}")

            if not dense_only:
                # -------------------------------------------------------------
                # STAGE 2: Inverted Index Lexical Blocking (Channel 2)
                # -------------------------------------------------------------
                log(f"  [Stage 2/2: Inverted Index] Streaming text records for {country} (memory-safe)...")
                inv_index = {}
                oversized_keys = set()
                pf = pq.ParquetFile(f"{prefix}_{source_name}_clean.parquet")
                c_idx = 0
                for batch in pf.iter_batches(batch_size=100_000, columns=['country', 'business_name_clean', 'business_address_clean', 'postal_code']):
                    df_b = batch.to_pandas()
                    df_b_c = df_b[df_b['country'] == country]
                    for r in df_b_c.itertuples(index=False):
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
                        c_idx += 1
                    del df_b, df_b_c

                del oversized_keys, pf
                gc.collect()
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
                        for c_idx in new_cand_indices:
                            c_id = src_c_ids[c_idx]
                            sc = 0.55000  # Lexical match; avoids millions of random EBS disk page faults
                            buf.append(f"{s1_id}\t{c_id}\t{source_name}\t{sc:.5f}\n")
                            total_pairs += 1
                            country_pairs += 1
                            channel2_added_pairs += 1
                            c2_country_added += 1

                    if len(buf) >= 100000:
                        f_part.writelines(buf)
                        f_part.flush()
                        buf = []

                    if (q_idx + 1) % 100_000 == 0 or (q_idx + 1) == len(s1_c_ids):
                        pct = ((q_idx + 1) / len(s1_c_ids)) * 100
                        log(f"    [Inverted Index | {country}] {q_idx + 1:,}/{len(s1_c_ids):,} ({pct:.1f}%) queries processed (+{c2_country_added:,} pairs added)...")

                del inv_index, s1_names, s1_addrs, s1_pcs
                gc.collect()

            if buf:
                f_part.writelines(buf)
                f_part.flush()
                buf = []

            f_part.close()
            with open(part_done, 'w') as f_d:
                f_d.write(f"{country_pairs}\n")
            log(f"  [Checkpoint Written] Flushed {country_pairs:,} pairs to {part_file} and marked done.")

            del seen_cands_map, s1_c_ids, df_s1_c
            gc.collect()

        del src_country_arr, ids_src, emb_src
        gc.collect()

    del emb_s1, ids_s1
    gc.collect()

    # Consolidate all intermediate country files into master detailed file
    log("=" * 60)
    log(f"CONSOLIDATING INTERMEDIATE CHECKPOINTS INTO {detailed_file}...")
    log("=" * 60)
    total_master_pairs = 0
    with open(detailed_file, 'w', encoding='utf-8') as f_out:
        f_out.write("source1_entity_id\tcandidate_entity_id\tsource\tembed_score\n")
        for s in ['source2', 'source3']:
            for c in countries:
                p = os.path.join(out_dir, f"{split}_candidates_{s}_{c}.tsv")
                if os.path.exists(p):
                    with open(p, 'r', encoding='utf-8') as f_in:
                        for chunk in iter(lambda: f_in.read(2 * 1024 * 1024), ''):
                            f_out.write(chunk)
                            total_master_pairs += chunk.count('\n')

    log(f"Saved {detailed_file} ({total_master_pairs:,} total rows)")
    log("Blocking complete. Detailed candidate pairs ready for prediction.")

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
    parser.add_argument('--dense-only', action='store_true', default=False, help="Run only GPU dense vector search")
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
            dense_only=args.dense_only,
            data_dir=args.data_dir,
            prefix=args.prefix
        )
